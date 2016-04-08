#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "expr-xtra.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace cnn;
using namespace std;
using namespace boost::program_options;

template <class Builder>
struct WordContextModel {
    explicit WordContextModel(Model& model, unsigned layers, unsigned vocab_size, 
	    unsigned hidden_dim, unsigned stochastic_dim, unsigned embedding_dim, unsigned window);
	    
    Expression build_graph(const vector<int> &sentence, ComputationGraph& cg, 
	    unsigned num_samples, unsigned &events);

    Expression build_graph_deterministic(const vector<int> &sentence, 
	    ComputationGraph &cg, unsigned &num_events);

    LookupParameters* p_C;	    // word embeddings -- FIXME: add character n-gram embeddings also
    Parameters* p_Wh1, *p_bh1;	    // mapping from word embedding to stochastic unit, layer 1
    Parameters* p_Wh2, *p_bh2;	    // layer 2 of the above, to form stochastic hidden outputs "h"
    Parameters* p_Whl, *p_bl;	    // mapping from h to left RNN
    Parameters* p_Whr, *p_br;	    // mapping from h to right RNN
    Parameters* p_Wo, *p_bo;	    // output embeddings, shared between right and left RNNs
    Builder builder_left;
    Builder builder_right;
    unsigned window;

    vector<vector<float>> sample_vecs;
};

#define WTF(expression) \
    cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << endl;
#define KTHXBYE(expression) \
    cout << *cg.get_value(expression.i) << endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

template <class Builder>
WordContextModel<Builder>::WordContextModel(Model& model, unsigned layers, unsigned vocab_size, 
	    unsigned hidden_dim, unsigned stochastic_dim, unsigned embedding_dim, unsigned _window)
  : builder_left(layers, embedding_dim, hidden_dim, &model),
    builder_right(layers, embedding_dim, hidden_dim, &model),
    window(_window)
{
    p_C = model.add_lookup_parameters(vocab_size, {embedding_dim}); 

    p_Wh1 = model.add_parameters({hidden_dim, embedding_dim});
    p_bh1 = model.add_parameters({hidden_dim});

    p_Wh2 = model.add_parameters({stochastic_dim, hidden_dim});
    p_bh2 = model.add_parameters({stochastic_dim});

    // might need #layers of these:
    p_Whl = model.add_parameters({hidden_dim, stochastic_dim});
    p_bl = model.add_parameters({hidden_dim});

    // might need #layers of these:
    p_Whr = model.add_parameters({hidden_dim, stochastic_dim});
    p_br = model.add_parameters({hidden_dim});

    p_Wo = model.add_parameters({vocab_size, hidden_dim});
    p_bo = model.add_parameters({vocab_size});
}

template <class Builder>
Expression WordContextModel<Builder>::build_graph(const vector<int> &sentence, ComputationGraph &cg, 
	unsigned num_samples, unsigned &num_events)
{
    Expression i_Wh1 = parameter(cg, p_Wh1);
    Expression i_bh1 = parameter(cg, p_bh1);
    Expression i_Wh2 = parameter(cg, p_Wh2);
    Expression i_bh2 = parameter(cg, p_bh2);
    Expression i_Whl = parameter(cg, p_Whl);
    Expression i_bl = parameter(cg, p_bl);
    Expression i_Whr = parameter(cg, p_Whr);
    Expression i_br = parameter(cg, p_br);
    Expression i_Wo = parameter(cg, p_Wo);
    Expression i_bo = parameter(cg, p_bo);

    vector<Expression> word_embeddings;
    for (auto token: sentence)
	word_embeddings.push_back(lookup(cg, p_C, token));

    auto slen = sentence.size(); 
    vector<Expression> errs;
    for (unsigned i=1; i < slen-1; ++i) {
	builder_left.new_graph(cg);
	builder_right.new_graph(cg);

	Expression &x = word_embeddings[i];
	Expression h1 = tanh(i_Wh1 * x + i_bh1);
	Expression h2 = logistic(i_Wh2 * h1 + i_bh2);

	// draw several samples from h2
	auto hdist = as_vector(cg.incremental_forward());
	vector<Expression> samples;
	vector<Expression> sample_errs;
	sample_vecs.resize(num_samples);
	for (unsigned s=0; s < num_samples; ++s) {
	    // draw vector of reals [0, 1] and threshold against hdist
	    Expression hs;
	    {
		vector<float> rands(hdist.size());
		uniform_real_distribution<float> distribution(0, 1);
		auto b = [&] {return distribution(*rndeng);};
		generate(rands.begin(), rands.end(), b);

		vector<float> &h_vec = sample_vecs[s];
		h_vec.resize(hdist.size());
		for (unsigned k=0; k < hdist.size(); ++k) 
		    h_vec[k] = (rands[k] < hdist[k]) ? 1 : 0;

		hs = input(cg, Dim({(int)h_vec.size()}), &h_vec);
	    }

	    // score p(y | h), i.e., using full model
	    Expression es;
	    {
		vector<Expression> local_errs;
		// FIXME: need to ensure this matches #layers and double that for LSTM
		Expression xl = tanh(i_Whl * hs + i_bl);
		builder_left.start_new_sequence({xl});
		for (int j=i; j >= max(1, (int)i-(int)window); --j) {
		    Expression yj = builder_left.add_input(word_embeddings[j]);
		    Expression rj = softmax(i_Wo * yj + i_bo);
		    Expression err = pickneglogsoftmax(rj, sentence[j-1]);
		    local_errs.push_back(err);
		}

		// FIXME: need to ensure this matches #layers and double that for LSTM
		Expression xr = tanh(i_Whr * hs + i_br);
		builder_right.start_new_sequence({xr});
		auto slen = sentence.size(); 
		for (int j=i; j <= min((int)slen-2, (int)i+(int)window); ++j) {
		    Expression yj = builder_right.add_input(word_embeddings[j]);
		    Expression rj = softmax(i_Wo * yj + i_bo);
		    Expression err = pickneglogsoftmax(rj, sentence[j+1]);
		    local_errs.push_back(err);
		}
		es = sum(local_errs);
		if (s == 0) num_events += local_errs.size();
	    }

	    samples.push_back(hs);
	    sample_errs.push_back(es);
	}
	Expression ws = softmax(concatenate(sample_errs));

	// add error terms for p(h | x) and p(y | h), weighted by ws
	for (int s=0; s < num_samples; ++s) {
	    Expression w = pick(ws, s);
	    Expression herr = w * binary_log_loss(h2, samples[s]);
	    errs.push_back(herr);
	    Expression yerr = w * sample_errs[s];
	    errs.push_back(yerr);
	}
    }

    Expression nerr = sum(errs);
    return nerr;
}

template <class Builder>
Expression WordContextModel<Builder>::build_graph_deterministic(const vector<int> &sentence, 
	ComputationGraph &cg, unsigned &num_events)
{
    Expression i_Wh1 = parameter(cg, p_Wh1);
    Expression i_bh1 = parameter(cg, p_bh1);
    Expression i_Wh2 = parameter(cg, p_Wh2);
    Expression i_bh2 = parameter(cg, p_bh2);
    Expression i_Whl = parameter(cg, p_Whl);
    Expression i_bl = parameter(cg, p_bl);
    Expression i_Whr = parameter(cg, p_Whr);
    Expression i_br = parameter(cg, p_br);
    Expression i_Wo = parameter(cg, p_Wo);
    Expression i_bo = parameter(cg, p_bo);

    vector<Expression> word_embeddings;
    for (auto token: sentence)
	word_embeddings.push_back(lookup(cg, p_C, token));

    auto slen = sentence.size(); 
    vector<Expression> errs;
    for (unsigned i=1; i < slen-1; ++i) {
	builder_left.new_graph(cg);
	builder_right.new_graph(cg);

	Expression &x = word_embeddings[i];
	Expression h1 = tanh(i_Wh1 * x + i_bh1);
	Expression h2 = logistic(i_Wh2 * h1 + i_bh2);

	// FIXME: need to ensure this matches #layers and double that for LSTM
	Expression xl = tanh(i_Whl * h2 + i_bl);
	builder_left.start_new_sequence({xl});
	for (int j=i; j >= max(1, (int)i-(int)window); --j) {
	    Expression yj = builder_left.add_input(word_embeddings[j]);
	    Expression rj = softmax(i_Wo * yj + i_bo);
	    Expression err = pickneglogsoftmax(rj, sentence[j-1]);
	    errs.push_back(err);
	}

	// FIXME: need to ensure this matches #layers and double that for LSTM
	Expression xr = tanh(i_Whr * h2 + i_br);
	builder_right.start_new_sequence({xr});
	auto slen = sentence.size(); 
	for (int j=i; j <= min((int)slen-2, (int)i+(int)window); ++j) {
	    Expression yj = builder_right.add_input(word_embeddings[j]);
	    Expression rj = softmax(i_Wo * yj + i_bo);
	    Expression err = pickneglogsoftmax(rj, sentence[j+1]);
	    errs.push_back(err);
	}
    }

    num_events += errs.size();
    Expression nerr = sum(errs);
    return nerr;
}

template <class rnn_t>
int main_body(variables_map vm);

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);

    // command line processing
    variables_map vm; 
    options_description opts("Allowed options");
    opts.add_options()
        ("help", "print help message")
        ("config,c", value<string>(), "config file specifying additional command line options")
        ("train,t", value<string>(), "file containing training sentences")
        ("devel,d", value<string>(), "file containing development sentences")
        ("layers,l", value<unsigned>()->default_value(1), "use <num> layers for hidden components")
        ("embedding,e", value<unsigned>()->default_value(32), "use <num> dimensions for word embeddings")
        ("stochastic,s", value<unsigned>()->default_value(16), "use <num> dimensions for stochastic units")
        ("hidden,h", value<unsigned>()->default_value(64), "use <num> dimensions for recurrent hidden states")
        ("window,w", value<unsigned>()->default_value(5), "predict up to <num> words to left and right")
        ("samples,S", value<unsigned>()->default_value(10), "number of stochastic samples used in E step")
        ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
        ("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
        ("verbose,v", "be extremely chatty")
    ;
    store(parse_command_line(argc, argv, opts), vm); 
    if (vm.count("config") > 0)
    {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, opts), vm); 
    }
    notify(vm);
    
    if (vm.count("help") || vm.count("train") != 1 || vm.count("devel") != 1) {
        cout << opts << "\n";
        return 1;
    }

    if (vm.count("lstm"))
	return main_body<LSTMBuilder>(vm);
    else if (vm.count("gru"))
	return main_body<GRUBuilder>(vm);
    else
	return main_body<SimpleRNNBuilder>(vm);
}

typedef vector<vector<int>> Corpus;
Corpus read_corpus(const string &filename, cnn::Dict &dict);

template <class rnn_t>
int main_body(variables_map vm)
{
    Dict dict;
    bool verbose = vm.count("verbose");

    string line;
    cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
    Corpus training = read_corpus(vm["train"].as<string>(), dict);
    dict.Freeze(); // no new word types allowed

    unsigned LAYERS = vm["layers"].as<unsigned>(); 
    unsigned EMBEDDING_DIM = vm["embedding"].as<unsigned>(); 
    unsigned STOCHASTIC_DIM = vm["stochastic"].as<unsigned>(); 
    unsigned HIDDEN_DIM = vm["hidden"].as<unsigned>(); 
    unsigned window = vm["window"].as<unsigned>();
    unsigned num_samples = vm["samples"].as<unsigned>();
    unsigned VOCAB_SIZE = dict.size();

    string flavour = "RNN";
    if (vm.count("lstm"))	flavour = "LSTM";
    else if (vm.count("gru"))	flavour = "GRU";

    cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
    Corpus devel = read_corpus(vm["devel"].as<string>(), dict);

    ostringstream os;
    os << "wcm"
	<< '_' << LAYERS
	<< '_' << HIDDEN_DIM
	<< '_' << EMBEDDING_DIM
	<< '_' << STOCHASTIC_DIM
	<< '_' << window
	<< '_' << flavour
	<< "-pid" << getpid() << ".params";
    string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;

    Model model;
    Trainer* sgd = new SimpleSGDTrainer(&model);

    cerr << "%% Using " << flavour << " recurrent units" << endl;
    WordContextModel<rnn_t> wcm(model, LAYERS, VOCAB_SIZE, HIDDEN_DIM, STOCHASTIC_DIM, EMBEDDING_DIM, window);

    double best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 500; 
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    bool first = true;
    int report = 0;
    unsigned lines = 0;

    while(1) {
        Timer iteration("completed in");
        double loss = 0;
        unsigned chars = 0;

        for (unsigned iter = 0; iter < report_every_i; ++iter) {
	    if (si == training.size()) {
		si = 0;
		if (first) { first = false; } else { sgd->update_epoch(); }
		if (verbose) cerr << "**SHUFFLE\n";
		shuffle(order.begin(), order.end(), *rndeng);
	    }

            // build graph for this instance
	    const auto& sent = training[order[si]];
	    ComputationGraph cg;
            ++si;
	    if (first) // skip over the slow sampling for the first epoch
		wcm.build_graph_deterministic(sent, cg, chars);
	    else
		wcm.build_graph(sent, cg, num_samples, chars);
            loss += as_scalar(cg.forward());
            
            cg.backward();
            sgd->update();
            ++lines;

	    if (verbose)
		cerr << "chug " << iter << "\r" << flush;
        }
        sgd->status();
        cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0;
            unsigned dchars = 0;
            for (auto& sent : devel) {
                ComputationGraph cg;
                wcm.build_graph(sent, cg, num_samples, dchars);
                dloss += as_scalar(cg.forward());
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }
    }

    delete sgd;

    return EXIT_SUCCESS;
}

Corpus read_corpus(const string &filename, Dict &dict)
{
    int kSOS = dict.Convert("<s>");
    int kEOS = dict.Convert("</s>");

    ifstream in(filename);
    assert(in);
    Corpus corpus;
    string line;
    int lc = 0, toks = 0;
    while(getline(in, line)) {
	//cerr << "line: '" << line << "'" << endl;
        ++lc;
        vector<int> sent = ReadSentence(line, &dict);
	if (sent.empty()) continue; // skip empty lines
        corpus.push_back(sent);
        toks += sent.size();

        if (sent.front() != kSOS && sent.back() != kEOS) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
    }
    cerr << lc << " lines, " << toks << " tokens " << dict.size() << " types\n";
    return corpus;
}

