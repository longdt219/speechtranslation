#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
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

Dict dict;

struct WordContextModel {
    explicit WordContextModel(Model& model, unsigned vocab_size, unsigned hidden_dim, 
	    unsigned stochastic_dim, unsigned embedding_dim, unsigned window);
	    
    Expression build_graph(const vector<int> &sentence, ComputationGraph& cg, 
	    unsigned num_samples, unsigned &events);

    LookupParameters* p_C;	    // word embeddings -- FIXME: add character n-gram embeddings also
    Parameters* p_Wh1, *p_bh1;	    // mapping from word embedding to stochastic unit, layer 1
    Parameters* p_Wh2, *p_bh2;	    // layer 2 of the above, to form stochastic hidden outputs "h"

    std::vector<Parameters*> p_Wc;  // context mapping weights -- -w, ..., -1, 1, ..., w
    std::vector<Parameters*> p_bc;  // and bias terms
    Parameters* p_Wo, *p_bo;	    // output embeddings
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

WordContextModel::WordContextModel(Model& model, unsigned vocab_size, 
	    unsigned hidden_dim, unsigned stochastic_dim, unsigned embedding_dim, unsigned _window)
    : window(_window)
{
    // mapping from 1-hot vocabulary (input)
    p_C = model.add_lookup_parameters(vocab_size, {embedding_dim}); 

    // stochastic components -- MLP with one hidden layer
    p_Wh1 = model.add_parameters({hidden_dim, embedding_dim});
    p_bh1 = model.add_parameters({hidden_dim});
    p_Wh2 = model.add_parameters({stochastic_dim, hidden_dim});
    p_bh2 = model.add_parameters({stochastic_dim});

    // windowing components
    for (unsigned w = 0; w < 2*window; ++w) {
	Parameters *p_W = model.add_parameters({embedding_dim, embedding_dim+stochastic_dim});
	Parameters *p_b = model.add_parameters({embedding_dim});
	p_Wc.push_back(p_W);
	p_bc.push_back(p_b);
    }

    // mapping into 1-hot vocabulary (output)
    p_Wo = model.add_parameters({vocab_size, embedding_dim});
    p_bo = model.add_parameters({vocab_size});
}

Expression WordContextModel::build_graph(const vector<int> &sentence, ComputationGraph &cg, 
	unsigned num_samples, unsigned &num_events)
{
    // input embeddings
    vector<Expression> word_embeddings;
    for (auto token: sentence) 
	word_embeddings.push_back(lookup(cg, p_C, token));
    // stochastic model params
    Expression i_Wh1 = parameter(cg, p_Wh1);
    Expression i_bh1 = parameter(cg, p_bh1);
    Expression i_Wh2 = parameter(cg, p_Wh2);
    Expression i_bh2 = parameter(cg, p_bh2);
    // windowing params
    std::vector<Expression> i_Wc, i_bc;
    for (unsigned w = 0; w < 2*window; ++w) {
	i_Wc.push_back(parameter(cg, p_Wc[w]));
	i_bc.push_back(parameter(cg, p_bc[w]));
    }
    // output params
    Expression i_Wo = parameter(cg, p_Wo);
    Expression i_bo = parameter(cg, p_bo);

    //LOLCAT(transpose(i_bh1));
    //LOLCAT(transpose(i_bh2));

    auto slen = sentence.size(); 
    vector<Expression> errs;
    for (unsigned i=1; i < slen-1; ++i) {
	// find sufficient stats for stochastic units
	Expression &x = word_embeddings[i];
	Expression h1 = rectify(i_Wh1 * x + i_bh1);
	Expression h2 = logistic(i_Wh2 * h1 + i_bh2);

	if (num_samples > 0) {
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

		// score p(y | h, x)
		Expression hx = concatenate({hs, x});
		vector<Expression> local_errs;
		for (unsigned w = 0; w < 2*window; ++w) {
		    int j = (w < window) ? (i - w - 1) : (i + w - window + 1);
		    if (j < 0 || j >= (int) slen) continue;

		    Expression &i_Wcw = i_Wc[w];
		    Expression &i_bcw = i_bc[w];

		    Expression cw = rectify(i_Wcw * hx + i_bcw);
		    Expression rw = softmax(i_Wo * cw + i_bo);
		    Expression err = pickneglogsoftmax(rw, sentence[j]); // FIXME: consider negative sampling?
		    local_errs.push_back(err);
		}
		Expression es = sum(local_errs);
		if (s == 0) num_events += local_errs.size();

		samples.push_back(hs);
		sample_errs.push_back(es);
	    }
	    Expression ws = softmax(concatenate(sample_errs));

	    // add error terms for p(h | x) and p(y | h), weighted by ws
	    for (unsigned s=0; s < num_samples; ++s) {
		Expression w = pick(ws, s);
		Expression herr = w * binary_log_loss(h2, samples[s]);
		errs.push_back(herr);
		Expression yerr = w * sample_errs[s];
		errs.push_back(yerr);
	    }

	} else {
	    // just do deterministic training, using h2 vector
	    Expression hx = concatenate({h2, x});
	    for (unsigned w = 0; w < 2*window; ++w) {
		int j = (w < window) ? i - window + w : i - window + w + 1;
		if (j < 0 || j >= (int) slen) continue;
		//cerr << "i " << i << " len " << slen << " w " << w << " j " << j;
		//cerr << "\tw[i] " << dict.Convert(sentence[i]) << " w[j] " << dict.Convert(sentence[j]) << endl;

		Expression &i_Wcw = i_Wc[w];
		Expression &i_bcw = i_bc[w];

		Expression cw = rectify(i_Wcw * hx + i_bcw);
		Expression rw = softmax(i_Wo * cw + i_bo);
		Expression err = pickneglogsoftmax(rw, sentence[j]); // FIXME: consider negative sampling?
		errs.push_back(err);
		++num_events;
	    }
	}
    }

    Expression nerr = sum(errs);
    return nerr;
}

typedef vector<vector<int>> Corpus;
Corpus read_corpus(const string &filename);

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
        ("embedding,e", value<unsigned>()->default_value(32), "use <num> dimensions for word embeddings")
        ("stochastic,s", value<unsigned>()->default_value(16), "use <num> dimensions for stochastic units")
        ("hidden,h", value<unsigned>()->default_value(64), "use <num> dimensions for recurrent hidden states")
        ("window,w", value<unsigned>()->default_value(2), "predict up to <num> words to left and right")
        ("samples,S", value<unsigned>()->default_value(10), "number of stochastic samples used in E step")
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

    bool verbose = vm.count("verbose");

    string line;
    cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
    Corpus training = read_corpus(vm["train"].as<string>());
    dict.Freeze(); // no new word types allowed

    unsigned EMBEDDING_DIM = vm["embedding"].as<unsigned>(); 
    unsigned STOCHASTIC_DIM = vm["stochastic"].as<unsigned>(); 
    unsigned HIDDEN_DIM = vm["hidden"].as<unsigned>(); 
    unsigned window = vm["window"].as<unsigned>();
    unsigned num_samples = vm["samples"].as<unsigned>();
    unsigned VOCAB_SIZE = dict.size();

    cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
    Corpus devel = read_corpus(vm["devel"].as<string>());

    ostringstream os;
    os << "wcm"
	<< '_' << HIDDEN_DIM
	<< '_' << EMBEDDING_DIM
	<< '_' << STOCHASTIC_DIM
	<< '_' << window
	<< "-pid" << getpid() << ".params";
    string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;

    Model model;
    //Trainer* sgd = new SimpleSGDTrainer(&model, 1e-6, 10);
    //Trainer* sgd = new AdamTrainer(&model);
    Trainer* sgd = new RmsPropTrainer(&model);
    //Trainer* sgd = new AdaGradTrainer(&model);
    //Trainer* sgd = new AdadeltaTrainer(&model);

    WordContextModel wcm(model, VOCAB_SIZE, HIDDEN_DIM, STOCHASTIC_DIM, EMBEDDING_DIM, window);

    double best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 500; 
    unsigned start_sampling_after_i_reports = 2000;
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    bool first = true;
    unsigned report = 0;
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
	    if (report < start_sampling_after_i_reports)
		wcm.build_graph(sent, cg, 0, chars);
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

Corpus read_corpus(const string &filename)
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

