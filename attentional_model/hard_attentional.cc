#include "hard_attentional.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace cnn;
using namespace boost::program_options;

unsigned LAYERS = 1; // 2
unsigned HIDDEN_DIM = 64;  // 1024
unsigned ALIGN_DIM = 32;   // 128
unsigned SRC_VOCAB_SIZE = 0;
unsigned TGT_VOCAB_SIZE = 0;

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;
bool verbose;

typedef vector<int> Sentence;
typedef pair<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> Corpus;

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

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
        ("train,t", value<string>(), "file containing training sentences, with "
            "each line consisting of source ||| target.")
        ("devel,d", value<string>(), "file containing development sentences.")
        ("test,T", value<string>(), "file containing testing source sentences (no training)")
        ("kbest,K", value<string>(), "test on kbest inputs using mononlingual Markov model")
        ("initialise,i", value<string>(), "load initial parameters from file")
        ("parameters,p", value<string>(), "save best parameters to this file")
        ("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
        ("align,a", value<int>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
        ("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
        ("samples,s", value<int>()->default_value(5), "use <num> stochastic samples")
        ("topk,k", value<int>()->default_value(100), "use <num> top kbest entries, used with --kbest")
        ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
        ("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
        ("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
        ("giza", "use GIZA++ style features in attentional components")
        ("curriculum", "use 'curriculum' style learning, focusing on easy problems in earlier epochs")
        ("verbose,v", "be extremely chatty")
    ;
    store(parse_command_line(argc, argv, opts), vm); 
    if (vm.count("config") > 0)
    {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, opts), vm); 
    }
    notify(vm);
    
    if (vm.count("help") || vm.count("train") != 1 || (vm.count("devel") != 1 && (vm.count("test") != 1 || vm.count("kbest") != 1))) {
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

void initialise(Model &model, const string &filename);

template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string out_file, bool curriculum, int samples);

template <class AM_t>
void test(Model &model, AM_t &am, string test_file, int samples);

template <class AM_t>
void test_kbest_arcs(Model &model, AM_t &am, string test_file, int top_k);

Corpus read_corpus(const string &filename);

template <class rnn_t>
int main_body(variables_map vm)
{
    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
    kTGT_SOS = td.Convert("<s>");
    kTGT_EOS = td.Convert("</s>");
    verbose = vm.count("verbose");

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    Corpus training, devel;
    string line;
    cerr << "Reading training data from " << vm["train"].as<string>() << "...\n";
    training = read_corpus(vm["train"].as<string>());
    sd.Freeze(); // no new word types allowed
    td.Freeze(); // no new word types allowed
    
    LAYERS = vm["layers"].as<int>(); 
    ALIGN_DIM = vm["align"].as<int>(); 
    HIDDEN_DIM = vm["hidden"].as<int>(); 
    bool bidir = vm.count("bidirectional");
    bool giza = vm.count("giza");

    string flavour = "RNN";
    if (vm.count("lstm"))	flavour = "LSTM";
    else if (vm.count("gru"))	flavour = "GRU";
    SRC_VOCAB_SIZE = sd.size();
    TGT_VOCAB_SIZE = td.size();

    if (vm.count("devel")) {
	cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
	devel = read_corpus(vm["devel"].as<string>());
    }

    string fname;
    if (vm.count("parameters")) {
	fname = vm["parameters"].as<string>();
    } else {
	ostringstream os;
	os << "am"
	    << '_' << LAYERS
	    << '_' << HIDDEN_DIM
	    << '_' << ALIGN_DIM
	    << '_' << flavour
	    << "_b" << bidir
	    << "_g" << giza
	    << "-pid" << getpid() << ".params";
	fname = os.str();
    }
    cerr << "Parameters will be written to: " << fname << endl;

    Model model;
    //bool use_momentum = false;
    Trainer* sgd = nullptr;
    //if (use_momentum)
        //sgd = new MomentumSGDTrainer(&model);
    //else
        sgd = new SimpleSGDTrainer(&model);
    //sgd = new AdadeltaTrainer(&model);

    cerr << "%% Using " << flavour << " recurrent units" << endl;
    HardAttentionalModel<rnn_t> am(model, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
	    LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza);

    if (vm.count("initialise"))
	initialise(model, vm["initialise"].as<string>());

    if (!vm.count("test") && !vm.count("kbest"))
	train(model, am, training, devel, *sgd, fname, vm.count("curriculum"), vm["samples"].as<int>());
    else if (vm.count("kbest"))
	test_kbest_arcs(model, am, vm["kbest"].as<string>(), vm["topk"].as<int>());
    else
	test(model, am, vm["test"].as<string>(), vm["samples"].as<int>());

    delete sgd;

    return EXIT_SUCCESS;
}

template <class AM_t>
void test(Model &model, AM_t &am, string test_file, int samples)
{
    double tloss = 0;
    int tchars = 0;
    int lno = 0;

    cerr << "Reading test examples from " << test_file << endl;
    ifstream in(test_file);
    assert(in);
    string line;
    while(getline(in, line)) {
	Sentence source, target;
	ReadSentencePair(line, &source, &sd, &target, &td);
	if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
		(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
	    cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
	    abort();
	}

	ComputationGraph cg;
	am.BuildGraph(source, target, samples, cg);
	double loss = as_scalar(cg.forward());
	for (auto &w: source)
	    cout << sd.Convert(w) << " ";
	cout << "|||";
	for (auto &w: target)
	    cout << " " << td.Convert(w);
	cout << " ||| " << loss << endl;
	tloss += loss;
	tchars += target.size() - 1;

	if (verbose)
	    cerr << "chug " << lno++ << "\r" << flush;
    }

    cerr << "\n***TEST E = " << (tloss / tchars) << " ppl=" << exp(tloss / tchars) << ' ';
    return;
}

template <class AM_t>
void test_kbest_arcs(Model &model, AM_t &am, string test_file, int top_k)
{
    // only suitable for monolingual setting, of predicting a sentence given preceeding sentence
    cerr << "Reading test examples from " << test_file << endl;
    unsigned lno = 0;
    ifstream in(test_file);
    assert(in);
    string line, last_id, last_last_id = "-";
    const std::string sep = "|||";
    vector<SentencePair> items, last_items;
    last_items.push_back(SentencePair(Sentence({ kSRC_SOS, kSRC_EOS }), Sentence({ kTGT_SOS, kTGT_EOS })));

    while(getline(in, line)) {
	Sentence source, target;

	istringstream in(line);
	string id, word;
	in >> id >> word;
	assert(word == sep);
	while(in) {
	    in >> word;
	    if (word.empty() || word == sep) break;
	    source.push_back(sd.Convert(word));
	    target.push_back(td.Convert(word));
	}

	if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
		(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
	    cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
	    abort();
	}

	if (id != last_id && !items.empty()) {
	    if (items.size() > top_k)
		items.resize(top_k);

	    unsigned count = 0;
	    for (auto &prev: last_items) {
		ComputationGraph cg;
		auto &source = prev.first;
		am.start_new_instance(source, cg);

		// FIXME: no sampling here...

		for (auto &curr: items) {
		    std::vector<Expression> errs;
		    auto &target = curr.second;
		    const unsigned tlen = target.size() - 1;
		    for (unsigned t = 0; t < tlen; ++t) {
			Expression i_r_t = am.add_input(target[t], t, cg);
			Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
			errs.push_back(i_err);
		    }
		    Expression i_nerr = sum(errs);
		    double loss = as_scalar(cg.incremental_forward());

		    cout << last_last_id << ":" << last_id << " |||";
		    for (auto &w: source) cout << " " << sd.Convert(w);
		    cout << " |||";
		    for (auto &w: target) cout << " " << td.Convert(w);
		    cout << " ||| " << loss << "\n";

		    ++count;
		}
	    }

	    last_items = items;
	    last_last_id = last_id;
	    last_id = id;
	    items.clear();

	    if (verbose)
		cerr << "chug " << lno++ << " [" << count << " pairs]\r" << flush;
	}

	last_id = id;
	items.push_back(SentencePair(source, target));
    }

    return;
}

template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel, 
	Trainer &sgd, string out_file, bool curriculum, int samples)
{
    double best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 500; 
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

    vector<vector<unsigned>> order_by_length; 
    const unsigned curriculum_steps = 10;
    if (curriculum) {
	// simple form of curriculum learning: for the first K epochs, use only
	// the shortest examples from the training set. E.g., K=10, then in
	// epoch 0 using the first decile, epoch 1 use the first & second
	// deciles etc. up to the full dataset in k >= 9.
	multimap<size_t, unsigned> lengths;
	for (unsigned i = 0; i < training.size(); ++i) 
	    lengths.insert(make_pair(training[i].first.size(), i));

	order_by_length.resize(curriculum_steps);
	unsigned i = 0;
	for (auto& landi: lengths) {
	    for (unsigned k = i * curriculum_steps / lengths.size(); k < curriculum_steps; ++k)  
		order_by_length[k].push_back(landi.second);
	    ++i;
	}
    }

    bool first = true;
    int report = 0;
    unsigned lines = 0;
    int epoch = 0;

    while(1) {
        Timer iteration("completed in");
        double loss = 0;
        unsigned chars = 0;

        for (unsigned iter = 0; iter < report_every_i; ++iter) {

            if (si == training.size()) {
                si = 0;
                if (first) { first = false; } else { sgd.update_epoch(); }

		if (curriculum && epoch < order_by_length.size()) {
		    order = order_by_length[epoch++];
		    cerr << "Curriculum learning, with " << order.size() << " examples\n";
		} 
	    }

            if (si % order.size() == 0) {
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
	    }

	    if (verbose && iter+1 == report_every_i) {
		auto& spair = training[order[si % order.size()]];
		ComputationGraph cg;
                cerr << "\nDecoding source, greedy Viterbi: ";
                am.decode(spair.first, cg, 1, td);

                cerr << "\nDecoding source, sampling: ";
                am.sample(spair.first, cg, td);
	    }

            // build graph for this instance
	    auto& spair = training[order[si % order.size()]];
	    ComputationGraph cg;
            chars += spair.second.size() - 1;
            ++si;
            Expression alignment;
            am.BuildGraph(spair.first, spair.second, samples, cg, &alignment);
            loss += as_scalar(cg.forward());
            
            cg.backward();
            sgd.update();
            ++lines;

	    if (verbose) {
		cerr << "chug " << iter << "\r" << flush;
		if (iter+1 == report_every_i) {
		    // display the alignment
		    am.display(spair.first, spair.second, cg, alignment, sd, td);
		}
	    }
        }
        sgd.status();
        cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0;
            int dchars = 0;
            for (auto& spair : devel) {
                ComputationGraph cg;
                am.BuildGraph(spair.first, spair.second, samples, cg);
                dloss += as_scalar(cg.forward());
                dchars += spair.second.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(out_file);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }
    }
}

Corpus read_corpus(const string &filename)
{
    ifstream in(filename);
    assert(in);
    Corpus corpus;
    string line;
    int lc = 0, stoks = 0, ttoks = 0;
    while(getline(in, line)) {
        ++lc;
        Sentence source, target;
        ReadSentencePair(line, &source, &sd, &target, &td);
        corpus.push_back(SentencePair(source, target));
        stoks += source.size();
        ttoks += target.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
                (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
    }
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
    return corpus;
}

void initialise(Model &model, const string &filename)
{
    cerr << "Initialising model parameters from file: " << filename << endl;
    ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> model;
}
