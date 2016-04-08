#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>

#include "cnn/gru.h"

#include "encoder.hh"
#include "corpus.hh"
#include "markov_mt_rnn.h"
#include "markov_train.h"

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace cnn;
using namespace boost::program_options;
using namespace boost::filesystem;

unsigned LAYERS = 2;         // 2 recommended
unsigned EMBEDDING_DIM = 64; // 256 recommended
unsigned HIDDEN_DIM = 128;   // 1024 recommended

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;

template <class rnn_t> int main_body(variables_map vm);
void initialise(Model &model, const string &filename);
template<class TM_t>
std::vector<std::pair<double,double> >
test_rescore(Model &model, TM_t &mm, AlignedEncodedCorpus &corpus, FeatureFactory &feat_factory);


int main(int argc, char **argv)
{
    cnn::Initialize(argc, argv);

    // command line processing
    variables_map vm; 
    options_description opts("Allowed options");
    opts.add_options()
        ("help", "print help message")
        ("config,c", value<string>(), "config file specifying additional command line options")
        ("input,i", value<string>(), "file containing training sentences. Either a single file with "
            "each line consisting of source ||| target ||| align; or "
            "three files with suffix .src/.trg/.align, one sentence per line")
        ("devel,d", value<string>(), "file containing development sentences (see --input)")
        ("test", value<string>(), "file containing test sentences (see -- input)")
        ("initialise", value<string>(), "file containing the saved model parameters")
        //("reverse", "reverse the source input file, the target input file and the alignment")
        //("intersect-reverse", "reverse the source input file and the target input file but not reverse the alignment")
        ("threshold-src,s", value<int>(), "keep only the <num> most frequent words (source)")
        ("threshold-trg,t", value<int>(), "keep only the <num> most frequent words (target)")
        ("treport,r", value<int>(), "report training every i iterations")
        ("dreport,R", value<int>(), "report dev every i iterations")
        ("notsave,S", "NOT save the best RNN")
        ("epochs,e", value<int>(), "max number of epochs")
        ("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
        ("embedding,E", value<int>()->default_value(EMBEDDING_DIM), "use <num> dimensions for word embeddings")
        ("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
        ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
        ("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
        ("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
	("abs-freqs,a", "threshold by absolute frequency instead of top N")
	("document-context,D", "use left and right sentence as document context")
        ("unsup-train", "treat alignment as latent variables in stochastic neural Markov TM")
        ("sup-train", "treat alignment as observed variables in stochastic neural Markov TM")
        ("rescore", value<string>(), "produce features for the sentence pairs in the test set (used for re-ranking kbest lists)")
        ("decode", value<string>(), "decode sentences in the test set") 
    ;

    store(parse_command_line(argc, argv, opts), vm); 
    if (vm.count("config") > 0)
    {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, opts), vm); 
    }

    notify(vm);

//    if (vm.count("help") || vm.count("sup-train") != 1 || (vm.count("unsup-train") != 1 && !(vm.count("test") == 0 || vm.count("kbest") == 0))) {
//        cout << opts << "\n";
//        return 1;
//    }

    if (vm.count("lstm")) {
        cout << "%% Using LSTM recurrent units" << endl;
        return main_body<LSTMBuilder>(vm);
    } else if (vm.count("gru")) {
        cout << "%% Using GRU recurrent units" << endl; 
        return main_body<GRUBuilder>(vm);
    } else {
        cout << "%% Using Simple RNN recurrent units" << endl;
        return main_body<SimpleRNNBuilder>(vm);
    }
}

template <class rnn_t>
int main_body(variables_map vm)
{    
    unsigned MAX_EPOCH = 100;
    unsigned WRITE_EVERY_I=1000;
    unsigned report = 1000;

    unsigned batch_size=10;
    unsigned sample_size_K=5;
    ALGN_SAMPLING sampling_method=BEAM_SEARCH;
    unsigned batch_iter=7;

    //----
    if (vm.count("dreport")) WRITE_EVERY_I = vm["dreport"].as<int>();
    if (vm.count("treport")) report = vm["treport"].as<int>();
    if (vm.count("epochs")) MAX_EPOCH = vm["epochs"].as<int>();

    if (vm.count("layers")) LAYERS = vm["layers"].as<int>();
    if (vm.count("embedding")) EMBEDDING_DIM = vm["embedding"].as<int>();
    if (vm.count("hidden")) HIDDEN_DIM = vm["hidden"].as<int>();
    bool bidir = vm.count("bidirectional");
    bool save_model = true;
    if (vm.count("notsave")) save_model = false;

    // ---- read training sentences
    AlignedEncodedCorpus training;
    bool fabs = vm.count("abs-freqs");
    bool document = vm.count("document-context");

    if (exists(vm["input"].as<string>())) {
	if (!document)
	    training.read_single_file(vm["input"].as<string>());
	else
	    training.read_document_file(vm["input"].as<string>());
    } else
        training.read_component_files(vm["input"].as<string>());

    //---- buiilding the vocab
    unsigned svsz = training.src_vocab()->size();
    unsigned tvsz = training.trg_vocab()->size();
    // apply frequency limits on threshold size
    if (vm.count("threshold-src"))
        training.threshold_source_vocabulary(vm["threshold-src"].as<int>(), fabs);
    if (vm.count("threshold-trg"))
        training.threshold_target_vocabulary(vm["threshold-trg"].as<int>(), fabs);

    EncoderPtr src_vocab = training.src_vocab();
    EncoderPtr trg_vocab = training.trg_vocab();
    src_vocab->freeze();
    trg_vocab->freeze();
    // FIXME: should really write both vocab structures to disk

    //---- read devel sentences
    AlignedEncodedCorpus devel(src_vocab, trg_vocab);
    if (exists(vm["devel"].as<string>())) {
	if (!document)
	    devel.read_single_file(vm["devel"].as<string>());
	else
	    devel.read_document_file(vm["devel"].as<string>());
    } else
	devel.read_component_files(vm["devel"].as<string>());

    //---- read test sentences
    AlignedEncodedCorpus test_corpus(src_vocab, trg_vocab);
    if (exists(vm["test"].as<string>())) {
        if (!document)
            test_corpus.read_single_file(vm["test"].as<string>());
        else
            test_corpus.read_document_file(vm["test"].as<string>());
    } else
        test_corpus.read_component_files(vm["test"].as<string>());
   
    // ---- output vocab, corpus stats
    cout << "%% Training has " << training.size() << " sentence pairs\n";
    cout << "%% Development has " << devel.size() << " sentence pairs\n";
    cout << "%% Test has " << test_corpus.size() << " sentence pairs\n";
    cout << "%% source vocab " << src_vocab->size() << " unique words, from " << svsz << "\n";
    cout << "%% target vocab " << trg_vocab->size() << " unique words, from " << tvsz << "\n";

    //---- file name for saving the parameters
    ostringstream os;
    os << "mm"
        << '_' << LAYERS
        << '_' << EMBEDDING_DIM
        << '_' << HIDDEN_DIM
        << "_b" << bidir
        << '_' << ((vm.count("lstm")) ? "lstm" : (vm.count("gru")) ? "gru" : "rnn")
        << "-pid" << getpid() << ".params";
    const string fname = os.str();

    cout << "%% layers " << LAYERS << " embedding " << EMBEDDING_DIM << " hidden " << HIDDEN_DIM << endl;
    if (bidir)
        cout << "%% using bidirectional RNN states for source context" << endl;
    else
        cout << "%% using word embeddings for source context" << endl;

    //----
    Model model;
    Trainer* sgd = new SimpleSGDTrainer(&model);
    FeatureFactory feat_factory;

    typedef RecurrentMarkovTranslationModel<rnn_t> TM_t;
    TM_t  mm(model, src_vocab->size(), trg_vocab->size(),
          LAYERS, EMBEDDING_DIM, HIDDEN_DIM, bidir, document, feat_factory.feat_dim());

    if (vm.count("initialise")) {
        cout << "initialising the model from: " << vm["initialise"].as<string>() << endl;
        initialise(model, vm["initialise"].as<string>());
    }

    if (vm.count("sup-train")) {
       cout << "supervise training ...." << endl;
       if (save_model)
          cerr << "Parameters will be written to: " << fname << endl;
       else
          cerr << "The parameters will NOT be saved" << endl;

        MarkovTrain<TM_t> mtrain(&training, &devel, &model, &mm, &feat_factory);
        mtrain.sup_train(report, WRITE_EVERY_I, MAX_EPOCH, *sgd, save_model, fname); 

       //FIXME remove later
       //vector<pair<double, double> > costs = test_rescore(model, mm, test_corpus, feat_factory);
       //string outf_name("feats2");  
      // ofstream outs(outf_name);
     //  for (auto i = 0; i < costs.size(); i++)
     //     outs << costs[i].first << " " << costs[i].second << endl;
     //  cout << "the (word,align) features are written into: " << outf_name << endl;
        
    }

    if (vm.count("unsup-train")) {
       cout << "unsupervise training ...." << endl;
       if (save_model)
          cerr << "Parameters will be written to: " << fname << endl;
       else
          cerr << "The parameters will NOT be saved" << endl;

       MarkovTrain<TM_t> mtrain(&training, &devel, &model, &mm, &feat_factory);
        mtrain.unsup_train(WRITE_EVERY_I, MAX_EPOCH, batch_size,
               sample_size_K, sampling_method, batch_iter, *sgd, fname);
    }

    if (vm.count("rescore")) {
       vector<pair<double, double> > costs = test_rescore(model, mm, test_corpus, feat_factory);
       string outf_name = vm["rescore"].as<string>(); 
       ofstream outs(outf_name);
       for (auto i = 0; i < costs.size(); i++)
          outs << costs[i].first << " " << costs[i].second << endl;
       cout << "the (word,align) features are written into: " << outf_name << endl;        
    }

    if (vm.count("decode")) {
    }

    delete sgd;

    return 0;
}

void initialise(Model &model, const string &filename)
{
    cerr << "Initialising model parameters from file: " << filename << endl;
    ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> model;
}

template<class TM_t>
std::vector<std::pair<double,double> >
test_rescore(Model &model, TM_t &mm, AlignedEncodedCorpus &corpus, FeatureFactory &feat_factory) {

    std::vector<std::pair<double,double> > costs;

    for (auto sent_num = 0; sent_num < corpus.size(); sent_num++) {
        FEATS_SENT phi_sent;
        feat_factory.add_feats_sent(*corpus.at(sent_num), phi_sent);
        ComputationGraph cg;
        auto err = mm.BuildGraph(corpus, sent_num, cg, phi_sent);
        auto total_err = err.first + err.second;
        cg.forward();
        double lossW = as_scalar(cg.get_value(err.first.i));
        double lossJ = as_scalar(cg.get_value(err.second.i));
        double len = corpus.at(sent_num)->trgSentence.size() - 1;
        costs.push_back(pair<double,double>(lossW/len,lossJ/len));
   }
   return costs;
}

