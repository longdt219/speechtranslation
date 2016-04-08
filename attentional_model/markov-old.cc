#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "cnn/gru.h"

#include "encoder.hh"
#include "corpus.hh"
#include "markov_mt_rnn.h"

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;
using namespace boost::program_options;
using namespace boost::filesystem;

unsigned LAYERS = 1;         // 2 recommended
unsigned EMBEDDING_DIM = 64; // 256 recommended
unsigned HIDDEN_DIM = 128;   // 1024 recommended

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;

cnn::Dict mdic; //morphemes

template <class TM_t>
void train(Model &model, TM_t &mm, AlignedEncodedCorpus &training, AlignedEncodedCorpus &devel, Trainer &sgd, const string &fname);



typedef std::map<int, vector<int>> MorphCorpus;
unsigned read_morphs(const std::string &filename, EncoderPtr &src_vocab, MorphCorpus &morphemes)
{
    ifstream in(filename.c_str());
    //FIX ME: return morpheme size
    string buf, token;
    int morpheme;
    bool state=false;
    while (getline(in, buf))
    {  
        istringstream ss(buf);
        while(ss >> token){
	    if (!state) {
		morpheme=src_vocab->encode(token); 	
		morphemes.insert(std::make_pair(morpheme, vector<int>()));
              }
	    else {
	 	morphemes[morpheme].push_back(mdic.Convert(token));	// replace []
	 }
	  state=true;
	}
         state=false;
    }   
  return mdic.size();
}


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
        //("reverse", "reverse the source input file, the target input file and the alignment")
        //("intersect-reverse", "reverse the source input file and the target input file but not reverse the alignment")
        ("threshold-src,s", value<int>(), "keep only the <num> most frequent words (source)")
        ("threshold-trg,t", value<int>(), "keep only the <num> most frequent words (target)")
        ("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
        ("embedding,e", value<int>()->default_value(EMBEDDING_DIM), "use <num> dimensions for word embeddings")
        ("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
        ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
        ("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
        ("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
	("abs-freqs,a", "threshold by absolute frequency instead of top N")
	("morphology,M", value<string>(), "use morphology")
    ;
    store(parse_command_line(argc, argv, opts), vm); 
    if (vm.count("config") > 0)
    {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, opts), vm); 
    }
    notify(vm);
    
    if (vm.count("help") || vm.count("input") != 1 || vm.count("devel") != 1) {
        cout << opts << "\n";
        return 1;
    }

    // read training sentences
    AlignedEncodedCorpus training;
    bool fabs=vm.count("abs-freqs") ? true : false;	
    //if (vm.count("reverse")) 
        //training.SetReverse(Reverse);
    //else if (vm.count("intersect-reverse")) 
        //training.SetReverse(Intersect_Reverse);
    if (exists(vm["input"].as<string>()))
        training.read_single_file(vm["input"].as<string>());
    else
        training.read_component_files(vm["input"].as<string>());

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
    MorphCorpus morphemes; 
    if (vm.count("morphology")){
	unsigned msize = read_morphs(vm["morphology"].as<string>(), src_vocab, morphemes);
	cerr << "Morphology has " << msize << "  morphemes"<<endl;}
    // FIXME: should really write both vocab structures to disk

    AlignedEncodedCorpus devel(src_vocab, trg_vocab);
    if (exists(vm["devel"].as<string>()))
	devel.read_single_file(vm["devel"].as<string>());
    else
	devel.read_component_files(vm["devel"].as<string>());
   
    // output vocab, corpus stats
    cout << "%% Training has " << training.size() << " sentence pairs\n";
    cout << "%% Development has " << devel.size() << " sentence pairs\n";
    cout << "%% source vocab " << src_vocab->size() << " unique words, from " << svsz << "\n";
    cout << "%% target vocab " << trg_vocab->size() << " unique words, from " << tvsz << "\n";

    Model model;
    Trainer* sgd = new SimpleSGDTrainer(&model);

    LAYERS = vm["layers"].as<int>(); 
    EMBEDDING_DIM = vm["embedding"].as<int>(); 
    HIDDEN_DIM = vm["hidden"].as<int>(); 
    bool bidir = vm.count("bidirectional");

    cout << "%% layers " << LAYERS << " embedding " << EMBEDDING_DIM << " hidden " << HIDDEN_DIM << endl;
    if (bidir)
        cout << "%% using bidirectional RNN states for source context" << endl;
    else
        cout << "%% using word embeddings for source context" << endl;

    ostringstream os;
    os << "mm"
        << '_' << LAYERS
        << '_' << EMBEDDING_DIM
        << '_' << HIDDEN_DIM
        << "_b" << bidir
        << '_' << ((vm.count("lstm")) ? "lstm" : (vm.count("gru")) ? "gru" : "rnn")
        << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;

    if (vm.count("lstm")) {
        cout << "%% Using LSTM recurrent units" << endl;
        RecurrentMarkovTranslationModel<LSTMBuilder> mm(model, 
                src_vocab->size(), trg_vocab->size(), mdic.size(),
                LAYERS, EMBEDDING_DIM, HIDDEN_DIM, bidir, vm.count("morphology"), morphemes);
        train(model, mm, training, devel, *sgd, fname);
    } else if (vm.count("gru")) {
        cout << "%% Using GRU recurrent units" << endl;
        RecurrentMarkovTranslationModel<GRUBuilder> mm(model, 
                src_vocab->size(), trg_vocab->size(), mdic.size(),
                LAYERS, EMBEDDING_DIM, HIDDEN_DIM, bidir, vm.count("morphology"), morphemes);
        train(model, mm, training, devel, *sgd, fname);
    } else {
        cout << "%% Using RNN recurrent units" << endl;
        RecurrentMarkovTranslationModel<SimpleRNNBuilder> mm(model, 
                src_vocab->size(), trg_vocab->size(), mdic.size(),
                LAYERS, EMBEDDING_DIM, HIDDEN_DIM, bidir, vm.count("morphology"), morphemes);
        train(model, mm, training, devel, *sgd, fname);
    }

    delete sgd;

    return 0;
}

template <class TM_t>
void train(Model &model, TM_t &mm, AlignedEncodedCorpus &training, AlignedEncodedCorpus &devel, Trainer &sgd, const string &fname)
{

    unsigned report_every_i = 20000;
    unsigned dev_every_i_reports = 20000;
    double best = 9e+99;
    unsigned si = training.size();
    vector<unsigned> order(training.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    bool first = true;
    int report = 0;
    unsigned lines = 0;
    while(1) {
        Timer iteration("completed in");
        double loss = 0, lossJ = 0;
        unsigned chars = 0;
        for (unsigned i = 0; i < report_every_i; ++i) {
            if (si == training.size()) {
                si = 0;
                if (first) { first = false; } else { sgd.update_epoch(); }
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
            }

            // build graph for this instance
            ComputationGraph cg;
            auto spair = training.at(order[si]);
            chars += spair->trgSentence.size() - 1;
            ++si;
	    auto err = mm.BuildGraph(*spair, cg);
	    auto total_err = err.first + err.second;
	    cg.forward();
            loss += as_scalar(cg.get_value(err.first.i));
            lossJ += as_scalar(cg.get_value(err.second.i));
            cg.backward();
            sgd.update();
            ++lines;
        }
        sgd.status();
        cerr << " ppl=" << exp(loss / chars) << " pplJ=" << exp(lossJ / chars) << ' ';

        // show score on dev data
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0, dlossJ = 0;
            int dchars = 0;
            for (auto& spair : devel) {
                ComputationGraph cg;
		auto err = mm.BuildGraph(*spair, cg);
		auto total_err = err.first + err.second;
		cg.forward();
		dloss += as_scalar(cg.get_value(err.first.i));
		dlossJ += as_scalar(cg.get_value(err.second.i));
                dchars += spair->trgSentence.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] ppl=" << exp(dloss / dchars) << " pplJ=" << exp(dlossJ / dchars) << ' ';
        }
    }
}
