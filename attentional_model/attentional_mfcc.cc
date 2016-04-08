#include "attentional_mfcc.h"
#include <glob.h>
#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>
#include <string.h>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/zlib.hpp>

using namespace std;
using namespace cnn;
using namespace boost::program_options;

unsigned LAYERS = 1; // 2
unsigned HIDDEN_DIM = 64;  // 1024
unsigned ALIGN_DIM = 32;   // 128

int MAX_SOURCE_SIZE = 1500 ; // Maximum 15 seconds (= 1500 frames, each 10 ms)
// Similar with limiting source sentence with 80 words

unsigned TGT_VOCAB_SIZE = 0;
float dropout_prob = 0;
bool is_only_alignment = false;
std::string alignment_output = "";
float threshold_alignment = 0.01;
int smooth_window = 0;
bool pyramid = false;
bool train_all_data = false;
bool mask_flag = false;
float smooth_softmax = 1;
float learning_rate = 1e-6;
bool output_translation_dev = false;

cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;
bool verbose;

typedef vector<vector<float>> MFCC_FILE;
typedef vector<int> Sentence;
//typedef pair<Sentence, Sentence> SentencePair;
typedef vector<int> BOUNDARY;
//typedef tuple<MFCC_FILE, Sentence, BOUNDARY, int> SentencePair; // includes document id (optional)
typedef tuple<MFCC_FILE, Sentence, int> SentencePair; // includes document id (optional)
typedef vector<SentencePair> Corpus;
// Long Duong : define new tuple

std::map<int,MFCC_FILE> mfcc_map;
std::map<int,BOUNDARY> bound_map;

// Long Duong : define new tuple (should be delete)
typedef tuple<int,int, std::string> Phoneme;
typedef vector<Phoneme> PhonemeList;
typedef std::map<int,PhonemeList> PhonemeMap;

PhonemeMap phoneme_map;
vector<int> train_idx;
vector<int> dev_idx;
vector<int> test_idx;

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
		("strain,st", value<string>(), "source (mfcc directory) use for training.")
		("ttrain,tt", value<string>(), "target (english folder) use for training.")
		("reftvcb,rtv", value<int>()->default_value(63), "reference target size vocabulary. i.e. number of phone type in the target")
		("refinitialise,ri", value<string>(), "load reference parameters from file")

        ("devel,d", value<string>(), "file containing development sentences.")
        ("test,T", value<string>(), "file containing testing sentences")
        ("rescore,r", "rescore (source, target) pairs in testing, default: translate source only")
        ("beam,b", value<int>()->default_value(0), "size of beam in decoding; 0=greedy")
        ("kbest,K", value<string>(), "test on kbest inputs using mononlingual Markov model")
        ("initialise,i", value<string>(), "load initial parameters from file")
		("boundary", value<string>(), "load the presegment boundary files")
        ("parameters,p", value<string>(), "save best parameters to this file")
		("trainer", value<string>()->default_value("sgd"), "Trainer algorithm either (adadelta,adagrad,momentum or sgd)")
		("learnrate", value<float>()->default_value(0.000001), "The lambda learning rate for trainer")
		("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
        ("align,a", value<int>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
		("dropout", value<float>()->default_value(0), "Dropout all the non-recurrent of RNN")
        ("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
        ("topk,k", value<int>()->default_value(100), "use <num> top kbest entries, used with --kbest")
        ("epochs,e", value<int>()->default_value(50), "maximum number of training epochs")
        ("gru", "use Gated Recurrent Unit (GRU) for recurrent structure; default RNN")
        ("lstm", "use Long Short Term Memory (GRU) for recurrent structure; default RNN")
		("local",value<int>()->default_value(-1), "Using the local attentional model (the anchor and width), default = -1 (none), 0: fix , 1: predict  ")
        ("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
        ("giza", "use GIZA++ style features in attentional components")
        ("curriculum", "use 'curriculum' style learning, focusing on easy problems in earlier epochs")
        ("swap", "swap roles of source and target, i.e., learn p(source|target)")
        ("document,D", "use previous sentence as document context; requires document id prefix in input files")
        ("coverage,C", value<float>()->default_value(0.0f), "impose alignment coverage penalty in training, with given coefficient")
        ("display", "just display alignments instead of training or decoding")
		("mask", "Mask the function to compute alignment matrix to encourage monotonicity")
		("translation", "take the pretrained model and output the translation on the development data")
		("trainall", "train on all data")
		("alignment", "take the pretrained model and output the alignment wrt phone alignment")
		("aliout,A", value<string>(), "the alignment output folder.")
		("split", value<string>(), "The split file (train,dev,test). Each line is a set of ID")
		("smooth", value<int>()->default_value(0), "Use smoothing for alignment generation, put (smooth) frame before and after current frame together")
		("ssize", value<int>()->default_value(39), "Size of source embedding (default = 39 dim MFCC)")
		("smoothsm", value<float>()->default_value(1), "Use the softmax smoothing constant ")
		("pyramid", "Reduce the size of source by 8 times = 4 * 2 using pyramidal structure")
		("threshold,H", value<float>()->default_value(0.0f), "Threshold for choosing the alignment (confidence must greater than threshold)")
		("verbose,v", "be extremely chatty")
    ;
    store(parse_command_line(argc, argv, opts), vm); 
    if (vm.count("config") > 0)
    {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, opts), vm); 
    }
    notify(vm);
    
    // Compusary parameters

    if (vm.count("help") || vm.count("strain") != 1 || vm.count("ttrain") !=1 || (vm.count("devel") != 1 && !(vm.count("test") == 0 || vm.count("kbest") == 0))) {
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
template <class AM_t> void translation_output(AM_t &am, Corpus &data, int size);

template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel, Corpus & testing,
	Trainer &sgd, string out_file, bool curriculum, int max_epochs, 
        bool doco, float coverage, bool display);

void generate_alignment(const std::vector<vector<float>> &source, const std::vector<int>& target,
        ComputationGraph &cg, const Expression &alignment, int docid, int smooth);
template <class rnn_t> void copy_rnn_builder(rnn_t &reference, rnn_t &target);
//template <class AM_t> void test_rescore(Model &model, AM_t &am, string test_file, bool doco);
//template <class AM_t> void test_decode(Model &model, AM_t &am, string test_file, bool doco, int beam);
//template <class AM_t> void test_kbest_arcs(Model &model, AM_t &am, string test_file, int top_k);

const Sentence* context(const Corpus &corpus, unsigned i);

tuple<Corpus,Corpus,Corpus> read_corpus_from_vars(std::map<int,string>* eng_sents);

void read_mfcc_directory(const string& foldername);
void read_split(const string& filename);
void read_boundary_directory(const string& foldername);
std::map<int,string> read_target_directory(const string& foldername);

//std::vector<int> ReadNumberedSentence(const std::string& line, Dict* sd, std::vector<int> &ids);
//void ReadNumberedSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, std::vector<int> &ids);
template <class rnn_t> void copy_rnn_builder(rnn_t &reference, rnn_t &target){
	assert(reference.params.size()==target.params.size());
	for (int i =0; i< reference.params.size(); i++){
		assert(reference.params[i].size() == target.params[i].size());
		for (int j =0; j< reference.params[i].size(); j++){
			// Copy
			Parameters* ref = reference.params[i][j];
			Parameters* trg = target.params[i][j];
			memcpy(trg->values.v, &ref->values.v[0], sizeof(cnn::real) * ref->values.d.size());
		}
	}
}

template <class rnn_t>
int main_body(variables_map vm)
{
    kTGT_SOS = td.Convert("<s>");
    kTGT_EOS = td.Convert("</s>");
    verbose = vm.count("verbose");

    LAYERS = vm["layers"].as<int>();
    dropout_prob = vm["dropout"].as<float>();
    ALIGN_DIM = vm["align"].as<int>(); 
    HIDDEN_DIM = vm["hidden"].as<int>();

    // If we only want alignment to check with the output
    is_only_alignment = vm.count("alignment");
    output_translation_dev  = vm.count("translation");

    if (is_only_alignment){
    		alignment_output = vm["aliout"].as<string>();
    		threshold_alignment = vm["threshold"].as<float>();
    }
    learning_rate = vm["learnrate"].as<float>();
    cerr << "Using learning rate = " << learning_rate << "\n";
    smooth_window = vm["smooth"].as<int>();
    cerr << "Using smoothing window =" << smooth_window << "\n";
    smooth_softmax = vm["smoothsm"].as<float>();
    cerr << "Using smoothing softmax =" << smooth_softmax << "\n";

    pyramid = vm.count("pyramid");
    mask_flag = vm.count("mask");
    cerr << "===> Using Mask for source (encourage monotonic) = " << mask_flag << "\n";
    cerr << "===> Using paramid structure for source =" << pyramid << "\n";
    bool bidir = vm.count("bidirectional");
    bool giza = vm.count("giza");
    if (giza) cerr << " Using giza features \n";
    bool swap = vm.count("swap");
    bool doco = vm.count("document");
    int local = vm["local"].as<int>();

    if (vm.count("trainall")){
    	train_all_data = true;
    }
    int source_emb_size = vm["ssize"].as<int>();
    string flavour = "RNN";
    if (vm.count("lstm"))	flavour = "LSTM";
    else if (vm.count("gru"))	flavour = "GRU";

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    Corpus training, devel,testing;
    string line;
    cerr << "Reading source training data from " << vm["strain"].as<string>() << "...\n";
    cerr << "Reading target training data from " << vm["ttrain"].as<string>() << "...\n";
    // Long Duong
    read_mfcc_directory(vm["strain"].as<string>());

    // If split is available
    if (vm.count("split")){
    	read_split(vm["split"].as<string>());
    	cerr << "Using split with (train,dev,test) =  " << train_idx.size() << " & "<< dev_idx.size() << " & "<< test_idx.size() << endl;
    }

    std::map<int,string> eng_sents = read_target_directory(vm["ttrain"].as<string>());
    // End
    if (vm.count("boundary")){
    		// Read the boundary
    		read_boundary_directory(vm["boundary"].as<string>());
    }
    tie(training,devel,testing) = read_corpus_from_vars(&eng_sents);

    td.Freeze(); // no new word types allowed
    TGT_VOCAB_SIZE = td.size();

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
	    << "_d" << doco
	    << "-pid" << getpid() << ".params";
	fname = os.str();
    }
    cerr << "Parameters will be written to: " << fname << endl;

    Model model;
    //bool use_momentum = false;


    Trainer* sgd = nullptr;
    cerr << "Using trainer : " << vm["trainer"].as<string>() << "\n";

    if (vm["trainer"].as<string>() == "adagrad")
    		sgd = new AdagradTrainer(&model);
    else
    		if (vm["trainer"].as<string>() == "adadelta")
    			sgd = new AdadeltaTrainer(&model);
    		else
    			if (vm["trainer"].as<string>() == "momentum")
    				sgd = new MomentumSGDTrainer(&model);
    			else
    				if (vm["trainer"].as<string>() == "sgd")
    					sgd = new SimpleSGDTrainer(&model,learning_rate, 0.1);
    				else
    				{
    					cerr << " DON'T RECOGNIZE TRAINER ";
    					abort();
    				}

    cerr << "%% Using " << flavour << " recurrent units" << endl;
    AttentionalModel<rnn_t> am(&model,  TGT_VOCAB_SIZE,
	    LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza, doco, local, mask_flag, source_emb_size);

    if (vm.count("initialise"))
	initialise(model, vm["initialise"].as<string>());

    // ##################### Load the reference model ##################
    if (vm.count("refinitialise")){
    	cerr << "===> Reference model initialization : " << endl ;
    	int ref_TGT_VOCAB_SIZE = vm["reftvcb"].as<int>();
    	// create the same object as the usual attentional model
    	Model ref_model;
    	AttentionalModel<rnn_t> ref_am(&ref_model, ref_TGT_VOCAB_SIZE,
    		    LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza, doco, local, mask_flag, source_emb_size);
    	initialise(ref_model, vm["refinitialise"].as<string>());
    	// Print some of the ref_model
    	float x = ref_am.builder_src_fwd.params[0][1]->values.v[0];
    	// Copy all the recurrent parameters
    	copy_rnn_builder<rnn_t>(ref_am.builder_src_fwd, am.builder_src_fwd);
    	copy_rnn_builder<rnn_t>(ref_am.builder_src_fwd_1, am.builder_src_fwd_1);
    	copy_rnn_builder<rnn_t>(ref_am.builder_src_fwd_2, am.builder_src_fwd_2);
    	copy_rnn_builder<rnn_t>(ref_am.builder_src_bwd, am.builder_src_bwd);
    	copy_rnn_builder<rnn_t>(ref_am.builder_src_bwd_1, am.builder_src_bwd_1);
    	copy_rnn_builder<rnn_t>(ref_am.builder_src_bwd_2, am.builder_src_bwd_2);
    	assert(x == am.builder_src_fwd.params[0][1]->values.v[0]);
    }
    // ####################### END Load reference model ################

    if (vm["coverage"].as<float>() > 0) cerr << "Using coverage with penalty = " << vm["coverage"].as<float>() << "\n";

    if (!vm.count("test") && !vm.count("kbest"))
	train(model, am, training, devel, testing, *sgd, fname, vm.count("curriculum"),
                vm["epochs"].as<int>(), doco, vm["coverage"].as<float>(), vm.count("display"));
//    else if (vm.count("kbest"))
//	test_kbest_arcs(model, am, vm["kbest"].as<string>(), vm["topk"].as<int>());
//    else {
//        if (vm.count("rescore"))
//            test_rescore(model, am, vm["test"].as<string>(), doco);
//        else
//            test_decode(model, am, vm["test"].as<string>(), doco, vm["beam"].as<int>());
//    }

    delete sgd;

    return EXIT_SUCCESS;
}

//template <class AM_t>
//void test_rescore(Model &model, AM_t &am, string test_file, bool doco)
//{
//    double tloss = 0;
//    int tchars = 0;
//    int lno = 0;
//
//    cerr << "Reading test examples from " << test_file << endl;
//    ifstream in(test_file);
//    assert(in);
//    string line;
//    Sentence last_source;
//    int last_docid = -1;
//    while(getline(in, line)) {
//	Sentence source, target;
//        vector<int> num;
//	ReadNumberedSentencePair(line, &source, &sd, &target, &td, num);
//	if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
//		(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
//	    cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
//	    abort();
//	}
//
//	ComputationGraph cg;
//	am.BuildGraph(source, target, cg, nullptr, (doco && num[0] == last_docid) ? &last_source : nullptr);
//
//	double loss = as_scalar(cg.forward());
//        for (auto &n: num)
//            cout << n << ' ';
//	cout << "|||";
//	for (auto &w: source)
//	    cout << " " << sd.Convert(w);
//	cout << " |||";
//	for (auto &w: target)
//	    cout << " " << td.Convert(w);
//	cout << " ||| " << loss << endl;
//	tloss += loss;
//	tchars += target.size() - 1;
//
//	if (verbose)
//	    cerr << "chug " << lno++ << "\r" << flush;
//
//        if (doco) {
//            last_source = source;
//            last_docid = num[0];
//        }
//    }
//
//    cerr << "\n***TEST E = " << (tloss / tchars) << " ppl=" << exp(tloss / tchars) << ' ';
//    return;
//}
//
//template <class AM_t>
//void test_decode(Model &model, AM_t &am, string test_file, bool doco, int beam)
//{
//    double tloss = 0;
//    int tchars = 0;
//    int lno = 0;
//
//    cerr << "Reading test examples from " << test_file << endl;
//    ifstream in(test_file);
//    assert(in);
//    string line;
//    Sentence last_source;
//    Sentence source;
//    int last_docid = -1;
//    while (getline(in, line)) {
//        vector<int> num;
//        if (doco)
//            source = ReadNumberedSentence(line, &sd, num);
//        else
//            source = ReadSentence(line, &sd);
//
//	if (source.front() != kSRC_SOS && source.back() != kSRC_EOS) {
//	    cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
//	    abort();
//	}
//
//	ComputationGraph cg;
//        std::vector<int> target;
//
//        if (beam > 0)
//            target = am.beam_decode(source, cg, beam, td, (doco && num[0] == last_docid) ? &last_source : nullptr);
//        else
//            target = am.greedy_decode(source, cg, td, (doco && num[0] == last_docid) ? &last_source : nullptr);
//
//        bool first = true;
//	for (auto &w: target) {
//            if (!first) cout << " ";
//            cout << td.Convert(w);
//            first = false;
//        }
//	cout << endl;
//
//	if (verbose)
//	    cerr << "chug " << lno++ << "\r" << flush;
//
//        if (doco) {
//            last_source = source;
//            last_docid = num[0];
//        }
//    }
//    return;
//}
//
//template <class AM_t>
//void test_kbest_arcs(Model &model, AM_t &am, string test_file, int top_k)
//{
//    // only suitable for monolingual setting, of predicting a sentence given preceeding sentence
//    cerr << "Reading test examples from " << test_file << endl;
//    unsigned lno = 0;
//    ifstream in(test_file);
//    assert(in);
//    string line, last_id;
//    const std::string sep = "|||";
//    vector<SentencePair> items, last_items;
//    last_items.push_back(SentencePair(Sentence({ kSRC_SOS, kSRC_EOS }), Sentence({ kTGT_SOS, kTGT_EOS }), -1));
//    unsigned snum = 0;
//    unsigned count = 0;
//
//    auto process = [&am, &snum, &last_items, &items, &count]() {
//        for (unsigned i = 0; i < last_items.size(); ++i) {
//            ComputationGraph cg;
//            auto &source = get<0>(last_items[i]);
//            am.start_new_instance(source, cg);
//
//            for (unsigned j = 0; j < items.size(); ++j) {
//                std::vector<Expression> errs;
//                auto &target = get<1>(items[j]);
//                const unsigned tlen = target.size() - 1;
//                for (unsigned t = 0; t < tlen; ++t) {
//                    Expression i_r_t = am.add_input(target[t], t, cg);
//                    Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
//                    errs.push_back(i_err);
//                }
//                Expression i_nerr = sum(errs);
//                double loss = as_scalar(cg.incremental_forward());
//
//                //cout << last_last_id << ":" << last_id << " |||";
//                //for (auto &w: source) cout << " " << sd.Convert(w);
//                //cout << " |||";
//                //for (auto &w: target) cout << " " << td.Convert(w);
//                //cout << " ||| " << loss << "\n";
//                cout << snum << '\t' << i << '\t' << j << '\t' << loss << '\n';
//                ++count;
//            }
//        }
//    };
//
//    while (getline(in, line)) {
//	Sentence source, target;
//
//	istringstream in(line);
//	string id, word;
//	in >> id >> word;
//	assert(word == sep);
//	while(in) {
//	    in >> word;
//	    if (word.empty() || word == sep) break;
//	    source.push_back(sd.Convert(word));
//	    target.push_back(td.Convert(word));
//	}
//
//	if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
//		(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
//	    cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
//	    abort();
//	}
//
//	if (id != last_id && !items.empty()) {
//	    if (items.size() > top_k)
//		items.resize(top_k);
//
//            process();
//
//	    last_items = items;
//	    last_id = id;
//	    items.clear();
//            snum++;
//
//	    if (verbose)
//		cerr << "chug " << lno++ << " [" << count << " pairs]\r" << flush;
//	}
//
//	last_id = id;
//	items.push_back(SentencePair(source, target, -1));
//    }
//
//    if (!items.empty())
//        process();
//
//    return;
//}

// Long Duong : function to generate the alignment
void generate_alignment(const std::vector<vector<float>> &source, const std::vector<int>& target,
        ComputationGraph &cg, const Expression &alignment, int docid, int smooth)
{
    using namespace std;
    //Sentence source = ssent;
    //Sentence target = tsent;
    // display the alignment
    int I = target.size();
    int J = source.size();
    if ((pyramid) && (J > 32))
    		J = J / 8 ; // Reduce by 8 times

    // Get the value of variableindex from computational graph
    const Tensor &a = cg.get_value(alignment.i);

    // Open output file
    stringstream ss1;
    ss1 << setw(3) << setfill('0') << docid / 1000;
    string s1 = ss1.str();
    stringstream ss2;
    ss2 << setw(3) << setfill('0') << docid % 1000;
    string s2 = ss2.str();
    string out_file_name = s1+"."+s2+".words";
    ofstream out;
    ::mkdir(alignment_output.c_str(), 0770);
    out.open(alignment_output+'/'+out_file_name);

    std::map<int,int> map_target_start;
    std::map<int,int> map_target_end;

    vector<vector<float>> alm_weight;

    // Transform the value first
    for (int i=0; i<I; ++i){
    		vector<float> this_word;
    		vector<float> this_word_2;
    		if (smooth>0){
    			for (int j=0; j< J; ++j){
        			int start = (j-smooth >= 0) ? j-smooth:0;
        			int end = (j+smooth) < J ? j+smooth:J-1;
        			float sum = 0;
        			for (int k = start; k<=end; ++k){
        				sum += TensorTools::AccessElement(a, Dim(k, i));
        			}
        			this_word.push_back(sum / (1.0 * (end-start +1)));
    			}

    			for (int j=0; j< J; ++j){
        			int start = (j-smooth >= 0) ? j-smooth:0;
        			int end = (j+smooth) < J ? j+smooth:J-1;
        			float sum = 0;
        			for (int k = start; k<=end; ++k){
        				sum += this_word[k];
        			}
        			this_word_2.push_back(sum / (1.0 * (end-start +1)));
    			}
    			this_word = this_word_2;
    		}
		else {
			for (int j=0; j< J; ++j)
				this_word.push_back(TensorTools::AccessElement(a, Dim(j, i)));
		}

    		alm_weight.push_back(this_word);
    }

    for (int j = 0; j < J; ++j) {

    		float max_j = 0;
    		int max_idx = -1;
        	for (int i = 1; i < I-1; ++i) {
        		// Also skip if this target word is the punctuation
        		string target_word = td.Convert(target[i]);
        		if ((target_word.length() ==1) && (ispunct(target_word.at(0)))) continue;
            //float v = TensorTools::AccessElement(a, Dim(j, i));
        		float v = alm_weight[i][j];
            if ((v >= threshold_alignment) && (v > max_j)) {
            		max_j = v;
            		max_idx = i;
            }
        }
		if (max_idx != -1){
			if ((pyramid) && (source.size() > 32))
				out << td.Convert(target[max_idx]) << " " << j * 8  * 10 << " " << (j+1) * 8 * 10 <<  "\n";
			else
				out << td.Convert(target[max_idx]) << " " << j * 10 << " " << (j+1) * 10 <<  "\n";
			// Heuristic, accept all the tokens between start and end
//			if (map_target_start.count(max_idx)){
//				if (start < map_target_start[max_idx])
//					map_target_start[max_idx] = start;
//				if (end > map_target_end[max_idx])
//					map_target_end[max_idx] = end;
//			}
//			else {
//				map_target_start[max_idx] = start;
//				map_target_end[max_idx] = end;
//			}
		}
    }

    // Write to output file
//    for (int i = 1; i < I-1; ++i) {
//    		// From start to end align with word target[i]
//    		if (map_target_start.count(i))
//    			out << td.Convert(target[i]) << " " << map_target_start[i] << " " << map_target_end[i] <<  "\n";
//    }
    out.close();
}

template <class AM_t> void translation_output(AM_t &am, Corpus &data, int size){
    int docid;
	MFCC_FILE ssent;
    Sentence tsent;
    if (size > data.size()) size = data.size();

	for (unsigned i = 0; i < size; ++i) {
		tie(ssent, tsent, docid) = data[i];
		cerr << "Looking for translation for " << docid << "\n";
		cerr << "Target length = " << tsent.size() << "\n";
		ComputationGraph cg;
		std::vector<int> target;
		int beam = 5; // Always use beam decode
		if (beam > 0)
			target = am.beam_decode(ssent, cg, beam, td,  nullptr);
		else
			target = am.greedy_decode(ssent, cg, td, nullptr);

		bool first = true;
		cout << "Doc ID " << docid << "\n";
		//------------------------
		cout << "Hypothesis : \n";
		first = true;
		for (auto &w: target) {
			if (!first) cout << " ";
			cout << td.Convert(w);
			first = false;
		}
		cout << endl;
		//------------------------
		cout << "Gold : \n";
		first = true;
		for (auto &w: tsent) {
			if (!first) cout << " ";
			cout << td.Convert(w);
			first = false;
		}
		cout << endl;
		cout << "---------------------------------------------------\n";
	}

}
// Long Duong: this is the function to actually train the model
template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel, Corpus& testing,
	Trainer &sgd, string out_file, bool curriculum, int max_epochs, 
        bool doco, float coverage, bool display)
{
    double best = 9e+99;
    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 200;
    if (training.size() / 50 < dev_every_i_reports) dev_every_i_reports = training.size() / 50;
    if (training.size() < 1000) dev_every_i_reports = 5;

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
		// Order by source length
	    lengths.insert(make_pair(get<0>(training[i]).size(), i));

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
    MFCC_FILE ssent;
    Sentence tsent;
//    BOUNDARY boundary;
    int docid;
    int MAX_DISPLAY_DEV = 500 ;
    cerr << "Using dropout with probability = " << dropout_prob << "\n";
    // FIXME: move this into sep function
    if (display) {
        // display the alignments
        //
        // Because it's not training
        am.set_dropoutprob(dropout_prob,false);
        am.set_smoothing_softmax(smooth_softmax);
        am.set_pyramid(pyramid);


    		for (unsigned i = 0; i < devel.size(); ++i) {
        		if (i > MAX_DISPLAY_DEV) continue;
//            tie(ssent, tsent, boundary, docid) = devel[i];
            tie(ssent, tsent,  docid) = devel[i];
            ComputationGraph cg;
            Expression alignment;

            am.BuildGraph(ssent, tsent, cg, &alignment, (doco) ? context(devel, i) : nullptr);
            cg.forward();

            cout << "\n====== SENTENCE " << docid << " =========\n";
            am.display_ascii(ssent, tsent, cg, alignment, td);
            cout << "\n";

            // Don't need to display_tikz hereq
            cout << "\n";
            am.display_tikz(ssent, tsent, cg, alignment, td);
            cout << "\n";
        }
        return;
    }

	if (output_translation_dev){
		// Here we output the translation on development data
		am.set_dropoutprob(dropout_prob,false);
		am.set_smoothing_softmax(smooth_softmax);
		am.set_pyramid(pyramid);


		cerr << "Output the translation of the development data \n";
		// Output the development data
		if (testing.size() ==0){
			cout << " ################ DEV DATA ################\n";
			translation_output(am, devel, devel.size());
		} else {
			cout << " ################ TEST DATA ################\n";
			translation_output(am, testing, testing.size());
		}
		cout << " ################ TRAINING DATA ################\n";
		translation_output(am,training, 500); // output first 500 sentences
		return;
	}

    if (is_only_alignment){
    		am.set_dropoutprob(dropout_prob,false);
    		am.set_smoothing_softmax(smooth_softmax);
    		am.set_pyramid(pyramid);

    		cerr << "Decode with alignment threshold = " << threshold_alignment << "\n";
    	cerr << "Train size : " << training.size() << " Dev size " << devel.size() << endl;

        // Generate for training example and development examples
    	for (unsigned i = 0; i < training.size(); ++i) {
        		if (i % 100 == 0) cerr << i << "..";
//            tie(ssent, tsent, boundary, docid) = training[i];
            tie(ssent, tsent,  docid) = training[i];
            ComputationGraph cg;
            Expression alignment;
            am.BuildGraph(ssent, tsent, cg, &alignment, (doco) ? context(training, i) : nullptr);
            cg.forward();
            // Generate the alignment
            // Long Duong : Modify here
            generate_alignment(ssent,tsent,cg,alignment,docid, smooth_window);
        }
    	for (unsigned i = 0; i < devel.size(); ++i) {
    		if (train_all_data) continue; // Completely skip the dev file because we already include in train
        	if (i % 100 == 0) cerr << i << "..";
//            tie(ssent, tsent, boundary, docid) = training[i];
            tie(ssent, tsent,  docid) = devel[i];
            ComputationGraph cg;
            Expression alignment;
            am.BuildGraph(ssent, tsent, cg, &alignment, (doco) ? context(training, i) : nullptr);
            cg.forward();
            // Generate the alignment
            // Long Duong : Modify here
            generate_alignment(ssent,tsent,cg,alignment,docid, smooth_window);
        }

    	return;
    }

#if 0
        if (true) {
            double dloss = 0;
            int dchars = 0;
            for (unsigned i = 0; i < devel.size(); ++i) {
                tie(ssent, tsent, boundary, docid) = devel[i];
                ComputationGraph cg;
                am.BuildGraph(ssent, tsent, cg, nullptr, (doco) ? context(devel, i) : nullptr);
                dloss += as_scalar(cg.forward());
                dchars += tsent.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(out_file);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }
#endif
    // Long Duong: Training for each epoch
    while (sgd.epoch < max_epochs) {
    		// Default = Training here
    		am.set_dropoutprob(dropout_prob,true);
    		am.set_smoothing_softmax(smooth_softmax);
    		am.set_pyramid(pyramid);

        Timer iteration("completed in");
        double loss = 0;
        double penalty = 0;
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
//		tie(ssent, tsent, boundary, docid) = training[order[si % order.size()]];
	    	tie(ssent, tsent,  docid) = training[order[si % order.size()]];
		ComputationGraph cg;
                cerr << "\nDecoding source, greedy Viterbi: ";
//                am.greedy_decode(ssent, cg, td, (doco) ? context(training, order[si % order.size()]) : nullptr);

                cerr << "\nDecoding source, sampling: ";
//                am.sample(ssent, cg, td, (doco) ? context(training, order[si % order.size()]) : nullptr);
	    }

        // build graph for this instance

//	    tie(ssent, tsent, boundary, docid) = training[order[si % order.size()]];
	    tie(ssent, tsent,  docid) = training[order[si % order.size()]];
//	    cerr << "Buid graph for this instance " << docid << "\n";
	    ComputationGraph cg;
		chars += tsent.size() - 1;
		++si;
		Expression alignment;
		if (coverage > 0) {
			Expression coverage_penalty;
			Expression xent = am.BuildGraph(ssent, tsent, cg, &alignment, (doco) ? context(training, order[si % order.size()]) : nullptr, &coverage_penalty);
			Expression objective = xent + coverage * coverage_penalty;
			loss += as_scalar(cg.forward());
			penalty += as_scalar(cg.get_value(coverage_penalty.i));
		} else {
			am.BuildGraph(ssent, tsent, cg, &alignment, (doco) ? context(training, order[si % order.size()]) : nullptr);
//			cerr<< "Finish an iteration \n";
			loss += as_scalar(cg.forward());
//			cerr<< "After forward \n";
		}
//		cerr << "Lost = " << loss << "\n";

		cg.backward();
		sgd.update();
		++lines;

	    if (verbose) {
		cerr << "chug " << iter << "\r" << flush;
		if (iter+1 == report_every_i) {
		    // display the alignment
//		    am.display_ascii(ssent, tsent, cg, alignment, sd, td);
                    cout << "\n";
//		    am.display_tikz(ssent, tsent, cg, alignment, sd, td);
                    cout << "\n";
			}
	    	}
        }
        sgd.status();
        cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
        if (coverage > 0) 
            cerr << "cover_pnl " << penalty/chars << ' ';

        // show score on dev data?
        report++;
        // Show report if this is the last epoches
        if ((sgd.epoch == max_epochs) || (report % dev_every_i_reports == 0)) {
            double dloss = 0;
            int dchars = 0;
            for (unsigned i = 0; i < devel.size(); ++i) {
//                tie(ssent, tsent, boundary, docid) = devel[i];
            		tie(ssent, tsent, docid) = devel[i];
                ComputationGraph cg;
                am.BuildGraph(ssent, tsent, cg, nullptr, (doco) ? context(devel, i) : nullptr);
                dloss += as_scalar(cg.forward());
                dchars += tsent.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                cerr << "\n\n ====== Save to output file : " << out_file <<  "====== \n";
                ofstream out(out_file);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }
    }
}
std::vector<string> globVector(const string& pattern){
    glob_t glob_result;
    glob(pattern.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> files;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}
int find_real_file_name(const string& file_name, const string& prefix){
	std::vector<string> substrs;
	boost::split(substrs, file_name, boost::is_any_of("/ "));
	string last_tokens = substrs[substrs.size() - 1];
	last_tokens.replace(last_tokens.find(prefix), string(prefix).size(), "");
	if (last_tokens.find(".") != std::string::npos)
		last_tokens.replace(last_tokens.find("."), string(".").size(), "");
	return atoi(last_tokens.c_str());

}
std::map<int,string> read_target_directory(const string& foldername){
	std::vector<string> files = globVector(foldername + "/*");

	const string prefix = ".en";
	std::map<int,string> result;
	for (unsigned int i=0; i<files.size(); ++i){
		// Find the real file name (not the path)
		int file_idx = find_real_file_name(files[i], prefix);

		// Open files[i]
		ifstream in(files[i]);
		assert(in);
		string line;

		while (getline(in,line)){
			// for a line, we should tokenize it probably
			result[file_idx] = line;
		}
		// cerr << file_idx <<  " with line " << result[file_idx] << "\n";
	}
	return result;
}

void read_split(const string& filename){
	ifstream in(filename);
	assert(in);
	string line;
	int count = 0;
	while (getline(in,line)){
		count++;
		std::vector<string> substrs;
		boost::split(substrs, line, boost::is_any_of("\t "));
		cerr << "Substring size " << substrs.size() << "\n";
		for (int i=0; i<substrs.size(); ++i){
			int v = atoi(substrs[i].c_str());
			if (count ==1) train_idx.push_back(v);
			if (count ==2) dev_idx.push_back(v);
			if (count ==3) test_idx.push_back(v);
		}
	}
}

void read_mfcc_directory(const string& foldername){
	std::vector<string> files = globVector(foldername + "/*");

	const string prefix = ".mfc";

	for (unsigned int i=0; i<files.size(); ++i){
		// Find the real file name (not the path)
		//cerr << "read file : " << files[i] << endl;
		int file_idx = find_real_file_name(files[i], prefix);
		ifstream in(files[i]);
		assert(in);
		string line;
		vector<vector<float>> this_file;
		while (getline(in,line)){
			std::vector<string> substrs;
			//cerr << line << "\n";
			vector<float> this_line;
			boost::split(substrs, line, boost::is_any_of("\t "));
			for (int i=0; i<substrs.size(); ++i){
				float v = atof(substrs[i].c_str());
				this_line.push_back(v);
			}
			this_file.push_back(this_line);
		}
		mfcc_map[file_idx] = this_file;
	}
}

void read_boundary_directory(const string& foldername){
	std::vector<string> files = globVector(foldername + "/*");

	const string prefix = ".bounds";

	for (unsigned int i=0; i<files.size(); ++i){
		// Find the real file name (not the path)
		int file_idx = find_real_file_name(files[i], prefix);
		ifstream in(files[i]);
		assert(in);
		string line;
		while (getline(in,line)){
			std::vector<string> substrs;
			//cerr << line << "\n";
			vector<int> this_line;
			boost::split(substrs, line, boost::is_any_of("\t "));
			for (int i=0; i<substrs.size(); ++i){
				int v = atoi(substrs[i].c_str());
				this_line.push_back(v);
			}
			bound_map[file_idx] = this_line;
		}

	}
}


tuple<Corpus,Corpus,Corpus> read_corpus_from_vars(std::map<int,string>* eng_sents)
{
    Corpus training,develop,testing;
    string line;
    int lc = 0, stoks = 0, ttoks = 0;
	int SIZE_DEV = 500;
    if (SIZE_DEV > eng_sents->size() / 2) SIZE_DEV = eng_sents->size() / 2;

	int count = 0;
    for (std::map<int,string>::iterator iter = eng_sents->begin(); iter != eng_sents->end(); ++iter){
		int key_idx = iter->first;
//		cerr << "read file : " << key_idx << endl;
		++lc;
		if (mfcc_map.count(key_idx)){
			count ++;
			// Contain the same key
			string target_sent = "<s> " + iter->second + " </s>";
			MFCC_FILE source_sent = mfcc_map[key_idx];
//			BOUNDARY boundary = bound_map[key_idx];
			//--------------------------
			Sentence target = ReadSentence(target_sent, &td);
			if (source_sent.size() == 0) continue;
			if (source_sent.size() > MAX_SOURCE_SIZE) continue;
//			corpus.push_back(SentencePair(source_sent,target,boundary, key_idx));
			if (train_idx.size() ==0) {
				// Default split, only split  to train and dev
				if (eng_sents->size() - count <= SIZE_DEV){
					develop.push_back(SentencePair(source_sent,target,key_idx));
					// All data will belong to training
					if (train_all_data) training.push_back(SentencePair(source_sent,target,key_idx));
				}
				else
					training.push_back(SentencePair(source_sent,target,key_idx));
			}else {
				// Predefined split based on key_idx
				if (std::find(train_idx.begin(),train_idx.end(), key_idx) != train_idx.end())
					training.push_back(SentencePair(source_sent,target,key_idx));
				if (std::find(dev_idx.begin(),dev_idx.end(), key_idx) != dev_idx.end())
					develop.push_back(SentencePair(source_sent,target,key_idx));
				if (std::find(test_idx.begin(),test_idx.end(), key_idx) != test_idx.end())
					testing.push_back(SentencePair(source_sent,target,key_idx));
			}

	        ttoks += target.size();
	        if  ((target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
	            cerr << "Sentence in file " << key_idx << " didn't start or end with <s>, </s>\n";
	            abort();
	        }
		}
	}

    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << " & " << td.size() << " (target) types\n";
	cerr << "Size of (train,dev,test) = " << training.size() << " & " << develop.size() << " & "<< testing.size() << endl;
	return tie(training,develop,testing);
}

//std::vector<int> ReadNumberedSentence(const std::string& line, Dict* sd, vector<int> &identifiers) {
//    std::istringstream in(line);
//    std::string word;
//    std::vector<int> res;
//    std::string sep = "|||";
//    if (in) {
//        identifiers.clear();
//        while (in >> word) {
//            if (!in || word.empty()) break;
//            if (word == sep) break;
//            identifiers.push_back(atoi(word.c_str()));
//        }
//    }
//
//    while(in) {
//        in >> word;
//        if (!in || word.empty()) break;
//        res.push_back(sd->Convert(word));
//    }
//    return res;
//}
//
//
//void ReadNumberedSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, vector<int> &identifiers)
//{
//    std::istringstream in(line);
//    std::string word;
//    std::string sep = "|||";
//    Dict* d = sd;
//    std::vector<int>* v = s;
//
//    if (in) {
//        identifiers.clear();
//        while (in >> word) {
//            if (!in || word.empty()) break;
//            if (word == sep) break;
//            identifiers.push_back(atoi(word.c_str()));
//        }
//    }
//
//    while(in) {
//        in >> word;
//        if (!in) break;
//        if (word == sep) { d = td; v = t; continue; }
//        v->push_back(d->Convert(word));
//    }
//}

void initialise(Model &model, const string &filename)
{
    cerr << "Initialising model parameters from file: " << filename << endl;
    ifstream in(filename);
    boost::archive::text_iarchive ia(in);
    ia >> model;
    // Calculate number of parameters
    int total_para = 0;
    for (const auto &p : model.parameters_list()){
    	total_para = total_para + p->values.d.size();
    }
    int total_lookup_para = 0;
	for (const auto &p : model.lookup_parameters_list())  {
			for (unsigned i = 0; i < p->values.size(); ++i)
				total_lookup_para = total_lookup_para + p->values[i].d.size();
	}
	cerr << " ==>>> Number of params : " << total_para << endl;
	cerr << " ==>>> Number of lookup params : " << total_lookup_para << endl;
}

const Sentence* context(const Corpus &corpus, unsigned i)
{
	// Don't alow to use document in the context of MFCC
	return nullptr;
}
