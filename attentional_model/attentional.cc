#include "attentional.h"
#include <glob.h>
#include <vector>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iostream>
#include <sys/stat.h>

#include <boost/algorithm/string.hpp>
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
float dropout_prob = 0;
bool is_only_alignment = false;
bool output_translation_dev = false;

std::string alignment_output = "";
float threshold_alignment = 0.2;
int smooth_window = 0;
float smooth_softmax = 1;
bool is_swap = 0;

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;
bool verbose;
string prefix_split = "";

typedef vector<int> Sentence;
//typedef pair<Sentence, Sentence> SentencePair;
typedef std::tuple<Sentence, Sentence, int> SentencePair; // includes document id (optional)
typedef std::vector<SentencePair> Corpus ;
// Long Duong : define new tuple
typedef tuple<int,int, std::string> Phoneme;
typedef vector<Phoneme> PhonemeList;
typedef std::map<int,PhonemeList> PhonemeMap;

PhonemeMap phoneme_map;

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
						("strain,st", value<string>(), "source (phone directory) use for training.")
						("ttrain,tt", value<string>(), "target (english folder) use for training.")
						("dsize", value<int>()->default_value(500), "Use this much data from train as dev (-1 : use all for training)")
						("devel,d", value<string>(), "file containing development sentences.")
						("test,T", value<string>(), "file containing testing sentences")
						("rescore,r", "rescore (source, target) pairs in testing, default: translate source only")
						("beam,b", value<int>()->default_value(0), "size of beam in decoding; 0=greedy")
						("kbest,K", value<string>(), "test on kbest inputs using mononlingual Markov model")
						("initialise,i", value<string>(), "load initial parameters from file")
						("parameters,p", value<string>(), "save best parameters to this file")
						("trainer", value<string>()->default_value("sgd"), "Trainer algorithm either (adadelta,adagrad,momentum or sgd)")
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
						("alignment", "take the pretrained model and output the alignment wrt phone alignment")
						("out_split", value<string>(),"output the split of train and develop to files prefix.(train|dev).(s|t)")
						("translation", "take the pretrained model and output the translation on the development data")
						("aliout,A", value<string>(), "the alignment output folder.")
						("smooth", value<int>()->default_value(0), "Use smoothing for alignment generation, put (smooth) frame before and after current frame together")
						("smoothsm", value<float>()->default_value(1), "Use the softmax smoothing constant ")
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

	if (vm.count("help") || (vm.count("train") != 1 && vm.count("strain") != 1)  || (vm.count("devel") != 1 && !(vm.count("test") == 0 || vm.count("kbest") == 0))) {
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
		Trainer &sgd, string out_file, bool curriculum, int max_epochs,
		bool doco, float coverage, bool display);
//template <class AM_t> void translation_output(AM_t &am, Corpus &data, int size);

void generate_alignment(const std::vector<int> &source, const std::vector<int>& target,
		ComputationGraph &cg, const Expression &alignment, int docid, int smooth);

template <class AM_t> void test_rescore(Model &model, AM_t &am, string test_file, bool doco);
template <class AM_t> void test_decode(Model &model, AM_t &am, string test_file, bool doco, int beam);
template <class AM_t> void test_kbest_arcs(Model &model, AM_t &am, string test_file, int top_k);


std::vector<int> find_high_tfidf_target_word(Corpus &training, Corpus& devel);
template <class AM_t> std::vector<std::tuple<int,int,float>> beam_decode_word(AM_t &am, Corpus &data, int size);

const Sentence* context(const Corpus &corpus, unsigned i);

Corpus read_corpus(const string &filename, bool doco);
tuple<Corpus,Corpus> read_corpus_from_vars(std::map<int,string>* eng_sents, int dsize);

void read_phone_directory(const string& foldername);
std::map<int,string> read_target_directory(const string& foldername);

std::vector<int> ReadNumberedSentence(const std::string& line, Dict* sd, std::vector<int> &ids);
void ReadNumberedSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, std::vector<int> &ids);
void write_output_file(Corpus corpus, string prefix, string type);

void write_output_file(Corpus corpus, string prefix, string type){
	ofstream enFile, spFile;
	enFile.open(prefix+"." + type + ".en");
	spFile.open(prefix+"." + type + ".es");
	Sentence ssent,tsent;
	int docid;
	for (int i=0; i<corpus.size(); i++){
		tie(ssent,tsent,docid) = corpus[i];
		for (int j =0; j<ssent.size(); j++){
			spFile << sd.Convert(ssent[j]) << " ";
		}
		spFile << endl;
		for (int j =0; j<tsent.size(); j++){
			enFile << td.Convert(tsent[j]) << " ";
		}
		enFile << endl;
	}
	spFile.close();
	enFile.close();

}
template <class rnn_t>
int main_body(variables_map vm)
{
	kSRC_SOS = sd.Convert("<s>");
	kSRC_EOS = sd.Convert("</s>");
	kTGT_SOS = td.Convert("<s>");
	kTGT_EOS = td.Convert("</s>");
	int SRC_UNK = sd.Convert("<unk>");
	int TGT_UNK = td.Convert("<unk>");
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

	smooth_window = vm["smooth"].as<int>();
	cerr << "Using smoothing window =" << smooth_window << "\n";
	smooth_softmax = vm["smoothsm"].as<float>();
	cerr << "Using smoothing softmax =" << smooth_softmax << "\n";

	bool bidir = vm.count("bidirectional");
	bool giza = vm.count("giza");
	if (giza) cerr << " Using giza features \n";
	is_swap = vm.count("swap");
	bool doco = vm.count("document");
	int local = vm["local"].as<int>();

	string flavour = "RNN";
	if (vm.count("lstm"))	flavour = "LSTM";
	else if (vm.count("gru"))	flavour = "GRU";

	Corpus training, devel;
	if (vm.count("strain")) {
		cerr << "Reading source training data from " << vm["strain"].as<string>() << "...\n";
		cerr << "Reading target training data from " << vm["ttrain"].as<string>() << "...\n";
		// Long Duong
		read_phone_directory(vm["strain"].as<string>());
		std::map<int,string> eng_sents = read_target_directory(vm["ttrain"].as<string>());
		// End
		cerr << "Using " << vm["dsize"].as<int>() << " for development data";
		tie(training,devel) = read_corpus_from_vars(&eng_sents, vm["dsize"].as<int>());
	}

	if (vm.count("train")){
		cerr << "Reading train data from " << vm["train"].as<string>() << "...\n";
		training = read_corpus(vm["train"].as<string>(), doco);
	}
	sd.Freeze(); // no new word types allowed
	td.Freeze(); // no new word types allowed

	SRC_VOCAB_SIZE = sd.size();
	TGT_VOCAB_SIZE = td.size();

	if (vm.count("devel")) {
		cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
		devel = read_corpus(vm["devel"].as<string>(), doco);
	}

	if (devel.size() <= 0){
		// Use 500 from train as dev
		for (int i=0; i<500 && i < training.size() ; i++){
			devel.push_back(training[i]);
		}
	}

	if (vm.count("out_split")){
		// Here output the split of according to the prefix of what is train what is dev
		string prefix_split = vm["out_split"].as<string>();
		write_output_file(training,prefix_split,"train");
		write_output_file(devel,prefix_split,"dev");
	}


	if (is_swap) {
		cerr << "Swapping role of source and target (for testing) \n";
		std::swap(sd, td);
		std::swap(kSRC_SOS, kTGT_SOS);
		std::swap(kSRC_EOS, kTGT_EOS);
		std::swap(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE);
		for (auto &sent: training)
			std::swap(get<0>(sent), get<1>(sent));
		for (auto &sent: devel)
			std::swap(get<0>(sent), get<1>(sent));
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
					sgd = new SimpleSGDTrainer(&model);
				else
				{
					cerr << " DON'T RECOGNIZE TRAINER ";
					abort();
				}

	cerr << "%% Using " << flavour << " recurrent units" << endl;
	AttentionalModel<rnn_t> am(&model, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
			LAYERS, HIDDEN_DIM, ALIGN_DIM, bidir, giza, doco, local);

	if (vm.count("initialise"))
		initialise(model, vm["initialise"].as<string>());

	if (vm["coverage"].as<float>() > 0) cerr << "Using coverage with penalty = " << vm["coverage"].as<float>() << "\n";

	if (!vm.count("test") && !vm.count("kbest"))
		train(model, am, training, devel, *sgd, fname, vm.count("curriculum"),
				vm["epochs"].as<int>(), doco, vm["coverage"].as<float>(), vm.count("display"));
	else if (vm.count("kbest"))
		test_kbest_arcs(model, am, vm["kbest"].as<string>(), vm["topk"].as<int>());
	else {
		if (vm.count("rescore"))
			test_rescore(model, am, vm["test"].as<string>(), doco);
		else
			test_decode(model, am, vm["test"].as<string>(), doco, vm["beam"].as<int>());
	}

	delete sgd;

	return EXIT_SUCCESS;
}

template <class AM_t>
void test_rescore(Model &model, AM_t &am, string test_file, bool doco)
{
	double tloss = 0;
	int tchars = 0;
	int lno = 0;

	cerr << "Reading test examples (for rescoring) from " << test_file << endl;
	if (is_swap) cerr << "Swap the role of source and target" << endl;
	ifstream in(test_file);
	assert(in);
	string line;
	Sentence last_source;
	int last_docid = -1;
	string original_line = "";
	while(getline(in, line)) {
		Sentence source, target;
		original_line = line ;
		// Get the Idx here
		string delimiter = "|||";
		int pos = line.find(delimiter);
		string idx1 = line.substr(0,pos);
		line.erase(0, pos + delimiter.length());

		vector<int> num;
//		cerr << "Index of sentence " << idx1 << endl;
//		cerr << "With content " << original_line << endl;
//		cerr << "Size of dictionary " << sd.size() << " target " << td.size() << endl;
//		cerr << "FOR TESTING : " << td.Convert("no") << endl;
		//ReadNumberedSentencePair(line, &source, &sd, &target, &td, num);
		if (is_swap)
			ReadSentencePair(line, &target, &td, &source, &sd);
		else
			ReadSentencePair(line, &source, &sd, &target, &td);

		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
				(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
			abort();
		}

		ComputationGraph cg;
		// Long Duong: set the options here for rescoring
		am.set_dropoutprob(dropout_prob,false);
		am.set_smoothing_softmax(smooth_softmax);
		// Build graph
		am.BuildGraph(source, target, cg, nullptr,  nullptr);

		double loss = as_scalar(cg.forward());

		// LD: Need to think more about this, base sololy on length is abit underestimating
		loss = loss / (1.0 * (target.size()-1));
		cout << original_line << " ||| " << loss << endl;

		tloss += loss;
		tchars += target.size() - 1;

		if (verbose)
			cerr << "chug " << lno++ << "\r" << flush;

		if (doco) {
			last_source = source;
			last_docid = num[0];
		}
	}

	cerr << "\n***TEST E = " << (tloss / tchars) << " ppl=" << exp(tloss / tchars) << ' ';
	return;
}

template <class AM_t>
void test_decode(Model &model, AM_t &am, string test_file, bool doco, int beam)
{
	double tloss = 0;
	int tchars = 0;
	int lno = 0;

	cerr << "Reading test examples from " << test_file << endl;
	ifstream in(test_file);
	assert(in);
	string line;
	Sentence last_source;
	Sentence source;
	int last_docid = -1;
	while (getline(in, line)) {
		vector<int> num;
		if (doco)
			source = ReadNumberedSentence(line, &sd, num);
		else
			source = ReadSentence(line, &sd);

		if (source.front() != kSRC_SOS && source.back() != kSRC_EOS) {
			cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
			abort();
		}

		ComputationGraph cg;
		std::vector<int> target;

		if (beam > 0)
			target = am.beam_decode(source, cg, beam, td, (doco && num[0] == last_docid) ? &last_source : nullptr);
		else
			target = am.greedy_decode(source, cg, td, (doco && num[0] == last_docid) ? &last_source : nullptr);

		bool first = true;
		for (auto &w: target) {
			if (!first) cout << " ";
			cout << td.Convert(w);
			first = false;
		}
		cout << endl;

		if (verbose)
			cerr << "chug " << lno++ << "\r" << flush;

		if (doco) {
			last_source = source;
			last_docid = num[0];
		}
	}
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
	string line, last_id;
	const std::string sep = "|||";
	vector<SentencePair> items, last_items;
	last_items.push_back(SentencePair(Sentence({ kSRC_SOS, kSRC_EOS }), Sentence({ kTGT_SOS, kTGT_EOS }), -1));
	unsigned snum = 0;
	unsigned count = 0;

	auto process = [&am, &snum, &last_items, &items, &count]() {
		for (unsigned i = 0; i < last_items.size(); ++i) {
			ComputationGraph cg;
			auto &source = get<0>(last_items[i]);
			am.start_new_instance(source, cg);

			for (unsigned j = 0; j < items.size(); ++j) {
				std::vector<Expression> errs;
				auto &target = get<1>(items[j]);
				const unsigned tlen = target.size() - 1;
				for (unsigned t = 0; t < tlen; ++t) {
					Expression i_r_t = am.add_input(target[t], t, cg);
					Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
					errs.push_back(i_err);
				}
				Expression i_nerr = sum(errs);
				double loss = as_scalar(cg.incremental_forward());

				//cout << last_last_id << ":" << last_id << " |||";
				//for (auto &w: source) cout << " " << sd.Convert(w);
				//cout << " |||";
				//for (auto &w: target) cout << " " << td.Convert(w);
				//cout << " ||| " << loss << "\n";
				cout << snum << '\t' << i << '\t' << j << '\t' << loss << '\n';
				++count;
			}
		}
	};

	while (getline(in, line)) {
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

			process();

			last_items = items;
			last_id = id;
			items.clear();
			snum++;

			if (verbose)
				cerr << "chug " << lno++ << " [" << count << " pairs]\r" << flush;
		}

		last_id = id;
		items.push_back(SentencePair(source, target, -1));
	}

	if (!items.empty())
		process();

	return;
}

// Long Duong : function to generate the alignment
void generate_alignment(const std::vector<int> &source, const std::vector<int>& target,
		ComputationGraph &cg, const Expression &alignment, int docid, int smooth)
{
	using namespace std;
	//Sentence source = ssent;
	//Sentence target = tsent;
	// display the alignment
	int I = target.size();
	int J = source.size();

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

	// Skip the first and the last element which is <s> and </s> for both source and target
	std::map<int,int> map_target_start;
	std::map<int,int> map_target_end;

	vector<vector<float>> alm_weight;

	// TEMP ------
	//    std::map<int,float> prob_w;
	//    for (int i=0; i<I; ++i){
	//		float value = 0;
	//		if (prob_w.count(target[i])){
	//			value = prob_w[target[i]];
	//		}
	//		prob_w[target[i]] = value + 1.0 / I;
	//    }
	//    std::map<int,float> prob_p;
	//    for (int j=0; j<J; ++j){
	//		float value = 0;
	//		if (prob_p.count(source[j])){
	//			value = prob_p[source[j]];
	//		}
	//		prob_p[source[j]] = value + 1.0 / J;
	//    }
	// TEMP ----------


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
					// The phone : source[k], word target[i]
					float p_p_w = TensorTools::AccessElement(a, Dim(k, i));
					//        				float p_w_p = p_p_w * prob_w[target[i]] /  prob_p[source[k]];
					//        				sum += p_w_p;
					sum += p_p_w;
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
			for (int j=0; j< J; ++j){
				float p_p_w = TensorTools::AccessElement(a, Dim(j, i));
				//				float p_w_p = p_p_w * prob_w[target[i]] /  prob_p[source[j]];
				//				this_word.push_back(p_w_p);
				this_word.push_back(p_p_w);
			}
		}

		alm_weight.push_back(this_word);
	}

	for (int j = 1; j < J-1; ++j) {
		int start, end;
		string phone;
		tie(start,end,phone) =  phoneme_map[docid][j-1];
		// special case when we dont care about silence and short pause char
		if ((phone=="sil") || (phone=="sp")) continue;
		assert(phone == sd.Convert(source[j]));

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
			out << td.Convert(target[max_idx]) << " " << start << " " << end <<  "\n";
		}
	}

	out.close();
}

template <class AM_t> std::vector<std::tuple<int,int,float>> beam_decode_word(AM_t &am, Corpus &data, int size){

    int docid;
    Sentence ssent, tsent;
    if (size > data.size()) size = data.size();
    std::vector<std::tuple<int,int,float>> result;
	for (unsigned i = 0; i < size; ++i) {
		tie(ssent, tsent, docid) = data[i];
		ComputationGraph cg;
		int beam = 5; // Always use greedy decoder
		std::map<int,float> word_score = am.beam_keyword(ssent, cg, beam, td,  nullptr);
		for (auto const word : word_score){
			result.push_back(tie(word.first,docid,word.second));
		}
	}
	return result;
}

//template <class AM_t> void translation_output(AM_t &am, Corpus &data, int size){
//    int docid;
//    Sentence ssent, tsent;
//    if (size > data.size()) size = data.size();
//
//	for (unsigned i = 0; i < size; ++i) {
//		tie(ssent, tsent, docid) = data[i];
//		cerr << "Generate translation for : " << docid << "\n";
//		ComputationGraph cg;
//		std::vector<int> target;
//		int beam = 5; // Always use greedy decoder
//		if (beam > 0)
//			target = am.beam_decode(ssent, cg, beam, td,  nullptr);
//		else
//			target = am.greedy_decode(ssent, cg, td, nullptr);
//
//		bool first = true;
//		cout << "Source phone \n";
//		for (auto &w: ssent) {
//			if (!first) cout << " ";
//			cout << sd.Convert(w);
//			first = false;
//		}
//		cout << endl;
//		//------------------------
//		cout << "Hypothesis : \n";
//		first = true;
//		for (auto &w: target) {
//			if (!first) cout << " ";
//			cout << td.Convert(w);
//			first = false;
//		}
//		cout << endl;
//		//------------------------
//		cout << "Gold : \n";
//		first = true;
//		for (auto &w: tsent) {
//			if (!first) cout << " ";
//			cout << td.Convert(w);
//			first = false;
//		}
//		cout << endl;
//		cout << "---------------------------------------------------\n";
//	}
//
//}

std::vector<int> find_high_tfidf_target_word(Corpus &training, Corpus& devel)
{
	// This function find the high tf.idf word in the training data that presented in the development data
	std::map<int,float> ftd;
	std::map<int,float> df;
	for (int i=0; i< training.size(); i++){
		std::vector<int> target = std::get<1>(training[i]); // get the target sentence
		std::map<int,int> unique;
		for (int j=0; j<target.size(); j++){
			float value = 0;
			if (ftd.find(target[j]) != ftd.end()) value = ftd[target[j]];
			ftd[target[j]] = value + 1;
			unique[target[j]] = 1;
		}
		// Now loop over unique
		for (auto const & idx : unique){
			int target_idx = idx.first;
			float value = 0;
			if (df.find(target_idx) != df.end()) value = df[target_idx];
			df[target_idx] = value + 1;
		}
	}
	// Now read the development data
	std::map<int,int> present_dev ;
	for (int i=0; i<devel.size(); i++){
		std::vector<int> target = std::get<1>(devel[i]);
		for (int j=0; j<target.size(); j++){
			present_dev[target[j]] = 1;
		}
	}
	// Now calculate tf.idf wrt development data
	std::vector<std::tuple<int,float>> score_words;
	for (auto const& idx : ftd){
		int target_idx = idx.first;
		if (present_dev.find(target_idx) != present_dev.end()){
			float ftd_value = idx.second;
			float df_value = df[target_idx];
			float tf_idf = std::log(1 + ftd_value) * std::log(1 + training.size() / (1.0 * df_value));
			score_words.push_back(tie(target_idx,tf_idf));
		}
	}
	// Now sort the score_words and get the first 100 words
	cout <<"WORDS size =  " << score_words.size() << endl;

    int TOP_N_WORDS = 200;
    if (TOP_N_WORDS > score_words.size()) TOP_N_WORDS = score_words.size()-10;

	std::partial_sort(score_words.begin(), score_words.begin()+TOP_N_WORDS, score_words.end(),
            [](std::tuple<int,float> &h1, std::tuple<int,float> &h2) { return std::get<1>(h1) > std::get<1>(h2); });


	std::vector<int> result ;
    cout << "OUTPUT The high  tfidf words \n";
    for (int i =0; i< TOP_N_WORDS; i++){
    	int idx;
    	float tfidf;
    	tie(idx,tfidf) = score_words[i];
    	result.push_back(idx);
//    	cout <<i << "\t"<< td.Convert(idx) << "\t" << tfidf << endl;
    }
    // output
    return result;
}

// Long Duong: this is the function to actually train the model
template <class AM_t>
void train(Model &model, AM_t &am, Corpus &training, Corpus &devel,
		Trainer &sgd, string out_file, bool curriculum, int max_epochs,
		bool doco, float coverage, bool display)
{
	double best = 9e+99;
	unsigned report_every_i = 50;
	unsigned dev_every_i_reports = 200;
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
	Sentence ssent, tsent;
	int docid;
	int MAX_DISPLAY_DEV = 100 ;
	cerr << "Using dropout with probability = " << dropout_prob << "\n";
	// FIXME: move this into sep function
	if (display) {
		// display the alignments
		//
		// Because it's not training
		am.set_dropoutprob(dropout_prob,false);
		am.set_smoothing_softmax(smooth_softmax);
		// Display the data on the training
		for (unsigned i = 0; i < training.size(); ++i) {
			if (i > MAX_DISPLAY_DEV) continue;
			tie(ssent, tsent, docid) = training[i];
			ComputationGraph cg;
			Expression alignment;

			am.BuildGraph(ssent, tsent, cg, &alignment, (doco) ? context(devel, i) : nullptr);
			cg.forward();

			cout << "\nSENTENCE : " << docid << "\n";

			am.display_ascii(ssent, tsent, cg, alignment, sd, td);
			cout << "\n";
			// Don't need to display_tikz hereq
//			am.display_tikz(ssent, tsent, cg, alignment, sd, td);
//			cout << "\n";
//			am.display_csv(ssent,tsent,cg,alignment,sd,td);
//			cout << "\n";
		}

		return;
	}
	if (output_translation_dev){
		// Here we output the translation on development data
		am.set_dropoutprob(dropout_prob,false);
		am.set_smoothing_softmax(smooth_softmax);
		cerr << "Output the translation of the development data \n";
		cout << " ################ DEV DATA ################\n";
	    am.translation_output(devel, devel.size(), sd, td);
		cout << " ################ TRAINING DATA ################\n";
		am.translation_output(training, 200, sd, td); // output first 200 sentences (for viewing only)

		////////// RETRIEVAL TASK ##########
		// Find target queries
		std::vector<int> word_list = find_high_tfidf_target_word(training, devel);
		// Get the set of retrieved list and score
		// Tuple<wordID, sentID, confident score>: that is the translation of source sentence (with ID = sentID)
		// has the wordID with the score
		std::vector<std::tuple<int,int,float>> output_beam = beam_decode_word(am,devel,devel.size());

		// Sort the output_beam according to the last value
		std::partial_sort(output_beam.begin(), output_beam.begin()+output_beam.size(), output_beam.end(),
	            [](std::tuple<int,int,float> &h1, std::tuple<int,int,float> &h2) { return std::get<2>(h1) > std::get<2>(h2); });

//		std::sort(output_beam.begin(), output_beam.end(),
//		            [](std::tuple<int,int,float> &h1, std::tuple<int,int,float> &h2) { return std::get<2>(h1) > std::get<2>(h2); });

		// Get the hash of the development data based on id
		std::map<int,Sentence> map_docid_to_target;
		std::map<int,int> map_word_to_occurance;
		for (int i=0;i<devel.size(); i++){
			tie(ssent,tsent,docid) = devel[i];
			map_docid_to_target[docid] = tsent;
			std::map<int,int> flag;
			for (int j=0; j<tsent.size(); j++){
				flag[tsent[j]] = 1;
			}
			for (auto const ele : flag){
				int value = 0;
				if (map_word_to_occurance.find(ele.first) != map_word_to_occurance.end()) value = map_word_to_occurance[ele.first];
				map_word_to_occurance[ele.first] = value + 1;
			}
		}

		cout << " ################ RETRIEVAL TASK ################\n";

		int total_prec_1 = 0;
		int total_prec_3 = 0;
		int total_prec_5 = 0;
		float total_prec = 0;
		float total_recall = 0;
		float total_fscore = 0;
		for (int i=0; i<word_list.size(); i++){
			cout<< "\nQuery (" << i << ")\t" << td.Convert(word_list[i]) << endl;
			int count = 0;
			int total_correct = 0;
			for (int j=0; j<output_beam.size(); j++){
				if (get<0>(output_beam[j]) == word_list[i]){
					count ++;
					// candidate of the match, get the score
					int docid = get<1>(output_beam[j]);
					// Output the actual sentence based on docid
					Sentence target_sent = map_docid_to_target[docid];
					cout << "Retrieved : \t";
					int check_prec_1 = 0;
					int check_prec_3 = 0;
					int check_prec_5 = 0;
					int check_prec_n = 0;
					for (int k=0; k<target_sent.size(); k++){
						cout<< td.Convert(target_sent[k]) << " ";
						if ((count <= 1) && (target_sent[k] == word_list[i])) check_prec_1 = 1;
						if ((count <= 3) && (target_sent[k] == word_list[i])) check_prec_3 = 1;
						if ((count <= 5) && (target_sent[k] == word_list[i])) check_prec_5 = 1;
						if (target_sent[k] == word_list[i]) check_prec_n = 1;
					}
					 total_prec_1 += check_prec_1;
					 total_prec_3 += check_prec_3;
					 total_prec_5 += check_prec_5;
					 total_correct += check_prec_n;
					cout << endl;
				}
			}
			// Calculate recal for this word
			float this_prec = 0;
			float this_recall = 0;
			if (count > 0)
				this_prec = total_correct / (1.0 * count);
			else
				this_prec = 0;

			cout << "Precision = " << this_prec << endl;
			total_prec +=  this_prec;
			if (map_word_to_occurance.find(word_list[i]) != map_word_to_occurance.end()){
				this_recall = total_correct / (1.0 * map_word_to_occurance[word_list[i]]);
				cout << "Recall = " << this_recall << endl;
				total_recall += this_recall;
			}
			else {
				cout << "Recall = 0 (EXPECT ERROR)" << endl;
			}
			float this_fscore = 0;
			if (this_prec + this_recall >0){
				this_fscore = 2 * this_prec * this_recall / (this_prec + this_recall) ;
			}
			cout << "F measure = " << this_fscore << endl;

			total_fscore += this_fscore;
		}

		// Calculate precision at 3
		cout << "\n============RETRIEVAL EVALUATION==================\n";
		cout << "PRECISION @ 1 = " << total_prec_1 / (word_list.size() * 1.0) << endl;
		cout << "PRECISION @ 3 = " << total_prec_3 / (word_list.size() * 3.0) << endl;
		cout << "PRECISION @ 5 = " << total_prec_5 / (word_list.size() * 5.0) << endl;
		float prec = total_prec / (word_list.size());
		float recall = total_recall / (word_list.size());
		float fscore = total_fscore / word_list.size();
		cout << "PRECISION = " <<  prec << endl;
		cout << "RECALL = " <<  recall << endl;
		cout << "FSCORE = " <<  fscore << endl;
		return;
	}

	if (is_only_alignment){
		am.set_dropoutprob(dropout_prob,false);
		am.set_smoothing_softmax(smooth_softmax);
		cerr << "Decode with alignment threshold = " << threshold_alignment << "\n";
		for (unsigned i = 0; i < training.size(); ++i) {
			if (i % 100 == 0) cerr << i << "..";
			tie(ssent, tsent, docid) = training[i];
			ComputationGraph cg;
			Expression alignment;
			am.BuildGraph(ssent, tsent, cg, &alignment, (doco) ? context(devel, i) : nullptr);
			cg.forward();
			// Generate the alignment
			generate_alignment(ssent,tsent,cg,alignment,docid, smooth_window);
		}
		return;
	}

#if 0
	if (true) {
		double dloss = 0;
		int dchars = 0;
		for (unsigned i = 0; i < devel.size(); ++i) {
			tie(ssent, tsent, docid) = devel[i];
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
				tie(ssent, tsent, docid) = training[order[si % order.size()]];
				ComputationGraph cg;
				cerr << "\nDecoding source, greedy Viterbi: ";
				am.greedy_decode(ssent, cg, td, (doco) ? context(training, order[si % order.size()]) : nullptr);

				cerr << "\nDecoding source, sampling: ";
				am.sample(ssent, cg, td, (doco) ? context(training, order[si % order.size()]) : nullptr);
			}

			// build graph for this instance
			tie(ssent, tsent, docid) = training[order[si % order.size()]];
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
				loss += as_scalar(cg.forward());
			}

			cg.backward();
			sgd.update();
			++lines;

			if (verbose) {
				cerr << "chug " << iter << "\r" << flush;
				if (iter+1 == report_every_i) {
					// display the alignment
					am.display_ascii(ssent, tsent, cg, alignment, sd, td);
					cout << "\n";
					am.display_tikz(ssent, tsent, cg, alignment, sd, td);
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
		if (report % dev_every_i_reports == 0) {
			double dloss = 0;
			int dchars = 0;
//			cerr << " DEV SIZE = " << devel.size() << "\n";
			for (unsigned i = 0; i < devel.size(); ++i) {
				tie(ssent, tsent, docid) = devel[i];
//				cerr << " Doc ID = " << docid << "\n";
				ComputationGraph cg;
				am.BuildGraph(ssent, tsent, cg, nullptr, (doco) ? context(devel, i) : nullptr);
				dloss += as_scalar(cg.forward());
//				cerr << " Lost = " << dloss << "\n";
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

void read_phone_directory(const string& foldername){
	std::vector<string> files = globVector(foldername + "/*");

	const string prefix = ".phones";

	for (unsigned int i=0; i<files.size(); ++i){
		// Find the real file name (not the path)
		int file_idx = find_real_file_name(files[i], prefix);

		// Open files[i]
		ifstream in(files[i]);
		assert(in);
		string line;
		PhonemeList phoneList;
		while (getline(in,line)){
			std::vector<string> substrs;
			boost::split(substrs, line, boost::is_any_of("\t "));
			int start_idx = atoi(substrs[0].c_str()) / 10000;
			int end_idx = atoi(substrs[1].c_str()) / 10000;
			phoneList.push_back(Phoneme(start_idx, end_idx, substrs[2]));
		}
		phoneme_map[file_idx] = phoneList;
		// cerr << file_idx <<  " with end " << std::get<1>(phoneme_map[file_idx][0]) << "\n";
	}
}

tuple<Corpus,Corpus> read_corpus_from_vars(std::map<int,string>* eng_sents, int dsize)
		{
	Corpus training,develop;
	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	int SIZE_DEV = dsize;
//	cerr << "Dsize " << dsize << endl;
	if (SIZE_DEV > eng_sents->size() / 2) SIZE_DEV = eng_sents->size() / 2;
//	cerr << "SIZE DEV = " << SIZE_DEV << endl;
	int count = 0;
	for (std::map<int,string>::iterator iter = eng_sents->begin(); iter != eng_sents->end(); ++iter){
		count ++;
		int key_idx = iter->first;
		++lc;
		if (phoneme_map.count(key_idx)){
			// Contain the same key
			string target_sent = iter->second;
			string source_sent = "";
			PhonemeList phoneme_list = phoneme_map[key_idx];
			for (std::vector<Phoneme>::iterator iter_phone = phoneme_list.begin(); iter_phone != phoneme_list.end(); ++ iter_phone){
				string phoneme_value = get<2>(*iter_phone);
				source_sent = source_sent + " " + phoneme_value;
			}
			//--------------------------
			// Add the start and end
			string sent_pair = "<s> " + source_sent + " </s> ||| <s> " + target_sent + " </s>";
			Sentence source, target;
			ReadSentencePair(sent_pair, &source, &sd, &target, &td);
			if (SIZE_DEV ==0)
				training.push_back(SentencePair(source,target,key_idx));
			else
				if (eng_sents->size() - count <= SIZE_DEV)
					develop.push_back(SentencePair(source,target,key_idx));
				else
					training.push_back(SentencePair(source,target,key_idx));

			stoks += source.size();
			ttoks += target.size();
			if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
					(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
				cerr << "Sentence in file " << key_idx << " didn't start or end with <s>, </s>\n";
				abort();
			}
		}
	}

	cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
	cerr << "Pick " << develop.size() << " sentence from train as development (also use for showing alignment) \n";
	return tie(training,develop);
	}


Corpus read_corpus(const string &filename, bool doco)
{
	ifstream in(filename);
	assert(in);
	Corpus corpus;
	string line;
	int lc = 0, stoks = 0, ttoks = 0;
	vector<int> identifiers({ -1 });
	while (getline(in, line)) {
		++lc;
		Sentence source, target;
		if (doco)
			ReadNumberedSentencePair(line, &source, &sd, &target, &td, identifiers);
		else
			ReadSentencePair(line, &source, &sd, &target, &td);

		//cerr << "identifier " << identifiers[0] << '\n';

		//corpus.push_back(SentencePair(source, target, identifiers[0]));
		corpus.push_back(SentencePair(source, target, lc)); // Rather use the lc as the index
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

std::vector<int> ReadNumberedSentence(const std::string& line, Dict* sd, vector<int> &identifiers) {
	std::istringstream in(line);
	std::string word;
	std::vector<int> res;
	std::string sep = "|||";
	if (in) {
		identifiers.clear();
		while (in >> word) {
			if (!in || word.empty()) break;
			if (word == sep) break;
			identifiers.push_back(atoi(word.c_str()));
		}
	}

	while(in) {
		in >> word;
		if (!in || word.empty()) break;
		res.push_back(sd->Convert(word));
	}
	return res;
}


void ReadNumberedSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, vector<int> &identifiers) 
{
	cerr << "Read numbered Sentence Pair " << line << endl;
	std::istringstream in(line);
	std::string word;
	std::string sep = "|||";
	Dict* d = sd;
	std::vector<int>* v = s;

	if (in) {
		identifiers.clear();
		while (in >> word) {
			if (!in || word.empty()) break;
			if (word == sep) break;
			identifiers.push_back(atoi(word.c_str()));
		}
	}

	while(in) {
		in >> word;
		if (!in) break;
		cerr << word << " ";
		if (word == sep) {
			cerr << " Switch to target DICT" << endl;
			d = td; v = t; continue;
		}
		v->push_back(d->Convert(word));
	}
}

void initialise(Model &model, const string &filename)
{
	cerr << "Initialising model parameters from file: " << filename << endl;
	ifstream in(filename);
	boost::archive::text_iarchive ia(in);
	ia >> model;
}

const Sentence* context(const Corpus &corpus, unsigned i)
{
	if (i > 0) {
		int docid = get<2>(corpus.at(i));
		int prev_docid = get<2>(corpus.at(i-1));
		if (docid == prev_docid)
			return &get<0>(corpus.at(i-1));
	}
	return nullptr;
}
