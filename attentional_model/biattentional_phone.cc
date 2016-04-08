#include "attentional.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <glob.h>
#include <vector>
#include <sys/stat.h>

#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>


using namespace std;
using namespace cnn;
using namespace boost::program_options;

unsigned LAYERS = 4;
unsigned HIDDEN_DIM = 128;  // 1024
unsigned ALIGN_DIM = 128;  // 1024
unsigned VOCAB_SIZE_SRC = 0;
unsigned VOCAB_SIZE_TGT = 0;
bool BIDIR = true;
bool GIZA = true;

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;
float dropout_prob = 0;
int smooth_softmax = 1;
#define WTF(expression) \
		cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << endl;

template <class Builder>
struct BidirAttentionalModel {
	AttentionalModel<Builder> s2t_model;
	AttentionalModel<Builder> t2s_model;
	double m_trace_weight;
	Expression s2t_align, t2s_align;
	Expression s2t_xent, t2s_xent, trace_bonus;

	explicit BidirAttentionalModel(Model *model, double trace_weight, double _smooth_sm)
	: s2t_model(model, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, LAYERS,
			HIDDEN_DIM, ALIGN_DIM, BIDIR, GIZA, false, -1), // the last one is the local attentional model
			t2s_model(model, VOCAB_SIZE_TGT, VOCAB_SIZE_SRC, LAYERS,
					HIDDEN_DIM, ALIGN_DIM, BIDIR, GIZA, false, -1),
					m_trace_weight(trace_weight)
	{
		m_trace_weight = trace_weight;

		// If we want to initialise the attentional model aside from construction, do it here
		s2t_model.set_smoothing_softmax(_smooth_sm);
		t2s_model.set_smoothing_softmax(_smooth_sm);
	}

	// return Expression of total loss
	Expression build_graph(const vector<int> &source, const vector<int>& target, ComputationGraph& cg)
	{
		// FIXME: slightly wasteful, the embeddings of the source and target are done twice each

		s2t_xent = s2t_model.BuildGraph(source, target, cg, &s2t_align);
		t2s_xent = t2s_model.BuildGraph(target, source, cg, &t2s_align);

		trace_bonus = trace_of_product(t2s_align, transpose(s2t_align));
		//cout << "xent: src=" << *cg.get_value(src_xent.i) << " tgt=" << *cg.get_value(tgt_xent.i) << endl;
		//cout << "trace bonus: " << *cg.get_value(trace_bonus.i) << endl;

		return s2t_xent + t2s_xent - m_trace_weight * trace_bonus; // What is the
		//return src_xent + tgt_xent;
	}

	void initialise(const std::string &src_file, const std::string &tgt_file, Model &model)
	{
		Model sm, tm;
		AttentionalModel<Builder> smb(&sm, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT,
				LAYERS, HIDDEN_DIM, ALIGN_DIM, BIDIR, GIZA, false, -1);
		AttentionalModel<Builder> tmb(&tm, VOCAB_SIZE_TGT, VOCAB_SIZE_SRC,
				LAYERS, HIDDEN_DIM, ALIGN_DIM, BIDIR, GIZA, false, -1);

		//for (const auto &p : sm.lookup_parameters_list())
		//std::cerr << "\tlookup size: " << p->values[0].d << " number: " << p->values.size() << std::endl;
		//for (const auto &p : sm.parameters_list())
		//std::cerr << "\tparam size: " << p->values.d << std::endl;

		std::cerr << "... loading " << src_file << " ..." << std::endl;
		{
			ifstream in(src_file);
			boost::archive::text_iarchive ia(in);
			ia >> sm;
		}

		std::cerr << "... loading " << tgt_file << " ..." << std::endl;
		{
			ifstream in(tgt_file);
			boost::archive::text_iarchive ia(in);
			ia >> tm;
		}
		std::cerr << "... merging parameters ..." << std::endl;

		unsigned lid = 0;
		auto &lparams = model.lookup_parameters_list();
		assert(lparams.size() == 2*sm.lookup_parameters_list().size());
		for (const auto &p : sm.lookup_parameters_list())  {
			for (unsigned i = 0; i < p->values.size(); ++i)
				memcpy(lparams[lid]->values[i].v, &p->values[i].v[0], sizeof(cnn::real) * p->values[i].d.size());
			lid++;
		}
		for (const auto &p : tm.lookup_parameters_list()) {
			for (unsigned i = 0; i < p->values.size(); ++i)
				memcpy(lparams[lid]->values[i].v, &p->values[i].v[0], sizeof(cnn::real) * p->values[i].d.size());
			lid++;
		}
		assert(lid == lparams.size());

		unsigned did = 0;
		auto &dparams = model.parameters_list();
		for (const auto &p : sm.parameters_list())
			memcpy(dparams[did++]->values.v, &p->values.v[0], sizeof(cnn::real) * p->values.d.size());
		for (const auto &p : tm.parameters_list())
			memcpy(dparams[did++]->values.v, &p->values.v[0], sizeof(cnn::real) * p->values.d.size());
		assert(did == dparams.size());
	}
};

//########################### END HEADER DEFINITION #########################


typedef vector<int> Sentence;
//typedef pair<Sentence, Sentence> SentencePair;
typedef std::tuple<Sentence, Sentence, int> SentencePair; // includes document id (optional)
typedef std::vector<SentencePair> Corpus ;
// Long Duong : define new tuple
typedef tuple<int,int, std::string> Phoneme;
typedef vector<Phoneme> PhonemeList;
typedef std::map<int,PhonemeList> PhonemeMap;

PhonemeMap phoneme_map;

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

tuple<Corpus,Corpus> read_corpus_from_vars(std::map<int,string>* eng_sents, int dsize){

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


//template <class AM_t> // Fix LSTM Builder
void test_rescore(Model &model, AttentionalModel<LSTMBuilder> &am, string test_file, bool doco)
{
	double tloss = 0;
	int tchars = 0;
	int lno = 0;

	cerr << "Reading test examples (for rescoring) from " << test_file << endl;
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
		ReadSentencePair(line, &source, &sd, &target, &td);
		if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
				(target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
			cerr << "Sentence in " << test_file << ":" << lno << " didn't start or end with <s>, </s>\n";
			abort();
		}

		ComputationGraph cg;
		// Build graph
		am.BuildGraph(source, target, cg, nullptr, nullptr);

		double loss = as_scalar(cg.forward());

		// LD: Need to think more about this, base sololy on length is abit underestimating
		loss = loss / (1.0 * (target.size()-1));
		cout << original_line << " ||| " << loss << endl;

		tloss += loss;
		tchars += target.size() - 1;

		if (doco) {
			last_source = source;
			last_docid = num[0];
		}
	}

	cerr << "\n***TEST E = " << (tloss / tchars) << " ppl=" << exp(tloss / tchars) << ' ';
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

int main(int argc, char** argv) {
	cnn::Initialize(argc, argv);

	// command line processing
	variables_map vm;
	options_description opts("Allowed options");
	opts.add_options()
        						("help", "print help message")
								("strain,st", value<string>(), "source (phone directory) use for training.")
								("train,t", value<string>(), "file containing training sentences, with ")
								("ttrain,tt", value<string>(), "target (english folder) use for training.")
								("test,T", value<string>(), "file containing testing sentences")
								("rescore,r", "rescore (source, target) pairs in testing, default: translate source only")
								("devel,d", value<string>(), "file containing development sentences.")
								("dsize", value<int>()->default_value(500), "Use this much data from train as dev (-1 : use all for training)")
								("initialise_s,i", value<string>(), "load initial parameters for (source -> target) from file")
								("initialise_t,i", value<string>(), "load initial parameters for (target -> source) from file")
								("initialise_a,i", value<string>(), "load initial parameters for both direction from file")
								("parameters,p", value<string>(), "save best parameters to this file")
								("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
								("align,a", value<int>()->default_value(ALIGN_DIM), "use <num> dimensions for alignment projection")
								("dropout", value<float>()->default_value(0), "Dropout all the non-recurrent of RNN")
								("hidden,h", value<int>()->default_value(HIDDEN_DIM), "use <num> dimensions for recurrent hidden states")
								("epochs,e", value<int>()->default_value(50), "maximum number of training epochs")
								("bidirectional", "use bidirectional recurrent hidden states as source embeddings, rather than word embeddings")
								("giza", "use GIZA++ style features in attentional components")
								("curriculum", "use 'curriculum' style learning, focusing on easy problems in earlier epochs")
								("swap", "swap roles of source and target, i.e., learn p(source|target)")
								("document,D", "use previous sentence as document context; requires document id prefix in input files")
								("coverage,C", value<float>()->default_value(0.0f), "impose alignment coverage penalty in training, with given coefficient")
								("display", "just display alignments instead of training or decoding")
								("alignment", "take the pretrained model and output the alignment wrt phone alignment")
								("translation", "take the pretrained model and output the translation on the development data")
								("aliout,A", value<string>(), "the alignment output folder.")
								("smooth", value<int>()->default_value(0), "Use smoothing for alignment generation, put (smooth) frame before and after current frame together")
								("smoothsm", value<float>()->default_value(1), "Use the softmax smoothing constant ")
								("threshold,H", value<float>()->default_value(0.0f), "Threshold for choosing the alignment (confidence must greater than threshold)")
								("verbose,v", "be extremely chatty")
								;
	// Compusary parameters
	store(parse_command_line(argc, argv, opts), vm);
	notify(vm);
	if (vm.count("help") || (vm.count("train") != 1 && vm.count("strain") != 1)) {
		cout << opts << "\n";
		return 1;
	}

	kSRC_SOS = sd.Convert("<s>");
	kSRC_EOS = sd.Convert("</s>");
	kTGT_SOS = td.Convert("<s>");
	kTGT_EOS = td.Convert("</s>");

	LAYERS = vm["layers"].as<int>();
	ALIGN_DIM = vm["align"].as<int>();
	HIDDEN_DIM = vm["hidden"].as<int>();

	Corpus training, dev;
	if (vm.count("strain")) {
	cerr << "Reading source training data from " << vm["strain"].as<string>() << "...\n";
	cerr << "Reading target training data from " << vm["ttrain"].as<string>() << "...\n";
	// Long Duong
	read_phone_directory(vm["strain"].as<string>());
	std::map<int,string> eng_sents = read_target_directory(vm["ttrain"].as<string>());
	// End
	cerr << "Using " << vm["dsize"].as<int>() << " for development data" << endl;
	// training = read_corpus(vm["train"].as<string>(), doco);
	tie(training,dev) = read_corpus_from_vars(&eng_sents, vm["dsize"].as<int>());
	}
	if (vm.count("train")){
		cerr << "Reading train data from " << vm["train"].as<string>() << "...\n";
		training = read_corpus(vm["train"].as<string>(), false);
	}

	sd.Freeze(); // no new word types allowed
	td.Freeze(); // no new word types allowed
	VOCAB_SIZE_SRC = sd.size();
	VOCAB_SIZE_TGT = td.size();

	if (vm.count("devel")) {
		cerr << "Reading dev data from " << vm["devel"].as<string>() << "...\n";
		dev = read_corpus(vm["devel"].as<string>(), false);
	}

	if (dev.size() <= 0){
		// Use 500 from train as dev
		for (int i=0; i<500 && i < training.size() ; i++){
			dev.push_back(training[i]);
		}
	}

	string fname;
	if (vm.count("parameters")) {
		fname = vm["parameters"].as<string>();
	} else {
	ostringstream os;
	os << "bam"
			<< '_' << LAYERS
			<< '_' << HIDDEN_DIM
			<< '_' << ALIGN_DIM
			<< "_lstm"
			<< "_b" << BIDIR
			<< "_g" << GIZA
			<< "-pid" << getpid() << ".params";
	fname = os.str();
	}

	cerr << "Parameters will be written to: " << fname << endl;
	double best = 9e+99;

	Model model;
	Trainer* sgd = new SimpleSGDTrainer(&model);

	BidirAttentionalModel<LSTMBuilder> am(&model, 0.1, vm["smoothsm"].as<float>());

	if (vm.count("initialise_a")){
		string fname = vm["initialise_a"].as<string>();
		cerr << "Parameters will be loaded form file " << fname << endl;
		ifstream in(fname);
		boost::archive::text_iarchive ia(in);
		ia >> model;
	}
	if (vm.count("initialise_s") && vm.count("initialise_t")){
		string s_file = vm["initialise_s"].as<string>();
		string t_file = vm["initialise_t"].as<string>();
		cerr << "initialising from " << s_file << " and " << t_file << endl;
		am.initialise(s_file, t_file, model);
	}

	// This will do for translation
	if (vm.count("translation")){
		cerr << "Output the translation of the development data \n";
		cout << " ################ DEV DATA ################\n";
		am.s2t_model.translation_output(dev,dev.size(), sd,td);
		cout << " ################ TRAINING DATA ################\n";
		am.s2t_model.translation_output(training, 500, sd, td); // output first 500 sentences
		return 1;
	}

	if (vm.count("rescore")){
		// Do rescoring for file in test
		test_rescore(model, am.s2t_model,  vm["test"].as<string>(), false);
		return 1;
	}

	// THIS IS FOR TESTING ONLY
	if (false) {
		double dloss = 0, dloss_s2t = 0, dloss_t2s = 0, dloss_trace = 0;
		unsigned dchars_s = 0, dchars_t = 0, dchars_tt = 0;
		unsigned i = 0;
		for (auto& spair : dev) {
			ComputationGraph cg;
			am.build_graph(get<0>(spair), get<1>(spair), cg);
			dloss += as_scalar(cg.incremental_forward());
			dloss_s2t += as_scalar(cg.get_value(am.s2t_xent.i));
			dloss_t2s += as_scalar(cg.get_value(am.t2s_xent.i));
			dloss_trace += as_scalar(cg.get_value(am.trace_bonus.i));
			dchars_s += get<0>(spair).size() - 1;
			dchars_t += get<1>(spair).size() - 1;
			dchars_tt += std::max(get<0>(spair).size(), get<1>(spair).size()) - 1;

			if (true)
			{
				cout << "\n===== SENTENCE " << i++ << " =====\n";
				am.s2t_model.display_tikz(get<0>(spair), get<1>(spair), cg, am.s2t_align, sd, td);
				cout << "\n";
				am.t2s_model.display_tikz(get<1>(spair), get<0>(spair), cg, am.t2s_align, td, sd);
				cout << "\n";
			}
		}
		if (dloss < best) {
			best = dloss;
			ofstream out(fname);
			boost::archive::text_oarchive oa(out);
			oa << model;
		}
		cerr << "\n***DEV [epoch=0]";
		cerr << " E = " << (dloss / (dchars_s + dchars_t)) << " ppl=" << exp(dloss / (dchars_s + dchars_t)) << ' '; // kind of hacky, as trace should be normalised differently
		cerr << " ppl_s = " << exp(dloss_t2s / dchars_s) << ' ';
		cerr << " ppl_t = " << exp(dloss_s2t / dchars_t) << ' ';
		cerr << " trace = " << exp(dloss_trace / dchars_tt) << ' ';
	}

	unsigned report_every_i = 50;
	unsigned dev_every_i_reports = 200;
	if (training.size() < 1000) dev_every_i_reports = 5;

	unsigned si = training.size();
	vector<unsigned> order(training.size());
	for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

	bool first = true;
	int report = 0;
	unsigned lines = 0;
	int max_epochs = vm["epochs"].as<int>();
	while (sgd->epoch < max_epochs) {
		Timer iteration("completed in");
		double loss = 0, loss_s2t = 0, loss_t2s = 0, loss_trace = 0;
		unsigned chars_s = 0, chars_t = 0, chars_tt = 0;
		for (unsigned i = 0; i < report_every_i; ++i) {
			if (si == training.size()) {
				si = 0;
				if (first) { first = false; } else { sgd->update_epoch(); }
				cerr << "**SHUFFLE\n";
				shuffle(order.begin(), order.end(), *rndeng);
			}

			// build graph for this instance
			ComputationGraph cg;
			auto& spair = training[order[si]];
			chars_s += get<0>(spair).size() - 1;
			chars_t += get<1>(spair).size() - 1;
			chars_tt += std::max(get<0>(spair).size(), get<1>(spair).size()) - 1; // max or min?
			++si;
			am.build_graph(get<0>(spair),get<1>(spair), cg);
			loss += as_scalar(cg.forward());
			loss_s2t += as_scalar(cg.get_value(am.s2t_xent.i));
			loss_t2s += as_scalar(cg.get_value(am.t2s_xent.i));
			loss_trace += as_scalar(cg.get_value(am.trace_bonus.i));
			cg.backward();
			sgd->update();
			++lines;

			//if ((i+1) == report_every_i) {
//			if (si == 1) {
//				//Expression aligns = concatenate({transpose(am.src_align), am.tgt_align});
//				//cerr << cg.get_value(aligns.i) << "\n";
//				am.s2t_model.display_ascii(get<0>(spair), get<1>(spair), cg, am.s2t_align, sd, td);
//				am.t2s_model.display_ascii(get<1>(spair), get<0>(spair), cg, am.t2s_align, td, sd);
//				cerr << "\txent_s2t " << as_scalar(cg.get_value(am.s2t_xent.i))
//		    						 << "\txent_t2s " << as_scalar(cg.get_value(am.t2s_xent.i))
//									 << "\ttrace " << as_scalar(cg.get_value(am.trace_bonus.i)) << endl;
//			}
		}
		sgd->status();
		cerr << " E = " << (loss / (chars_s + chars_t)) << " ppl=" << exp(loss / (chars_s + chars_t)) << ' '; // kind of hacky, as trace should be normalised differently
		cerr << " ppl_s = " << exp(loss_t2s / chars_s) << ' ';
		cerr << " ppl_t = " << exp(loss_s2t / chars_t) << ' ';
		cerr << " trace = " << exp(loss_trace / chars_tt) << ' ';

		// show score on dev data?
		report++;
		if (report % dev_every_i_reports == 0) {
			double dloss = 0, dloss_s2t = 0, dloss_t2s = 0, dloss_trace = 0;
			unsigned dchars_s = 0, dchars_t = 0, dchars_tt = 0;
			for (auto& spair : dev) {
				ComputationGraph cg;
				am.build_graph(get<0>(spair), get<1>(spair), cg);
				dloss += as_scalar(cg.incremental_forward());
				dloss_s2t += as_scalar(cg.get_value(am.s2t_xent.i));
				dloss_t2s += as_scalar(cg.get_value(am.t2s_xent.i));
				dloss_trace += as_scalar(cg.get_value(am.trace_bonus.i));
				dchars_s += get<0>(spair).size() - 1;
				dchars_t += get<1>(spair).size() - 1;
				dchars_tt += std::max(get<0>(spair).size(), get<1>(spair).size()) - 1;
			}
			if (dloss < best) {
				cerr << "\n\n ====== Save to output file : " << fname <<  "====== \n";
				best = dloss;
				ofstream out(fname);
				boost::archive::text_oarchive oa(out);
				oa << model;
			}
			cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "]";
			cerr << " E = " << (dloss / (dchars_s + dchars_t)) << " ppl=" << exp(dloss / (dchars_s + dchars_t)) << ' '; // kind of hacky, as trace should be normalised differently
			cerr << " ppl_s = " << exp(dloss_t2s / dchars_s) << ' ';
			cerr << " ppl_t = " << exp(dloss_s2t / dchars_t) << ' ';
			cerr << " trace = " << exp(dloss_trace / dchars_tt) << ' ';
		}
	}
	delete sgd;
}

