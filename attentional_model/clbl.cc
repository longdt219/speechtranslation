#include <tuple>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>

#include "clbl.h"

#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 


using namespace std;
using namespace cnn;

using namespace boost::program_options;
using namespace boost::filesystem;

unsigned LAYERS = 2;         // 2 recommended
unsigned EMBEDDING_DIM = 64; // 256 recommended
unsigned HIDDEN_DIM = 128;   // 1024 recommended

unsigned MAX_EPOCH = 100;
unsigned WRITE_EVERY_I=1000;


typedef std::vector<int> Sentence;
typedef vector<Sentence> Corpus;
typedef std::map<int, Sentence> MorphCorpus;
cnn::Dict sdic; //words
cnn::Dict mdic; //morphemes

template <class LM_t>
void train(Model &model, LM_t &lm, Corpus &training, Corpus &devel, MorphCorpus &morphs,
	Trainer &sgd, string out_file, bool curriculum);

// Monolingual corpus: one sentences per line
Corpus read_corpus(const string &filename)
{
    ifstream in(filename);
    assert(in);
    Corpus corpus;
    string line;
    int lc = 0, stoks = 0;
    while(getline(in, line)) {
        ++lc;
        Sentence sent;
	std::vector<int> encoded=ReadSentence(line, &sdic);
        corpus.push_back(encoded);
        stoks += encoded.size();
         }
    cerr << lc << " lines\n";
    return corpus;
}


//  INPUT FILE FORMAT:  [word morph_0 morph_1 morph_2 ... morph_n]
MorphCorpus read_morphs(const std::string &filename)
{
    MorphCorpus morphemes; 
    ifstream in(filename.c_str());
    
    string buf, token;
    int morpheme;
    bool state=false;
    while (getline(in, buf))
    {  
        istringstream ss(buf);
        while(ss >> token){
	    if (!state) {
		morpheme=sdic.Convert(token);	
		morphemes.insert(std::make_pair(morpheme, Sentence()));
              }
	    else {
	 	morphemes[morpheme].push_back(mdic.Convert(token));	// replace []
	 }
	  state=true;
	}
         state=false;
    }
    return morphemes;
}



int main(int argc, char **argv)
{
    cnn::Initialize(argc, argv);

    // command line processing
    Corpus training, devel;
    MorphCorpus morphemes;
    variables_map vm; 
    options_description opts("Allowed options");
    opts.add_options()
        ("help", "print help message")
        ("config,c", value<string>(), "config file specifying additional command line options")
        ("input,w", value<string>(), "file containing training sentences. ")
	("morphs,m", value<string>(), "file containing morphs. ")
        ("devel,d", value<string>(), "file containing development sentences (see --input)")
	("parameters,p", value<string>(), "save best parameters to this file")
        ("epochs,e", value<int>(), "max number of epochs")
        ("layers,l", value<int>()->default_value(LAYERS), "use <num> layers for RNN components")
        ("embedding,n", value<int>()->default_value(EMBEDDING_DIM), "use <num> dimensions for word embeddings")
    ;
    store(parse_command_line(argc, argv, opts), vm); 
    if (vm.count("config") > 0)
    {
        ifstream config(vm["config"].as<string>().c_str());
        store(parse_config_file(config, opts), vm); 
    }
    notify(vm);
    
    if (vm.count("help") || vm.count("input") != 1 || vm.count("morphs") != 1) {
        cout << opts << "\n";
        return 1;
    }
    if (vm.count("input"))          //Fixme: check the existence of file with exists()
       training = read_corpus(vm["input"].as<string>());
    if (vm.count("morphs"))          //Fixme: check the existence of file with exists()
       morphemes = read_morphs(vm["morphs"].as<string>());
    if (vm.count("devel"))          //Fixme: check the existence of file with exists()
       devel = read_corpus(vm["devel"].as<string>());
   
   // if (vm.count("report")) WRITE_EVERY_I = vm["report"].as<int>();
    if (vm.count("epochs")) MAX_EPOCH = vm["epochs"].as<int>();
   
    cout << "%% Training has " << training.size() << " sentences\n";
    cout << "%% Development has " << devel.size() << " sentences\n";
    cout << "%% source vocab " << morphemes.size() << " unique morphemes\n";
   
    Model model;
    Trainer* sgd = new SimpleSGDTrainer(&model);
   /* string init_file;
    if (vm.count("initialise"))
	init_file = vm["initialise"].as<string>();*/

    string test;  //
    if (vm.count("test")) 
	test = vm["test"].as<string>();

    if (vm.count("layers")) LAYERS = vm["layers"].as<int>(); 
    if (vm.count("embedding")) EMBEDDING_DIM = vm["embedding"].as<int>(); 
    if (vm.count("hidden")) HIDDEN_DIM = vm["hidden"].as<int>(); 
    cout << "%% layers " << LAYERS << " embedding " << EMBEDDING_DIM << endl;
   
    string fname;
    if (vm.count("parameters")) {
	fname = vm["parameters"].as<string>();
    } else {    
    	ostringstream os;
    	os << "clbl"
   	   << '_' << LAYERS
	   << '_' << HIDDEN_DIM
       	   << '_' << EMBEDDING_DIM
           << "-pid" << getpid() << ".params";
    	  fname = os.str();
    }
    cerr << "Parameters will be written to: " << fname << endl;

    cout << "%% Using CLBL with recurrent units %%" << endl;
    CLBLRNN<SimpleRNNBuilder> lm(model, sdic.size(), mdic.size(), LAYERS, EMBEDDING_DIM, HIDDEN_DIM);
    train(model, lm, training, devel, morphemes, *sgd, test, vm.count("curriculum"));
    

    delete sgd;

    return EXIT_SUCCESS;
}

template <class LM_t>
void train(Model &model, LM_t &lm, Corpus &training, Corpus &devel, MorphCorpus &morphs,
	Trainer &sgd, string out_file, bool curriculum)
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
	    lengths.insert(make_pair(training[i].size(), i));

	order_by_length.resize(curriculum_steps);
	unsigned i = 0;
	for (auto& landi: lengths) {
	    for (unsigned k = i * curriculum_steps / lengths.size(); k < curriculum_steps; ++k)  
		order_by_length[k].push_back(landi.second);
	    ++i;
	}
    }
    //bool verbose = false; //nb
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
            // build graph for this instance
	    auto& sent = training[order[si]];
            if (sent.size()<2) { ++si; continue;} //NB! should we allow size = 1?
	    cout<< "\nSentence: " ;
	    for (auto word: sent){
	   	cout << " " << sdic.Convert(word);}//<< " Morphemes: ";
 	//  	for (auto morpheme: morphs.at(word))
	//  	cout << mdic.Convert(morpheme) << ",";
	//	}
	  //  cout << "FIN" << endl;
	    ComputationGraph cg;
            chars += sent.size(); //used to be -1
            ++si;
            lm.BuildLMGraph(sent, morphs, cg);
            loss += as_scalar(cg.forward());
            
            cg.backward();
            sgd.update();
            ++lines;

	    
        }
	if (chars!=0){
        sgd.status();
	cout << "    LOSS=" << loss << " Chars=" << chars << endl;
        cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0;
            int dchars = 0;
            for (auto& sent : devel) {
                ComputationGraph cg;
                lm.BuildLMGraph(sent, morphs, cg);
                dloss += as_scalar(cg.forward());
                dchars += sent.size() - 1;
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
}
