#include "dam.h"

#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <set>
#include <map>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned LAYERS = 2;      // 2
unsigned INPUT_DIM = 64;  // 256
unsigned HIDDEN_DIM = 128;// 1024
unsigned ALIGN_DIM = 64;  // 512
unsigned VOCAB_SIZE = 0;

cnn::Dict d;
int kSOS;
int kEOS;

typedef vector<int> Sentence;
typedef vector<Sentence> Document;
typedef vector<Document> Corpus;

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;

void read_documents(const std::string &filename, Corpus &text);


int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);
    if (argc != 3 && argc != 4) {
        cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]" << endl;
        return 1;
    }
    kSOS = d.Convert("<s>");
    kEOS = d.Convert("</s>");

    // load the corpora
    Corpus training, dev;
    cerr << "Reading training data from " << argv[1] << "..." << endl;
    read_documents(argv[1], training);
    d.Freeze(); // no new word types allowed
    VOCAB_SIZE = d.size();
    cerr << "Reading dev data from " << argv[2] << "..." << endl;
    read_documents(argv[2], dev);

    ostringstream os;
    os << "dam"
        << '_' << LAYERS
        << '_' << INPUT_DIM
        << '_' << HIDDEN_DIM
        << '_' << ALIGN_DIM
        << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;
    double best = 9e+99;

    Model model;
    Trainer* sgd = new SimpleSGDTrainer(&model);

    DocumentAttentionalModel<LSTMBuilder> lm(model,
            VOCAB_SIZE, LAYERS, INPUT_DIM, HIDDEN_DIM, ALIGN_DIM);
    if (argc == 4) {
        string fname = argv[3];
        ifstream in(fname);
        boost::archive::text_iarchive ia(in);
        ia >> model;
    }

    unsigned report_every_i = 5;
    unsigned dev_every_i_reports = 100;
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
        for (unsigned i = 0; i < report_every_i; ++i) {
            if (si == training.size()) {
                si = 0;
                if (first) { first = false; } else { sgd->update_epoch(); }
                cerr << "**SHUFFLE" << endl;
                shuffle(order.begin(), order.end(), *rndeng);
            }

            // build graph for this instance
            ComputationGraph cg;
            auto& doc = training[order[si]];
            for (auto &sent: doc)
                chars += sent.size() - 1;
            ++si;
            //cerr << "sent length " << sent.size();
            lm.BuildGraph(doc, cg);
            loss += as_scalar(cg.forward());
            cg.backward();
            sgd->update();
            ++lines;
        }
        sgd->status();
        // FIXME: is chars incorrect?
        cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';
        //lm.RandomSample(); // why???

        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0;
            int dchars = 0;
            for (unsigned i = 0; i < dev.size(); ++i) {
                const auto& doc = dev[i];
                ComputationGraph cg;
                lm.BuildGraph(doc, cg);
                dloss += as_scalar(cg.forward());
                for (auto &sent: doc)
                    dchars += sent.size() - 1;
            }
            if (dloss < best) {
                best = dloss;
                ofstream out(fname);
                boost::archive::text_oarchive oa(out);
                oa << model;
            }
            cerr << "***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << ' ';
        }
    }
    delete sgd;
}

void read_documents(const std::string &filename, Corpus &corpus) {
    ifstream in(filename);
    assert(in);
    int toks = 0, lno = 0;
    string line;
    Document doc;
    while(std::getline(in, line)) {
        ++lno;
        auto sentence = ReadSentence(line, &d);
        if (sentence.empty()) {
            // empty lines separate documents
            corpus.push_back(doc);
            doc.clear();
        } else {
            if (sentence.front() != kSOS && sentence.back() != kEOS) {
                cerr << "Sentence in " << filename << ":" << lno << " didn't start or end with <s>, </s>" << endl;
                abort();
            }
            doc.push_back(sentence);
            toks += sentence.size();
        }
    }
    assert(doc.empty());

    cerr << corpus.size() << " documents, " << (lno-corpus.size()) << " sentences, " << toks << " tokens, " << d.size() << " types" << endl;
}
