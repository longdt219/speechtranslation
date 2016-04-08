#include "attentional_constraints.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned LAYERS = 2;
unsigned HIDDEN_DIM = 256;  // 1024
unsigned ALIGN_DIM = 64;   // 128
unsigned VOCAB_SIZE_SRC = 0;
unsigned VOCAB_SIZE_TGT = 0;

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;

typedef vector<int> Sentence;
typedef pair<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> Corpus;

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

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);
    if (argc != 3 && argc != 4) {
        cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
        return 1;
    }
    kSRC_SOS = sd.Convert("<s>");
    kSRC_EOS = sd.Convert("</s>");
    kTGT_SOS = td.Convert("<s>");
    kTGT_EOS = td.Convert("</s>");

    typedef vector<int> Sentence;
    typedef pair<Sentence, Sentence> SentencePair;
    vector<SentencePair> training, dev;
    string line;
    cerr << "Reading training data from " << argv[1] << "...\n";
    training = read_corpus(argv[1]);
    sd.Freeze(); // no new word types allowed
    td.Freeze(); // no new word types allowed
    VOCAB_SIZE_SRC = sd.size();
    VOCAB_SIZE_TGT = td.size();

    cerr << "Reading dev data from " << argv[2] << "...\n";
    dev = read_corpus(argv[2]);

    ostringstream os;
    os << "am"
        << '_' << LAYERS
        << '_' << HIDDEN_DIM
        << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;
    double best = 9e+99;

    Model model;
    bool use_momentum = false;
    SimpleSGDTrainer* sgd = nullptr;
    //if (use_momentum)
        //sgd = new MomentumSGDTrainer(&model);
    //else
    sgd = new SimpleSGDTrainer(&model);

    //AttentionalConstraintModel<SimpleRNNBuilder> am(model, LAYERS, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, HIDDEN_DIM, ALIGN_DIM, 1);
    AttentionalConstraintModel<LSTMBuilder> am(model, LAYERS, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, HIDDEN_DIM, ALIGN_DIM, 2);
    //AttentionalConstraintModel<GRUBuilder> am(model, LAYERS, VOCAB_SIZE_SRC, VOCAB_SIZE_TGT, HIDDEN_DIM, ALIGN_DIM, 1);

    // record the list of model parameters
    vector<Parameters*> model_parameters(model.parameters_list());
    vector<LookupParameters*> model_lookup_parameters(model.lookup_parameters_list());

    // for the constraints
    vector<Parameters*> lagrange_multipliers;
    for (const auto &instance: training) {
        unsigned slen = instance.first.size()-1;
        lagrange_multipliers.push_back(model.add_parameters({slen}));
    }

    if (argc == 4) {
        string fname = argv[3];
        ifstream in(fname);
        boost::archive::text_iarchive ia(in);
        ia >> model;
    }

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
        double violations = 0;
        unsigned chars = 0;
        bool minimising_loss = true;
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
            auto& lagrange = lagrange_multipliers[order[si]];
            chars += spair.second.size() - 1;
            ++si;
            auto penalised_loss = am.BuildGraph(spair.first, spair.second, cg, lagrange);
            if (!minimising_loss) {
                // maximising loss wrt lagrange
                Expression obj = -1 * (penalised_loss.first + penalised_loss.second);
                cg.forward();
                cg.backward();
                sgd->update({}, lagrange_multipliers);
                minimising_loss = true;
            } else {
                // minimising loss wrt lagrange
                Expression obj = penalised_loss.first + penalised_loss.second; // this is needed, cg.forward() operates over the last Expression
                cg.forward();
                cg.backward();
                sgd->update(model_lookup_parameters, model_parameters);
                minimising_loss = false;
            }
            loss += as_scalar(cg.get_value(penalised_loss.first.i));
            violations += as_scalar(cg.get_value(penalised_loss.second.i));
            ++lines;
        }
        sgd->status();
        cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << " violations = " << (violations / chars);

        // show score on dev data?
        report++;
        if (report % dev_every_i_reports == 0) {
            double dloss = 0;
            int dchars = 0;
            for (auto& spair : dev) {
                ComputationGraph cg;
                am.BuildGraph(spair.first, spair.second, cg, nullptr);
                dloss += as_scalar(cg.forward());
                dchars += spair.second.size() - 1;
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
}

