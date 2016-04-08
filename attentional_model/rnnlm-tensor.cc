#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "rtnn.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace std;
using namespace cnn;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 512;    //512
unsigned HIDDEN_DIM = 512;  //512
unsigned FACTOR_DIM = 128;  //128
unsigned SRC_VOCAB_SIZE = 0;
unsigned TGT_VOCAB_SIZE = 0;

cnn::Dict sd;
cnn::Dict td;
int kSRC_SOS;
int kSRC_EOS;
int kTGT_SOS;
int kTGT_EOS;

struct RTNNLanguageModel {
    LookupParameters* p_S;
    LookupParameters* p_T;
    Parameters* p_R;
    Parameters* p_bias;
    RTNNBuilder builder;
    explicit RTNNLanguageModel(Model& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, INPUT_DIM, FACTOR_DIM, &model) {
        p_S = model.add_lookup_parameters(SRC_VOCAB_SIZE, {INPUT_DIM}); 
        p_T = model.add_lookup_parameters(TGT_VOCAB_SIZE, {INPUT_DIM}); 
        p_R = model.add_parameters({TGT_VOCAB_SIZE, HIDDEN_DIM});
        p_bias = model.add_parameters({TGT_VOCAB_SIZE});
    }

    // return Expression of total loss
    Expression BuildLMGraph(const vector<int>& source, const vector<int> &target, ComputationGraph& cg) {
        builder.new_graph(cg);  // reset RNN builder for new graph

        // embed the source
        vector<Expression> semb;
        for (auto tok: source)
            semb.push_back(lookup(cg, p_S, tok));
        Expression svec = average(semb);

        builder.start_new_sequence_aux(svec);
        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias
        vector<Expression> errs;
        const unsigned slen = target.size() - 1;
        for (unsigned t = 0; t < slen; ++t) {
            Expression i_x_t = lookup(cg, p_T, target[t]);
            // y_t = RNN(x_t)
            Expression i_y_t = builder.add_input(i_x_t);
            Expression i_r_t =  i_bias + i_R * i_y_t;
            Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
            errs.push_back(i_err);
        }
        Expression i_nerr = sum(errs);
        return i_nerr;
    }

    // return Expression for total loss
    void RandomSample(const vector<int> &source, int max_len = 150) {
        cerr << endl;
        ComputationGraph cg;
        builder.new_graph(cg);  // reset RNN builder for new graph
        // embed the source
        vector<Expression> semb;
        cerr << "\tsource:";
        for (auto tok: source) {
            cerr << " " << sd.Convert(tok);
            semb.push_back(lookup(cg, p_S, tok));
        }
        cerr << "\n";
        Expression svec = average(semb);
        builder.start_new_sequence_aux(svec);

        Expression i_R = parameter(cg, p_R);
        Expression i_bias = parameter(cg, p_bias);
        vector<Expression> errs;
        int len = 0;
        int cur = kTGT_SOS;
        cerr << "\tsample:";
        while(len < max_len && cur != kTGT_EOS) {
            ++len;
            Expression i_x_t = lookup(cg, p_T, cur);
            // y_t = RNN(x_t)
            Expression i_y_t = builder.add_input(i_x_t);
            Expression i_r_t = i_bias + i_R * i_y_t;

            Expression ydist = softmax(i_r_t);

            unsigned w = 0;
            while (w == 0 || (int)w == kTGT_SOS) {
                auto dist = as_vector(cg.incremental_forward());
                double p = rand01();
                for (; w < dist.size(); ++w) {
                    p -= dist[w];
                    if (p < 0.0) { break; }
                }
                if (w == dist.size()) w = kTGT_EOS;
            }
            cerr << " " << td.Convert(w);
            cur = w;
        }
        cerr << endl;
    }
};

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
    vector<pair<vector<int>,vector<int>>> training, dev;
    string line;
    int tlc = 0;
    int tttoks = 0;
    int tstoks = 0;
    cerr << "Reading training data from " << argv[1] << "...\n";
    {
        ifstream in(argv[1]);
        assert(in);
        while(getline(in, line)) {
            ++tlc;
            vector<int> source, target;
            ReadSentencePair(line, &source, &sd, &target, &td);
            training.push_back(make_pair(source, target));
            tstoks += source.size();
            tttoks += target.size();
            if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
                    (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
                cerr << "Sentence in " << argv[1] << ":" << tlc << " didn't start or end with <s>, </s>\n";
                abort();
            }
        }
        cerr << tlc << " lines, " << tstoks << " & " << tttoks << " tokens (s & t), " << sd.size() << " & " << td.size() << " types\n";
    }
    sd.Freeze(); // no new word types allowed
    td.Freeze(); // no new word types allowed
    SRC_VOCAB_SIZE = sd.size();
    TGT_VOCAB_SIZE = td.size();

    int dlc = 0;
    int dstoks = 0;
    int dttoks = 0;
    cerr << "Reading dev data from " << argv[2] << "...\n";
    {
        ifstream in(argv[2]);
        assert(in);
        while(getline(in, line)) {
            ++dlc;
            vector<int> source, target;
            ReadSentencePair(line, &source, &sd, &target, &td);
            dev.push_back(make_pair(source, target));
            dstoks += source.size();
            dttoks += target.size();
            if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS) ||
                    (target.front() != kTGT_SOS && target.back() != kTGT_EOS)) {
                cerr << "Dev sentence in " << argv[2] << ":" << dlc << " didn't start or end with <s>, </s>\n";
                abort();
            }
        }
        cerr << dlc << " lines, " << dstoks << " & " << dttoks << " tokens (s & t)\n";
    }
    ostringstream os;
    os << "lm"
        << '_' << LAYERS
        << '_' << INPUT_DIM
        << '_' << HIDDEN_DIM
        << '_' << FACTOR_DIM
        << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;
    double best = 9e+99;

    Model model;
    bool use_momentum = false;
    Trainer* sgd = nullptr;
    //if (use_momentum)
    //  sgd = new MomentumSGDTrainer(&model);
    //else
    sgd = new SimpleSGDTrainer(&model);

    RTNNLanguageModel lm(model);
    if (argc == 4) {
        string fname = argv[3];
        ifstream in(fname);
        boost::archive::text_iarchive ia(in);
        ia >> model;
    }

    unsigned report_every_i = 50;
    unsigned dev_every_i_reports = 500;
    unsigned sample_every_i_reports = 100;
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
                cerr << "**SHUFFLE\n";
                shuffle(order.begin(), order.end(), *rndeng);
            }

            // build graph for this instance
            ComputationGraph cg;
            auto& sent = training[order[si]];
            chars += sent.second.size() - 1;
            ++si;
            lm.BuildLMGraph(sent.first, sent.second, cg);
            loss += as_scalar(cg.forward());
            cg.backward();
            sgd->update();
            ++lines;
        }
        sgd->status();
        cerr << " E = " << (loss / chars) << " ppl=" << exp(loss / chars) << ' ';

        // show score on dev data?
        report++;
        if (report % sample_every_i_reports == 0) 
            lm.RandomSample(training[order[si % order.size()]].first);

        if (report % dev_every_i_reports == 0) {
            double dloss = 0;
            int dchars = 0;
            for (auto& sent : dev) {
                ComputationGraph cg;
                lm.BuildLMGraph(sent.first, sent.second, cg);
                dloss += as_scalar(cg.forward());
                dchars += sent.second.size() - 1;
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

