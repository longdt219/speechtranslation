#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include "expr-xtra.h"

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

unsigned LAYERS_FWD = 2;
unsigned LAYERS_BWD = 1;
unsigned INPUT_DIM = 16;  //256
unsigned HIDDEN_DIM_FWD = 48;  // 512
unsigned HIDDEN_DIM_BWD = 48;  // 512
unsigned VOCAB_SIZE = 0;

cnn::Dict d;
int kSOS;
int kEOS;

typedef vector<int> Sentence;
typedef vector<Sentence> Document;
typedef vector<Document> Corpus;

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;

template <class Builder>
struct iDCLM {
    LookupParameters* p_C;
    Parameters* p_Ro;
    Parameters* p_Rc_f;
    Parameters* p_Rc_b;
    Parameters* p_bo;
    Parameters* p_H_f;
    Parameters* p_H_b;
    Parameters* p_bh;
    Builder builder_forward;
    Builder builder_backward;

    //
    // Model works as follows:
    //
    //	a_i = rnn encoding of sentence t in forward direction
    //	a'_i = rnn encoding of sentence t in backward direction
    //	
    //	p(w_{i,t} | w_{i,t-1}, h_{i,t-1}) =
    //	    softmax [ f(w_{i,t-1}, h_{i,t-1}, a_{i-1}, a'_{i-1}) ]_{w_i_t}
    // 
    //  and
    //
    //	h_{i,t-1} = g(w_{i,t-1}, h_{i,t-2})
    //	h_0 = i(a_{i-1}, a'_{i-1})
    //
    // FIXME: consider having a hidden layer at the output with a
    //	fairly high capacity to merge the RNN state and the contextual
    //	state.
    //

    explicit iDCLM(Model& model) 
        : builder_forward(LAYERS_FWD, INPUT_DIM, HIDDEN_DIM_FWD, &model),
          builder_backward(LAYERS_BWD, INPUT_DIM, HIDDEN_DIM_BWD, &model) 
    {
        p_C = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); 
        p_Ro = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM_FWD});
        p_Rc_f = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM_FWD});
        p_Rc_b = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM_BWD});
        p_bo = model.add_parameters({VOCAB_SIZE});
        p_H_f = model.add_parameters({HIDDEN_DIM_FWD, HIDDEN_DIM_FWD});
        p_H_b = model.add_parameters({HIDDEN_DIM_FWD, HIDDEN_DIM_BWD});
        p_bh = model.add_parameters({HIDDEN_DIM_FWD});
    }

    // return Expression of total loss
    Expression BuildGraph(const Document& doc, ComputationGraph& cg) 
    {
        Expression i_Ro = parameter(cg, p_Ro); 
        Expression i_Rc_f = parameter(cg, p_Rc_f); 
        Expression i_Rc_b = parameter(cg, p_Rc_b); 
        Expression i_bo = parameter(cg, p_bo); 
        Expression i_H_f = parameter(cg, p_H_f); 
        Expression i_H_b = parameter(cg, p_H_b); 
        Expression i_bh = parameter(cg, p_bh); 

	builder_forward.new_graph(cg);
	builder_backward.new_graph(cg);

	Expression h_last_f, h_last_b;
        vector<Expression> errs;
        for (unsigned i = 0; i < doc.size(); ++i) {
	    auto &sent = doc[i];

            // perform forward pass 
	    if (i > 0) {
		// FIXME: treat each layer differently, instead of "cloning" input
		// -- removing this initialisation makes it first-order Markov
		Expression hctx = affine_transform({i_bh, i_H_f, h_last_f, i_H_b, h_last_b});
		builder_forward.start_new_sequence(vector<Expression>(builder_forward.num_h0_components(), hctx));
	    } else {
		builder_forward.start_new_sequence();
	    }

	    Expression octx;
	    if (i > 0) {
		octx = affine_transform({i_bo, i_Rc_f, h_last_f, i_Rc_b, h_last_b});
	    } else {
		octx = i_bo;
	    }
		
            for (unsigned t = 0; t < sent.size()-1; ++t) {
                Expression i_x_t = lookup(cg, p_C, sent[t]);
		Expression i_y_t = builder_forward.add_input(i_x_t);
                Expression i_r_t = affine_transform({octx, i_Ro, i_y_t});
                Expression i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
                errs.push_back(i_err);
            }
	    h_last_f = builder_forward.back(); // FIXME: use other layers

            // perform backwards pass 
            builder_backward.start_new_sequence();
            for (int t = sent.size()-1; t > 0; --t) {
                Expression i_x_t = lookup(cg, p_C, sent[t]);
		Expression i_y_t = builder_backward.add_input(i_x_t);
            }
	    h_last_b = builder_backward.back(); // FIXME: use other layers
        }
        Expression i_nerr = sum(errs);
        return i_nerr;
    }
};

void read_documents(const std::string &filename, Corpus &text);

int main(int argc, char** argv) {
    cnn::Initialize(argc, argv);
    if (argc != 3 && argc != 4) {
        cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
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
    os << "idclm"
        << '_' << LAYERS_FWD
        << '_' << LAYERS_BWD
        << '_' << INPUT_DIM
        << '_' << HIDDEN_DIM_FWD
        << '_' << HIDDEN_DIM_BWD
        << "-pid" << getpid() << ".params";
    const string fname = os.str();
    cerr << "Parameters will be written to: " << fname << endl;
    double best = 9e+99;

    Model model;
    bool use_momentum = false;
    Trainer* sgd = nullptr;
    if (use_momentum)
        sgd = new MomentumSGDTrainer(&model);
    else
        sgd = new SimpleSGDTrainer(&model);

    iDCLM<LSTMBuilder> lm(model);
    if (argc == 4) {
        string fname = argv[3];
        ifstream in(fname);
        boost::archive::text_iarchive ia(in);
        ia >> model;
    }

    unsigned report_every_i = 5;
    unsigned dev_every_i_reports = 200;
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
                cerr << "Sentence in " << filename << ":" << lno << " didn't start or end with <s>, </s>\n";
                abort();
            }
            doc.push_back(sentence);
            toks += sentence.size();
        }
    }

    assert(doc.empty());

    cerr << corpus.size() << " documents, " << (lno-corpus.size()) << " sentences, " << toks << " tokens, " << d.size() << " types\n";
}
