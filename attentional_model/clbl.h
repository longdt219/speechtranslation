#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <tuple>
#include <set>
#include <map>

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;

namespace cnn{
using namespace std;

template <class Builder> 
struct CLBLRNN {
    LookupParameters* p_R;  //morphemes
    LookupParameters* p_Q_T; // words
    Parameters* p_Q; // words
    Parameters* p_bias;
    Builder builder;

    explicit CLBLRNN(Model& model,unsigned vocab_size,unsigned morph_vocab_size,  
	    unsigned layers, unsigned embed_dim, unsigned hidden_dim) 
	: builder(layers, embed_dim, hidden_dim, &model) // let embed_dim=hidden_dim for now
    {
	p_R = model.add_lookup_parameters(morph_vocab_size, {hidden_dim});
	p_Q_T = model.add_lookup_parameters(vocab_size, {hidden_dim});
	p_Q = model.add_parameters({vocab_size, hidden_dim}); 
	p_bias = model.add_parameters({vocab_size});
    }

    // return Expression of total loss; will work with words (not ids) for now
    Expression BuildLMGraph(const vector<int> &sentence,const map<int, vector<int>> &morphemes, ComputationGraph& cg) {
	Expression i_bias = parameter(cg, p_bias);  // word bias
	Expression i_Q = parameter(cg, p_Q);

	vector<Expression> errs;
	builder.new_graph(cg); 
	builder.start_new_sequence();

	for (auto t = 0u; t < sentence.size()-1; ++t) {
	    Expression i_x_t = lookup(cg, p_Q_T, sentence[t]);
	    std::vector<Expression> inputs({ i_x_t });
	    for (auto k=0u; k < morphemes.at(sentence[t]).size(); ++k){
		Expression i_m_t = lookup(cg, p_R, morphemes.at(sentence[t])[k]);
		inputs.push_back(i_m_t);
	    }
	    Expression i_r_t = sum(inputs);
	    Expression i_y_t = builder.add_input(i_r_t);
	    Expression i_pred = i_bias + i_Q * i_y_t;

	    Expression i_err = pickneglogsoftmax(i_pred, sentence[t+1]);
	    errs.push_back(i_err);

	}

	Expression i_nerr = sum(errs);
	return i_nerr;
    }
};

#undef WTF

}; // namespace cnn


