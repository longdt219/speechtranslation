#include "rtnn.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << expression.pg->nodes[expression.i]->dim << std::endl;


using namespace std;
using namespace cnn::expr;
using namespace cnn;

namespace cnn {

enum { X2F=0, H2F, A2F, HB };

RTNNBuilder::RTNNBuilder(unsigned layers, unsigned input_dim, unsigned hidden_dim, 
        unsigned auxiliary_dim, unsigned factor_dim, Model* model) 
    : layers(layers) 
{
    unsigned layer_input_dim = input_dim;
    for (unsigned i = 0; i < layers; ++i) {
        Parameters* p_x2f = model->add_parameters({factor_dim, layer_input_dim});
        Parameters* p_h2f = model->add_parameters({factor_dim, hidden_dim});
        Parameters* p_a2f = model->add_parameters({factor_dim, auxiliary_dim});
        Parameters* p_hb = model->add_parameters({hidden_dim});
        vector<Parameters*> ps = {p_x2f, p_h2f, p_a2f, p_hb};
        params.push_back(ps);
        layer_input_dim = hidden_dim;
    }
}

void RTNNBuilder::new_graph_impl(ComputationGraph& cg) {
    param_vars.clear();
    for (unsigned i = 0; i < layers; ++i) {
        vector<Expression> vars;
        for (auto &p: params[i]) 
            vars.push_back(parameter(cg, p));
        param_vars.push_back(vars);
    }
}

void RTNNBuilder::start_new_sequence_aux(const Expression &aux, const vector<Expression>& h_0) {
    // precompute auxiliary component
    auxiliary.clear();
    for (unsigned i = 0; i < layers; ++i) {
        Expression ai = param_vars[i][A2F] * aux;
        auxiliary.push_back(ai);
    }
    this->start_new_sequence(h_0);
}

void RTNNBuilder::start_new_sequence_impl(const vector<Expression>& h_0) {
    h.clear();
    h0 = h_0;
    if (h0.size()) { assert(h0.size() == layers); }
}

Expression RTNNBuilder::add_input_impl(int prev, const Expression &in) {
    const unsigned T = h.size();
    h.push_back(vector<Expression>(layers));

    Expression x = in;

    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];

        bool got_s = true;
        Expression s;
        if (prev == -1 && h0.size() > 0)
            s = vars[H2F] * h0[i];
        else if (prev >= 0)
            s = vars[H2F] * h[prev][i];
        else
            got_s = false;

        Expression t = vars[X2F] * x;
        Expression &u = auxiliary[i];
        Expression r = cwise_multiply((got_s) ? s + t : t, u); 
        // FIXME: 3-way, i.e., s . t . u?
        //Expression r = cwise_multiply(s, cwise_multiply(t, u)); 
        // FIXME: tanh(s+t) in the above?
        //Expression r = cwise_multiply(tanh(s + t), u); 

        Expression y = affine_transform({vars[HB], transpose(vars[H2F]), r});
        x = h[T][i] = tanh(y); 
        // FIXME: do we need the non-linearity?
        //x = h[t][i] = y; 
    }
    return h[T].back();
}

} // namespace cnn

#undef WTF
