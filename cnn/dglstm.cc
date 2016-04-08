#include "cnn/dglstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

//#define USE_STANDARD_LSTM_DEFINE 
/**
the standard definition has a forget gate computed using its own parameter. 
however, in CNN, the forget gate is computed as f_t = 1 - i_t, where i_t is the input gate. 
this is also used in LSTM.cc.

*/

using namespace std;
using namespace cnn::expr;

namespace cnn {

    enum { X2I, H2I, C2I, BI, 
#ifdef USE_STANDARD_LSTM_DEFINE
        X2F, H2F, C2F, BF, 
#endif
        X2O, H2O, C2O, BO, X2C, H2C, BC, X2K, C2K, Q2K, BK, STAB, X2K0 };

DGLSTMBuilder::DGLSTMBuilder(unsigned ilayers,
    unsigned input_dim,
    unsigned hidden_dim,
    Model* model) 
{
    layers = ilayers;
    Parameters * p_x2k, *p_c2k, *p_q2k, *p_bk, *p_x2k0;
    long layer_input_dim = input_dim;
    input_dims = vector<unsigned>(layers, layer_input_dim);
    p_x2k0 = model->add_parameters({ long(hidden_dim), long(layer_input_dim) });
    for (unsigned i = 0; i < layers; ++i) {
        input_dims[i] = layer_input_dim;
        // i
        Parameters* p_x2i = model->add_parameters({ long(hidden_dim), layer_input_dim });
        Parameters* p_h2i = model->add_parameters({ long(hidden_dim), long(hidden_dim) });
        Parameters* p_c2i = model->add_parameters({ long(hidden_dim), long(hidden_dim) });
        Parameters* p_bi = model->add_parameters({ long(hidden_dim) });
#ifdef USE_STANDARD_LSTM_DEFINE
        // f
        Parameters* p_x2f = model->add_parameters({ long(hidden_dim), layer_input_dim });
        Parameters* p_h2f = model->add_parameters({ long(hidden_dim), long(hidden_dim) });
        Parameters* p_c2f = model->add_parameters({ long(hidden_dim), long(hidden_dim) });
        Parameters* p_bf = model->add_parameters({ long(hidden_dim) });
#endif
        // o
        Parameters* p_x2o = model->add_parameters({ long(hidden_dim), layer_input_dim });
        Parameters* p_h2o = model->add_parameters({ long(hidden_dim), long(hidden_dim) });
        Parameters* p_c2o = model->add_parameters({ long(hidden_dim), long(hidden_dim) });
        Parameters* p_bo = model->add_parameters({ long(hidden_dim) });

        // c
        Parameters* p_x2c = model->add_parameters({ long(hidden_dim), layer_input_dim });
        Parameters* p_h2c = model->add_parameters({ long(hidden_dim), long(hidden_dim) });
        Parameters* p_bc = model->add_parameters({ long(hidden_dim) });

        p_x2k = model->add_parameters({ long(hidden_dim), long(layer_input_dim) });
        p_c2k = model->add_parameters({ long(hidden_dim) });
        p_bk = model->add_parameters({ long(hidden_dim) });
        p_q2k = model->add_parameters({ long(hidden_dim) });

        layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

        Parameters * p_stab = model->add_parameters({ 1 });
        p_stab->reset_to_zero();

        vector<Parameters*> ps;
        if (i == 0)
            ps = { p_x2i, p_h2i, p_c2i, p_bi, 
#ifdef USE_STANDARD_LSTM_DEFINE
            p_x2f, p_h2f, p_c2f, p_bf, 
#endif
            p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc, p_x2k, p_c2k, p_q2k, p_bk, p_stab, p_x2k0 };
        else
            ps = { p_x2i, p_h2i, p_c2i, p_bi, 
#ifdef USE_STANDARD_LSTM_DEFINE
            p_x2f, p_h2f, p_c2f, p_bf, 
#endif
            p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc, p_x2k, p_c2k, p_q2k, p_bk, p_stab };

        params.push_back(ps);
    }  // layers
}

void DGLSTMBuilder::new_graph_impl(ComputationGraph& cg){
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];

    //i
    Expression i_x2i = parameter(cg,p[X2I]);
    Expression i_h2i = parameter(cg,p[H2I]);
    Expression i_c2i = parameter(cg,p[C2I]);
    Expression i_bi = parameter(cg,p[BI]);
#ifdef USE_STANDARD_LSTM_DEFINE
    //f
    Expression i_x2f = parameter(cg, p[X2F]);
    Expression i_h2f = parameter(cg, p[H2F]);
    Expression i_c2f = parameter(cg, p[C2F]);
    Expression i_bf = parameter(cg, p[BF]);
#endif
    //o
    Expression i_x2o = parameter(cg,p[X2O]);
    Expression i_h2o = parameter(cg,p[H2O]);
    Expression i_c2o = parameter(cg,p[C2O]);
    Expression i_bo = parameter(cg,p[BO]);
    //c
    Expression i_x2c = parameter(cg,p[X2C]);
    Expression i_h2c = parameter(cg,p[H2C]);
    Expression i_bc = parameter(cg,p[BC]);

    vector<Expression> vars;

    //k
    Expression i_x2k = parameter(cg, p[X2K]);
    Expression i_q2k = parameter(cg, p[Q2K]);
    Expression i_c2k = parameter(cg, p[C2K]);
    Expression i_bk = parameter(cg, p[BK]);

    Expression i_stab = parameter(cg, p[STAB]);

    if (i == 0)
    {
        Expression i_x2k0 = parameter(cg, p[X2K0]);
        vars = { i_x2i, i_h2i, i_c2i, i_bi, 
#ifdef USE_STANDARD_LSTM_DEFINE
            i_x2f, i_h2f, i_c2f, i_bf, 
#endif
            i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc, i_x2k, i_c2k, i_q2k, i_bk, i_stab, i_x2k0 };
    }
    else
        vars = { i_x2i, i_h2i, i_c2i, i_bi, 
#ifdef USE_STANDARD_LSTM_DEFINE
        i_x2f, i_h2f, i_c2f, i_bf, 
#endif
        i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc, i_x2k, i_c2k, i_q2k, i_bk, i_stab };

    param_vars.push_back(vars);
  }
  set_data_in_parallel(data_in_parallel());
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void DGLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();

  if (hinit.size() > 0) {
    assert(layers*2 == hinit.size());
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }

  set_data_in_parallel(data_in_parallel());
}

void DGLSTMBuilder::set_data_in_parallel(int n)
{
    RNNBuilder::set_data_in_parallel(n);

    biases.clear();
    for (unsigned i = 0; i < layers; ++i) {
        const vector<Expression>& vars = param_vars[i];
        Expression bimb = concatenate_cols(vector<Expression>(data_in_parallel(), vars[BI]));
        Expression bcmb = concatenate_cols(vector<Expression>(data_in_parallel(), vars[BC]));
        Expression bomb = concatenate_cols(vector<Expression>(data_in_parallel(), vars[BO]));
        Expression bkmb = concatenate_cols(vector<Expression>(data_in_parallel(), vars[BK]));
        Expression mc2kmb = concatenate_cols(vector<Expression>(data_in_parallel(), vars[C2K]));
        Expression mq2kmb = concatenate_cols(vector<Expression>(data_in_parallel(), vars[Q2K]));
#ifdef USE_STANDARD_LSTM_DEFINE
        Expression bfmb = concatenate_cols(vector<Expression>(data_in_parallel(), vars[BF]));
#endif
        Expression i_stabilizer = exp(vars[STAB]);
        Expression i_stab = concatenate(vector<Expression>(input_dims[i], i_stabilizer));  /// self stabilizer

        vector<Expression> b = { bimb, 
            bcmb, bomb, bkmb, mc2kmb, mq2kmb, i_stab 
#ifdef USE_STANDARD_LSTM_DEFINE
            , bfmb
#endif
        };
        biases.push_back(b);
    }
}

Expression DGLSTMBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();

  Expression lower_layer_c = x;  /// at layer 0, no lower memory but observation
  Expression in = x;
  Expression in_stb;

  int nutt = data_in_parallel();

  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    Expression i_stabilizer = biases[i][6];
    Expression i_v_stab = concatenate_cols(vector<Expression>(nutt, i_stabilizer));
    in_stb = cwise_multiply(i_v_stab, in); 
    Expression i_h_tm1, i_c_tm1;

    bool has_prev_state = (prev >= 0 || has_initial_state);
    if (prev < 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }

    // input
    Expression i_ait;
    Expression bimb = biases[i][0];
    if (has_prev_state)
        i_ait = affine_transform({ bimb, vars[X2I], in_stb, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1 });
    else
        i_ait = affine_transform({ bimb, vars[X2I], in_stb });
    Expression i_it = logistic(i_ait);
    
    // forget
    // input
#ifdef USE_STANDARD_LSTM_DEFINE
    Expression i_aft;
    Expression bfmb = biases[i][7];
    if (has_prev_state)
        i_aft = affine_transform({ bfmb, vars[X2F], in_stb, vars[H2F], i_h_tm1, vars[C2F], i_c_tm1 });
    else
        i_aft = affine_transform({ bfmb, vars[X2F], in_stb });
    Expression i_ft = logistic(i_aft);
#else
    Expression i_ft = 1.0 - i_it;
#endif

    // write memory cell
    Expression bcmb = biases[i][1];
    Expression i_awt;
    if (has_prev_state)
        i_awt = affine_transform({ bcmb, vars[X2C], in_stb, vars[H2C], i_h_tm1 });
    else
        i_awt = affine_transform({ bcmb, vars[X2C], in_stb });
    Expression i_wt = tanh(i_awt);

    // output
    Expression i_before_add_with_lower_linearly;
    if (has_prev_state) {
      Expression i_nwt = cwise_multiply(i_it,i_wt);
      Expression i_crt = cwise_multiply(i_ft,i_c_tm1);
      i_before_add_with_lower_linearly = i_crt + i_nwt;
    } else {
      i_before_add_with_lower_linearly = cwise_multiply(i_it, i_wt);
    }

    /// add lower layer memory cell
    Expression i_k_t; 
    Expression bkmb = biases[i][3];
    Expression i_k_lowerc = bkmb;
    if (i > 0)
    {
        Expression mc2kmb = biases[i][4];
        i_k_lowerc = i_k_lowerc + cwise_multiply(mc2kmb, lower_layer_c); 
    }

    if (has_prev_state)
    {
        Expression q2kmb = biases[i][5];
        i_k_t = logistic(i_k_lowerc + vars[X2K] * in_stb + cwise_multiply(q2kmb, i_c_tm1));
    }
    else
        i_k_t = logistic(i_k_lowerc + vars[X2K] * in_stb);
    ct[i] = i_before_add_with_lower_linearly + cwise_multiply(i_k_t, (i == 0) ? vars[X2K0] * lower_layer_c : lower_layer_c);


    Expression i_aot;
    Expression bomb = biases[i][2];
    if (has_prev_state)
        i_aot = affine_transform({bomb, vars[X2O], in_stb, vars[H2O], i_h_tm1, vars[C2O], ct[i]});
    else
        i_aot = affine_transform({ bomb, vars[X2O], in_stb });
    Expression i_ot = logistic(i_aot);
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cwise_multiply(i_ot,ph_t);
    lower_layer_c = ct[i];
  }
  return ht.back();
}

void DGLSTMBuilder::copy(const RNNBuilder & rnn) {
  const DGLSTMBuilder & rnn_lstm = (const DGLSTMBuilder&)rnn;
  assert(params.size() == rnn_lstm.params.size());
  for(size_t i = 0; i < params.size(); ++i)
      for(size_t j = 0; j < params[i].size(); ++j)
        params[i][j]->copy(*rnn_lstm.params[i][j]);
}

} // namespace cnn
