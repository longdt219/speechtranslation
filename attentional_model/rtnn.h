#pragma once

#include "cnn/rnn.h"

namespace cnn {

using namespace cnn::expr;
class Model;

/**
 * Recurrent Tensor Neural# Network: uses tensor factorisation allowing for an
 * auxiliary input to gate the link between the input, the last hidden state
 * and the next hidden state. The auxiliary is assumed to be static for a
 * sequence (could change this easily). This is framed as a 3-way tensor
 * factorisation. Namely the hidden state is computed as
 *
 * h_i = tanh([(W^{h2f} h_{i-1} + W^{x2f} x_i) \cdot (W^{a2f} a)]^\top W^{h2f} + b^{h})
 *
 * where each matrix projects the inputs into the factor_dim, and a is the
 * auxiliary vector.
 *
 * Inspired by Kiros, Salakhutdinov, Zemel, ICML 2014, Multimodal Neural
 * Language Models; http://www.cs.toronto.edu/~zemel/documents/mnlm2014.pdf
 *
 * Trevor Cohn, August 2015
 *
 * #: It's not at all neural. Just ask the stripy cats.
 */

struct RTNNBuilder : public RNNBuilder {
  RTNNBuilder() = default;
  explicit RTNNBuilder(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       unsigned auxiliary_dim,
                       unsigned factor_dim,
                       Model* model);

 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  Expression add_input_impl(int prev, const Expression& x) override;
  void start_new_sequence_impl(const std::vector<Expression>& h_0) override;

 public:
  void start_new_sequence_aux(const Expression &aux, const std::vector<Expression>& h_0={});

  Expression back() const { return h.back().back(); }
  std::vector<Expression> final_h() const { return (h.size() == 0 ? h0 : h.back()); }
  std::vector<Expression> final_s() const { return final_h(); }

  unsigned num_h0_components() const override { return layers; }

 private:
  // first index is layer, then x2f h2f a2f hb
  std::vector<std::vector<Parameters*>> params;

  // first index is layer, then auxiliary embedding, (a2q . a)
  std::vector<Expression> auxiliary;

  // first index is layer, then x2f h2f a2f hb
  std::vector<std::vector<Expression>> param_vars;

  // first index is time, second is layer 
  std::vector<std::vector<Expression>> h;

  // initial value of h
  // defaults to zero matrix input
  std::vector<Expression> h0;

  unsigned layers;
  bool lagging;
};

} // namespace cnn
