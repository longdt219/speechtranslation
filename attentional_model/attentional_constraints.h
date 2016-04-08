#pragma once

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

namespace cnn {

template <class Builder>
struct AttentionalConstraintModel {
    explicit AttentionalConstraintModel(Model& model, 
            unsigned layers, unsigned vocab_size_src, unsigned vocab_size_tgt, 
            unsigned hidden_dim, unsigned align_dim, unsigned hidden_replicates=1, 
            LookupParameters* cs=0, LookupParameters *ct=0);
    std::pair<Expression,Expression> 
        BuildGraph(const std::vector<int> &source, const std::vector<int>& target, 
            ComputationGraph& cg, Parameters *p_fertility_lagrange, Expression *alignment=0);

    LookupParameters* p_cs;
    LookupParameters* p_ct;
    std::vector<Parameters*> p_h0;
    Parameters* p_R;
    Parameters* p_bias;
    Parameters* p_Wa;
    Parameters* p_Ua;
    Parameters* p_va;
    Parameters* p_Ta;
    Builder builder;
};

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

template <class Builder>
AttentionalConstraintModel<Builder>::AttentionalConstraintModel(cnn::Model& model,
    unsigned layers, unsigned vocab_size_src, unsigned vocab_size_tgt, unsigned hidden_dim, 
    unsigned align_dim, unsigned hidden_replicates, 
    LookupParameters* cs, LookupParameters *ct)
: builder(layers, 2*hidden_dim, hidden_dim, &model) 
{
    p_cs = (cs) ? cs : model.add_lookup_parameters(vocab_size_src, {hidden_dim}); 
    p_ct = (ct) ? ct : model.add_lookup_parameters(vocab_size_tgt, {hidden_dim}); 
    p_R = model.add_parameters({vocab_size_tgt, hidden_dim});
    p_bias = model.add_parameters({vocab_size_tgt});
    for (auto l = 0; l < layers * hidden_replicates; ++l) 
        p_h0.push_back(model.add_parameters({hidden_dim}));
    //p_M = model.add_parameters({hidden_dim, hidden_dim});
    p_Wa = model.add_parameters({align_dim, hidden_dim});
    p_Ua = model.add_parameters({align_dim, hidden_dim});
#ifdef ATTENTIONAL_EXTENSIONS
    p_Ta = model.add_parameters({align_dim, 9});
#endif
    p_va = model.add_parameters({align_dim});
}

template <class Builder>
std::pair<Expression,Expression> 
AttentionalConstraintModel<Builder>::BuildGraph(const std::vector<int> &source, const std::vector<int>& target, ComputationGraph& cg,
        Parameters *p_fertility_lagrange, Expression *alignment)
{
    //cout << "source sentence length: " << source.size() << " target: " << target.size() << endl;

    // first compile the source sentence
    const unsigned slen = source.size() - 1; 
    std::vector<Expression> source_embeddings;
    for (unsigned s = 0; s < slen; ++s) {
        source_embeddings.push_back(lookup(cg, p_cs, source[s]));
    }
    Expression src = concatenate_cols(source_embeddings); 

    // now for the target sentence
    const unsigned tlen = target.size() - 1; // -1 for predicting the next symbol
    builder.new_graph(cg); 
    std::vector<Expression> i_h0;
    for (const auto &p: p_h0)
        i_h0.push_back(parameter(cg, p));
    builder.start_new_sequence(i_h0);
    Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression i_bias = parameter(cg, p_bias);  // word bias
    Expression i_Wa = parameter(cg, p_Wa); 
    Expression i_Ua = parameter(cg, p_Ua);
    Expression i_va = parameter(cg, p_va);
#ifdef ATTENTIONAL_EXTENSIONS
    Expression i_Ta = parameter(cg, p_Ta);   
#endif
    Expression i_uax = i_Ua * src;

#ifdef ATTENTIONAL_EXTENSIONS
    auto i_src_idx = arange(cg, 0, slen, true);
    //LOLCAT(i_src_idx);
    auto i_src_len = repeat(cg, slen, log(1.0 + slen));
#endif

    std::vector<Expression> errs;
    std::vector<Expression> aligns;
    for (unsigned t = 0; t < tlen; ++t) {
        // alignment input -- FIXME: just done for top layer
        auto i_h_tm1 = (t == 0) ? i_h0.back() : builder.final_h().back();
        //LOLCAT(i_h_tm1);
        //Expression i_e_t = tanh(i_src_M * i_h_tm1); 
        Expression i_wah = i_Wa * i_h_tm1;
        //LOLCAT(i_wah);
        // want numpy style broadcasting, but have to do this manually
        Expression i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
        //LOLCAT(i_wah_rep);
#ifdef ATTENTIONAL_EXTENSIONS
        std::vector<Expression> alignment_context;
        if (t >= 1) {
            auto i_aprev = concatenate_cols(aligns);
            auto i_asum = sum_cols(i_aprev);
            auto i_asum_pm = dither(cg, i_asum);
            //LOLCAT(i_asum_pm);
            alignment_context.push_back(i_asum_pm);
            auto i_alast_pm = dither(cg, aligns.back());
            //LOLCAT(i_alast_pm);
            alignment_context.push_back(i_alast_pm);
        } else {
            // just 6 repeats of the 0 vector
            auto zeros = repeat(cg, slen, 0);
            //LOLCAT(zeros);
            alignment_context.push_back(zeros); 
            alignment_context.push_back(zeros);
            alignment_context.push_back(zeros);
            alignment_context.push_back(zeros);
            alignment_context.push_back(zeros);
            alignment_context.push_back(zeros);
        }
        //LOLCAT(i_src_idx);
        alignment_context.push_back(i_src_idx);
        //LOLCAT(i_src_len);
        alignment_context.push_back(i_src_len);
        auto i_tgt_idx = repeat(cg, slen, log(1.0 + t));
        //LOLCAT(i_tgt_idx);
        alignment_context.push_back(i_tgt_idx);
        auto i_context = concatenate_cols(alignment_context);
        //LOLCAT(i_context); // fails to KTHXBYE on this

        auto i_e_t_input = i_wah_rep + i_uax + i_Ta * transpose(i_context); 
        //LOLCAT(i_e_t_input);
        auto i_e_t = transpose(tanh(i_e_t_input)) * i_va;
        //LOLCAT(i_e_t);
#else
        Expression i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
#endif
        Expression i_alpha_t = softmax(i_e_t);
        aligns.push_back(i_alpha_t);
        Expression i_c_t = src * i_alpha_t; 
        // word input
        Expression i_x_t = lookup(cg, p_ct, target[t]);
        Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t})); // vstack/hstack?
        // y_t = RNN([x_t, a_t])
        Expression i_y_t = builder.add_input(input);
        Expression i_r_t = i_bias + i_R * i_y_t;
        // errors
        Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
        errs.push_back(i_err);
    }
    // save the alignment for later
    if (alignment != 0) {
        *alignment = concatenate_cols(aligns);
    }

    Expression i_penalty;
    if (p_fertility_lagrange) {
        auto i_lagrange = parameter(cg, p_fertility_lagrange);
        auto i_fertility = sum_cols(concatenate_cols(aligns));
        auto i_constraint_violations = rectify(1.0f - i_fertility);
        auto i_total_violations = sum_cols(i_constraint_violations);
        //LOLCAT(i_total_violations);
        i_penalty = dot_product(i_lagrange, i_total_violations); 
    }

    Expression i_xent = sum(errs);
    return std::make_pair(i_xent, i_penalty);
}

#undef WTF
#undef KTHXBYE
#undef LOLCAT

}; // namespace cnn
