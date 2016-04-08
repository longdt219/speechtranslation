#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"

#include <iostream>

#include "corpus.hh"
#include "expr-xtra.h"
#include "feature_factory.h"

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define WTFv(expression) \
    std::cout << #expression << " has dimensions "; \
    for (auto t: expression)  \
        std::cout << " " << cg.nodes[t.i]->dim; \
    std::cout << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

struct AlignHypothesis {
     Alignment align; // init with the alignment of <s> in the target
     double cost;
     RNNPointer last_h;
     int t = 0; //last aligned target position

     AlignHypothesis() {align.push_back(0);} // alignment for <s>
};  


namespace cnn {

using namespace std;
using namespace cnn::expr;

template <class Builder>
struct RecurrentMarkovTranslationModel {
    LookupParameters* p_E_src;
    LookupParameters* p_E_tgt;
    //LookupParameters* p_h0; //no need since NULL and <s> are now separated
    Parameters* p_null;
    Parameters* p_R_tgt;
    Parameters* p_b_tgt;
    Parameters* p_L_src;
    Parameters* p_M_src;
    Parameters* p_e;
    //Parameters* p_v_a;
    vector<Parameters*> p_doc_ctx_l;
    vector<Parameters*> p_doc_ctx_r;
    Builder builder_tgt;
    Builder builder_src_fwd;
    Builder builder_src_bwd;
    bool rnn_src_embeddings;
    unsigned layers;
    unsigned embedding_dim;
    unsigned hidden_dim;
    unsigned  feat_dim;
    unsigned num_init_params;
    bool document_context;
    std::vector<cnn::real> h_zeros, e_zeros_src, e_zeros_trg;
    //Expression i_src_zero;
    std::vector<Expression> source_embeddings_next;
    Expression i_C_src_next;

    explicit RecurrentMarkovTranslationModel(Model& model, 
            unsigned src_vocab_size, unsigned tgt_vocab_size, 
            unsigned layers, unsigned _embedding_dim, unsigned hidden_dim,
            bool _rnn_src_embeddings, bool _document_context, unsigned feat_dim) 
        : builder_tgt(layers, (_rnn_src_embeddings) ? _embedding_dim+2*hidden_dim : 2*_embedding_dim, hidden_dim, &model),
          builder_src_fwd(1, _embedding_dim, hidden_dim, &model),
          builder_src_bwd(1, _embedding_dim, hidden_dim, &model),
          rnn_src_embeddings(_rnn_src_embeddings), 
          embedding_dim(_embedding_dim), 
          document_context(_document_context)
    {
        p_E_src = model.add_lookup_parameters(src_vocab_size, {embedding_dim}); 
        p_E_tgt = model.add_lookup_parameters(tgt_vocab_size, {embedding_dim}); 
        p_R_tgt = model.add_parameters({tgt_vocab_size, hidden_dim});
        p_b_tgt = model.add_parameters({tgt_vocab_size});

        num_init_params = builder_tgt.num_h0_components();
        //if (!document_context) 
        //    p_h0 = model.add_lookup_parameters(num_init_params, {hidden_dim}); 

        for (int i = 0; i < this->num_init_params; i++) {
            if (document_context) {
                if (!rnn_src_embeddings) {
                    p_doc_ctx_l.push_back(model.add_parameters({hidden_dim, embedding_dim}));
                    p_doc_ctx_r.push_back(model.add_parameters({hidden_dim, embedding_dim}));
                } else {
                    int parts = builder_src_fwd.num_h0_components() + builder_src_bwd.num_h0_components();
                    p_doc_ctx_l.push_back(model.add_parameters({hidden_dim, parts*hidden_dim}));
                    p_doc_ctx_r.push_back(model.add_parameters({hidden_dim, parts*hidden_dim}));
                }
            }
        }

        e_zeros_trg.resize(embedding_dim, 0.0f);
        h_zeros.resize(hidden_dim, 0.0f);
        if (!rnn_src_embeddings) {
            p_M_src = model.add_parameters({embedding_dim, hidden_dim});
            p_L_src = model.add_parameters({embedding_dim, embedding_dim});
            p_null = model.add_parameters({embedding_dim,1});
            e_zeros_src.resize(embedding_dim, 0.0f);
        } else {
            p_M_src = model.add_parameters({2*hidden_dim, hidden_dim});
            p_L_src = model.add_parameters({2*hidden_dim, 2*hidden_dim});
            p_null = model.add_parameters({2*hidden_dim,1});
            e_zeros_src.resize(2*hidden_dim, 0.0f);
        }
        //p_b_src = model.add_lookup_parameters(src_vocab_size, {1});
        p_e = model.add_parameters({feat_dim});
        //p_v_a = model.add_parameters({hidden_dim});

        this->embedding_dim = embedding_dim;
        this->hidden_dim = hidden_dim;
        this->layers = layers;
        this->feat_dim = feat_dim;
    }

    void build_source_embedding(ComputationGraph& cg, const EncodedSentence & srcsen, 
            std::vector<Expression> &source_embeddings, bool fixed_dimension = false) {

        if (!rnn_src_embeddings) {
            // just lookup the embeddings for each source token
            for (auto i = 0u; i < srcsen.size(); ++i)
                source_embeddings.push_back(lookup(cg, p_E_src, srcsen[i]));

            if (fixed_dimension) {
                Expression avg = average(source_embeddings);
                source_embeddings.clear();
                source_embeddings.push_back(avg);
            }
        } else {
            // run a RNN backward and forward over the source sentence
            // and stack the top-level hidden states from each model as 
            // the representation at each position
            std::vector<Expression> src_fwd(srcsen.size());
            std::vector<Expression> src_bwd(srcsen.size());
            // now for the rest of the sentence ...
            builder_src_fwd.new_graph(cg);
            builder_src_bwd.new_graph(cg);
            builder_src_fwd.start_new_sequence(); // can add a parameter for the <start> symbol in forward!
            builder_src_bwd.start_new_sequence();
            for (auto i = 0u; i < srcsen.size(); ++i) 
                src_fwd[i] = builder_src_fwd.add_input(lookup(cg, p_E_src, srcsen[i]));
            for (int i = srcsen.size()-1; i >= 0; --i) 
                src_bwd[i] = builder_src_bwd.add_input(lookup(cg, p_E_src, srcsen[i]));

            // place null (i.e. <s>) embedding in position 0 -- but this is a different dimensionality!
            //src_fwd[0] = lookup(cg, p_E_src, srcsen[0]); // FIXME: should be special "NULL" token, not <S>
            //src_bwd[0] = src_fwd[0];

            if (!fixed_dimension) {
                for (auto i = 0u; i < srcsen.size(); ++i) 
                    source_embeddings.push_back(concatenate(vector<Expression>({src_fwd[i], src_bwd[i]})));
            } else {
                source_embeddings = builder_src_fwd.final_h();
                auto last_bwd = builder_src_bwd.final_h();
                source_embeddings.insert(source_embeddings.begin(), last_bwd.begin(), last_bwd.end());
            }
        }
        //WTFv(source_embeddings);
    }


    void h0_doc_context(ComputationGraph& cg, const AlignedEncodedCorpus &corpus, 
            unsigned sent_num, const AlignedEncodedSentence &sent_pair, std::vector<Expression> &i_h0) {

        std::vector<Expression> i_ctx_left, i_ctx_right;
        if (document_context) {
            for (auto p: p_doc_ctx_l)
                i_ctx_left.push_back(parameter(cg, p));
            for (auto p: p_doc_ctx_r)
                i_ctx_right.push_back(parameter(cg, p));
        }

        bool h0_initialised = false;

        if (!document_context) {
            //no need to the initial params
            //because of the introduction of <s> and NULL
            for (int i = 0; i < num_init_params; i++)  
               i_h0[i] = input(cg, {hidden_dim} , &h_zeros); //lookup(cg, p_h0, i);
               
        } else {
            // find the sentence to the left and right and embed
            if (sent_num >= 1) {
                auto &left = *corpus.at(sent_num-1);
                if (left.document_id == sent_pair.document_id) {
                    vector<Expression> context;
                    build_source_embedding(cg, left.srcSentence, context, true);
                    Expression i_left = concatenate(context);
                    for (int i = 0; i < num_init_params; i++)
                        i_h0[i] = i_ctx_left[i] * i_left;
                    h0_initialised = true;
                }
            }
            if (sent_num+1 < corpus.size()) {
                auto &right = *corpus.at(sent_num+1);
                if (right.document_id == sent_pair.document_id) {
                    vector<Expression> context;
                    build_source_embedding(cg, right.srcSentence, context, true);
                    Expression i_right = concatenate(context);
                    for (int i = 0; i < num_init_params; i++) {
                        if (h0_initialised)
                            i_h0[i] = affine_transform({i_h0[i], i_ctx_right[i], i_right});
                        else
                            i_h0[i] = i_ctx_right[i] * i_right;
                    }
                }
            }
        }
    }


    Expression next_rnn_state(ComputationGraph& cg, const AlignedEncodedCorpus &corpus, unsigned sent_num, 
               FeatureFactory &feat_factory, AlignHypothesis &cur_hyp, Expression &algn_distrib) 
    {

        const AlignedEncodedSentence &sent_pair = *corpus.at(sent_num);
        const auto &srcsen = sent_pair.srcSentence;
        const auto &trgsen = sent_pair.trgSentence;
        const auto &align = sent_pair.alignment;
        auto & t = cur_hyp.t;

        Expression i_R_tgt = parameter(cg, p_R_tgt);
        Expression i_b_tgt = parameter(cg, p_b_tgt);
        Expression i_L_src = parameter(cg, p_L_src);
        Expression i_M_src = parameter(cg, p_M_src);
        Expression i_e = parameter(cg, p_e);
        Expression i_t_m1, i_s; 
        Expression i_null = parameter(cg, p_null);

        if (t == 0) {
            source_embeddings_next.clear();
            build_source_embedding(cg, srcsen, source_embeddings_next);
            std::vector<Expression> null_source_embeddings = source_embeddings_next;
            null_source_embeddings[0] = i_null;
            i_C_src_next = concatenate_cols(null_source_embeddings);

            // the very first state in the RNN  sequence for all layers/chains
            std::vector<Expression> i_h0(num_init_params);
            h0_doc_context(cg, corpus, sent_num, sent_pair, i_h0); 

            builder_tgt.new_graph(cg);
            builder_tgt.start_new_sequence(i_h0);

            i_t_m1 = input(cg, {embedding_dim}, &e_zeros_trg); 
            //i_s = source_embeddings[0];
        } else {
            int ttok = trgsen[t-1];
            i_t_m1 = lookup(cg, p_E_tgt, ttok);
        }

        int i = cur_hyp.align[t];
        if (i < 0) 
            i_s = i_null; // assign NULL alignments to the start token
        else 
            i_s = source_embeddings_next[i];            
        
        std::vector<Expression> minput;
        minput.push_back(i_t_m1); 
        minput.push_back(i_s);
        Expression i_x_j = concatenate(minput);

        // next RNN state
        Expression i_y_j;
        if (t == 0)
           i_y_j = builder_tgt.add_input(i_x_j);
        else
           i_y_j = builder_tgt.add_input(cur_hyp.last_h, i_x_j);    
        cur_hyp.last_h = builder_tgt.state();

        //word cost
        Expression i_err; 
        if (t > 0) {
           Expression i_r_j = i_b_tgt + i_R_tgt * i_y_j;
           i_err = pickneglogsoftmax(i_r_j, trgsen[t]);           
        } else
           i_err = input(cg, .0f);  
      
        //alignment distribution
        std::vector<cnn::real> srcsen_zeros(srcsen.size(), 0.0f);
        if (t < (trgsen.size()-1)) {
           // first term
           Expression first_term = transpose(i_C_src_next) * (i_L_src*i_s + i_M_src*i_y_j);
           // second term  FIXME
           // need to build phi incrementally in here using feat_factory! FIXME
           //Expression i_phi = input(cg, {this->feat_dim, srcsen.size()}, &(phi_sent[j+1]));
           Expression second_term = input(cg, {srcsen.size()}, &srcsen_zeros); //transpose(i_phi) * i_e;  
           //put all together            
           algn_distrib = log_softmax(first_term + second_term);
        } else
           algn_distrib = input(cg, {srcsen.size()}, &srcsen_zeros);
        return i_err;
    }

    // return Expression of total loss -- currently just the translation sequence
    std::pair<Expression, Expression>
    //BuildGraph(const AlignedEncodedSentence &sent_pair, ComputationGraph& cg, FEATS_SENT &phi_sent) {
    BuildGraph(const AlignedEncodedCorpus &corpus, unsigned sent_num, ComputationGraph& cg, FEATS_SENT &phi_sent) {

            const AlignedEncodedSentence &sent_pair = *corpus.at(sent_num);

            Expression i_R_tgt = parameter(cg, p_R_tgt);
            Expression i_b_tgt = parameter(cg, p_b_tgt);
            Expression i_L_src = parameter(cg, p_L_src);
            Expression i_M_src = parameter(cg, p_M_src);
            Expression i_e = parameter(cg, p_e);
            Expression i_null = parameter(cg, p_null);

            vector<Expression> errsTarget, errsJump;

            const auto &srcsen = sent_pair.srcSentence;
            const auto &trgsen = sent_pair.trgSentence;
            const auto &align = sent_pair.alignment;

            // build source embedding by bidirectionl RNN, i.e. two RNNS in forward and backward directions
            std::vector<Expression> source_embeddings;
            build_source_embedding(cg, srcsen, source_embeddings);

            // build i_C_src: column zero should have the null mebedding
            std::vector<Expression> null_source_embeddings = source_embeddings;
            null_source_embeddings[0] = i_null;
            Expression i_C_src = concatenate_cols(null_source_embeddings); 

            // the very first state in the RNN  sequence for all layers/chains coming from surronding context
            std::vector<Expression> i_h0(num_init_params);
            h0_doc_context(cg, corpus, sent_num, sent_pair, i_h0);
            //WTFv(i_h0);

            builder_tgt.new_graph(cg);
            builder_tgt.start_new_sequence(i_h0);
            // WTF(i_h0[0]);

            // now range over the target sentence
            for (auto j = 0u; j < trgsen.size(); ++j) {

                // gather embedding for previous target  words
                Expression i_t_m1;
                if (j ==0) 
                   i_t_m1 = input(cg, {embedding_dim}, &e_zeros_trg); 
                else
                   i_t_m1 =  lookup(cg, p_E_tgt, trgsen[j-1]);

                // gather embedding for aligned source 
                Expression i_s;
                int i = align[j]; 
                if (i < 0) 
                   i_s = i_null;  
                else
                   i_s = source_embeddings[i];

                // fed embeddings prev target word and aligned soource wird as input into the RNN
                Expression i_x_j = concatenate({i_t_m1, i_s});
                Expression i_y_j = builder_tgt.add_input(i_x_j);
                Expression i_r_j = i_b_tgt + i_R_tgt * i_y_j;
  
                // no need to generate the '<s>' token
                // the output is used to generate the next alignment, which is
                // part of the training objective 
                if (j > 0) { 
                   Expression i_err = pickneglogsoftmax(i_r_j, trgsen[j]);
                   errsTarget.push_back(i_err);
                }

                // no need to continue if we've generated the '</s>' token
                if (j+1 == trgsen.size()) break;

                // now for the alignment predictions, framed as si -> si'
                // ----- feature based "phi" term 
                Expression i_phi = input(cg, {this->feat_dim, srcsen.size()}, &(phi_sent[j+1]));  
                Expression phi_term = transpose(i_phi) * i_e; 
                // ----- other term using bigram embedding and RNN hidden state
                Expression rnn_term = transpose(i_C_src) * (i_L_src*i_s + i_M_src*i_y_j);
                //------ put all together     
                int target = align[j+1]; 
                if (target < 0) target = 0; // reassign NULL to position 0
                Expression i_a_err = pickneglogsoftmax(phi_term + rnn_term, target);
                errsJump.push_back(i_a_err);
                //LOLCAT(i_actions);
            }
            Expression i_nerrTarget = sum(errsTarget);
            Expression i_nerrJump = sum(errsJump);
            return make_pair(i_nerrTarget, i_nerrJump);
        }
};

} // namespace cnn

#undef WTF
#undef KTHXBYE
#undef LOLCAT
