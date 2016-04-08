#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"

#include "corpus.hh"
#include "expr-xtra.h"

#define WTF(expression) \
    std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
    std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
    WTF(expression) \
    KTHXBYE(expression) 

namespace cnn {

using namespace cnn::expr;
using namespace std;

template <class Builder>
struct RecurrentMarkovTranslationModel {
    LookupParameters* p_E_src;
    LookupParameters* p_E_tgt;
    LookupParameters* p_U_src; 
    Parameters* p_R_tgt;
    Parameters* p_b_tgt;
    Parameters* p_L_src;
    Parameters* p_M_src;
    Parameters* p_e;
    Parameters* p_v_a;
    Builder builder_tgt;
    Builder builder_src_fwd;
    Builder builder_src_bwd;
    unsigned embedding_dim;
    bool rnn_src_embeddings, morphology;
    map<int, vector<int>> morphemes;

    explicit RecurrentMarkovTranslationModel(Model& model, 
            unsigned src_vocab_size, unsigned tgt_vocab_size, unsigned src_morph_size,
            unsigned layers, unsigned _embedding_dim, unsigned hidden_dim,
            bool _rnn_src_embeddings, bool _morphology, const map<int, vector<int>>& _morphemes) 
        : builder_tgt(layers, (_rnn_src_embeddings) ? _embedding_dim+2*hidden_dim : 2*_embedding_dim, hidden_dim, &model),
          builder_src_fwd(1, _embedding_dim, hidden_dim, &model),
          builder_src_bwd(1, _embedding_dim, hidden_dim, &model),
          embedding_dim(_embedding_dim), rnn_src_embeddings(_rnn_src_embeddings)
    {
        p_E_src = model.add_lookup_parameters(src_vocab_size, {embedding_dim}); 
        p_E_tgt = model.add_lookup_parameters(tgt_vocab_size, {embedding_dim}); 
        p_R_tgt = model.add_parameters({tgt_vocab_size, hidden_dim});
        p_b_tgt = model.add_parameters({tgt_vocab_size});
	
	if (_morphology)
	   p_U_src = model.add_lookup_parameters(src_morph_size, {embedding_dim});
        
	morphology = _morphology;
	morphemes = _morphemes;

        if (!rnn_src_embeddings) {
            p_M_src = model.add_parameters({embedding_dim, hidden_dim});
        } else {
            p_M_src = model.add_parameters({2*hidden_dim, hidden_dim});
        }
        p_L_src = model.add_parameters({embedding_dim, hidden_dim});
        //p_b_src = model.add_lookup_parameters(src_vocab_size, {1});

        p_e = model.add_parameters({6});
        p_v_a = model.add_parameters({hidden_dim});

        this->embedding_dim = embedding_dim;
    }

    Expression word2embedding(ComputationGraph& cg, int word_id) {
      
      if (!morphology) return lookup(cg, p_E_src, word_id);
 
      std::vector<Expression> res;
      res.push_back(lookup(cg, p_E_src, word_id));
      //source_embeddings.push_back(i_x_t);
      for (auto k=0u; k < morphemes.at(word_id).size(); ++k)
	   res.push_back(lookup(cg, p_U_src, morphemes.at(word_id)[k]));
      return sum(res);
   }
    // return Expression of total loss -- currently just the translation sequence
    std::pair<Expression, Expression>
    BuildGraph(const AlignedEncodedSentence &sent_pair, ComputationGraph& cg) {
        using namespace cnn;
        using namespace std;

        Expression i_R_tgt = parameter(cg, p_R_tgt);
        Expression i_b_tgt = parameter(cg, p_b_tgt);
        Expression i_L_src = parameter(cg, p_L_src);
        Expression i_M_src = parameter(cg, p_M_src);
        Expression i_e = parameter(cg, p_e);
        Expression i_v_a = parameter(cg, p_v_a);
        Expression i_zero = repeat(cg, embedding_dim, 0.0f);
        //WTF(i_zero);

        vector<Expression> errsTarget, errsJump;

        const auto &srcsen = sent_pair.srcSentence;
        const auto &trgsen = sent_pair.trgSentence;
        const auto &align = sent_pair.alignment;
        int last_i = -1;
        bool start = true;

        //cout << "srcsen.size() " << srcsen.size() << endl;
        //cout << "trgsen.size() " << trgsen.size() << endl;
        
        Expression i_0vec = repeat(cg, srcsen.size()-1, 0);
        Expression i_1vec = 1 - i_0vec;
        Expression i_range = arange(cg, 0, srcsen.size()-1, false);
        Expression i_10vec = leq(i_range, 0, i_1vec);

        // embed the source sentence using a bidirectional RNN
        std::vector<Expression> source_embeddings;
        if (!rnn_src_embeddings) {
            // just lookup the embeddings for each source token
            for (auto i = 0u; i < srcsen.size()-1; ++i) 
                source_embeddings.push_back(word2embedding(cg, srcsen[i]));
        } else {
            // run a RNN backward and forward over the source sentence
            // and stack the top-level hidden states from each model as 
            // the representation at each position
            std::vector<Expression> src_fwd(srcsen.size()-1);
            std::vector<Expression> src_bwd(srcsen.size()-1);
            builder_src_fwd.new_graph(cg);
            builder_src_fwd.start_new_sequence();
            builder_src_bwd.new_graph(cg);
            builder_src_bwd.start_new_sequence();
            for (auto i = 0u; i < srcsen.size()-1; ++i) {
                src_fwd[i] = builder_src_fwd.add_input(word2embedding(cg, srcsen[i]));
                auto ri = srcsen.size()-1-i;
                // offset by 1; only includes the info beyond but not including the current token
                src_bwd[ri-1] = builder_src_bwd.add_input(word2embedding(cg, srcsen[ri]));
            }
	  
            for (auto i = 0u; i < srcsen.size()-1; ++i) {
                //WTF(src_fwd[i]);
                //WTF(src_bwd[i]);
                source_embeddings.push_back(concatenate(vector<Expression>({src_fwd[i], src_bwd[i]})));
                //WTF(source_embeddings.back());
            }
        }
        Expression i_C_src = concatenate_cols(source_embeddings);
        //WTF(i_C_src);

        builder_tgt.new_graph(cg);
        builder_tgt.start_new_sequence();

        // now range over the target sentence
        for (auto j = 0u; j < trgsen.size()-1; ++j) {
            int ttok = trgsen[j];
            int i = align[j];
            if (i < 0) i = 0; // assign NULL alignments to the start token

            // gather together embedding for source and target aligned words
            // fed as input into the RNN
            Expression i_w_j = lookup(cg, p_E_tgt, ttok);
            std::vector<Expression> input({ i_w_j });
            if (i >= 0) {
                input.push_back(source_embeddings[i]);
            } else {
                input.push_back(i_zero);
            }
            //WTF(input[0]);
            //WTF(input[1]);
            Expression i_x_j = concatenate(input);
            //WTF(i_x_j);
            Expression i_y_j = builder_tgt.add_input(i_x_j);
            //WTF(i_y_j);

            // the output is used to generate the next target word, which is
            // part of the training objective
            Expression i_r_j = i_b_tgt + i_R_tgt * i_y_j;
            //WTF(i_r_j);
            Expression i_err = pickneglogsoftmax(i_r_j, trgsen[j+1]);
            //WTF(i_err);
            errsTarget.push_back(i_err);

            // no need to continue if we've generated the '</s>' token
            if (j+2 == trgsen.size()) break;

            // now for the alignment predictions, framed as si -> si'
            // Chris Dyer -- this following thing can be done more efficiently using affine transform
            Expression jth_part = transpose(i_w_j) * i_L_src;
            Expression jth_rep = concatenate(vector<Expression>(srcsen.size()-1, jth_part));
            //WTF(jth_rep);
            Expression ith_part = transpose(i_C_src) * i_M_src;
            //WTF(ith_part);
            Expression i_a_j = tanh(jth_rep + ith_part) * i_v_a;
            //WTF(i_a_j);
            // and action type specific features
            vector<Expression> actions;
            if (start) { 
                //cout << "start" << endl;
                actions.push_back(i_1vec); // START action
                actions.push_back(i_0vec); // BACKWARD action
                actions.push_back(i_0vec); // STAY action
                actions.push_back(i_0vec); // MONOTONE action
                actions.push_back(i_0vec); // FORWARD action
                actions.push_back(i_10vec); // NULL action
            } else {
                //cout << "not start" << endl;
                actions.push_back(i_0vec); // START 
                //WTF(actions.back());
                Expression i_dist = i_range - last_i;
                //LOLCAT(i_dist);
                actions.push_back(leq(i_dist, -1, i_1vec)); // BACKWARD 
                //WTF(actions.back());
                //cout << "foeey" << endl;
                actions.push_back(eq(i_dist, 0)); // STAY 
                //cout << "gluey" << endl;
                //WTF(actions.back());
                actions.push_back(eq(i_dist, 1)); // MONOTONE 
                //WTF(actions.back());
                actions.push_back(geq(i_dist, 2, i_1vec)); // FORWARD 
                //WTF(actions.back());
                actions.push_back(i_10vec); // NULL 
                //WTF(actions.back());
            }
            //cout << "cating" << endl;
            Expression i_actions = concatenate_cols(actions);
            //LOLCAT(i_actions);
            Expression i_at_j = i_actions * i_e;
            //WTF(i_at_j);

            // FIXME: I think I prefer this:
            //  1) f(last_i, j, srcsen.size()) -- project to space nq (broadcast to nq x I)
            //                                      using a matrix of (3 x nq)
            //  2) g(i_range)                  -- project to space nq x I
            //                                      using a vector of (1 x nq)
            //  3) h(last_i - i_range)         -- project to space nq x I
            //                                      using a vector of (1 x nq)
            //  4) l = v . tanh(f + g + h)     -- project into I with non-linearity
            //                                      using a vector of (1 x nq)
            // where we tanh or log transform all the index based features, to prevent
            // numerical problems; nq needs to be 128 or so; f, g, and h are
            // linear transforms
            int target = align[j+1];
            if (target < 0) target = 0; // reassign NULL to '<s>' token
            //cout << "align[j+1]=" << align[j+1] << " while |srcsen|=" << srcsen.size() << endl;
            Expression i_a_err = pickneglogsoftmax(i_a_j + i_at_j, target);
            //WTF(i_a_err);
            errsJump.push_back(i_a_err);

            if (i >= 0) {
                start = false;
                last_i = i;
            }
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
