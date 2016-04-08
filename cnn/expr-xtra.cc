#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/expr-xtra.h"

using namespace cnn;
using namespace std;

// Chris -- this should be a library function
Expression arange(ComputationGraph &cg, unsigned begin, unsigned end, bool log_transform, std::vector<float> *aux_mem) 
{
    aux_mem->clear();
    for (unsigned i = begin; i < end; ++i) 
        aux_mem->push_back((log_transform) ? log(1.0 + i) : i);
    return Expression(&cg, cg.add_input(Dim({(long) (end-begin)}), aux_mem));
}

// Chris -- this should be a library function
Expression repeat(ComputationGraph &cg, unsigned num, float value, std::vector<float> *aux_mem) 
{
    aux_mem->clear();
    aux_mem->resize(num, value);
    return Expression(&cg, cg.add_input(Dim({long(num)}), aux_mem));
}

// Chris -- this should be a library function
Expression dither(ComputationGraph &cg, const Expression &expr, float pad_value, std::vector<float> *aux_mem)
{
    const auto& shape = cg.nodes[expr.i]->dim;
    aux_mem->clear();
    aux_mem->resize(shape.cols(), pad_value);
    Expression padding(&cg, cg.add_input(Dim({shape.cols()}), aux_mem));
    Expression padded = concatenate(std::vector<Expression>({padding, expr, padding}));
    Expression left_shift = pickrange(padded, 2, shape.rows()+2);
    Expression right_shift = pickrange(padded, 0, shape.rows());
    return concatenate_cols(std::vector<Expression>({left_shift, expr, right_shift}));
}

// these expressions can surely be implemented much more efficiently than this
Expression abs(const Expression &expr) 
{
    return rectify(expr) + rectify(-expr); 
}

// binary boolean functions, is it better to use a sigmoid?
Expression eq(const Expression &expr, float value, float epsilon) 
{
    return min(rectify(expr - (value - epsilon)), rectify(-expr + (value + epsilon))) / epsilon; 
}

Expression geq(const Expression &expr, float value, Expression &one, float epsilon) 
{
    return min(one, rectify(expr - (value - epsilon)) / epsilon);
        //rectify(1 - rectify(expr - (value - epsilon)));
}

Expression leq(const Expression &expr, float value, Expression &one, float epsilon) 
{
    return min(one, rectify((value + epsilon) - expr) / epsilon);
    //return rectify(1 - rectify((value + epsilon) - expr));
}

/// do forward and backward embedding
template<class Builder>
Expression bidirectional(int slen, const vector<int>& source, ComputationGraph& cg, LookupParameters* p_cs,
    Builder & encoder_fwd, Builder& encoder_bwd)
{

    std::vector<Expression> source_embeddings;

    std::vector<Expression> src_fwd(slen);
    std::vector<Expression> src_bwd(slen);

    for (int t = 0; t < source.size(); ++t) {
        Expression i_x_t = lookup(cg, p_cs, source[t]);
        src_fwd[t] = encoder_fwd.add_input(i_x_t);
    }
    for (int t = source.size() - 1; t >= 0; --t) {
        Expression i_x_t = lookup(cg, p_cs, source[t]);
        src_bwd[t] = encoder_bwd.add_input(i_x_t);
    }

    for (unsigned i = 0; i < slen - 1; ++i)
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i + 1] })));
    source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[slen - 1], src_bwd[slen - 1] })));
    Expression src = concatenate_cols(source_embeddings);

    return src;
}

/// source [1..T][1..NUTT] is time first and then content from each utterance
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
vector<Expression> embedding(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero, size_t feat_dim)
{
    size_t nutt = source.size();
    /// get the maximum length of utternace from all speakers
    slen = 0;
    for (auto p : source)
        slen = (slen < p.size()) ? p.size() : slen;

    std::vector<Expression> source_embeddings;

    Expression i_x_t;

    for (int t = 0; t < slen; ++t) {
        vector<Expression> vm;
        for (size_t k = 0; k < nutt; k++)
        {
            if (source[k].size() > t)
                vm.push_back(lookup(cg, p_cs, source[k][t]));
            else
                vm.push_back(input(cg, { (long)feat_dim }, &zero));
        }
        i_x_t = concatenate_cols(vm);
        source_embeddings.push_back(i_x_t);
    }

    return source_embeddings;
}

/// return an expression for the time embedding weight
Expression time_embedding_weight(size_t t, size_t feat_dim, size_t slen, ComputationGraph& cg, map<size_t, map<size_t, tExpression>> & m_time_embedding_weight)
{
    if (m_time_embedding_weight.find(t) == m_time_embedding_weight.end() || m_time_embedding_weight[t].find(feat_dim) == m_time_embedding_weight[t].end()
        || m_time_embedding_weight[t][feat_dim].find(slen) == m_time_embedding_weight[t][feat_dim].end()){

        vector<cnn::real> lj(feat_dim, 1 - (t +1) / (slen + 0.0));
        for (size_t k = 0; k < lj.size(); k++)
        {
            lj[k] -= (k + 1.0) / feat_dim * (1 - 2.0 * (t  + 1.0) / slen );
        }
        Expression wgt = input(cg, { (long)feat_dim }, &lj);
        cg.incremental_forward();
        m_time_embedding_weight[t][feat_dim][slen] = wgt;
    }
    return m_time_embedding_weight[t][feat_dim][slen] ;
}

/// representation of a sentence using a single vector
vector<Expression> time_embedding(unsigned & slen, const vector<vector<int>>& source, ComputationGraph& cg, LookupParameters* p_cs, vector<cnn::real>& zero, size_t feat_dim, map<size_t, map<size_t, tExpression>>  &m_time_embedding_weight)
{
    size_t nutt = source.size();
    /// get the maximum length of utternace from all speakers
    slen = 0;
    for (auto p : source)
        slen = (slen < p.size()) ? p.size() : slen;

    std::vector<Expression> source_embeddings;

    Expression i_x_t;

    for (size_t k = 0; k < nutt; k++)
    {
        vector<Expression> vm;
        int t = 0;
        while (t < source[k].size())
        {
            Expression xij = lookup(cg, p_cs, source[k][t]);
            Expression wgt = time_embedding_weight(t, feat_dim, slen, cg, m_time_embedding_weight); 
            vm.push_back(cwise_multiply(wgt, xij));

            t++;
        }
        i_x_t = sum(vm);
        source_embeddings.push_back(i_x_t);
    }
    return source_embeddings;
}


vector<size_t> each_sentence_length(const vector<vector<int>>& source)
{
    /// get each sentence length
    vector<size_t> slen;
    for (auto p : source)
        slen.push_back(p.size());
    return slen;
}

bool similar_length(const vector<vector<int>>& source)
{
    int imax = -1;
    int imin = 10000;
    /// get each sentence length
    vector<int> slen;
    for (auto p : source)
    {
        imax = std::max<int>(p.size(), imax);
        imin = std::min<int>(p.size(), imin);
    }

    return (fabs((float)(imax - imin)) < 3.0);
}

vector<Expression> attention_to_source(vector<Expression> & v_src, const vector<size_t>& v_slen,
    Expression i_U, Expression src, Expression i_va, Expression i_Wa,
    Expression i_h_tm1, size_t a_dim, size_t feat_dim, size_t nutt)
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = 0;
    vector<Expression> i_wah_rep;

    for (auto p : v_slen)
        slen += p;

    Expression i_wah = i_Wa * i_h_tm1;  /// [d nutt]
    Expression i_wah_reshaped = reshape(i_wah, { long(nutt * a_dim) });
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_wah_each = pickrange(i_wah_reshaped, k * a_dim, (k + 1)*a_dim);  /// [d]
        /// need to do subsampling
        i_wah_rep.push_back(concatenate_cols(std::vector<Expression>(v_slen[k], i_wah_each)));  /// [d v_slen[k]]
    }
    Expression i_wah_m = concatenate_cols(i_wah_rep);  // [d \sum_k v_slen[k]]

    i_e_t = transpose(tanh(i_wah_m + src)) * i_va;  // [\sum_k v_slen[k] 1]

    Expression i_alpha_t;

    vector<Expression> v_input;
    int istt = 0;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_input;
        int istp = istt + v_slen[k];

        i_input = v_src[k] * softmax(pickrange(i_e_t, istt, istp));  // [D v_slen[k]] x[v_slen[k] 1] = [D 1]
        v_input.push_back(i_input);

        istt = istp;
    }

    return v_input;
}

vector<Expression> local_attention_to(ComputationGraph& cg, vector<int> v_slen,
    Expression i_Wlp, Expression i_blp, Expression i_vlp,
    Expression i_h_tm1, size_t nutt)
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = v_slen[0];
    vector<Expression> v_attention_to;

    Expression i_wah = i_Wlp * i_h_tm1;
    Expression i_wah_bias = concatenate_cols(vector<Expression>(nutt, i_blp));
    Expression i_position = logistic(i_vlp * tanh(i_wah + i_wah_bias));

    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_position_each = pick(i_position, k) * v_slen[k];

        /// need to do subsampling
        v_attention_to.push_back(i_position_each);
    }
    return v_attention_to;
}


vector<Expression> convert_to_vector(Expression & in, size_t dim, size_t nutt)
{
    Expression i_d = reshape(in, { long(dim * nutt) });
    vector<Expression> v_d;

    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_t_kk = pickrange(i_d, k * dim, (k + 1) * dim);
        v_d.push_back(i_t_kk);
    }
    return v_d;
}

/// use key to find value, return a vector with element for each utterance
vector<Expression> attention_weight(const vector<size_t>& v_slen, const Expression& src_key, Expression i_va, Expression i_Wa,
    Expression i_h_tm1, size_t a_dim, size_t nutt)
{
    Expression i_c_t;
    Expression i_e_t;
    int slen = 0;
    vector<Expression> i_wah_rep;

    for (auto p : v_slen)
        slen += p;

    Expression i_wah = i_Wa * i_h_tm1;  /// [d nutt]
    Expression i_wah_reshaped = reshape(i_wah, { long(nutt * a_dim) });

    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_wah_each = pickrange(i_wah_reshaped, k * a_dim, (k + 1)*a_dim);  /// [d]
        /// need to do subsampling
        i_wah_rep.push_back(concatenate_cols(std::vector<Expression>(v_slen[k], i_wah_each)));  /// [d v_slen[k]]
    }
    Expression i_wah_m = concatenate_cols(i_wah_rep);  // [d \sum_k v_slen[k]]

    /// compare the input with key for every utterance
    i_e_t = transpose(tanh(i_wah_m + concatenate_cols(vector<Expression>(nutt, src_key)))) * i_va;  // [\sum_k v_slen[k] 1]

    Expression i_alpha_t;

    vector<Expression> v_input;
    int istt = 0;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_input;
        int istp = istt + v_slen[k];

        i_input = softmax(pickrange(i_e_t, istt, istp));  // [v_slen[k] 1] 
        v_input.push_back(i_input);

        istt = istp;
    }

    return v_input;
}

/// use key to find value, return a vector with element for each utterance
vector<Expression> attention_to_key_and_retreive_value(const Expression& M_t, const vector<size_t>& v_slen,
    const vector<Expression> & i_attention_weight, size_t nutt)
{

    vector<Expression> v_input;
    int istt = 0;
    for (size_t k = 0; k < nutt; k++)
    {
        Expression i_input;
        int istp = istt + v_slen[k];

        i_input = M_t * i_attention_weight[k];  // [D v_slen[k]] x[v_slen[k] 1] = [D 1]
        v_input.push_back(i_input);

        istt = istp;
    }

    return v_input;
}


Expression bidirectional(int slen, const vector<vector<cnn::real>>& source, ComputationGraph& cg, std::vector<Expression>& src_fwd, std::vector<Expression>& src_bwd)
{

    assert(slen == source.size());
    std::vector<Expression> source_embeddings;

    src_fwd.resize(slen);
    src_bwd.resize(slen);

    for (int t = 0; t < source.size(); ++t) {
        long fdim = source[t].size();
        src_fwd[t] = input(cg, { fdim }, &source[t]);
    }
    for (int t = source.size() - 1; t >= 0; --t) {
        long fdim = source[t].size();
        src_bwd[t] = input(cg, { fdim }, &source[t]);
    }

    for (unsigned i = 0; i < slen - 1; ++i)
        source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[i], src_bwd[i + 1] })));
    source_embeddings.push_back(concatenate(std::vector<Expression>({ src_fwd[slen - 1], src_bwd[slen - 1] })));
    Expression src = concatenate_cols(source_embeddings);

    return src;
}

vector<cnn::real> get_value(Expression nd, ComputationGraph& cg)
{
    /// get the top output
    vector<cnn::real> vm;

    vm = as_vector(cg.get_value(nd));

    return vm;
}

vector<cnn::real> get_error(Expression nd, ComputationGraph& cg)
{
    /// get the top output
    vector<cnn::real> vm;

    vm = as_vector(cg.get_error(nd.i));

    return vm;
}
