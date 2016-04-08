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

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/range/irange.hpp>


namespace cnn {
// Builder is the class of RNN (either lstm or gru), kind of polymorphism here
template <class Builder>
struct AttentionalModel {
	explicit AttentionalModel(Model* model, unsigned vocab_size_tgt, unsigned layers,
			unsigned hidden_dim, unsigned align_dim, bool rnn_src_embeddings,
			bool giza_extensions, bool doc_context, int _local, bool _mask_flag, int src_ebd_size,
			LookupParameters* cs=0, LookupParameters *ct=0);

	~AttentionalModel();

	Expression BuildGraph(const std::vector<std::vector<float>>& source, const std::vector<int>& target,
			ComputationGraph& cg, Expression* alignment=0, const std::vector<int>* ctx=0,
			Expression *coverage=0);

	void display_ascii(const std::vector<std::vector<float>> &source, const std::vector<int>& target,
			ComputationGraph& cg, const Expression& alignment, Dict &td);

	void display_tikz(const std::vector<std::vector<float>> &source, const std::vector<int>& target,
			ComputationGraph& cg, const Expression& alignment, Dict &td);

	std::vector<int> greedy_decode(const std::vector<std::vector<float>> &source, ComputationGraph& cg,
			Dict &tdict, const std::vector<int>* ctx=0);

	std::vector<int> beam_decode(const std::vector<std::vector<float>> &source, ComputationGraph& cg,
			int beam_width, Dict &tdict, const std::vector<int>* ctx=0);

	std::vector<int> sample(const std::vector<int> &source, ComputationGraph& cg,
			Dict &tdict, const std::vector<int>* ctx=0);

	LookupParameters* p_cs;
	LookupParameters* p_ct;
	Parameters* p_R;
	Parameters* p_Q;
	Parameters* p_P;
	Parameters* p_S;
	Parameters* p_bias;
	Parameters* p_Wa;
	// Params for local model
	Parameters* p_Wpa;
	Parameters* p_Vpa;
	Parameters* p_Wpw;
	Parameters* p_Vpw;
	// Params for the masking
	Parameters* p_Wmp;
	Parameters* p_Wmn;
	Parameters* p_bias_mp;
//	Parameters* p_bias_mn;


	std::vector<Parameters*> p_Wh0;
	Parameters* p_Ua;
	Parameters* p_va;
	Parameters* p_Ta;
	Builder builder;
	Builder builder_src_fwd;
	Builder builder_src_fwd_1;
	Builder builder_src_fwd_2;
	Builder builder_src_bwd;
	Builder builder_src_bwd_1;
	Builder builder_src_bwd_2;
	bool rnn_src_embeddings;
	bool giza_extensions;
	bool doc_context;
	unsigned vocab_size_tgt;
	int local_attention;
	bool is_train = true;
	float dropout_prob = 0;
	bool pyramid = false;
	bool mask_flag = false;
	int SIZE_EMBEDDING_SRC = 39;
	float smooth_softmax = 1;
	// statefull functions for incrementally creating computation graph, one
	// target word at a time
	void start_new_instance(const std::vector<std::vector<float>> &src, ComputationGraph &cg, const std::vector<int> *ctx=0);
	Expression add_input(int tgt_tok, int t, ComputationGraph &cg, RNNPointer *prev_state=0);
	std::vector<float> *auxiliary_vector(); // memory management

	// Set the dropout
	void set_dropoutprob(const float& _dropout_prob, const bool _is_train);
	void set_smoothing_softmax(const float _smooth_softmax);
	void set_pyramid(const bool _pyramid);

	Expression drop_based_on_status(Expression* x);

	// state variables used in the above two methods
	Expression src;
	Expression i_R;
	Expression i_Q;
	Expression i_P;
	Expression i_S;
	Expression i_bias;
	Expression i_Wa;
	Expression i_Ua;
	Expression i_va;
	Expression i_uax;
	Expression i_Ta;
	Expression i_src_idx;
	Expression i_src_len;
	Expression i_tt_ctx;
	// Expression for local attentaional model
	Expression i_Wpa;
	Expression i_Vpa;
	Expression i_Wpw;
	Expression i_Vpw;

	// Expression for the mask
	Expression i_Wmp;
	Expression i_Wmn;
	Expression i_bias_mp;
	Expression i_bias_mn;

	std::vector<Expression> aligns;
	std::vector<std::vector<float>*> aux_vecs; // special storage for constant vectors
	unsigned num_aux_vecs;
	unsigned slen;
	unsigned tlen;
	bool has_document_context;
};

#define WTF(expression) \
		std::cout << #expression << " has dimensions " << cg.nodes[expression.i]->dim << std::endl;
#define KTHXBYE(expression) \
		std::cout << *cg.get_value(expression.i) << std::endl;

#define LOLCAT(expression) \
		WTF(expression) \
		KTHXBYE(expression)

///--------------------------Actual Code ------------------------------///

// Constructor
template <class Builder>
AttentionalModel<Builder>::AttentionalModel(cnn::Model* model, unsigned _vocab_size_tgt, unsigned layers, unsigned hidden_dim,
		unsigned align_dim, bool _rnn_src_embeddings, bool _giza_extentions, bool _doc_context, int _local, bool _mask_flag, int src_emb_size,
		LookupParameters* cs, LookupParameters *ct)
		: // RNN for the target size (3 * hidden_dim) or 2*hidden_dim
		builder(layers, (_rnn_src_embeddings) ? 3*hidden_dim : 2*hidden_dim, hidden_dim, model),
		// source always using just 1 layer, hidden dim = 39 (MFCC features)
		// FIX CODE HERE ^_^
		builder_src_fwd(1, src_emb_size, hidden_dim, model), // Call the constructor of lstm/gru
		builder_src_fwd_1(1, hidden_dim, hidden_dim, model), // Call the constructor of lstm/gru
		builder_src_fwd_2(1, hidden_dim, hidden_dim, model), // Call the constructor of lstm/gru
		builder_src_bwd(1, src_emb_size, hidden_dim, model),
		builder_src_bwd_1(1, hidden_dim, hidden_dim, model),
		builder_src_bwd_2(1, hidden_dim, hidden_dim, model),
		// Using bidirectional or not, if bidirectional => rnn of source, other wise use source directly
		rnn_src_embeddings(_rnn_src_embeddings), // bool whether bidirectional or not
		giza_extensions(_giza_extentions),
		SIZE_EMBEDDING_SRC(src_emb_size),
		doc_context(_doc_context),
		local_attention(_local),
		mask_flag(_mask_flag),
		vocab_size_tgt(_vocab_size_tgt),
		num_aux_vecs(0)
		// LD: This is kind of initialize the variable only (not very important I suppose)
		{
	//std::cerr << "Attentionalmodel(" << vocab_size_src  << " " <<  _vocab_size_tgt  << " " <<  layers  << " " <<  hidden_dim << " " <<  align_dim  << " " <<  _rnn_src_embeddings  << " " <<  _giza_extentions  << " " <<  _doc_context << ")\n";
	// Long Duong: add a set of params here

	// This is the embedding of source and target
	// p_cs = (cs) ? cs : model->add_lookup_parameters(vocab_size_src, {hidden_dim});
	p_ct = (ct) ? ct : model->add_lookup_parameters(vocab_size_tgt, {hidden_dim});

	// RNN for target size
	p_R = model->add_parameters({vocab_size_tgt, hidden_dim});
	p_P = model->add_parameters({hidden_dim, hidden_dim});
	p_bias = model->add_parameters({vocab_size_tgt});

	//
	p_Wa = model->add_parameters({align_dim, layers*hidden_dim});

	// LD: Add params for local attentional model
	if (local_attention == 1){
		std::cerr << " Using local attentional model (Predict) \n";
		// For predicting the anchor point
		p_Wpa = model->add_parameters({layers*hidden_dim, layers*hidden_dim});
		p_Vpa = model->add_parameters({layers*hidden_dim});
		// For predicting the width
		p_Wpw = model->add_parameters({layers*hidden_dim, layers*hidden_dim});
		p_Vpw = model->add_parameters({layers*hidden_dim});
	}
	if (local_attention == 0)
		std::cerr << " Using local attentional model (Fix) \n";
	if (mask_flag){
		std::cerr << " Using masking \n";
		p_Wmp = model->add_parameters({hidden_dim});
		p_Wmn = model->add_parameters({hidden_dim});
//		p_bias_mn = model->add_parameters({hidden_dim});
		p_bias_mp = model->add_parameters({hidden_dim});
	}

	if (rnn_src_embeddings) {
		// Using the bidirectional (rnn for source)
		p_Ua = model->add_parameters({align_dim, 2*hidden_dim});
		p_Q = model->add_parameters({hidden_dim, 2*hidden_dim});
	} else {
		p_Ua = model->add_parameters({align_dim, hidden_dim});
		p_Q = model->add_parameters({hidden_dim, hidden_dim});
	}
	if (giza_extensions) {
		p_Ta = model->add_parameters({align_dim, 9});
	}
	p_va = model->add_parameters({align_dim});

	if (doc_context) {
		if (rnn_src_embeddings) {
			p_S = model->add_parameters({hidden_dim, 2*hidden_dim});
		} else {
			p_S = model->add_parameters({hidden_dim, hidden_dim});
		}
	}
	// What is this ?? hidden_layers probably is the number of hidden items in RNN
	int hidden_layers = builder.num_h0_components();
	for (int l = 0; l < hidden_layers; ++l) {
		if (rnn_src_embeddings)
			p_Wh0.push_back(model->add_parameters({hidden_dim, 2*hidden_dim}));
		else
			p_Wh0.push_back(model->add_parameters({hidden_dim, hidden_dim}));
	}
		}
// Destructor
template <class Builder>
AttentionalModel<Builder>::~AttentionalModel()
{
	for (auto v: aux_vecs)
		delete v;
}

template <class Builder>
std::vector<float>* AttentionalModel<Builder>::auxiliary_vector()
{
	while (num_aux_vecs >= aux_vecs.size())
		aux_vecs.push_back(new std::vector<float>());
	// NB, we return the last auxiliary vector, AND increment counter
	return aux_vecs[num_aux_vecs++];
}

template <class Builder>
Expression AttentionalModel<Builder>::drop_based_on_status(Expression* x){
	if ((is_train) && (dropout_prob > 0)){
		*x = dropout(*x,dropout_prob);
	}
	if ((!is_train) && (dropout_prob > 0))
		*x = (1-dropout_prob) * *x; // scale up this
	return *x;
}

template <class Builder>
void AttentionalModel<Builder>::start_new_instance(const std::vector<std::vector<float>> &source, ComputationGraph &cg, const std::vector<int> *ctx)
{
	//slen = source.size() - 1;
	slen = source.size();
	std::vector<Expression> source_embeddings;
	if (!rnn_src_embeddings) {
		for (unsigned s = 0; s < slen; ++s) {
			Expression temp;
			std::cerr << " Size of source " << source[s].size() << "\n";
			temp = Expression(&cg,cg.add_input(Dim({SIZE_EMBEDDING_SRC}), &source[s]));
			source_embeddings.push_back(temp);
		}
	} else {
		// run a RNN backward and forward over the source sentence
		// and stack the top-level hidden states from each model as
		// the representation at each position
		std::vector<Expression> src_fwd(slen);
		builder_src_fwd.new_graph(cg);
		builder_src_fwd.start_new_sequence();

		for (unsigned i = 0; i < slen; ++i){
			Expression temp = input(cg, Dim({SIZE_EMBEDDING_SRC}), &(source[i]));
			drop_based_on_status(&temp);
			src_fwd[i] = builder_src_fwd.add_input(temp);
		}

		std::vector<Expression> src_bwd(slen);
		builder_src_bwd.new_graph(cg);
		builder_src_bwd.start_new_sequence();
		for (int i = slen-1; i >= 0; --i) {
			// offset by one position to the right, to catch </s> and generally
			// not duplicate the w_t already captured in src_fwd[t]
			Expression temp = Expression(&cg,cg.add_input(Dim({SIZE_EMBEDDING_SRC}), &source[i]));
			drop_based_on_status(&temp);
			src_bwd[i] = builder_src_bwd.add_input(temp);
		}

		if ((pyramid) && (slen > 32)){
			// Reduce the size of source by 4
			int scalling_factor = 4 ;
			slen = slen / scalling_factor;
			std::vector<Expression> src_fwd_1(slen);
			builder_src_fwd_1.new_graph(cg);
			builder_src_fwd_1.start_new_sequence();
			for (unsigned i = 0; i < slen; ++i){
				Expression temp = src_fwd[scalling_factor * i];
				drop_based_on_status(&temp);
				src_fwd_1[i] = builder_src_fwd_1.add_input(temp);
			}

			std::vector<Expression> src_bwd_1(slen);
			builder_src_bwd_1.new_graph(cg);
			builder_src_bwd_1.start_new_sequence();
			for (int i = slen-1; i >= 0; --i) {
				// offset by one position to the right, to catch </s> and generally
				// not duplicate the w_t already captured in src_fwd[t]
				Expression temp = src_bwd[scalling_factor * i];
				drop_based_on_status(&temp);
				src_bwd_1[i] = builder_src_bwd_1.add_input(temp);
			}

			// Reduce the size of source by further 2
			scalling_factor = 2 ;
			slen = slen / scalling_factor;
			std::vector<Expression> src_fwd_2(slen);
			builder_src_fwd_2.new_graph(cg);
			builder_src_fwd_2.start_new_sequence();
			for (unsigned i = 0; i < slen; ++i){
				Expression temp = src_fwd_1[scalling_factor * i];
				drop_based_on_status(&temp);
				src_fwd_2[i] = builder_src_fwd_2.add_input(temp);
			}

			std::vector<Expression> src_bwd_2(slen);
			builder_src_bwd_2.new_graph(cg);
			builder_src_bwd_2.start_new_sequence();
			for (int i = slen-1; i >= 0; --i) {
				// offset by one position to the right, to catch </s> and generally
				// not duplicate the w_t already captured in src_fwd[t]
				Expression temp = src_bwd_1[scalling_factor * i];
				drop_based_on_status(&temp);
				src_bwd_2[i] = builder_src_bwd_2.add_input(temp);
			}
			//
			src_fwd = src_fwd_2;
			src_bwd = src_bwd_2;
			//		std::cerr << " New source length = " << slen << "\n";
		}

		for (unsigned i = 0; i < slen; ++i)
			// neat way to combine these 2 to form an Expression
			source_embeddings.push_back(concatenate(std::vector<Expression>({src_fwd[i], src_bwd[i]})));
	}
	src = concatenate_cols(source_embeddings); // Form a long column vectors

	// now for the target sentence
	i_R = parameter(cg, p_R); // hidden -> word rep parameter
	i_Q = parameter(cg, p_Q);
	i_P = parameter(cg, p_P);
	i_bias = parameter(cg, p_bias);  // word bias
	i_Wa = parameter(cg, p_Wa);
	i_Ua = parameter(cg, p_Ua);
	i_va = parameter(cg, p_va);

	if (local_attention == 1){
		i_Wpa = parameter(cg, p_Wpa);
		i_Vpa = parameter(cg, p_Vpa);
		i_Wpw = parameter(cg, p_Wpw);
		i_Vpw = parameter(cg, p_Vpw);
	}
	if (mask_flag){
		i_Wmp = parameter(cg, p_Wmp);
		i_Wmn = parameter(cg, p_Wmn);
//		i_bias_mn = parameter(cg, p_bias_mn);
		i_bias_mp = parameter(cg, p_bias_mp);
//		cerr << "Finish geting expression from params \n";
	}

	// Precompute i_uax (don't need the hidden variable ...)
	i_uax = i_Ua * src; // R(align * 2hid) * R(2hid * source_len) = R(align * source_len)

	// reset aux_vecs counter, allowing the memory to be reused
	num_aux_vecs = 0;

	if (giza_extensions) {
		i_Ta = parameter(cg, p_Ta);
		i_src_idx = arange(cg, 0, slen, true, auxiliary_vector());
		i_src_len = repeat(cg, slen, log(1.0 + slen), auxiliary_vector());
	}

	aligns.clear();
	// ???? DON"T know what this mean (allocate auxiliary vector)
	aligns.push_back(repeat(cg, slen, 0.0f, auxiliary_vector()));

	// initialilse h from global information of the source sentence
#ifndef RNN_H0_IS_ZERO
	std::vector<Expression> h0;
	Expression i_src = average(source_embeddings); // try max instead?
	int hidden_layers = builder.num_h0_components();
	for (int l = 0; l < hidden_layers; ++l) {
		Expression i_Wh0 = parameter(cg, p_Wh0[l]);
		h0.push_back(tanh(i_Wh0 * i_src));
	}
	builder.new_graph(cg);
	builder.start_new_sequence(h0);
#else
	// builder is RNN system for the target sentence
	builder.new_graph(cg);
	builder.start_new_sequence();
#endif

	// document context; n.b. use "0" context for the first sentence
	if (doc_context && ctx != 0) {
		const std::vector<int> &context = *ctx;

		std::vector<Expression> ctx_embed;
		if (!rnn_src_embeddings) {
			for (unsigned s = 1; s+1 < context.size(); ++s)
				ctx_embed.push_back(lookup(cg, p_cs, context[s]));
		} else {
			ctx_embed.resize(context.size()-1);
			builder_src_fwd.start_new_sequence();
			for (unsigned i = 0; i+1 < context.size(); ++i)
				ctx_embed[i] = builder_src_fwd.add_input(lookup(cg, p_cs, context[i]));
		}
		Expression avg_context = average(source_embeddings);
		i_S = parameter(cg, p_S);
		i_tt_ctx = i_S * avg_context;
		has_document_context = true;
	} else {
		has_document_context = false;
	}
}

template <class Builder>
Expression AttentionalModel<Builder>::add_input(int trg_tok, int t, ComputationGraph &cg, RNNPointer *prev_state)
{
	// alignment input
	Expression i_wah_rep;


	// Local attentional model
	// DEFINE VALUE OF MIN AND MAX WIDTH
	float MIN_WIDTH = 1.0;
	float MAX_WIDTH = 15.0;

	Expression anchor;
	Expression width;
	float window_size = slen / (1.0 * tlen);

	if (t > 0) {
		//auto i_h_tm1 = builder.final_h().back();
		auto i_h_tm1 = concatenate(builder.final_h());
		Expression i_wah = i_Wa * i_h_tm1;
		// want numpy style broadcasting, but have to do this manually
		//WTF(i_wah);
		//	std::cerr<< " Source length = " << slen << "\n";
		i_wah_rep = concatenate_cols(std::vector<Expression>(slen, i_wah));
		//	WTF(i_wah_rep);
		//WTF(i_wah_rep);
		// => i_wah_rep in R(align * source_len)

		if (local_attention == 1){
			anchor = slen * logistic(transpose(i_Vpa) *  tanh(i_Wpa * i_h_tm1)) ;
			width = MIN_WIDTH + (MAX_WIDTH - MIN_WIDTH) * logistic(transpose(i_Vpw) *  tanh(i_Wpw * i_h_tm1));
		}

	}

	if ((local_attention == 0) || (t == 0)){
		anchor = Expression(&cg, cg.add_input(t * window_size + window_size / 2.0)); // Fix the middle point
		width = Expression(&cg, cg.add_input(window_size));
	}

	//    WTF(anchor);
	//    WTF(width);

	Expression i_e_t;
	if (giza_extensions) {
		std::vector<Expression> alignment_context;
		if (t > 0) {
			auto i_aprev = concatenate_cols(aligns);
			auto i_asum = sum_cols(i_aprev);
			// LD: this get the -1, 0, +1 shift in the position (why though)
			auto i_asum_pm = dither(cg, i_asum, 0.0f, auxiliary_vector());

			alignment_context.push_back(i_asum_pm);
			auto i_alast_pm = dither(cg, aligns.back(), 0.0f, auxiliary_vector());
			alignment_context.push_back(i_alast_pm);
		} else {
			// just 6 repeats of the 0 vector
			auto zeros = repeat(cg, slen, 0, auxiliary_vector());
			alignment_context.push_back(zeros);
			alignment_context.push_back(zeros);
			alignment_context.push_back(zeros);
			alignment_context.push_back(zeros);
			alignment_context.push_back(zeros);
			alignment_context.push_back(zeros);
		}
		alignment_context.push_back(i_src_idx);
		alignment_context.push_back(i_src_len);
		auto i_tgt_idx = repeat(cg, slen, log(1.0 + t), auxiliary_vector());
		alignment_context.push_back(i_tgt_idx);
		auto i_context = concatenate_cols(alignment_context);

		// R(align * slen) + R(align * 9) * R(9 * slen) = R(align * slen)
		auto i_e_t_input = i_uax + i_Ta * transpose(i_context);
		if (t > 0) i_e_t_input = i_e_t_input + i_wah_rep;
		i_e_t = transpose(tanh(i_e_t_input)) * i_va;
	} else {
		if (t > 0)
			i_e_t = transpose(tanh(i_wah_rep + i_uax)) * i_va;
		else
			// LD: Important (alignment when we have t = 0
			i_e_t = transpose(tanh(i_uax)) * i_va;
	}
	// Masking i_e_t
	if (mask_flag){
		// Index start from 1
		Expression i_arr = arange(cg, 1, slen+1, false, auxiliary_vector());
//		// Previous alignment
		Expression k_arr = arange(cg, 1, slen+1, false, auxiliary_vector());
		Expression vec_expect = cwise_multiply(aligns[aligns.size() -1], k_arr);
		Expression e_a_m_1 = sum_cols(transpose(vec_expect));
		Expression temp = concatenate_cols(std::vector<Expression>(slen, e_a_m_1));
		Expression vector_delta = i_arr - transpose(temp);
//		WTF(vector_delta); => R(slen)
		// Now calculate the function
		Expression hid1 = concatenate_cols(std::vector<Expression>(slen, i_Wmp)) * vector_delta + i_bias_mp;
		Expression d_delta = logistic(transpose(concatenate_cols(std::vector<Expression>(slen, i_Wmn))) * hid1);
		i_e_t = cwise_multiply(i_e_t, d_delta);
	}
	// Local attentional model
	if ((t>0) && (local_attention==1)){
		// Choose the anchor point = max i_e_t
		//anchor =
	}
	Expression i_e_t_masked;
	Expression i_alpha_t;
	//    WTF(i_e_t);
	if (local_attention != -1){

		//std::cerr << "anchor " << anchor << " width " << width << "\n";
		Expression std_dv = width / 4.0;
		Expression mask_arrange = arange(cg, 0, slen, false, auxiliary_vector());

		//LD: Must do the manual broascasting
		Expression anchor_dup = transpose(concatenate_cols(std::vector<Expression>(slen, anchor)));
		Expression temp1 = square(mask_arrange - anchor_dup);
		Expression temp2 = transpose(concatenate_cols(std::vector<Expression>(slen, 2.0 * square(std_dv))));
		//    		WTF(temp1);
		//    		WTF(temp2);
		Expression mask_exp =  exp(-1.0 * cdiv(temp1, temp2));
		//    		WTF(mask_exp);

//		i_e_t_masked = cwise_multiply(i_e_t, mask_exp); // must be pointwise multiplication
		i_e_t_masked =  mask_exp; // Trevor's idea, completely disregard eij (just use anchor and width)
		i_alpha_t = softmax(smooth_softmax * i_e_t_masked);
	}
	else
		i_alpha_t = softmax(smooth_softmax * i_e_t); // FIXME: consider summing to less than one?




	// LD: aligns is the list the accumulate the alignment vector at each time t
	aligns.push_back(i_alpha_t);

	// Remember: src in R(2hid * source_len)
	// and i_alpha_t in R(source_len)

	Expression i_c_t = src * i_alpha_t; // FIXME: effectively summing here, consider maxing?

	// word input (embedding of target word)
	Expression i_x_t = lookup(cg, p_ct, trg_tok);

	// LD: join x_t and c_t
	Expression input = concatenate(std::vector<Expression>({i_x_t, i_c_t}));
	//    WTF(input);
	// If using dropout then corrupt the input
	input = drop_based_on_status(&input);

	// y_t = RNN([x_t, a_t])
	Expression i_y_t;
	if (prev_state)
		i_y_t = builder.add_input(*prev_state, input);
	else
		i_y_t = builder.add_input(input);
	// Note: i_y_t here belong to R(hid)


	if (doc_context && has_document_context)
		i_y_t = i_y_t + i_tt_ctx;
#ifndef VANILLA_TARGET_LSTM
	// Bahdanau does a max-out thing here; I do a tanh. Tomaatos tomateos.
	Expression i_tildet_t = tanh(affine_transform({i_y_t, i_Q, i_c_t, i_P, i_x_t}));
	Expression i_r_t = affine_transform({i_bias, i_R, i_tildet_t});
#else
	Expression i_r_t = affine_transform({i_bias, i_R, i_y_t});
	// i_R belong R(voc * hid)
	// i_bias be
#endif
	//    WTF(i_r_t);
	return i_r_t;
}

template <class Builder>
void AttentionalModel<Builder>::set_dropoutprob(const float& _dropout_prob, const bool _is_train){
	dropout_prob = _dropout_prob;
	is_train = _is_train;
}

template <class Builder>
void AttentionalModel<Builder>::set_smoothing_softmax(const float _smooth_softmax){
	smooth_softmax = _smooth_softmax;
}


template <class Builder>
void AttentionalModel<Builder>::set_pyramid(const bool _pyramid){
	pyramid = _pyramid;
}


template <class Builder>
Expression AttentionalModel<Builder>::BuildGraph(const std::vector<std::vector<float>> &source,
		const std::vector<int>& target, ComputationGraph& cg, Expression *alignment,
		const std::vector<int>* ctx, Expression *coverage)
		{
	//std::cout << "source sentence length: " << source.size() << " target: " << target.size() << std::endl;

	// ctx is used when we have the documents
//		std::cerr<< "Start new instance \n";
	start_new_instance(source, cg, ctx);
//		std::cerr << "Add input\n";
	std::vector<Expression> errs;
	tlen = target.size() - 1;
	for (unsigned t = 0; t < tlen; ++t) {
//		std::cerr << "with t = " << t << " out of " << tlen << "\n";
		Expression i_r_t = add_input(target[t], t, cg);
		Expression i_err = pickneglogsoftmax(i_r_t, target[t+1]);
		errs.push_back(i_err);
	}
	//    std::cerr << "Finish add input\n";
	// save the alignment for later
	if (alignment != 0) {
		// pop off the last alignment column

		*alignment = concatenate_cols(aligns);
		//        std::cerr << "Reach here for computing alignment \n";
	}

	// AM paper (vision one) has a penalty over alignment rows deviating from 1
	if (coverage != nullptr) {
		Expression i_aligns = (alignment != 0) ? *alignment : concatenate_cols(aligns);
		Expression i_totals = sum_cols(i_aligns);

		// only care about the non-null entries,
		// LD: because for the source langauge the start and the end is <s> and <\s>
		Expression i_total_nonull = pickrange(i_totals, 1, slen-1);
		Expression i_ones = repeat(cg, slen-2, 1.0f, auxiliary_vector());
		Expression i_penalty = squared_distance(i_total_nonull, i_ones);
		*coverage = i_penalty;
	}

	Expression i_nerr = sum(errs);
	//    WTF(i_nerr);
	return i_nerr;
		}

template <class Builder>
void 
AttentionalModel<Builder>::display_ascii(const std::vector<std::vector<float>> &source, const std::vector<int>& target,
		ComputationGraph &cg, const Expression &alignment, Dict &td)
		{
	using namespace std;

	// display the alignment
	//float I = target.size() - 1;
	//float J = source.size() - 1;
	int I = target.size();
	int J = source.size();
	if ((pyramid) && (J > 32)) {
		J = J / 8 ;
	}

	//vector<string> symbols{"\u2588","\u2589","\u258A","\u258B","\u258C","\u258D","\u258E","\u258F"};
	vector<string> symbols{".","o","*","O","@"};
	int num_symbols = symbols.size();
	vector<float> thresholds;
	thresholds.push_back(0.8/I);
	float lgap = (0 - log(thresholds.back())) / (num_symbols - 1);
	for (auto rit = symbols.begin(); rit != symbols.end(); ++rit) {
		float thr = exp(log(thresholds.back()) + lgap);
		thresholds.push_back(thr);
	}
	// FIXME: thresholds > 1, what's going on?
	//cout << thresholds.back() << endl;

	const Tensor &a = cg.get_value(alignment.i);
	//    cerr << "Dim of alignment : " << a.d << "vs " << I <<","<< J << "\n";
	//cout << "I = " << I << " J = " << J << endl;

	cout.setf(ios_base::adjustfield, ios_base::left);
	cout << setw(12) << "source" << "  ";
	cout.setf(ios_base::adjustfield, ios_base::right);
	for (int j = 0; j < J; ++j)
		cout << setw(2) << j << ' ';
	cout << endl;

	for (int i = 0; i < I; ++i) {
		cout.setf(ios_base::adjustfield, ios_base::left);
		//cout << setw(12) << td.Convert(target[i+1]) << "  ";
		cout << setw(12) << td.Convert(target[i]) << "  ";
		cout.setf(ios_base::adjustfield, ios_base::right);
		float max_v = 0;
		int max_j = -1;
		for (int j = 0; j < J; ++j) {
			float v = TensorTools::AccessElement(a, Dim(j, i));
			string symbol;
			for (int s = 0; s <= num_symbols; ++s) {
				if (s == 0)
					symbol = ' ';
				else
					symbol = symbols[s-1];
				if (s != num_symbols && v < thresholds[s])
					break;
			}
			cout << setw(2) << symbol << ' ';
			if (v >= max_v) {
				max_v = v;
				max_j = j;
			}
		}
		cout << setw(20) << "max Pr=" << setprecision(3) << setw(5) << max_v << " @ " << max_j << endl;
	}
	cout << resetiosflags(ios_base::adjustfield);

	if ((pyramid) && (source.size() > 32))
		cout << "Using pyramidal structure \n";
	else
		cout << "Raw source mfcc frame \n";

	cout << endl;
		}

template <class Builder>
void 
AttentionalModel<Builder>::display_tikz(const std::vector<std::vector<float>> &source, const std::vector<int>& target,
		ComputationGraph &cg, const Expression &alignment,  Dict &td)
		{
	using namespace std;

	// display the alignment
	int I = target.size();
	int J = source.size();

	if ((pyramid) && (J > 32)) {
		J = J / 8 ;
	}

	const Tensor &a = cg.get_value(alignment.i);
	cout << a.d[0] << " x " << a.d[1] << endl;

	cout << "\\begin{tikzpicture}[scale=0.5]\n";
	for (int j = 0; j < J; ++j)
		cout << "\\node[anchor=west,rotate=90] at (" << j+0.5 << ", " << I+0.2 << ") { " << j << " };\n";
	for (int i = 0; i < I; ++i)
		cout << "\\node[anchor=west] at (" << J+0.2 << ", " << I-i-0.5 << ") { " << td.Convert(target[i]) << " };\n";

	float eps = 0.01;
	for (int i = 0; i < I; ++i) {
		for (int j = 0; j < J; ++j) {
			float v = TensorTools::AccessElement(a, Dim(j, i));
			//int val = int(pow(v, 0.3) * 100);
			int val = int(pow(v * 100, 1.5)); // Good ^-^
			if (val > 100) val = 100;
			//int val = int(v * 100);
			cout << "\\fill[black!" << val << "!white] (" << j+eps << ", " << I-i-1+eps << ") rectangle (" << j+1-eps << "," << I-i-eps << ");\n";
		}
	}
	cout << "\\draw[step=1cm,color=gray] (0,0) grid (" << J << ", " << I << ");\n";
	cout << "\\end{tikzpicture}\n";
		}


template <class Builder>
std::vector<int>
AttentionalModel<Builder>::greedy_decode(const std::vector<std::vector<float>>  &source, ComputationGraph& cg,
		cnn::Dict &tdict, const std::vector<int>* ctx)
		{
	const int sos_sym = tdict.Convert("<s>");
	const int eos_sym = tdict.Convert("</s>");

	std::vector<int> target;
	target.push_back(sos_sym);

	//std::cerr << tdict.Convert(target.back());
	int t = 0;
	start_new_instance(source, cg, ctx);
	while (target.back() != eos_sym)
	{
		Expression i_scores = add_input(target.back(), t, cg);
		Expression ydist = softmax(i_scores); // compiler warning, but see below

		// find the argmax next word (greedy)
		unsigned w = 0;
		auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
		auto pr_w = dist[w];
		for (unsigned x = 1; x < dist.size(); ++x) {
			if (dist[x] > pr_w) {
				w = x;
				pr_w = dist[x];
			}
		}

		// break potential infinite loop
		if (t > 70) {
			w = eos_sym;
			pr_w = dist[w];
		}

		//std::cerr << " " << tdict.Convert(w) << " [p=" << pr_w << "]";
		t += 1;
		target.push_back(w);
	}
	//std::cerr << std::endl;

	return target;
		}

struct Hypothesis {
	Hypothesis() {};
	Hypothesis(RNNPointer state, int tgt, float cst, std::vector<Expression> &al)
	: builder_state(state), target({tgt}), cost(cst), aligns(al) {}
	Hypothesis(RNNPointer state, int tgt, float cst, Hypothesis &last, std::vector<Expression> &al)
	: builder_state(state), target(last.target), cost(cst), aligns(al) {
		target.push_back(tgt);
	}
	RNNPointer builder_state;
	std::vector<int> target;
	float cost;
	std::vector<Expression> aligns;
};

template <class Builder>
std::vector<int>
AttentionalModel<Builder>::beam_decode(const std::vector<std::vector<float>> &source, ComputationGraph& cg, int beam_width,
		cnn::Dict &tdict, const std::vector<int>* ctx)
		{
	const int sos_sym = tdict.Convert("<s>");
	const int eos_sym = tdict.Convert("</s>");

	start_new_instance(source, cg, ctx);

	std::vector<Hypothesis> chart;
	chart.push_back(Hypothesis(builder.state(), sos_sym, 0.0f, aligns));

	std::vector<unsigned> vocab(boost::copy_range<std::vector<unsigned>>(boost::irange(0u, vocab_size_tgt)));
	std::vector<Hypothesis> completed;

	for (int steps = 0; completed.size() < beam_width && steps < 2*source.size() && steps < 100; ++steps) {
		std::vector<Hypothesis> new_chart;

		for (auto &hprev: chart) {
			//std::cerr << "hypo t[-1]=" << tdict.Convert(hprev.target.back()) << " cost " << hprev.cost << std::endl;
			if (giza_extensions) aligns = hprev.aligns;
			Expression i_scores = add_input(hprev.target.back(), hprev.target.size()-1, cg, &hprev.builder_state);
			Expression ydist = softmax(i_scores); // compiler warning, but see below

			// find the top k best next words
			unsigned w = 0;
			auto dist = as_vector(cg.incremental_forward()); // evaluates last expression, i.e., ydist
			std::partial_sort(vocab.begin(), vocab.begin()+beam_width, vocab.end(),
					[&dist](unsigned v1, unsigned v2) { return dist[v1] > dist[v2]; });

			// add to chart
			for (auto vi = vocab.begin(); vi < vocab.begin() + beam_width; ++vi) {
				//std::cerr << "\t++word " << tdict.Convert(*vi) << " prob " << dist[*vi] << std::endl;
//				if (new_chart.size() < beam_width) {
					Hypothesis hnew(builder.state(), *vi, hprev.cost-log(dist[*vi]), hprev, aligns);
					if (*vi == eos_sym)
						completed.push_back(hnew);
					else
						new_chart.push_back(hnew);
//				}
			}
		}

		if (new_chart.size() > beam_width) {
			// sort new_chart by score, to get kbest candidates
			std::partial_sort(new_chart.begin(), new_chart.begin()+beam_width, new_chart.end(),
					[](Hypothesis &h1, Hypothesis &h2) { return h1.cost < h2.cost; });
			new_chart.resize(beam_width);
		}
		chart.swap(new_chart);
	}

	// sort completed by score, adjusting for length -- not very effective, too short!
    if (completed.size() >0){
    	// sort completed by score, adjusting for length -- not very effective, too short! (normalized the cost according to length
    	// Probably better way to do it.
    	auto best = std::min_element(completed.begin(), completed.end(),
    	            [](Hypothesis &h1, Hypothesis &h2) { return h1.cost/h1.target.size() < h2.cost/h2.target.size(); });
    	return best->target;
    }
    else {
    	// If not just greedy pick the chart
    	// Probably better way to do it.
    	auto best = std::min_element(chart.begin(), chart.end(),
    	            [](Hypothesis &h1, Hypothesis &h2) { return h1.cost/h1.target.size() < h2.cost/h2.target.size(); });
    	return best->target;
    }

}

template <class Builder>
std::vector<int>
AttentionalModel<Builder>::sample(const std::vector<int> &source, ComputationGraph& cg, cnn::Dict &tdict,
		const std::vector<int> *ctx)
		{
	const int sos_sym = tdict.Convert("<s>");
	const int eos_sym = tdict.Convert("</s>");

	std::vector<int> target;
	target.push_back(sos_sym);

	std::cerr << tdict.Convert(target.back());
	int t = 0;
	start_new_instance(source, cg, ctx);
	while (target.back() != eos_sym)
	{
		Expression i_scores = add_input(target.back(), t, cg);
		Expression ydist = softmax(i_scores);

		// in rnnlm.cc there's a loop around this block -- why? can incremental_forward fail?
		auto dist = as_vector(cg.incremental_forward());
		double p = rand01();
		unsigned w = 0;
		for (; w < dist.size(); ++w) {
			p -= dist[w];
			if (p < 0) break;
		}
		// this shouldn't happen
		if (w == dist.size()) w = eos_sym;

		std::cerr << " " << tdict.Convert(w) << " [p=" << dist[w] << "]";
		t += 1;
		target.push_back(w);
	}
	std::cerr << std::endl;

	return target;
		}


#undef WTF
#undef KTHXBYE
#undef LOLCAT

}; // namespace cnn
