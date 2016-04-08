typedef enum {BEAM_SEARCH, IBM_MODEL1, PARTICLE_FILTER, NAIVE} ALGN_SAMPLING;

bool hyp_compare_func(AlignHypothesis &hyp1, AlignHypothesis &hyp2) { return (hyp1.cost<hyp2.cost); }

template <class TM_t>
class GenerateAlignment {
public:

   GenerateAlignment(TM_t *p_mm, FeatureFactory *p_feat_factory) {
      this->p_mm = p_mm;
      this->p_feat_factory = p_feat_factory;
   }

   void create_samples(MiniBatch &mb_train, unsigned K,  ALGN_SAMPLING method) {
      for (auto i = mb_train.lb; i < mb_train.rb; i++) {
         unsigned si = mb_train.order[i];
         std::vector<Alignment> algn;
         std::vector<double> weight;
         if (method == BEAM_SEARCH)
            beam_search(mb_train.corpus, si, K, algn, weight);
         else if (method == IBM_MODEL1)
            ibm_model1_sampling(mb_train.corpus, si, K, algn, weight); 
         else if (method == PARTICLE_FILTER) 
            particle_filtering(mb_train.corpus, si, K, algn, weight); //IE sequential importance sampling
         else 
            naive_sampling(mb_train.corpus, si, K, algn, weight);
         mb_train.algn.push_back(algn);
         mb_train.weight.push_back(weight);
      }
   }
   
   void beam_search(AlignedEncodedCorpus &corpus, unsigned sent_num, unsigned K, 
           std::vector<Alignment> &algn, std::vector<double> &weight) {

      ComputationGraph cg;

      const AlignedEncodedSentence &sent_pair = *corpus.at(sent_num);
      const auto &srcsen = sent_pair.srcSentence;
      const auto &trgsen = sent_pair.trgSentence;
      const auto &align = sent_pair.alignment;

      std::vector<AlignHypothesis> kbest;
      AlignHypothesis init_hyp;
      kbest.push_back(init_hyp);

      for (auto t = 0; t < trgsen.size(); t++) {
         // for each position t, consider alignment options and their scores etc
         std::vector<AlignHypothesis> q_hyp_list;
         for (auto cur_hyp : kbest) {
            Expression algn_distrib;
            auto err =  p_mm->next_rnn_state(cg, corpus, sent_num, *p_feat_factory, cur_hyp, algn_distrib);
            cg.incremental_forward(); //just compute the newly added RNN node
            auto cost_word = as_scalar(cg.get_value(err.i));
            auto log_prob_algn = as_vector(cg.get_value(algn_distrib.i));
            for (auto i = 0; i < log_prob_algn.size(); i++) {
               AlignHypothesis tmp_hyp = cur_hyp;
               tmp_hyp.cost += cost_word-log_prob_algn[i];
               tmp_hyp.align.push_back(i);
               tmp_hyp.t++; 
               q_hyp_list.push_back(tmp_hyp);
            }
         }
         sort(q_hyp_list.begin(), q_hyp_list.end(), hyp_compare_func);
         kbest.clear();
         for (auto i= 0; i < q_hyp_list.size(); i++) {
            if (i == K) break;
            kbest.push_back(q_hyp_list[i]);
         }
      }
      // now copy the kbest alignments and the weights onto the results 
      for (auto k = 0; k < kbest.size(); k++) {
         algn.push_back(kbest[k].align);
         weight.push_back(-kbest[k].cost);
      }
      weight2prob(weight); //normalizing the weights into probs
   }

   void ibm_model1_sampling(AlignedEncodedCorpus &corpus,unsigned sent_num, unsigned K,
                std::vector<Alignment> &algn, std::vector<double> &weight) {

   }

   void particle_filtering(AlignedEncodedCorpus &corpus,unsigned sent_num, unsigned K,
                std::vector<Alignment> &algn, std::vector<double> &weight) {

   }

   void naive_sampling(AlignedEncodedCorpus &corpus,unsigned sent_num, unsigned K,
                std::vector<Alignment> &algn, std::vector<double> &weight) {

   }

private:
   //Model model;
   TM_t *p_mm;
   FeatureFactory *p_feat_factory;

   void weight2prob(std::vector<double> &w) {
      double max_p = w[0];
      for (auto i = 0; i < w.size(); i++)
         if (w[i] > max_p) max_p = w[i];
      double expZ = 0;
      for (auto i = 0; i < w.size(); i++)
         expZ += exp(w[i]-max_p);
      double logZ = max_p + log(expZ);
      for (auto i = 0; i < w.size(); i++)
         w[i] = exp(w[i] - logZ);
   }

};

