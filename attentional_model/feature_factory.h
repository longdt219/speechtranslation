#include "corpus.hh"
#include <vector>

#include <iostream>

typedef std::vector<cnn::real> FEATS_WORD;
typedef std::vector< FEATS_WORD > FEATS_SENT;
typedef std::vector<FEATS_SENT> FEATS_CORPUS;

enum ACTIONS {START, STAY, BACKWARD, MONOTONE, FORWARD, NULLA};


class FeatureFactory {
public:
   unsigned feat_dim(void) {return num_feats;}

   //void freeze(void) {}
  
   void print(FEATS_CORPUS &corp) {
      std::cout << "\n the corpus size is: " << corp.size() << std::endl;
      for (auto k = 0u; k < corp.size(); k++) {
         FEATS_SENT sent = corp[k];
         std::cout << "== sents " << k << " has length " << sent.size() << std::endl;
         for (auto j = 0u; j < sent.size(); j++) {
            FEATS_WORD word = sent[j];
            std::cout << "    pos " << j << " len " << word.size() << " : ";
            for (auto i = 0u; i < word.size(); i++) std::cout << word[i] << " ";
            std::cout << std::endl; 
         }
      } 
   }

   void print_seq(FEATS_CORPUS &cphi, AlignedEncodedCorpus &corp) {
      for (auto k = 0u; k < corp.size(); k++) {
         auto align = (corp.at(k))->alignment;
         auto trgsen = (corp.at(k))->trgSentence;
         auto srcsen = (corp.at(k))->srcSentence;
         auto sphi = cphi[k];
         std::cout << "== sent_pair: " << k << std::endl;
         //---
         std::cout << " src_sen : ";  
         for (auto j = 0u; j < srcsen.size(); j++)
            std::cout << srcsen[j] << " ";
         std::cout << "\n trg_sen : ";
         for (auto j = 0u; j < trgsen.size(); j++)
            std::cout << trgsen[j] << " "; 
         std::cout << "\n aln : ";
         for (auto j = 0u; j < align.size(); j++)
            std::cout << align[j] << "_" << j << " ";
         std::cout << "\n";
         //---
         for (auto j = 1u; j < align.size(); j++) {
            auto wphi = sphi[j];
            int i = align[j]; if (i < 0) i=0;
            std::cout << action2string(wphi, i * this->num_feats) 
                      << " tgt_" << j << " src_" << i << std::endl;
         }
      }
   }

   std::string action2string(FEATS_WORD &fw, int base) {
      std::string a2s[] = {"START", "STAY", "BACKWARD", "MONOTONE", "FORWARD", "NULLA"};
      int res, sum=0;
      for (auto i = 0u; i < 6; i++) {
         sum += fw[base+i];
         if (fw[base+i] == 1) res=i;
      }
      if (sum == 0) return "NONE!";
      if (sum == 1) return a2s[res];
      return "MULTIPLE!";
   }
 
   void build_feats(AlignedEncodedCorpus &corp, FEATS_CORPUS &phi_corp) {

        for (auto i = 0u; i < corp.size(); i++) {
           FEATS_SENT phi_sent;
           add_feats_sent(*(corp.at(i)), phi_sent);  
           phi_corp.push_back(phi_sent); //check the memory!
        }
   }


   void add_feats_sent(AlignedEncodedSentence & sent_pair, FEATS_SENT &phi_sent) {

        const auto &srcsen = sent_pair.srcSentence;
        const auto &trgsen = sent_pair.trgSentence;
        const auto &align = sent_pair.alignment;
        int last_i = -1;
        bool start = true;
        //dumy PHI for <s> in the target
        FEATS_WORD dummy(this->num_feats*srcsen.size(), 0);
        phi_sent.push_back(dummy);
        //now PHI for each word in the target
        for (auto j = 1u; j < trgsen.size(); ++j) {
            FEATS_WORD feat_sent;
            add_feats_word(last_i, start, srcsen, feat_sent);
            phi_sent.push_back(feat_sent);
            //--
            if (align[j] > 0) {
               start = false;
               last_i = align[j];
            }
        }
    }


private:
   unsigned num_feats = 8;


   void add_feats_word(int last_i, bool start, const EncodedSentence & srcsen , FEATS_WORD & feats_word) {
       //features for aligning to position i are added, then for position i+1 etc 
       for (auto i = 0u; i < srcsen.size(); i++ ) {
           std::vector<float> temp(this->num_feats, 0); // 8 features with value 0
           //action features
           temp[next_action(last_i, i, start)] = 1;
           // length features
           //temp[6] = log(1+fabs(i-last_i));
           temp[6] = (int)i-(int)last_i;
           temp[7] = temp[6]/srcsen.size();
           // copying the features for this position from the temp to the result variable 
           copy(temp.begin(), temp.end(), back_inserter(feats_word));
       }
   }

   ACTIONS next_action(int last_i, int i, bool start) {
      if (i == 0) return NULLA;
      if (start) return START;
      int delta = i-last_i;
      if (delta == 0) return STAY;
      if (delta < 0) return BACKWARD;
      if (delta == 1) return MONOTONE;
      if (delta > 0) return FORWARD;
      assert(false && "can't reach this line");
      return NULLA;
   }

};

#if 0
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
#endif


