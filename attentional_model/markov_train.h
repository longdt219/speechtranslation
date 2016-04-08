#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

// minibatch is defined by indeces in order[lb] to order[rb-1]
// for all i in [lb, rb): D[i] = corpus[order[i]], algn[i] is sampled alignments, weight[i] are weights
struct MiniBatch {

   MiniBatch(AlignedEncodedCorpus &c, std::vector<unsigned> &o, unsigned l, unsigned r) {
      corpus = c; 
      lb = l; 
      rb = r; 
      order = o;
   }

   AlignedEncodedCorpus corpus;
   std::vector<unsigned> order;
   unsigned lb, rb;

   std::vector<std::vector<Alignment> > algn;
   std::vector<std::vector<double> > weight;
};

#include "markov_gen_alignment.hh"

template <class TM_t>
class MarkovTrain {

public:
    MarkovTrain(AlignedEncodedCorpus *training, AlignedEncodedCorpus *devel, 
            Model *model, TM_t *mm, FeatureFactory *feat_factory) 
    {
       this->p_training = training;
       this->p_devel = devel;
       this->p_mm = mm;
       this->p_model = model;
       this->p_feat_factory = feat_factory;
    }

   //generates the approximate K-best alignments
   // options are: BEAM_SEARCH, IBM_MODEL1, PARTICLE_FILTER, NAIVE
    void unsup_train(unsigned WRITE_EVERY_I, unsigned MAX_EPOCH, unsigned batch_size, 
               unsigned K, ALGN_SAMPLING sampling_method, unsigned max_mb_iter, Trainer &sgd, 
               const string &fname)
    {
       unsigned si = 0;
       double best = 9e+99;

       vector<unsigned> order(p_training->size());
       for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
     
      FEATS_CORPUS d_phi;
      p_feat_factory->build_feats(*p_devel, d_phi);
      
      for (auto t = 0; t < MAX_EPOCH; t++ ) {
         for (unsigned lb = 0; lb < p_training->size(); lb += batch_size) {
            unsigned rb = lb+batch_size;
            //cout << "im here 1" << endl;
            if (rb > p_training->size()) rb = p_training->size();
            MiniBatch mb_train(*p_training, order, lb, rb);
            //cout << "im here 2" << endl;
            GenerateAlignment<TM_t> align_generator(p_mm, p_feat_factory);
            //generates the approximate k-best alignments
            // options are: BEAM_SEARCH, IBM_MODEL1, PARTICLE_FILTER, NAIVE
            //cout << "im here 3" << endl;
            align_generator.create_samples(mb_train, K, sampling_method); 
            //cout << "im her 4e" << endl;
            std::clock_t start2 = std::clock();
            unsup_train_mb(mb_train,  max_mb_iter, sgd);
            //cout << "im here 5" << endl;
            si += rb - lb;
            if (si >= WRITE_EVERY_I)  {
                cout << ">>  MiniBatch   [completed in "
                     << ( std::clock() - start2 ) / (double) CLOCKS_PER_SEC << "]\n";
                double temp_best = dev_scores(*p_devel, d_phi);
                if (temp_best < best) {
                    cout << "# saving the model for " << temp_best  << flush;
                    best = temp_best;
                    ofstream out(fname);
                    boost::archive::text_oarchive oa(out);
                    oa << *p_model;
                    cout << "   ... now  done!" << endl;
                }
                si = 0;
            }
       
         }
         // at the end shuffle
         shuffle(order.begin(), order.end(), *rndeng);
      }
    }

    void unsup_train_mb(MiniBatch & mb_training,  unsigned max_iter, Trainer &sgd)
    {
       // forming the features PHI
       cout << "joon1" << endl;
       std::vector<FEATS_CORPUS> mb_phi;
       cout << "joon2" << endl;
       build_mb_feats(mb_training,  mb_phi); 
       std::cout << "now in minibatch ...\n";
       for (auto ep = 0; ep < max_iter; ep++) {
          double loss = 0, lossJ=0;
          for (auto i = 0; i < mb_training.rb-mb_training.lb; i++) {
             unsigned si = mb_training.order[i+mb_training.lb];
             auto sent_pair = mb_training.corpus.at(si);
             Alignment algn_back = sent_pair->alignment;
             for (auto k = 0; k < mb_training.algn[i].size(); k++) {
                sent_pair->alignment = mb_training.algn[i][k];
                ComputationGraph cg;
                auto err = p_mm->BuildGraph(mb_training.corpus, si, cg, mb_phi[i][k]);
                double instance_weight = mb_training.weight[i][k]/(mb_training.rb - mb_training.lb);
                //cout << "weights " << instance_weight << " p " << mb_training.weight[i][k] 
                //     << " rb-lb " << mb_training.rb - mb_training.lb << endl;
                auto total_err = (err.first + err.second) * instance_weight; 
                cg.forward();
                loss += as_scalar(cg.get_value(total_err.i));
                lossJ += as_scalar(cg.get_value(err.second.i)); //FIXME
                cg.backward();                  
             }
             sent_pair->alignment = algn_back;
          }
          sgd.update(); 
          std::cout << " minibatch err in iter " << ep << " is -->  word:" << loss << " J:" << lossJ << "\n";
      }
      std::cout << "now leaving the minibatch .. \n\n";
    }

    void sup_train(unsigned report, unsigned WRITE_EVERY_I, unsigned MAX_EPOCH, Trainer &sgd,  
            bool save_model, const string &fname)
    {
       unsigned mib = 0, ep = 0, chars = 0, tchars = 0, si = 0;
       double loss = 0, lossJ = 0, tloss = 0, tlossJ = 0, best = 9e+99;
       std::clock_t start = std::clock(), start2 = std::clock();

       vector<unsigned> order(p_training->size());
       for (unsigned i = 0; i < order.size(); ++i) order[i] = i;

       // forming the features PHI
       FEATS_CORPUS t_phi, d_phi;
       p_feat_factory->build_feats(*p_training, t_phi);
       //feat_factory.print(t_phi);
       //feat_factory.print_seq(t_phi, training);
       p_feat_factory->build_feats(*p_devel, d_phi);
       //feat_factory.print(d_phi);
       //feat_factory.print_seq(d_phi, devel);

       cout << "report DEV every " << WRITE_EVERY_I << " sentences" << endl;
       cout << "report TRAIN every " << report << " sentences" << endl;
       while(1) {
           if (si == p_training->size()) {
              cout << "\nEpoch " << ep << ")  [completed in "
                     << ( std::clock() - start ) / (double) CLOCKS_PER_SEC << "]\n";
              cout <<  "   TRN:   wrd_nll = " << (loss / chars) << "   wrd_ppl = " << exp(loss / chars)
                     <<  "   aln_nll = " << (lossJ / chars) << "   aln_ppl = " << exp(lossJ / chars) << endl;
              dev_scores(*p_devel, d_phi);
              cout << endl;
               //performance  on train
               //corpus_scores<TM_t>(mm, training, t_phi, "TRN");
               //performance  on dev
               //corpus_scores<TM_t>(mm, devel, d_phi, "DEV:");
               //cout << endl;
               ep++;
               if (ep == MAX_EPOCH) break;
               start = std::clock();
               si = 0; loss = 0; chars = 0; lossJ = 0;
               shuffle(order.begin(), order.end(), *rndeng);
           }

           {
               // build graph for this instance
               ComputationGraph cg;
               auto spair = p_training->at(order[si]);
               auto phi = t_phi[order[si]];
               chars += spair->trgSentence.size() - 1; 
	       tchars += spair->trgSentence.size() - 1;
               auto err = p_mm->BuildGraph(*p_training, order[si], cg, phi);
               auto total_err = err.first + err.second;
               cg.forward();
               loss += as_scalar(cg.get_value(err.first.i)); 
	       tloss += as_scalar(cg.get_value(err.first.i));
               lossJ += as_scalar(cg.get_value(err.second.i)); 
	       tlossJ += as_scalar(cg.get_value(err.second.i));
               cg.backward();
               sgd.update();

	       si++;
              //sgd.status();
           }
           if ((si % WRITE_EVERY_I) == 0) {
              double temp_best = dev_scores(*p_devel, d_phi);
              if ((temp_best < best) && save_model) {
                    cout << "# saving the model for " << temp_best  << flush;
                    best = temp_best;
                    ofstream out(fname);
                    boost::archive::text_oarchive oa(out);
                    oa << *p_model;
                    cout << "   ... now  done!" << endl;
              }
              //if (ep == MAX_EPOCH) break;
          }
          if ((si % report) == 0) {
              cout << ">>  MiniBatch " << mib++ << ")  [completed in "
                     << ( std::clock() - start2 ) / (double) CLOCKS_PER_SEC << "]\n";
              cout <<  "   TRN:   wrd_nll = " << (tloss / tchars) << "   wrd_ppl = " << exp(tloss / tchars)
                     <<  "   aln_nll = " << (tlossJ / tchars) << "   aln_ppl = " << exp(tlossJ / tchars)
                     << endl << flush;
              tchars = 0; tloss = tlossJ = 0;
              start2 = std::clock();       
          }
       }
   }

private:

   AlignedEncodedCorpus *p_training;
   AlignedEncodedCorpus *p_devel;
   TM_t *p_mm;
   FeatureFactory *p_feat_factory;
   Model *p_model;

    double dev_scores(AlignedEncodedCorpus &devel, FEATS_CORPUS &phi) {
        // show score on dev data
            double dloss = 0, dlossJ = 0;

            int dchars = 0;
            for (auto j = 0; j < devel.size(); j++) {
                 auto spair = devel.at(j);
                ComputationGraph cg;
                auto err = p_mm->BuildGraph(devel, j, cg, phi[j]);
                auto total_err = err.first + err.second;
                cg.forward();
                dloss += as_scalar(cg.get_value(err.first.i));
                dlossJ += as_scalar(cg.get_value(err.second.i));
                dchars += spair->trgSentence.size() - 1;
            }
            cout  <<  "   DEV:   wrd_nll = " << (dloss / dchars) << "   wrd_ppl = " << exp(dloss / dchars)
                  <<  "   aln_nll = " << (dlossJ / dchars) << "   aln_ppl = " << exp(dlossJ / dchars)
                  << endl << flush;
            return dloss + dlossJ;
    }

    void build_mb_feats(MiniBatch&  mb_training, std::vector<FEATS_CORPUS> &mb_phi) {
       //cout << "now in bf  lb=" << mb_training.lb << "  rb=" << mb_training.rb << endl; 
       for (auto i = 0; i < mb_training.rb-mb_training.lb; i++) {
          unsigned si = mb_training.order[i+mb_training.lb];
          auto sent_pair = mb_training.corpus.at(si);
          //cout << " iteri:" << i << endl;
          Alignment algn_back = sent_pair->alignment;
          //cout << "pooni 1" << endl;
          FEATS_CORPUS fc;
          //cout << "poon 2" << endl;
          for (auto k = 0; k < mb_training.algn[i].size(); k++) {
             sent_pair->alignment = mb_training.algn[i][k];
             //cout << " iterk:" << k << endl;
             FEATS_SENT fs;
             //cout << "poon 3" << endl;
             p_feat_factory->add_feats_sent(*sent_pair, fs);
             //cout << "poon 4" << endl;
             fc.push_back(fs);
             //cout << "poon 5" << endl;
          }
          mb_phi.push_back(fc);
          //cout << "poon 6" << endl;
          sent_pair->alignment = algn_back;
          //cout << "poon 7" << endl;
       }
       //cout << "now leaving bf" << endl;
    }

};
