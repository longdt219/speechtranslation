#pragma once

#include <map>
#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dglstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"

using namespace cnn;
using namespace std;

typedef vector<cnn::real> FVector;
typedef vector<FVector>   FMatrix;
typedef vector<FMatrix>   FCorpus;
typedef vector<FCorpus*>  FCorpusPointers;

typedef vector<int> Sentence;
typedef pair<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> Dialogue;
typedef vector<Dialogue> Corpus;

/// for parallel processing of data
typedef pair<Sentence, Sentence> SentencePair;
typedef vector<SentencePair> PTurn;  /// a turn consits of sentences pairs from difference utterances
typedef vector<PTurn> PDialogue;  /// a dialogue consists of many turns
typedef vector<PDialogue> PCorpus; /// a parallel corpus consists of many parallel dialogues

/// save the number of turns to dialogue id list
typedef map<int, vector<int>> NumTurn2DialogId;

typedef pair<vector<Sentence>, Sentence> StatementsQuery;
typedef pair<StatementsQuery, Sentence> FBTurns;
typedef vector<FBTurns> FBDialogue;
typedef vector<FBDialogue> FBCorpus;


/**
usually packs a matrix with real value element
this truncates both source and target 
@mbsize : number of samples
@nutt : number of sentences to process in parallel
*/
vector<vector<Expression>> pack_obs(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg, const vector<size_t>& rand_stt);

/// utterance first ordering of data
/// [s00 s01 s02 s10 s11 s12] where s1 is the second speaker, and s0 is the firest speaker
vector<vector<Expression>> pack_obs_uttfirst(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg, const vector<size_t>& rand_stt);

/// return the index of the selected dialogues
vector<int> get_same_length_dialogues(Corpus corp, size_t nbr_dialogues, size_t &min_nbr_turns, vector<bool>& used, PDialogue& selected, NumTurn2DialogId& info);


Corpus read_corpus(const string &filename, unsigned& min_diag_id, WDict& sd, int kSRC_SOS, int kSRC_EOS, int maxSentLength = 10000, bool appendBSandES = false);
int MultiTurnsReadSentencePair(const std::wstring& line, std::vector<int>* s, WDict* sd, std::vector<int>* t, WDict* td, bool appendSBandSE = false, int kSRC_SOS = -1, int kSRC_EOS = -1);
Corpus read_corpus(const string &filename, unsigned& min_diag_id, Dict& sd, int kSRC_SOS, int kSRC_EOS, int maxSentLength = 10000, bool appendBSandES = false);
int MultiTurnsReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, bool appendSBandSE = false, int kSRC_SOS = -1, int kSRC_EOS = -1);

NumTurn2DialogId get_numturn2dialid(Corpus corp);

/// shuffle the data from 
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
/// to 
/// [v_spk1_time0 v_spk1_tim1 | v_spk2_time0 v_spk2_time1]
Expression shuffle_data(Expression src, size_t nutt, size_t feat_dim, size_t slen);

std::vector<Expression> shuffle_data(Expression src, size_t nutt, size_t feat_dim, const vector<std::size_t>& slen);

void convertHumanQuery(const std::string& line, std::vector<int>& t, Dict& td);

void convertHumanQuery(const std::wstring& line, std::vector<int>& t, WDict& td);

std::wstring utf8_to_wstring(const std::string& str);

std::string wstring_to_utf8(const std::wstring& str);


/// utiles to read facebook data
int read_one_line_facebook_qa(const std::string& line, std::vector<int>& v, Dict& sd);
FBCorpus read_facebook_qa_corpus(const string &filename, size_t & diag_id, Dict& sd);
