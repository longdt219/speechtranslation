#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "cnn/data-util.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>
#include <boost/system/config.hpp>
#include <boost/locale.hpp>
#include <boost/locale/encoding_utf.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/regex.hpp>

using namespace cnn;
using namespace std;
using namespace boost::algorithm;
using boost::locale::conv::utf_to_utf;
using namespace boost::locale;

/// utterance first ordering of data
/// [s00 s01 s02 s10 s11 s12] where s1 is the second speaker, and s0 is the firest speaker
vector<vector<Expression>> pack_obs(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg, const vector<size_t>& randstt)
{
    int nCorpus = raw.size();
    long nutt = raw[0]->size();

    vector<vector<Expression>> ret;
    random_device rd;

    vector<vector<cnn::real>> tgt;
    assert(randstt.size() == nutt);

    int feat_dim = (*raw[0])[0][0].size();
    for (auto cc : raw)
    {
        vector<vector<cnn::real>> obs;
        for (size_t k = 0; k < mbsize; k++)
        {
            obs.push_back(vector<cnn::real>(feat_dim * nutt, 0.0));
            for (size_t u = 0; u < nutt; u++)
            {
                size_t nsamples = (*cc)[u].size();
                if (k + randstt[u] >= nsamples)
                    break;

                size_t stt = u * feat_dim;

                /// random starting position randstt
                vector<cnn::real>::iterator pobs = obs[k].begin();
                vector<cnn::real>::iterator pfrm = (*cc)[u][k + randstt[u]].begin();
                copy(pfrm, pfrm + feat_dim, pobs + stt);
            }
        }

        vector<Expression> vx(mbsize);
        for (unsigned i = 0; i < mbsize; ++i)
        {
            vx[i] = input(cg, { (long)feat_dim, nutt }, &obs[i]);
            cg.incremental_forward();
        }
        ret.push_back(vx); 
    }

    return ret;
}

/// utterance first ordering of data
/// [s00 s01 s02 s10 s11 s12] where s1 is the second speaker, and s0 is the firest speaker
vector<vector<Expression>> pack_obs_uttfirst(FCorpusPointers raw, size_t mbsize, ComputationGraph& cg, const vector<size_t>& randstt)
{
    int nCorpus = raw.size();
    long nutt = raw[0]->size();

    vector<vector<Expression>> ret;
    random_device rd;

    assert(randstt.size() == nutt);
    vector<vector<cnn::real>> tgt;
    int feat_dim = (*raw[0])[0][0].size();

    for (auto cc : raw)
    {
        vector<vector<cnn::real>> obs;
        for (size_t u = 0; u < nutt; u++)
        {
            obs.push_back(vector<cnn::real>(feat_dim * mbsize, 0.0));
            vector<cnn::real>::iterator pobs = obs[u].begin();

            for (size_t k = 0; k < mbsize; k++)
            {
                size_t nsamples = (*cc)[u].size();
                if (k + randstt[u] >= nsamples)
                    break;

                size_t stt = k * feat_dim;

                /// random starting position randstt
                vector<cnn::real>::iterator pfrm = (*cc)[u][k + randstt[u]].begin();
                copy(pfrm, pfrm + feat_dim, pobs + stt);
            }
        }

        vector<Expression> vx(nutt);
        for (unsigned i = 0; i < nutt; ++i)
        {
            vx[i] = input(cg, { (long)feat_dim, (long)mbsize}, &obs[i]);
            cg.incremental_forward();
        }
        ret.push_back(vx);
    }

    return ret;
}

/** 
extract from a dialogue corpus, a set of dialogues with the same number of turns
@corp : dialogue corpus
@nbr_dialogues : expected number of dialogues to extract
@stt_dialogue_id : starting dialogue id

return a vector of dialogues in selected, also the starting dialogue id is increased by one.
Notice that a dialogue might be used in multiple times

selected [ turn 0 : <query_00, answer_00> <query_10, answer_10>]
         [ turn 1 : <query_01, answer_01> <query_11, answer_11>]
*/
vector<int> get_same_length_dialogues(Corpus corp, size_t nbr_dialogues, size_t &min_nbr_turns, vector<bool>& used, PDialogue& selected, NumTurn2DialogId& info)
{
    /// ciruculum style training, start from short conversations
    /// start from short conversation with as few as one dialogue turn
    int nutt = 0;
    vector<int> v_sel_idx;
    int nbr_turn = -1;
    bool need_shuffle = false; 

    for (auto p : info)
    {
        if (p.first < min_nbr_turns) continue;
        for (auto k: p.second)
        {
            if (used[k] == false)
            {
                nbr_turn = p.first;
                if (nbr_turn != min_nbr_turns)
                {
                    need_shuffle = true;
                }
                break;
            }
        }
        if (nbr_turn != -1)
            break;
    }
    if (nbr_turn == -1)
        return v_sel_idx;

    selected.clear();
    selected.resize(nbr_turn);
    vector<int> vd = info[nbr_turn];

    if (need_shuffle)
    {
        random_shuffle(vd.begin(), vd.end());
        info[nbr_turn] = vd;
    }

    size_t nd = 0;
    for (auto k : vd)
    {
        if (used[k] == false && nd < nbr_dialogues)
        {
            size_t iturn = 0;
            for (auto p : corp[k])
            {
                selected[iturn].push_back(corp[k][iturn]);
                iturn++;
            }
            used[k] = true;
            v_sel_idx.push_back(k);
            nd++;
        }
    }

    min_nbr_turns = nbr_turn;
    return v_sel_idx; 
}

std::wstring utf8_to_wstring(const std::string& str)
{
    return utf_to_utf<wchar_t>(str.c_str(), str.c_str() + str.size());
}

std::string wstring_to_utf8(const std::wstring& str)
{
    return utf_to_utf<char>(str.c_str(), str.c_str() + str.size());
}

Corpus read_corpus(const string &filename, unsigned& min_diag_id, WDict& sd, int kSRC_SOS, int kSRC_EOS, int maxSentLength, bool appendBSandES)
{
    wifstream in(filename);
    generator gen;
    locale loc  = gen("zh-CN.UTF-8");
    // Create all locales

    in.imbue(loc); 
    wstring line;

    Corpus corpus;
    Dialogue diag;
    int prv_diagid = -1;
    int lc = 0, stoks = 0, ttoks = 0;
    min_diag_id = 99999;
    while (getline(in, line)) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            break;
        ++lc;
        Sentence source, target;
        int diagid = MultiTurnsReadSentencePair(line, &source, &sd, &target, &sd, appendBSandES, kSRC_SOS, kSRC_EOS);
        if (diagid == -1)
            break;
        if (diagid < min_diag_id)
            min_diag_id = diagid;
        if (diagid != prv_diagid)
        {
            if (diag.size() > 0)
                corpus.push_back(diag);
            diag.clear();
            prv_diagid = diagid;
        }
        if (source.size() > maxSentLength)
        {
            source.resize(maxSentLength - 1);
            source.push_back(kSRC_EOS);
        }
        if (target.size() > maxSentLength)
        {
            target.resize(maxSentLength - 1);
            target.push_back(kSRC_EOS);
        }
        diag.push_back(SentencePair(source, target));
        stoks += source.size();
        ttoks += target.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS)) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
    }

    if (diag.size() > 0)
        corpus.push_back(diag);
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << sd.size() << " types\n";
    return corpus;
}

Corpus read_corpus(const string &filename, unsigned& min_diag_id, Dict& sd, int kSRC_SOS, int kSRC_EOS, int maxSentLength, bool appendBSandES)
{
    ifstream in(filename);
    string line;

    Corpus corpus;
    Dialogue diag;
    int prv_diagid = -1;
    int lc = 0, stoks = 0, ttoks = 0;
    min_diag_id = 99999;
    while (getline(in, line)) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            break;
        ++lc;
        Sentence source, target;
        int diagid = MultiTurnsReadSentencePair(line, &source, &sd, &target, &sd, appendBSandES, kSRC_SOS, kSRC_EOS);
        if (diagid == -1)
            break;
        if (diagid < min_diag_id)
            min_diag_id = diagid;
        if (diagid != prv_diagid)
        {
            if (diag.size() > 0)
                corpus.push_back(diag);
            diag.clear();
            prv_diagid = diagid;
        }
        if (source.size() > maxSentLength)
        {
            source.resize(maxSentLength - 1);
            source.push_back(kSRC_EOS);
        }
        if (target.size() > maxSentLength)
        {
            target.resize(maxSentLength - 1);
            target.push_back(kSRC_EOS);
        }
        diag.push_back(SentencePair(source, target));
        stoks += source.size();
        ttoks += target.size();

        if ((source.front() != kSRC_SOS && source.back() != kSRC_EOS)) {
            cerr << "Sentence in " << filename << ":" << lc << " didn't start or end with <s>, </s>\n";
            abort();
        }
    }

    if (diag.size() > 0)
        corpus.push_back(diag);
    cerr << lc << " lines, " << stoks << " & " << ttoks << " tokens (s & t), " << sd.size() << " & " << sd.size() << " types\n";
    return corpus;
}

int MultiTurnsReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td, bool appendSBandSE, int kSRC_SOS, int kSRC_EOS)
{
    std::istringstream in(line);
    std::string word;
    std::string sep = "|||";
    Dict* d = sd;
    std::string diagid, turnid;

    std::vector<int>* v = s;

    if (line.length() == 0)
        return -1;

    in >> diagid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting diagid" << endl;
        abort();
    }

    in >> turnid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting turn id" << endl;
        abort();
    }

    if (appendSBandSE)
        v->push_back(kSRC_SOS);
    while (in) {
        in >> word;
        trim(word);
        if (!in) break;
        if (word == sep) {
            if (appendSBandSE)
                v->push_back(kSRC_EOS);
            d = td; v = t;
            if (appendSBandSE)
                v->push_back(kSRC_SOS);
            continue;
        }
        v->push_back(d->Convert(word));
    }
    if (appendSBandSE)
        v->push_back(kSRC_EOS);
    int res;

    stringstream(diagid) >> res;

    return res;
}

int MultiTurnsReadSentencePair(const std::wstring& line, std::vector<int>* s, WDict* sd, std::vector<int>* t, WDict* td, bool appendSBandSE, int kSRC_SOS, int kSRC_EOS)
{
    std::wistringstream in(line);
    std::wstring word;
    std::wstring sep = L"|||";
    WDict* d = sd;
    std::wstring diagid, turnid;

    std::vector<int>* v = s;

    if (line.length() == 0)
        return -1;

    in >> diagid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting diagid" << endl;
        abort();
    }

    in >> turnid;
    in >> word;
    if (word != sep)
    {
        cerr << "format should be <diagid> ||| <turnid> ||| src || tgt" << endl;
        cerr << "expecting turn id" << endl;
        abort();
    }

    if (appendSBandSE)
        v->push_back(kSRC_SOS);
    while (in) {
        in >> word;
        trim(word);
        if (!in) break;
        if (word == sep) {
            if (appendSBandSE)
                v->push_back(kSRC_EOS);
            d = td; v = t;
            if (appendSBandSE)
                v->push_back(kSRC_SOS);
            continue;
        }
        v->push_back(d->Convert(word));
    }
    if (appendSBandSE)
        v->push_back(kSRC_EOS);
    int res;

    wstringstream(diagid) >> res;
    return res;
}

NumTurn2DialogId get_numturn2dialid(Corpus corp)
{
    NumTurn2DialogId info;

    int id = 0;
    for (auto p : corp)
    {
        size_t d_turns = p.size();
        info[d_turns].push_back(id++);
    }
    return info; 
}


/// shuffle the data from 
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
/// to 
/// [v_spk1_time0 v_spk1_tim1 | v_spk2_time0 v_spk2_time1]
/// this assumes same length
Expression shuffle_data(Expression src, size_t nutt, size_t feat_dim, size_t slen)
{
    Expression i_src = reshape(src, {(long) (nutt * slen * feat_dim)});

    int stride = nutt * feat_dim;
    vector<Expression> i_all_spk;
    for (size_t k = 0; k < nutt; k++)
    {
        vector<Expression> i_each_spk;
        for (size_t t = 0; t < slen; t++)
        {
            long stt = k * feat_dim;
            long stp = (k + 1)*feat_dim;
            stt += (t * stride);
            stp += (t * stride);
            Expression i_pick = pickrange(i_src, stt, stp);
            i_each_spk.push_back(i_pick);
        }
        i_all_spk.push_back(concatenate_cols(i_each_spk));
    }
    return concatenate_cols(i_all_spk);
}

/// shuffle the data from 
/// [v_spk1_time0 v_spk2_time0 | v_spk1_time1 v_spk2_tim1 ]
/// to 
/// [v_spk1_time0 v_spk1_tim1 | v_spk2_time0 v_spk2_time1]
/// this assumes different source length
vector<Expression> shuffle_data(Expression src, size_t nutt, size_t feat_dim, const vector<size_t>& v_slen)
{
    /// the input data is arranged into a big matrix, assuming same length of utterance
    /// but they are different length
    size_t slen = *std::max_element(v_slen.begin(), v_slen.end());

    Expression i_src = reshape(src, { (long)(nutt * slen * feat_dim) });

    int stride = nutt * feat_dim;
    vector<Expression> i_all_spk;
    for (size_t k = 0; k < nutt; k++)
    {
        vector<Expression> i_each_spk;
        for (size_t t = 0; t < v_slen[k]; t++)
        {
            long stt = k * feat_dim;
            long stp = (k + 1)*feat_dim;
            stt += (t * stride);
            stp += (t * stride);
            Expression i_pick = pickrange(i_src, stt, stp);
            i_each_spk.push_back(i_pick);
        }
        i_all_spk.push_back(concatenate_cols(i_each_spk));
    }
    return i_all_spk;
}

void convertHumanQuery(const std::string& line, std::vector<int>& t, Dict& td)
{
    std::istringstream in(line);
    std::string word;
    t.clear();

    while (in) {
        in >> word;
        if (!in) break;
        t.push_back(td.Convert(word, true));
    }
}

void convertHumanQuery(const std::wstring& line, std::vector<int>& t, WDict& td)
{
    std::wistringstream in(line);
    std::wstring word;

    t.clear();

    while (in) {
        in >> word;
        if (!in) break;
        t.push_back(td.Convert(word, true));
    }
}

FBCorpus read_facebook_qa_corpus(const string &filename, size_t& diag_id, Dict& sd)
{
    ifstream in(filename);
    generator gen;

    string line;
    int turnid;

    diag_id = -1;
    FBCorpus corpus;
    FBDialogue diag;
    FBTurns turns;
    StatementsQuery sq;
    vector<Sentence> statements;

    int prv_turn = 9999, lc = 0, stoks = 0, ttoks = 0;

    while (getline(in, line)) {
        trim_left(line);
        trim_right(line);
        if (line.length() == 0)
            break;
        ++lc;
        Sentence source, query, target;
        vector<string> vstr; 
        string newline;
        string ans; 

        if (find(line.begin(), line.end(), '?') != line.end())
        {
            /// check if question
            boost::split(vstr, line, boost::is_any_of("\t"));
            ans = vstr[1];
            
            newline = vstr[0];
            std::replace(newline.begin(), newline.end(), '?', ' ');

            turnid = read_one_line_facebook_qa(newline, query, sd);
            sq = make_pair(statements, query);

            Sentence sans;
            sans.push_back(sd.Convert(ans));
            turns = make_pair(sq, sans);

            ttoks++;
            diag.push_back(turns);
            statements.clear();
        }
        else
        {
            newline = line;
            std::replace(newline.begin(), newline.end(), '.', ' '); 
            turnid = read_one_line_facebook_qa(newline, source, sd);
            statements.push_back(source);
        }

        if (turnid < prv_turn)
        {
            diag_id++;

            if (diag.size() > 0)
                corpus.push_back(diag);
            diag.clear();
        }
        prv_turn = turnid;
    }

    cerr << lc << " lines & " << diag_id << " dialogues & " << ttoks << " questions " << endl; 
    return corpus;
}

int read_one_line_facebook_qa(const std::string& line, std::vector<int>& v, Dict& sd)
{
    std::istringstream in(line);
    std::string word;
    std::string turnid;

    if (line.length() == 0)
        return -1;

    in >> turnid;

    while (in) {
        in >> word;
        trim(word);
        if (!in) break;
        v.push_back(sd.Convert(word));
    }

    return boost::lexical_cast<int, string>(turnid);
}

