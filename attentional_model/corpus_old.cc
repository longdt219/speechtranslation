#include "corpus.hh"

#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;
 
//////////////////////////////////////////////////
// Corpus
//////////////////////////////////////////////////

Corpus::Corpus()
{
}

unsigned 
Corpus::read(const std::string &filename)
{
    ifstream in(filename.c_str());

    string buf, token;
    unsigned count=0;
    while (getline(in, buf))
    {
        SentencePtr tokens(new Sentence());
        istringstream ss(buf);

        tokens->push_back("<S>");
        while(ss >> token) 
            tokens->push_back(token);
        tokens->push_back("</S>");

        m_sentences.push_back(tokens);

        count++;
    }

    return count;
}


//////////////////////////////////////////////////
// EncodedCorpus
//////////////////////////////////////////////////

EncodedCorpus::EncodedCorpus(EncoderPtr e)
: m_encoder(e)
{
}

unsigned 
EncodedCorpus::read(const std::string &filename)
{
    ifstream in(filename.c_str());

    string buf, token;
    unsigned count=0;
    while (getline(in, buf))
    {
        EncodedSentencePtr tokens(new EncodedSentence());
        istringstream ss(buf);

        while(ss >> token) 
            tokens->push_back(m_encoder->encode(token));

        m_sentences.push_back(tokens);

        count++;
    }

    return count;
}

///////////////////////////////////////////////////////////////
// AlignedEncodedCorpus
///////////////////////////////////////////////////////////////

AlignedEncodedCorpus::AlignedEncodedCorpus()
: m_srcEncoder(new Encoder()), m_trgEncoder(new Encoder())
{
    reverse = Normal;
}

AlignedEncodedCorpus::AlignedEncodedCorpus(EncoderPtr s, EncoderPtr t)
: m_srcEncoder(s), m_trgEncoder(t)
{
    reverse = Normal;
}

unsigned 
AlignedEncodedCorpus::read_single_file(const std::string &filename)
{
    //assert(reverse == Normal); // not sure how to handle this otherwise

    ifstream infile(filename.c_str());
    if(!infile.is_open()){
        cerr << "Failed to open file " << filename << endl;
        exit(1);
    }
    
    string buf, token;
    unsigned count=0;
    
    while (getline(infile, buf)) {
        
        AlignedEncodedSentencePtr sentencePair(new AlignedEncodedSentence());
        sentencePair->srcSentence.push_back(m_srcEncoder->LEFT_BOUNDARY);
        sentencePair->trgSentence.push_back(m_trgEncoder->LEFT_BOUNDARY);

        //cerr << "getline: " << buf << endl;
        istringstream ss(buf.c_str());        
        int state = 0;
        while (ss >> token) 
        {
	    //cerr << "\tstate: " << state << " token: " << token << endl;
            if (token == "|||") 
            {
                state += 1;
                assert(state <= 2);
                continue;
            }

            if (state == 0)
                sentencePair->srcSentence.push_back(m_srcEncoder->encode(token));
            else if (state == 1)
                sentencePair->trgSentence.push_back(m_trgEncoder->encode(token));
            else if (state == 2)
            {
                if (sentencePair->alignment.empty())
                    sentencePair->alignment.resize(sentencePair->trgSentence.size()+1, -1);

                int pos = token.find('-');
                assert(pos != -1);
                auto srcPos = atoi(token.substr(0, pos++).c_str());
                auto trgPos = atoi(token.substr(pos, (int)token.size()-pos).c_str());
                
                if (Intersect_Reverse == reverse)
                    sentencePair->alignment[srcPos] = trgPos;
                else
                    sentencePair->alignment[trgPos] = srcPos+1;

                sentencePair->alignment[sentencePair->trgSentence.size()] = sentencePair->srcSentence.size();
            }
        }
        
        sentencePair->trgSentence.push_back(m_trgEncoder->RIGHT_BOUNDARY);
        sentencePair->srcSentence.push_back(m_srcEncoder->RIGHT_BOUNDARY);
        m_sentences.push_back(sentencePair);
        count++;
    }
    
    return count;
}

unsigned 
AlignedEncodedCorpus::read_component_files(const std::string &filename)
{
    string srcFile, trgFile, alignFile;
    
    if (reverse == Normal){
        srcFile = filename + ".src";
        trgFile = filename + ".trg";
        alignFile = filename + ".align";
    }
    else {
        srcFile = filename + ".trg";
        trgFile = filename + ".src";
        alignFile = filename + ".align.reverse";
    }
    
    ifstream srcin(srcFile.c_str());
    ifstream trgin(trgFile.c_str());
    ifstream alignin(alignFile.c_str());
    if(!srcin.is_open()){
        cerr << "Failed to open file " << srcFile << endl;
        exit(1);
    }
    if(!trgin.is_open()){
        cerr << "Failed to open file " << trgFile << endl;
        exit(1);
    }
    if(!alignin.is_open()){
        cerr << "Failed to open file " << alignFile << endl;
        exit(1);
    }
    
    string buf, token;
    unsigned count=0;
    int srcPos, trgPos, pos;
    
    while (getline(srcin, buf)) {
        
       // if(count >= 20)
       //     break;
        
        AlignedEncodedSentencePtr sentencePair(new AlignedEncodedSentence());
        sentencePair->srcSentence.push_back(m_srcEncoder->LEFT_BOUNDARY);
        
        istringstream ss(buf.c_str());        
        while(ss >> token) 
            sentencePair->srcSentence.push_back(m_srcEncoder->encode(token));
        
        getline(trgin, buf);
        ss.clear();
        ss.str(buf.c_str());
        while(ss >> token) 
            sentencePair->trgSentence.push_back(m_trgEncoder->encode(token));
        
        getline(alignin, buf);
        ss.clear();
        ss.str(buf.c_str());
        sentencePair->alignment.resize(sentencePair->trgSentence.size()+1, -1);
        while(ss >> token) 
        {
            pos = token.find('-');
            assert(pos != -1);
            srcPos = atoi(token.substr(0, pos++).c_str());
            trgPos = atoi(token.substr(pos, (int)token.size()-pos).c_str());
            
            if (Intersect_Reverse == reverse) 
                sentencePair->alignment[srcPos] = trgPos;
            else
                sentencePair->alignment[trgPos] = srcPos+1;
        }
        sentencePair->alignment[sentencePair->trgSentence.size()] = sentencePair->srcSentence.size();
        
        sentencePair->trgSentence.push_back(m_trgEncoder->RIGHT_BOUNDARY);
        sentencePair->srcSentence.push_back(m_srcEncoder->RIGHT_BOUNDARY);
        m_sentences.push_back(sentencePair);
        count++;
    }
    
    return count;
}

unsigned 
AlignedEncodedCorpus::read_document_file(const std::string &filename)
{
    //assert(reverse == Normal); // not sure how to handle this otherwise

    ifstream infile(filename.c_str());
    if(!infile.is_open()){
        cerr << "Failed to open file " << filename << endl;
        exit(1);
    }
    
    string buf, token;
    unsigned count=0;
    
    while (getline(infile, buf)) {
        
        AlignedEncodedSentencePtr sentencePair(new AlignedEncodedSentence());
        sentencePair->srcSentence.push_back(m_srcEncoder->LEFT_BOUNDARY);
        sentencePair->trgSentence.push_back(m_trgEncoder->LEFT_BOUNDARY);

        istringstream ss(buf.c_str());        
        int state = 0;
        while (ss >> token) 
        {
            if (token == "|||") 
            {
                state += 1;
                assert(state <= 3);
                continue;
            }

	    if (state == 0)
                sentencePair->document_id = atoi(token.c_str());
            else if (state == 1)
                sentencePair->srcSentence.push_back(m_srcEncoder->encode(token));
            else if (state == 2)
                sentencePair->trgSentence.push_back(m_trgEncoder->encode(token));
            else if (state == 3)
            {
                if (sentencePair->alignment.empty())
                    sentencePair->alignment.resize(sentencePair->trgSentence.size()+1, -1); // +1: right boundary

                int pos = token.find('-');
                assert(pos != -1);
                auto srcPos = atoi(token.substr(0, pos++).c_str());
                auto trgPos = atoi(token.substr(pos, (int)token.size()-pos).c_str());
                
                if (Intersect_Reverse == reverse)
                    sentencePair->alignment[srcPos] = trgPos;
                else
                    sentencePair->alignment[trgPos] = srcPos+1;
            }
        }
        // aligning the </s> symbols
        if (sentencePair->alignment.empty())
            sentencePair->alignment.resize(sentencePair->trgSentence.size()+1, -1); // +1: right boundary
        sentencePair->alignment[sentencePair->trgSentence.size()] = sentencePair->srcSentence.size();
        // adding the right boundary </s> 
        sentencePair->trgSentence.push_back(m_trgEncoder->RIGHT_BOUNDARY);
        sentencePair->srcSentence.push_back(m_srcEncoder->RIGHT_BOUNDARY);
        // adding the sent to the corpus
        m_sentences.push_back(sentencePair);
        count++;
    }
    
    return count;
}

void 
AlignedEncodedCorpus::threshold_source_vocabulary(int threshold,bool abs)
{
    // count occurrences of each word type 
    map<unsigned, unsigned> wid_to_count;
    for (auto&& sentence: m_sentences)
        for (auto& enc_word: sentence->srcSentence) 
            wid_to_count[enc_word] += 1;

    // flip around to (count, word type) tuples
    multimap<unsigned, unsigned> count_to_wid;
    for (auto word_count: wid_to_count)
        count_to_wid.insert(make_pair(word_count.second, word_count.first));

    // read off <threshold> top entries and store these
    int i = 2; // start from 3 to cover UNK/s/</s>
    set<unsigned> whitelist;
    if (abs) {
        for (auto cwit = count_to_wid.rbegin();
            cwit != count_to_wid.rend() && cwit->first >=(unsigned) threshold; ++cwit)
	    whitelist.insert(cwit->second);
    } else {
	for (auto cwit = count_to_wid.rbegin(); 
		cwit != count_to_wid.rend() && i <= threshold; ++cwit, ++i)
	    whitelist.insert(cwit->second);
    }
    // construct new encoder and remap vocabulary
    EncoderPtr new_encoder(new Encoder(*m_srcEncoder, whitelist));
    for (auto&& sentence: m_sentences)
    {
        for (auto wit = sentence->srcSentence.begin(); wit != sentence->srcSentence.end(); ++wit) 
        {
            //cout << "\tbefore " << *wit << "/" << m_srcEncoder->decode(*wit) << "\n";
            *wit = new_encoder->encode(m_srcEncoder->decode(*wit), false);
            //cout << "\tnow " << *wit << "/" << new_encoder->decode(*wit) << "\n";
        }
    }

    // store new encoder and discard the old one
    m_srcEncoder.swap(new_encoder);
}

void 
AlignedEncodedCorpus::threshold_target_vocabulary(int threshold,bool abs)
{
    // count occurrences of each word type 
    map<unsigned, unsigned> wid_to_count;
    for (auto&& sentence: m_sentences)
        for (auto& enc_word: sentence->trgSentence) 
            wid_to_count[enc_word] += 1;

    // flip around to (count, word type) tuples
    multimap<unsigned, unsigned> count_to_wid;
    for (auto word_count: wid_to_count) {
        count_to_wid.insert(make_pair(word_count.second, word_count.first));
    }

    // read off <threshold> top entries and store these
    int i = 2; // start from 3 to cover UNK/s/</s>
    set<unsigned> whitelist;
    if (abs) {
	for (auto cwit = count_to_wid.rbegin();
		cwit != count_to_wid.rend() && cwit->first >=(unsigned) threshold; ++cwit)
	    whitelist.insert(cwit->second);
    } else {
	for (auto cwit = count_to_wid.rbegin(); 
		cwit != count_to_wid.rend() && i <= threshold; ++cwit, ++i)
	    whitelist.insert(cwit->second);
    }    
    
    // construct new encoder and remap vocabulary
    EncoderPtr new_encoder(new Encoder(*m_trgEncoder, whitelist));
    for (auto&& sentence: m_sentences)
        for (auto wit = sentence->trgSentence.begin(); wit != sentence->trgSentence.end(); ++wit) 
            *wit = new_encoder->encode(m_trgEncoder->decode(*wit), false);

    // store new encoder and discard the old one
    m_trgEncoder.swap(new_encoder);
}
