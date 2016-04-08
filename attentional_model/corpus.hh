#ifndef _CORPUS_HH
#define _CORPUS_HH

#include "encoder.hh"

#include <vector>
#include <string>
#include <tuple>
#include <boost/shared_ptr.hpp>

////////////////////////////////////////////////////////////////
// Corpus
////////////////////////////////////////////////////////////////
typedef std::vector<std::string> Sentence;
typedef boost::shared_ptr<Sentence> SentencePtr;

class Corpus
{
public:
    typedef std::vector<SentencePtr>::const_iterator const_iterator;

public:
    Corpus();
    virtual ~Corpus() {}

    virtual unsigned read(const std::string &filename);

    SentencePtr next();

    const_iterator begin() const { return m_sentences.begin(); }
    const_iterator end() const { return m_sentences.end(); }

    size_t num_sentences() const { return m_sentences.size(); }

    SentencePtr at(size_t i) {
      return m_sentences.at(i);
    }

protected:
    std::vector<SentencePtr> m_sentences; 
};
typedef boost::shared_ptr<Corpus> CorpusPtr;

////////////////////////////////////////////////////////////////
// EncodedCorpus
////////////////////////////////////////////////////////////////
typedef std::vector<unsigned>               EncodedSentence;
typedef boost::shared_ptr<EncodedSentence>  EncodedSentencePtr;

class EncodedCorpus
{
public:
    typedef std::vector<EncodedSentencePtr>::const_iterator const_iterator;

public:
    EncodedCorpus(EncoderPtr e);
    virtual ~EncodedCorpus() {}

    virtual unsigned read(const std::string &filename);

    const_iterator begin() const { return m_sentences.begin(); }
    const_iterator end() const { return m_sentences.end(); }

    SentencePtr decode(const EncodedSentencePtr e) const;

    EncodedSentencePtr at(size_t i) {
      return m_sentences.at(i);
    }

protected:
    EncoderPtr                         m_encoder;
    std::vector<EncodedSentencePtr>    m_sentences; 
};
typedef boost::shared_ptr<EncodedCorpus> EncodedCorpusPtr;
 
///////////////////////////////////////////////////////////////
// AlignedEncodedCorpus
///////////////////////////////////////////////////////////////
typedef std::vector<int> Alignment;
struct AlignedEncodedSentence
{
    EncodedSentence srcSentence;
    EncodedSentence trgSentence;
    Alignment alignment;
    unsigned document_id;
};

typedef boost::shared_ptr<AlignedEncodedSentence> AlignedEncodedSentencePtr;

enum REVERSE_TYPE {Normal, Reverse, Intersect_Reverse};

class AlignedEncodedCorpus
{
public:
    typedef std::vector<AlignedEncodedSentencePtr>::const_iterator const_iterator;
    
public:
    AlignedEncodedCorpus();
    AlignedEncodedCorpus(EncoderPtr f, EncoderPtr e);
    virtual ~AlignedEncodedCorpus() {}
    
    virtual unsigned read_single_file(const std::string &filename);
    virtual unsigned read_component_files(const std::string &filename);
    virtual unsigned read_document_file(const std::string &filename);
    
    const_iterator begin() const { return m_sentences.begin(); }
    const_iterator end() const { return m_sentences.end(); }
    
    void threshold_source_vocabulary(int threshold, bool abs);
    void threshold_target_vocabulary(int threshold, bool abs);
    
    AlignedEncodedSentencePtr at(size_t i) const {
        return m_sentences.at(i);
    }

    void SetReverse(REVERSE_TYPE type) {reverse = type;}
    
    size_t size() const {return m_sentences.size();}

    EncoderPtr src_vocab() { return m_srcEncoder; }
    EncoderPtr trg_vocab() { return m_trgEncoder; }

protected:
    REVERSE_TYPE reverse;
    EncoderPtr                         m_srcEncoder;
    EncoderPtr                         m_trgEncoder;
    std::vector<AlignedEncodedSentencePtr>   m_sentences;
};
typedef boost::shared_ptr<AlignedEncodedCorpus> AlignedEncodedCorpusPtr;
#endif // _CORPUS_HH
