#ifndef _ENCODER_HH
#define _ENCODER_HH

#include <map>
#include <set>
#include <string>
#include <boost/shared_ptr.hpp>

class Encoder
{
public:
    Encoder() :
        m_encoding(), m_rev_encoding(),
        UNKNOWN(0),
        LEFT_BOUNDARY(1),
        RIGHT_BOUNDARY(2)
    {
        m_encoding["<unk>"] = UNKNOWN;
        m_encoding["<s>"] = LEFT_BOUNDARY;
        m_encoding["</s>"] = RIGHT_BOUNDARY;
        m_rev_encoding[UNKNOWN] = "<unk>";
        m_rev_encoding[LEFT_BOUNDARY] = "<s>";
        m_rev_encoding[RIGHT_BOUNDARY] = "</s>";
	m_freeze = false;
    }

    Encoder(const Encoder &other, const std::set<unsigned> &whitelist) :
        m_encoding(), m_rev_encoding(),
        UNKNOWN(0),
        LEFT_BOUNDARY(1),
        RIGHT_BOUNDARY(2)
    {
        m_encoding["<unk>"] = UNKNOWN;
        m_encoding["<s>"] = LEFT_BOUNDARY;
        m_encoding["</s>"] = RIGHT_BOUNDARY;
        m_rev_encoding[UNKNOWN] = "<unk>";
        m_rev_encoding[LEFT_BOUNDARY] = "<s>";
        m_rev_encoding[RIGHT_BOUNDARY] = "</s>";
	m_freeze = false;

        for (auto word: whitelist) 
            encode(other.decode(word));
    }

    // En/Decoding functions
    std::string decode(unsigned i) const
    {
        std::map<unsigned, std::string>::const_iterator result 
            = m_rev_encoding.find(i);
        if (result == m_rev_encoding.end()) 
            return decode(UNKNOWN);
        return result->second;
    }

    unsigned encode(const std::string &s, bool add=true)
    {
        if (add && !m_freeze)
        {
            std::pair< std::map<std::string, unsigned>::const_iterator, bool > result 
                = m_encoding.insert(std::make_pair(s, m_encoding.size()));
            if (result.second)
                m_rev_encoding[m_encoding.size() - 1] = s;
            return result.first->second;
        }
        else
        {
            std::map<std::string, unsigned>::const_iterator result 
                = m_encoding.find(s);
            if (result == m_encoding.end()) 
                return UNKNOWN;
            return result->second;
        }
    }

    // useful operators
    unsigned operator()(const std::string &s)
    { return encode(s); }

    std::string operator()(unsigned i) const
    { return decode(i); }

    unsigned size() const
    { return m_encoding.size(); }

    void freeze()
    { m_freeze = true; }

protected:
    std::map<std::string, unsigned> m_encoding;
    std::map<unsigned, std::string> m_rev_encoding;
    bool m_freeze;

public:
    const unsigned UNKNOWN;
    const unsigned LEFT_BOUNDARY;
    const unsigned RIGHT_BOUNDARY;
};
typedef boost::shared_ptr<Encoder> EncoderPtr;

#endif // _ENCODER_HH
