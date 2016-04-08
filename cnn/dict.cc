#include "dict.h"

#include <string>
#include <vector>
#include <sstream>

using namespace std;

namespace cnn {

std::vector<int> ReadSentence(const std::string& line, Dict* sd) {
  std::istringstream in(line);
  std::string word;
  std::vector<int> res;
  while(in) {
    in >> word;
    if (!in || word.empty()) break;
    res.push_back(sd->Convert(word));
  }
  return res;
}

void ReadSentencePair(const std::string& line, std::vector<int>* s, Dict* sd, std::vector<int>* t, Dict* td) {
  std::istringstream in(line);
  std::string word;
  std::string sep = "|||";
  Dict* d = sd;
  std::vector<int>* v = s;
  while(in) {
    in >> word;
    if (!in) break;
    if (word == sep) { d = td; v = t; continue; }
    v->push_back(d->Convert(word));
  }
}

void ReadMultipleSentencePair(const std::string& line, std::vector<int>* s1, std::vector<int>* s2, std::vector<int>* s3,
		                                              Dict* sd, std::vector<int>* t, Dict* td) {
  std::istringstream in(line);
  std::string word;
  std::string sep = "|||";
  Dict* d = sd;
  int count = 0;
  std::vector<int>* v = s1;
  while(in) {
    in >> word;
    if (!in) break;
    if (word == sep) {
    	count +=1;
    	if (count == 1) v = s2;
    	if (count == 2) v = s3;
    	if (count == 3){
    		d = td; v = t;
    	}
    	continue;
    }
    v->push_back(d->Convert(word));
  }
}


} // namespace cnn

