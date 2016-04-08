#ifndef CNN_DECODE_H_
#define CNN_DECODE_H_

#include <cassert>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

#include <boost/version.hpp>
#if BOOST_VERSION >= 105600
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/string.hpp>
#endif

namespace cnn {

struct Hypothesis {
    Hypothesis(RNNPointer state, int tgt, float cst, int _t)
        : builder_state(state), target({tgt}), cost(cst), t(_t) {}
    Hypothesis(RNNPointer state, int tgt, float cst, Hypothesis &last)
        : builder_state(state), target(last.target), cost(cst), t(last.t+1) {
        target.push_back(tgt);
    }
    RNNPointer builder_state;
    std::vector<int> target;
    float cost;
    int t;
};

struct CompareHypothesis
{
    bool operator()(const Hypothesis& h1, const Hypothesis& h2)
    {
        if (h1.cost < h2.cost) return true;
        return false; 
    }
};

} // namespace cnn

#endif
