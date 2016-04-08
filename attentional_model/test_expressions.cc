#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"
#include "expr-xtra.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

  const unsigned HIDDEN_SIZE = 8;
  ComputationGraph cg;

  Expression as = arange(cg, 0, HIDDEN_SIZE, false);
  Expression bs = repeat(cg, HIDDEN_SIZE, -1.2345);
  Expression das = dither(cg, as);
  Expression sdas = sum_cols(das);

  cerr << "as =\n" << *cg.get_value(as.i) << endl;
  cerr << "bs =\n" << *cg.get_value(bs.i) << endl;
  cerr << "das =\n" << *cg.get_value(das.i) << endl;
  cerr << "sdas =\n" << *cg.get_value(sdas.i) << endl;
}

