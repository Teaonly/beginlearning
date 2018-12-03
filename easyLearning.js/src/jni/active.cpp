#include <assert.h>

#include "active.h"

namespace ecj {

static double SigmoidForward(double x) {
  return 1.0 / (1.0 + exp(-x));
}

static double SigmoidBackward(double x, double y) {
  return y * ( 1.0 - y);
}

static double TanhForward(double x) {
  double y = exp(2.0 * x);
  return (y - 1.0) / (y + 1.0);
}

static double TanhBackward(double x, double y) {
  return 1.0 - y*y;
}

static double ReluForward(double x) {
  if ( x > 0.0) {
    return x;
  } else {
    return 0.0;
  }
}

static double ReluBackward(double x, double y) {
  if ( y > 0.0) {
    return 1.0;
  } else {
    return 0.0;
  }
}

static double BypassForward(double x) {
  return x;
}

static double BypassBackward(double x, double y) {
  return 1;
}

Activation Activation::get(std::string& actType) {
  Activation ret;
  if ( actType == "sigmoid") {
    ret.forward = SigmoidForward;
    ret.backward = SigmoidBackward;
  } else if ( actType == "tanh") {
    ret.forward = TanhForward;
    ret.backward = TanhBackward;
  } else if ( actType == "relu") {
    ret.forward = ReluForward;
    ret.backward = ReluBackward;
  } else if ( actType == "bypass") {
    ret.forward = BypassForward;
    ret.backward = BypassBackward;
  } else {
    assert(false);
  }

  return ret;
}

}
