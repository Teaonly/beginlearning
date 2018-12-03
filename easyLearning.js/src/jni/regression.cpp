#include "regression.h"
#include <math.h>

namespace ecj {

L1Regression::L1Regression(unsigned int inputSize) {
  _inputSize = inputSize;

  Volume* v;

  v = new Volume(1,1,inputSize);
  _out = *v;
  delete v;

  v = new Volume(1,1,inputSize);
  _din = *v;
  delete v;
}

Volume L1Regression::forward(Volume& in) {
  for(unsigned int i = 0; i < _inputSize; i++) {
    _out[i] = in[i];
  }
  return _out;
}

double L1Regression::loss(std::vector<double>& y) {
  double sum = 0.0;
  for(unsigned int i = 0; i < _inputSize; i++) {
    sum += abs(_out[i] - y[i]);
  }
  return sum;
}

Volume L1Regression::backward(std::vector<double>& y) {
  for(unsigned int i = 0; i < _inputSize; i++) {
    if ( (_out[i] - y[i]) > 0) {
      _din[i] = 1;
    } else {
      _din[i] = -1;
    }
  }
  return _din;
}


L2Regression::L2Regression(unsigned int inputSize) {
  _inputSize = inputSize;

  Volume* v;

  v = new Volume(1,1,inputSize);
  _out = *v;
  delete v;

  v = new Volume(1,1,inputSize);
  _din = *v;
  delete v;
}

Volume L2Regression::forward(Volume& in) {
  for(unsigned int i = 0; i < _inputSize; i++) {
    _out[i] = in[i];
  }
  return _out;
}

double L2Regression::loss(std::vector<double>& y) {
  double sum = 0.0;
  for(unsigned int i = 0; i < _inputSize; i++) {
    sum += (_out[i] - y[i]) * (_out[i] - y[i]);
  }
  return sum*0.5;
}

Volume L2Regression::backward(std::vector<double>& y) {
  for(unsigned int i = 0; i < _inputSize; i++) {
    _din[i] = _out[i] - y[i];
  }
  return _din;
}


}
