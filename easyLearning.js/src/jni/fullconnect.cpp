#include "fullconnect.h"

namespace ecj {

FullConnectLayer::FullConnectLayer(std::string& actType, unsigned int number, unsigned int inputSize)
    : _number(number), _inputSize(inputSize) {

  Volume *v = NULL;

  _act = Activation::get(actType);

  for(unsigned int i = 0; i < _number; i++) {
    v = new Volume(1, 1, _inputSize, 1);
    _w.push_back(*v);
    delete v;
    v = new Volume(1, 1, _inputSize, 1);
    _dw.push_back(*v);
    delete v;
  }

  v = new Volume(1,1, _number);
  _dz = *v;
  delete v;
  v = new Volume(1, 1, _number);
  _out = *v;
  delete v;

  v = new Volume(1,1, _inputSize);
  _in = *v;
  delete v;
  v = new Volume(1, 1, _inputSize);
  _din = *v;
  delete v;

}

void FullConnectLayer::init() {
}

Volume FullConnectLayer::forward(Volume& inData, bool isTrain) {

  for(unsigned int j = 0; j < _inputSize; j++) {
    _in[j] = inData[j];
  }

  for (unsigned int i = 0; i < _number; i++) {

    double sum = _w[i][_inputSize];   // this is bais
    for(unsigned int j = 0; j < _inputSize; j++) {
      sum += _w[i][j]*inData[j];
    }
    _out[i] = _act.forward(sum);
    if(isTrain) {
      _dz[i] = _act.backward(sum, _out[i]);
    }
  }
  return _out;
}

Volume FullConnectLayer::backward(Volume& dout) {
  // 更新weight导数
  for (unsigned int i = 0; i < _number; i++) {
    _dz[i] = _dz[i] * dout[i];

    // bias
    _dw[i][_inputSize] = _dz[i];

    for(unsigned int j = 0; j < _inputSize; j++) {
      _dw[i][j] = _in[j] * _dz[i];
    }
  }

  // 更新输入导数
  for(unsigned int j = 0; j < _inputSize; j++) {
    double sum = 0;
    for (unsigned int i = 0; i < _number; i++) {
      sum += _w[i][j] * _dz[i];
    }
    _din[j] = sum;
  }

  return _din;
}


}
