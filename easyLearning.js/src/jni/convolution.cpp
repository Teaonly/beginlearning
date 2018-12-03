#include <assert.h>
#include "convolution.h"

namespace ecj {

ConvolutionLayer::ConvolutionLayer(std::string& actType, unsigned int filterSize, unsigned int filterNumber,
                                   unsigned int inputSx, unsigned int inputSy, unsigned int inputDepth,
                                   unsigned int padding, unsigned int stride)
    : _fs(filterSize), _number(filterNumber),
      _inputSx(inputSx), _inputSy(inputSy), _inputDepth(inputDepth),
      _padding(padding), _stride(stride) {

  // 检查参数是否合理
  if ( ((_inputSx + 2*_padding - _fs) % _stride) != 0 ) {
    assert(false);
  }

  _outputSx = (_inputSx + 2*_padding - _fs) / _stride + 1;
  _outputSy = (_inputSy + 2*_padding - _fs) / _stride + 1;


  Volume *v = NULL;

  _act = Activation::get(actType);

  for(unsigned int i = 0; i < _number; i++) {
   v = new Volume(_fs, _fs, _inputDepth, 1);
   _w.push_back(*v);
   delete v;
   v = new Volume(_fs, _fs, _inputDepth, 1);
   _dw.push_back(*v);
   delete v;
  }

  v = new Volume(_ouputSx, _outputSy, _number);
  _dz = *v;
  delete v;
  v = new Volume(_ouputSx, _outputSy, _number);
  _out = *v;
  delete v;

  v = new Volume(1, 1, _inputSize);
  _in = *v;
  delete v;
  v = new Volume(1, 1, _inputSize);
  _din = *v;
  delete v;

}


}
