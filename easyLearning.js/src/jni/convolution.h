#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include <vector>
#include <string>
#include "volume.h"

namespace ecj {

class ConvolutionLayer {
public:
  ConvolutionLayer(std::string& actType, unsigned int filterSize, unsigned int filterNumber,
                   unsigned int inputSx, unsigned int inputSy, unsigned int inputDepth,
                   unsigned int padding, unsigned int stride);
  void init();
  Volume forward(Volume& in, bool isTrain);
  Volume backward(Volume& dout);


public:
  std::vector<Volume> _w;
  std::vector<Volume> _dw;

private:
  unsigned int _fs;
  unsigned int _number;
  unsigned int _inputSx;
  unsigned int _inputSy;
  unsigned int _inputDepth;
  unsigned int _padding;
  unsigned int _stride;

  unsigned int _outputSx;
  unsigned int _outputSy;

  Volume _out;
  Volume _dz;

  Volume _in;
  Volume _din;

  Activation _act;
};

}
#endif
