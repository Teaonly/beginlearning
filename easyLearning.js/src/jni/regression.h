#ifndef _REGRESSION_H_
#define _REGRESSION_H_

#include <vector>
#include <string>
#include "volume.h"

namespace ecj {

class L1Regression {
public:
  L1Regression(unsigned int inputSize);
  Volume forward(Volume& in);
  double loss(std::vector<double>& y);
  Volume backward(std::vector<double>& y);


public:
  unsigned int _inputSize;

private:
  Volume _out;
  Volume _din;

};


class L2Regression {
public:
  L2Regression(unsigned int inputSize);
  Volume forward(Volume& in);
  double loss(std::vector<double>& y);
  Volume backward(std::vector<double>& y);


public:
  unsigned int _inputSize;

private:
  Volume _out;
  Volume _din;

};


}

#endif
