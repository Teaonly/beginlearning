#ifndef _FULLCONNECT_H_
#define _FULLCONNECT_H_

#include <vector>
#include <string>
#include "volume.h"
#include "active.h"

namespace ecj {

class FullConnectLayer {
public:
  FullConnectLayer(std::string& actType, unsigned int number, unsigned int inputSize);
  void init();
  Volume forward(Volume& in, bool isTrain);
  Volume backward(Volume& dout);

public:
  std::vector<Volume> _w;
  std::vector<Volume> _dw;

  unsigned int _number;
  unsigned int _inputSize;

private:
  Volume _out;
  Volume _dz;

  Volume _in;
  Volume _din;

  Activation _act;
};


}
#endif
