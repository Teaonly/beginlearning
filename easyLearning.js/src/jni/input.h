#ifndef _INPUT_H_
#define _INPUT_H_

#include <vector>
#include "volume.h"

namespace ecj {

class InputLayer {
public:
  InputLayer(unsigned int sx, unsigned int sy, unsigned int depth);
  void forward(std::vector<double>& input);

public:
  unsigned int _sx;
  unsigned int _sy;
  unsigned int _depth;
  Volume _data;
};



}
#endif
