#ifndef _ACTIVE_H_
#define _ACTIVE_H_

#include <string>
#include <math.h>
#include "volume.h"

// void  (*destruct)(struct AVPacket *);

namespace ecj {

class Activation {
public:
  Activation() {
    forward = NULL;
    backward = NULL;
  };

public:
  double (*forward)(double x);
  double (*backward)(double x, double y);

  static Activation get(std::string& actType);
};


}
#endif
