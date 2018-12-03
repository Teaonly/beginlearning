#include <stdlib.h>

#include "volume.h"

namespace ecj {

double* Volume::_heapMemory = NULL;
unsigned int Volume::_heapSize = 0;
unsigned int Volume::_heapIndex = 0;

Volume::Volume(unsigned int sx, unsigned int sy, unsigned int depth, unsigned int extend)
    : _sx(sx), _sy(sy), _depth(depth), _extend(extend) {

  _size = _sx * _sy * _depth + _extend;
  _offset = _heapIndex;
  _heapIndex += _size;

}


}
