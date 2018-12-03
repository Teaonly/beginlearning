#ifndef _VOLUME_H_
#define _VOLUME_H_

namespace ecj {

class Volume {
// 静态变量
public:
  static void setup(double* heap, unsigned int size) {
    Volume::_heapMemory = heap;
    Volume::_heapSize = size;
    Volume::_heapIndex = 0;
  }

public:
  static double* _heapMemory;
  static unsigned int _heapSize;
  static unsigned int _heapIndex;

  Volume() {
    _sx = 0;
    _sy = 0;
    _depth = 0;
    _extend = 0;
    _size = 0;
    _offset = -1;
  }

  Volume(unsigned int sx, unsigned int sy, unsigned int depth,
      unsigned int extend = 0);

  unsigned int size() {
    return _size;
  }

  double &operator[](int i) {
    return _heapMemory[_offset + i];
  }

  double &operator()(int x, int y, int d) {
    return _heapMemory[ (y*_sx + x)*_depth + d ];
  }

  void reset(const double value = 0.0) {
    for(unsigned int i = 0; i < _size; i++) {
      _heapMemory[_offset + i] = value;
    }
  }

public:
  unsigned int _sx;
  unsigned int _sy;
  unsigned int _depth;
  unsigned int _extend;

private:
  unsigned int _offset;
  unsigned int _size;
};


}

#endif
