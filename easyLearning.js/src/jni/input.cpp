#include "input.h"

namespace ecj {

InputLayer::InputLayer(unsigned int sx, unsigned int sy, unsigned int depth)
    : _sx(sx), _sy(sy), _depth(depth), _data(sx, sy, depth, 0) {

}

void InputLayer::forward(std::vector<double>& input) {
  for(unsigned int i = 0; i < input.size(); i++) {
    _data[i] = input[i];
  }
}

} // end of namespace ecj
