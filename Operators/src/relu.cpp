#include "relu.hpp"
#include <algorithm>

namespace Operators {

void ReLU::forward(const Tensor& input, Tensor& output) {
    if (output.shape != input.shape) {
        output = Tensor(input.shape, 0.0f);
    }
    size_t total = input.total_size();
    for (size_t i = 0; i < total; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

}