#pragma once
#include "tensor.hpp"

namespace Operators {

class ReLU {
public:
    static void forward(const Tensor& input, Tensor& output);
};

} // namespace Operators