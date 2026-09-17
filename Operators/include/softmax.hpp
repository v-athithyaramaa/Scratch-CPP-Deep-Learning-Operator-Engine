#pragma once
#include "tensor.hpp"

namespace Operators {

class Softmax {
public:
    static void forward(const Tensor& input, Tensor& output);
};

}