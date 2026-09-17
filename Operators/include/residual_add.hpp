#pragma once
#include "tensor.hpp"

namespace Operators {

class ResidualAdd {
public:
    static void forward(const Tensor& a, const Tensor& b, Tensor& output);
};

}