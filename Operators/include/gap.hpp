#pragma once
#include "tensor.hpp"

namespace Operators {

class GAP {
public:
    static void forward(const Tensor& input, Tensor& output);
};

}