#pragma once
#include "tensor.hpp"

namespace Operators {

class FullyConnected {
public:
    static void forward(
        const Tensor& input,
        Tensor& output,
        const Tensor& weights,
        const Tensor* bias
    );
};

}