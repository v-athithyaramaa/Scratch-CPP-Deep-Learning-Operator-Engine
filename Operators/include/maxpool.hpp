#pragma once

#include "tensor.hpp"

namespace Operators {

class MaxPool2D {
public:
    static void forward(
        const Tensor& input,
        Tensor& output,
        int pool_size = 2,
        int stride = 2,
        int padding = 0
    );
};

}