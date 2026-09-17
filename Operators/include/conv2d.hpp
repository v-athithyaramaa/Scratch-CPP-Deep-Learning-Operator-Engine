#pragma once

#include "tensor.hpp"

namespace Operators
{

    class Conv2D
    {
    public:
        static void forward(
            const Tensor &input,
            Tensor &output,
            const Tensor &weights,
            const Tensor *bias, // optional: pass nullptr if model uses BatchNorm without bias
            int stride = 1,
            int padding = 1);
    };

}