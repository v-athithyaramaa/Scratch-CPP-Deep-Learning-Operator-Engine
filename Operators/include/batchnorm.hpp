#pragma once
#include "tensor.hpp"

namespace Operators {

class BatchNorm {
public:
    static void forward(
        const Tensor& input,
        Tensor& output,
        const Tensor& gamma,
        const Tensor& beta,
        const Tensor& mean,
        const Tensor& var,
        float eps = 1e-5f
    );
};

}