#include "residual_add.hpp"
#include <stdexcept>

namespace Operators {

void ResidualAdd::forward(const Tensor& a, const Tensor& b, Tensor& output) {
    if (a.shape != b.shape) {
        throw std::runtime_error("[!] Shape mismatch in ResidualAdd!");
    }
    if (output.shape != a.shape) {
        output = Tensor(a.shape, 0.0f);
    }
    size_t total = a.total_size();
    for (size_t i = 0; i < total; ++i) {
        output[i] = a[i] + b[i];
    }
}

}