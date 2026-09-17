#include "batchnorm.hpp"
#include <cmath>

namespace Operators {

void BatchNorm::forward(
    const Tensor& input,
    Tensor& output,
    const Tensor& gamma,
    const Tensor& beta,
    const Tensor& mean,
    const Tensor& var,
    float eps
) {
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    if (output.shape != input.shape) {
        output = Tensor(input.shape, 0.0f);
    }

    for (int c = 0; c < C; ++c) {
        float std_dev = std::sqrt(var[c] + eps);
        float scale = gamma[c] / std_dev;
        float shift = beta[c] - (mean[c] * scale);

        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float x = input.at4D(n, c, h, w);
                    output.at4D(n, c, h, w) = x * scale + shift;
                }
            }
        }
    }
}

} // namespace Operators