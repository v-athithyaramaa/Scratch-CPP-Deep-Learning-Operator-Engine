#include "gap.hpp"

namespace Operators {

void GAP::forward(const Tensor& input, Tensor& output) {
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    // Output is [N, C, 1, 1] or flattened to [N, C]
    std::vector<int> expected_shape = {N, C, 1, 1};
    if (output.shape != expected_shape) {
        output = Tensor(expected_shape, 0.0f);
    }

    float spatial_elements = static_cast<float>(H * W);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    sum += input.at4D(n, c, h, w);
                }
            }
            output.at4D(n, c, 0, 0) = sum / spatial_elements;
        }
    }
}

}