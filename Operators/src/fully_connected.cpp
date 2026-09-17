#include "fully_connected.hpp"

namespace Operators {

void FullyConnected::forward(
    const Tensor& input,
    Tensor& output,
    const Tensor& weights,
    const Tensor* bias
) {
    // Weights are [Out_Features, In_Features]
    int out_features = weights.shape[0];
    int in_features = weights.shape[1];
    int N = input.shape[0];

    std::vector<int> expected_shape = {N, out_features};
    if (output.shape != expected_shape) {
        output = Tensor(expected_shape, 0.0f);
    }

    for (int n = 0; n < N; ++n) {
        for (int i = 0; i < out_features; ++i) {
            float sum = (bias != nullptr && !bias->data.empty()) ? bias->data[i] : 0.0f;
            for (int j = 0; j < in_features; ++j) {
                // Flattened input indexed at j
                sum += weights.at2D(i, j) * input.data[n * in_features + j];
            }
            output.at2D(n, i) = sum;
        }
    }
}

} 