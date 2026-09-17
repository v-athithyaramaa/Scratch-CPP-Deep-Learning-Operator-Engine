#include "conv2d.hpp"
#include <iostream>

namespace Operators
{

    void Conv2D::forward(
        const Tensor &input,
        Tensor &output,
        const Tensor &weights,
        const Tensor *bias,
        int stride,
        int padding)
    {
        // Input dimensions: [N, C_in, H_in, W_in]
        int N = input.shape[0];
        int C_in = input.shape[1];
        int H_in = input.shape[2];
        int W_in = input.shape[3];

        // Weight dimensions: [C_out, C_in, K_h, K_w]
        int C_out = weights.shape[0];
        int K_h = weights.shape[2];
        int K_w = weights.shape[3];

        // Compute expected output dimensions
        int H_out = ((H_in - K_h + 2 * padding) / stride) + 1;
        int W_out = ((W_in - K_w + 2 * padding) / stride) + 1;

        // Allocate output tensor if shape does not match
        std::vector<int> expected_shape = {N, C_out, H_out, W_out};
        if (output.shape != expected_shape)
        {
            output = Tensor(expected_shape, 0.0f);
        }
        else
        {
            // Reset output buffer to zero
            std::fill(output.data.begin(), output.data.end(), 0.0f);
        }

        // Cache line-friendly loop nesting:
        // Outer: Batch & Output Channel
        for (int n = 0; n < N; ++n)
        {
            for (int k = 0; k < C_out; ++k)
            {
                float b = (bias != nullptr && !bias->data.empty()) ? bias->data[k] : 0.0f;

                for (int h_out = 0; h_out < H_out; ++h_out)
                {
                    int h_in_base = h_out * stride - padding;

                    for (int w_out = 0; w_out < W_out; ++w_out)
                    {
                        int w_in_base = w_out * stride - padding;

                        float accumulator = b;

                        // Inner: Accumulate over input channels and kernel window
                        for (int c = 0; c < C_in; ++c)
                        {
                            for (int r = 0; r < K_h; ++r)
                            {
                                int h_in = h_in_base + r;

                                // Check vertical padding boundary
                                if (h_in < 0 || h_in >= H_in)
                                {
                                    continue;
                                }

                                for (int s = 0; s < K_w; ++s)
                                {
                                    int w_in = w_in_base + s;

                                    // Check horizontal padding boundary
                                    if (w_in >= 0 && w_in < W_in)
                                    {
                                        float in_val = input.at4D(n, c, h_in, w_in);
                                        float wt_val = weights.at4D(k, c, r, s);
                                        accumulator += in_val * wt_val;
                                    }
                                }
                            }
                        }

                        output.at4D(n, k, h_out, w_out) = accumulator;
                    }
                }
            }
        }
    }

}