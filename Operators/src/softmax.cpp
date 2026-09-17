#include "softmax.hpp"
#include <cmath>
#include <algorithm>

namespace Operators
{

    void Softmax::forward(const Tensor &input, Tensor &output)
    {
        if (output.shape != input.shape)
        {
            output = Tensor(input.shape, 0.0f);
        }

        int N = input.shape[0];
        int C = input.shape[1];

        for (int n = 0; n < N; ++n)
        {
            // Step 1: Find max logit for numerical stability
            float max_val = input.at2D(n, 0);
            for (int c = 1; c < C; ++c)
            {
                max_val = std::max(max_val, input.at2D(n, c));
            }

            // Step 2: Exponentiate and accumulate sum
            float sum = 0.0f;
            for (int c = 0; c < C; ++c)
            {
                float exp_val = std::exp(input.at2D(n, c) - max_val);
                output.at2D(n, c) = exp_val;
                sum += exp_val;
            }

            // Step 3: Normalize to probability distribution
            for (int c = 0; c < C; ++c)
            {
                output.at2D(n, c) /= sum;
            }
        }
    }

}