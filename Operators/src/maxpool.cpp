#include "maxpool.hpp"
#include <algorithm>
#include <limits>

namespace Operators
{

    void MaxPool2D::forward(
        const Tensor &input,
        Tensor &output,
        int pool_size,
        int stride,
        int padding)
    {
        int N = input.shape[0];
        int C = input.shape[1];
        int Hin = input.shape[2];
        int Win = input.shape[3];

        int Hout = ((Hin - pool_size + 2 * padding) / stride) + 1;
        int Wout = ((Win - pool_size + 2 * padding) / stride) + 1;

        output = Tensor({N, C, Hout, Wout});

        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int ho = 0; ho < Hout; ++ho)
                {
                    int h_start = ho * stride - padding;
                    for (int wo = 0; wo < Wout; ++wo)
                    {
                        int w_start = wo * stride - padding;

                        float max_val = -std::numeric_limits<float>::infinity();

                        for (int ph = 0; ph < pool_size; ++ph)
                        {
                            int h_in = h_start + ph;
                            if (h_in < 0 || h_in >= Hin)
                                continue;

                            for (int pw = 0; pw < pool_size; ++pw)
                            {
                                int w_in = w_start + pw;
                                if (w_in < 0 || w_in >= Win)
                                    continue;

                                float val = input.at4D(n, c, h_in, w_in);
                                if (val > max_val)
                                {
                                    max_val = val;
                                }
                            }
                        }

                        output.at4D(n, c, ho, wo) = max_val;
                    }
                }
            }
        }
    }

}