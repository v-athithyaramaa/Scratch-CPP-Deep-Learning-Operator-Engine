#include "test_suite.hpp"
#include "conv2d.hpp"
#include "batchnorm.hpp"
#include "relu.hpp"
#include "residual_add.hpp"
#include "gap.hpp"
#include "fully_connected.hpp"
#include "softmax.hpp"
#include "maxpool.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>

namespace Testing
{

    static void printHeader()
    {
        std::cout << "\n========================================================================================\n";
        std::cout << "                             OPERATOR UNIT TEST SUITE (40/40)                           \n";
        std::cout << "========================================================================================\n";
        std::cout << std::left << std::setw(28) << "Test Case"
                  << std::setw(16) << "Operator"
                  << std::setw(10) << "Status"
                  << std::setw(18) << "Max Abs Diff"
                  << std::setw(14) << "Time (ms)" << "\n";
        std::cout << "----------------------------------------------------------------------------------------\n";
    }

    static void printResult(const TestCase &tc)
    {
        std::cout << std::left << std::setw(28) << tc.test_name
                  << std::setw(16) << tc.operator_name;
        if (tc.passed)
        {
            std::cout << "\033[32m" << std::setw(10) << "PASS" << "\033[0m";
        }
        else
        {
            std::cout << "\033[31m" << std::setw(10) << "FAIL" << "\033[0m";
        }
        std::cout << std::scientific << std::setprecision(4) << std::setw(18) << tc.max_abs_diff
                  << std::fixed << std::setprecision(3) << std::setw(14) << tc.duration_ms << "\n";
    }

    TestCase UnitTester::testConv2D(int id, int N, int Cin, int Hin, int Win, int Cout, int Kh, int Kw, int stride, int padding, bool has_bias)
    {
        Tensor input({N, Cin, Hin, Win});
        Tensor weights({Cout, Cin, Kh, Kw});
        Tensor bias({Cout});

        for (size_t i = 0; i < input.total_size(); ++i)
            input[i] = std::sin(static_cast<float>(i + 1));
        for (size_t i = 0; i < weights.total_size(); ++i)
            weights[i] = std::cos(static_cast<float>(i + 1)) * 0.1f;
        for (size_t i = 0; i < bias.total_size(); ++i)
            bias[i] = has_bias ? 0.05f * (i + 1) : 0.0f;

        Tensor output_calculated;
        auto t0 = std::chrono::high_resolution_clock::now();
        Operators::Conv2D::forward(input, output_calculated, weights, has_bias ? &bias : nullptr, stride, padding);
        auto t1 = std::chrono::high_resolution_clock::now();

        int Hout = ((Hin - Kh + 2 * padding) / stride) + 1;
        int Wout = ((Win - Kw + 2 * padding) / stride) + 1;
        Tensor output_reference({N, Cout, Hout, Wout}, 0.0f);

        for (int n = 0; n < N; ++n)
        {
            for (int k = 0; k < Cout; ++k)
            {
                float b = has_bias ? bias[k] : 0.0f;
                for (int ho = 0; ho < Hout; ++ho)
                {
                    int h_base = ho * stride - padding;
                    for (int wo = 0; wo < Wout; ++wo)
                    {
                        int w_base = wo * stride - padding;
                        float acc = b;
                        for (int c = 0; c < Cin; ++c)
                        {
                            for (int r = 0; r < Kh; ++r)
                            {
                                int hi = h_base + r;
                                if (hi < 0 || hi >= Hin)
                                    continue;
                                for (int s = 0; s < Kw; ++s)
                                {
                                    int wi = w_base + s;
                                    if (wi >= 0 && wi < Win)
                                    {
                                        acc += input.at4D(n, c, hi, wi) * weights.at4D(k, c, r, s);
                                    }
                                }
                            }
                        }
                        output_reference.at4D(n, k, ho, wo) = acc;
                    }
                }
            }
        }

        auto cmp = IOUtils::compareTensors(output_calculated, output_reference, 1e-4f, 1e-3f);
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return {"Conv2D_Case_" + std::to_string(id), "Conv2D", cmp.pass, cmp.max_abs_diff, cmp.max_rel_diff, elapsed};
    }

    TestCase UnitTester::testBatchNorm(int id, int N, int C, int H, int W)
    {
        Tensor input({N, C, H, W});
        Tensor gamma({C}), beta({C}), mean({C}), var({C});

        for (size_t i = 0; i < input.total_size(); ++i)
            input[i] = static_cast<float>(i % 23) - 11.0f;
        for (int c = 0; c < C; ++c)
        {
            gamma[c] = 1.0f + 0.05f * c;
            beta[c] = -0.1f * c;
            mean[c] = 0.5f * c;
            var[c] = 1.2f + 0.1f * c;
        }

        Tensor output_calculated;
        auto t0 = std::chrono::high_resolution_clock::now();
        Operators::BatchNorm::forward(input, output_calculated, gamma, beta, mean, var, 1e-5f);
        auto t1 = std::chrono::high_resolution_clock::now();

        Tensor output_reference({N, C, H, W});
        for (int c = 0; c < C; ++c)
        {
            float inv_std = 1.0f / std::sqrt(var[c] + 1e-5f);
            for (int n = 0; n < N; ++n)
            {
                for (int h = 0; h < H; ++h)
                {
                    for (int w = 0; w < W; ++w)
                    {
                        float x = input.at4D(n, c, h, w);
                        output_reference.at4D(n, c, h, w) = gamma[c] * ((x - mean[c]) * inv_std) + beta[c];
                    }
                }
            }
        }

        auto cmp = IOUtils::compareTensors(output_calculated, output_reference, 1e-4f, 1e-3f);
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return {"BatchNorm_Case_" + std::to_string(id), "BatchNorm", cmp.pass, cmp.max_abs_diff, cmp.max_rel_diff, elapsed};
    }

    TestCase UnitTester::testReLU(int id, const std::vector<int> &shape, float min_val, float max_val)
    {
        Tensor input(shape);
        size_t total = input.total_size();
        float step = (max_val - min_val) / static_cast<float>(total > 1 ? total - 1 : 1);
        for (size_t i = 0; i < total; ++i)
            input[i] = min_val + i * step;

        Tensor output_calculated;
        auto t0 = std::chrono::high_resolution_clock::now();
        Operators::ReLU::forward(input, output_calculated);
        auto t1 = std::chrono::high_resolution_clock::now();

        Tensor output_reference(shape);
        for (size_t i = 0; i < total; ++i)
            output_reference[i] = input[i] > 0.0f ? input[i] : 0.0f;

        auto cmp = IOUtils::compareTensors(output_calculated, output_reference, 1e-5f, 1e-4f);
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return {"ReLU_Case_" + std::to_string(id), "ReLU", cmp.pass, cmp.max_abs_diff, cmp.max_rel_diff, elapsed};
    }

    TestCase UnitTester::testResidualAdd(int id, const std::vector<int> &shape)
    {
        Tensor a(shape), b(shape);
        for (size_t i = 0; i < a.total_size(); ++i)
        {
            a[i] = static_cast<float>(i) * 0.25f;
            b[i] = -0.15f * static_cast<float>(i);
        }

        Tensor output_calculated;
        auto t0 = std::chrono::high_resolution_clock::now();
        Operators::ResidualAdd::forward(a, b, output_calculated);
        auto t1 = std::chrono::high_resolution_clock::now();

        Tensor output_reference(shape);
        for (size_t i = 0; i < a.total_size(); ++i)
            output_reference[i] = a[i] + b[i];

        auto cmp = IOUtils::compareTensors(output_calculated, output_reference, 1e-5f, 1e-4f);
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return {"ResAdd_Case_" + std::to_string(id), "ResidualAdd", cmp.pass, cmp.max_abs_diff, cmp.max_rel_diff, elapsed};
    }

    TestCase UnitTester::testGAP(int id, int N, int C, int H, int W)
    {
        Tensor input({N, C, H, W});
        for (size_t i = 0; i < input.total_size(); ++i)
            input[i] = static_cast<float>(i % 17) + 0.5f;

        Tensor output_calculated;
        auto t0 = std::chrono::high_resolution_clock::now();
        Operators::GAP::forward(input, output_calculated);
        auto t1 = std::chrono::high_resolution_clock::now();

        Tensor output_reference({N, C, 1, 1});
        float spatial_area = static_cast<float>(H * W);
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                float sum = 0.0f;
                for (int h = 0; h < H; ++h)
                {
                    for (int w = 0; w < W; ++w)
                    {
                        sum += input.at4D(n, c, h, w);
                    }
                }
                output_reference.at4D(n, c, 0, 0) = sum / spatial_area;
            }
        }

        auto cmp = IOUtils::compareTensors(output_calculated, output_reference, 1e-4f, 1e-3f);
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return {"GAP_Case_" + std::to_string(id), "GAP", cmp.pass, cmp.max_abs_diff, cmp.max_rel_diff, elapsed};
    }

    TestCase UnitTester::testFullyConnected(int id, int N, int in_feat, int out_feat, bool has_bias)
    {
        Tensor input({N, in_feat});
        Tensor weights({out_feat, in_feat});
        Tensor bias({out_feat});

        for (size_t i = 0; i < input.total_size(); ++i)
            input[i] = static_cast<float>(i + 1) * 0.05f;
        for (size_t i = 0; i < weights.total_size(); ++i)
            weights[i] = ((i % 5) - 2.0f) * 0.1f;
        for (int i = 0; i < out_feat; ++i)
            bias[i] = has_bias ? 0.01f * (i + 1) : 0.0f;

        Tensor output_calculated;
        auto t0 = std::chrono::high_resolution_clock::now();
        Operators::FullyConnected::forward(input, output_calculated, weights, has_bias ? &bias : nullptr);
        auto t1 = std::chrono::high_resolution_clock::now();

        Tensor output_reference({N, out_feat});
        for (int n = 0; n < N; ++n)
        {
            for (int i = 0; i < out_feat; ++i)
            {
                float sum = has_bias ? bias[i] : 0.0f;
                for (int j = 0; j < in_feat; ++j)
                {
                    sum += weights.at2D(i, j) * input[n * in_feat + j];
                }
                output_reference.at2D(n, i) = sum;
            }
        }

        auto cmp = IOUtils::compareTensors(output_calculated, output_reference, 1e-4f, 1e-3f);
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return {"FC_Case_" + std::to_string(id), "FullyConnected", cmp.pass, cmp.max_abs_diff, cmp.max_rel_diff, elapsed};
    }

    TestCase UnitTester::testSoftmax(int id, int N, int C, float scale)
    {
        Tensor input({N, C});
        for (size_t i = 0; i < input.total_size(); ++i)
            input[i] = (static_cast<float>(i % 11) - 5.0f) * scale;

        Tensor output_calculated;
        auto t0 = std::chrono::high_resolution_clock::now();
        Operators::Softmax::forward(input, output_calculated);
        auto t1 = std::chrono::high_resolution_clock::now();

        Tensor output_reference({N, C});
        for (int n = 0; n < N; ++n)
        {
            float max_val = input.at2D(n, 0);
            for (int c = 1; c < C; ++c)
                max_val = std::max(max_val, input.at2D(n, c));
            float sum = 0.0f;
            for (int c = 0; c < C; ++c)
            {
                output_reference.at2D(n, c) = std::exp(input.at2D(n, c) - max_val);
                sum += output_reference.at2D(n, c);
            }
            for (int c = 0; c < C; ++c)
                output_reference.at2D(n, c) /= sum;
        }

        auto cmp = IOUtils::compareTensors(output_calculated, output_reference, 1e-4f, 1e-3f);
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return {"Softmax_Case_" + std::to_string(id), "Softmax", cmp.pass, cmp.max_abs_diff, cmp.max_rel_diff, elapsed};
    }

    TestCase UnitTester::testMaxPool2D(int id, int N, int C, int Hin, int Win, int pool_size, int stride, int padding)
    {
        Tensor input({N, C, Hin, Win});
        for (size_t i = 0; i < input.total_size(); ++i)
        {
            input[i] = static_cast<float>((static_cast<int>(i) % 37) - 18);
        }

        Tensor output_calculated;
        auto t0 = std::chrono::high_resolution_clock::now();
        Operators::MaxPool2D::forward(input, output_calculated, pool_size, stride, padding);
        auto t1 = std::chrono::high_resolution_clock::now();

        int Hout = ((Hin - pool_size + 2 * padding) / stride) + 1;
        int Wout = ((Win - pool_size + 2 * padding) / stride) + 1;
        Tensor output_reference({N, C, Hout, Wout});

        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int ho = 0; ho < Hout; ++ho)
                {
                    int h_base = ho * stride - padding;
                    for (int wo = 0; wo < Wout; ++wo)
                    {
                        int w_base = wo * stride - padding;
                        float max_val = -std::numeric_limits<float>::infinity();

                        for (int ph = 0; ph < pool_size; ++ph)
                        {
                            int hi = h_base + ph;
                            if (hi < 0 || hi >= Hin)
                                continue;

                            for (int pw = 0; pw < pool_size; ++pw)
                            {
                                int wi = w_base + pw;
                                if (wi < 0 || wi >= Win)
                                    continue;

                                float val = input.at4D(n, c, hi, wi);
                                if (val > max_val)
                                    max_val = val;
                            }
                        }
                        output_reference.at4D(n, c, ho, wo) = max_val;
                    }
                }
            }
        }

        auto cmp = IOUtils::compareTensors(output_calculated, output_reference, 1e-5f, 1e-4f);
        double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return {"MaxPool_Case_" + std::to_string(id), "MaxPool2D", cmp.pass, cmp.max_abs_diff, cmp.max_rel_diff, elapsed};
    }

    void UnitTester::runAllTests()
    {
        printHeader();
        size_t pass_count = 0;
        std::vector<TestCase> cases;

        // 5 Tests for Conv2D (varied kernel, stride, padding, bias)
        cases.push_back(testConv2D(1, 1, 3, 32, 32, 32, 3, 3, 1, 1, false));
        cases.push_back(testConv2D(2, 1, 32, 32, 32, 64, 3, 3, 2, 1, false));
        cases.push_back(testConv2D(3, 1, 64, 16, 16, 64, 3, 3, 1, 1, false));
        cases.push_back(testConv2D(4, 1, 16, 8, 8, 32, 5, 5, 1, 2, true));
        cases.push_back(testConv2D(5, 1, 8, 14, 14, 16, 3, 3, 2, 0, true));

        // 5 Tests for BatchNorm (varied channels and spatial sizes)
        cases.push_back(testBatchNorm(1, 1, 32, 32, 32));
        cases.push_back(testBatchNorm(2, 1, 64, 16, 16));
        cases.push_back(testBatchNorm(3, 1, 128, 8, 8));
        cases.push_back(testBatchNorm(4, 1, 16, 4, 4));
        cases.push_back(testBatchNorm(5, 2, 8, 10, 10));

        // 5 Tests for ReLU (varied ranges)
        cases.push_back(testReLU(1, {1, 32, 32, 32}, -5.0f, 5.0f));
        cases.push_back(testReLU(2, {1, 64, 16, 16}, -100.0f, -1.0f));
        cases.push_back(testReLU(3, {1, 128, 8, 8}, 0.5f, 50.0f));
        cases.push_back(testReLU(4, {1, 10}, -10.0f, 10.0f));
        cases.push_back(testReLU(5, {2, 16, 4, 4}, -1.0f, 1.0f));

        // 5 Tests for ResidualAdd (varied shapes)
        cases.push_back(testResidualAdd(1, {1, 64, 16, 16}));
        cases.push_back(testResidualAdd(2, {1, 32, 32, 32}));
        cases.push_back(testResidualAdd(3, {1, 128, 8, 8}));
        cases.push_back(testResidualAdd(4, {1, 10}));
        cases.push_back(testResidualAdd(5, {2, 16, 8, 8}));

        // 5 Tests for GAP (varied spatial sizes and channel counts)
        cases.push_back(testGAP(1, 1, 128, 8, 8));
        cases.push_back(testGAP(2, 1, 64, 16, 16));
        cases.push_back(testGAP(3, 1, 32, 32, 32));
        cases.push_back(testGAP(4, 1, 16, 4, 4));
        cases.push_back(testGAP(5, 2, 8, 6, 6));

        // 5 Tests for FullyConnected (varied input/output features, with and without bias)
        cases.push_back(testFullyConnected(1, 1, 128, 10, true));
        cases.push_back(testFullyConnected(2, 1, 128, 10, false));
        cases.push_back(testFullyConnected(3, 1, 64, 32, true));
        cases.push_back(testFullyConnected(4, 1, 512, 10, true));
        cases.push_back(testFullyConnected(5, 2, 16, 4, true));

        // 5 Tests for Softmax (varied scale and dimension)
        cases.push_back(testSoftmax(1, 1, 10, 1.0f));
        cases.push_back(testSoftmax(2, 1, 10, 20.0f));
        cases.push_back(testSoftmax(3, 1, 100, 0.5f));
        cases.push_back(testSoftmax(4, 2, 10, 2.0f));
        cases.push_back(testSoftmax(5, 1, 2, 5.0f));

        // 5 Tests for MaxPool2D (varied shapes, windows, strides, paddings, batches)
        cases.push_back(testMaxPool2D(1, 1, 32, 32, 32, 2, 2, 0));
        cases.push_back(testMaxPool2D(2, 1, 64, 16, 16, 2, 2, 0));
        cases.push_back(testMaxPool2D(3, 1, 16, 8, 8, 3, 1, 1));
        cases.push_back(testMaxPool2D(4, 1, 8, 14, 14, 3, 2, 0));
        cases.push_back(testMaxPool2D(5, 2, 16, 4, 4, 2, 2, 0));

        for (const auto &tc : cases)
        {
            printResult(tc);
            if (tc.passed)
                pass_count++;
        }

        std::cout << "----------------------------------------------------------------------------------------\n";
        std::cout << "Total Cases: " << cases.size() << " | Passed: " << pass_count << " | Failed: " << (cases.size() - pass_count) << "\n";
        std::cout << "========================================================================================\n\n";
    }

}