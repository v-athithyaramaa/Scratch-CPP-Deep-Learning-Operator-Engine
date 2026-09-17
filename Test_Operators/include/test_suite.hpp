#pragma once

#include "tensor.hpp"
#include "io_utils.hpp"
#include <string>
#include <functional>

namespace Testing
{

    struct TestCase
    {
        std::string test_name;
        std::string operator_name;
        bool passed;
        float max_abs_diff;
        float max_rel_diff;
        double duration_ms;
    };

    class UnitTester
    {
    public:
        static void runAllTests();

    private:
        static TestCase testConv2D(int id, int N, int Cin, int Hin, int Win, int Cout, int Kh, int Kw, int stride, int padding, bool has_bias);
        static TestCase testBatchNorm(int id, int N, int C, int H, int W);
        static TestCase testReLU(int id, const std::vector<int> &shape, float min_val, float max_val);
        static TestCase testResidualAdd(int id, const std::vector<int> &shape);
        static TestCase testGAP(int id, int N, int C, int H, int W);
        static TestCase testFullyConnected(int id, int N, int in_feat, int out_feat, bool has_bias);
        static TestCase testSoftmax(int id, int N, int C, float scale);
        static TestCase testMaxPool2D(int id, int N, int C, int Hin, int Win, int pool_size, int stride, int padding);
    };

}