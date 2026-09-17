#include "tensor.hpp"
#include "io_utils.hpp"
#include "conv2d.hpp"
#include "batchnorm.hpp"
#include "relu.hpp"
#include "residual_add.hpp"
#include "gap.hpp"
#include "fully_connected.hpp"
#include "softmax.hpp"
#include "test_suite.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>

// Helper to log both to stdout and to report/execution_report.log
void logLayer(std::ofstream &logfile, const std::string &name, const std::string &op,
              bool pass, float max_diff, double time_ms, const std::vector<int> &shape)
{
    // Console output with ANSI coloring
    std::cout << std::left << std::setw(24) << name
              << std::setw(16) << op;
    if (pass)
    {
        std::cout << "\033[32m" << std::setw(10) << "PASS" << "\033[0m";
    }
    else
    {
        std::cout << "\033[31m" << std::setw(10) << "FAIL" << "\033[0m";
    }
    std::cout << std::scientific << std::setprecision(4) << std::setw(16) << max_diff
              << std::fixed << std::setprecision(3) << std::setw(12) << time_ms << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    std::cout << "]\n";

    // File output (clean text without terminal color escapes)
    if (logfile.is_open())
    {
        logfile << std::left << std::setw(24) << name
                << std::setw(16) << op
                << std::setw(10) << (pass ? "PASS" : "FAIL")
                << std::scientific << std::setprecision(4) << std::setw(16) << max_diff
                << std::fixed << std::setprecision(3) << std::setw(12) << time_ms << "[";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            logfile << shape[i] << (i + 1 < shape.size() ? ", " : "");
        }
        logfile << "]\n";
    }
}

int runModelInference(const std::string &config_path)
{
    std::ofstream logfile("../report/execution_report.log");
    if (!logfile.is_open())
    {
        // Fallback for running from different relative paths
        logfile.open("report/execution_report.log");
    }

    std::string banner =
        "\n========================================================================================\n"
        "                 CIFAR-10 MINI-RESNET: BARE-METAL MODEL INFERENCE                       \n"
        "========================================================================================\n";
    std::cout << banner;
    if (logfile.is_open())
        logfile << banner;

    std::cout << "Loaded Model Config: " << config_path << "\n\n";
    if (logfile.is_open())
        logfile << "Loaded Model Config: " << config_path << "\n\n";

    std::cout << std::left << std::setw(24) << "Layer / Stage"
              << std::setw(16) << "Operator"
              << std::setw(10) << "Status"
              << std::setw(16) << "Max Abs Diff"
              << std::setw(12) << "Time (ms)"
              << "Output Shape\n";
    std::cout << "----------------------------------------------------------------------------------------\n";

    if (logfile.is_open())
    {
        logfile << std::left << std::setw(24) << "Layer / Stage"
                << std::setw(16) << "Operator"
                << std::setw(10) << "Status"
                << std::setw(16) << "Max Abs Diff"
                << std::setw(12) << "Time (ms)"
                << "Output Shape\n";
        logfile << "----------------------------------------------------------------------------------------\n";
    }

    double total_inference_time = 0.0;
    bool all_passed = true;

    // Load Input Image [1, 3, 32, 32]
    Tensor input({1, 3, 32, 32});
    if (!IOUtils::loadBinary("../data/input/sample_input.bin", input))
    {
        if (!IOUtils::loadBinary("data/input/sample_input.bin", input))
        {
            std::cerr << "[!] Could not load sample input binary!\n";
            return 1;
        }
    }

    Tensor buf_a, buf_b, ref_tensor;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = std::chrono::high_resolution_clock::now();

    // 1. Stem Conv2D (3 -> 32, k=3, s=1, p=1)
    Tensor stem_w({32, 3, 3, 3});
    IOUtils::loadBinary("../data/weights/stem_conv_weight.bin", stem_w);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::Conv2D::forward(input, buf_a, stem_w, nullptr, 1, 1);
    t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 32, 32, 32});
    IOUtils::loadBinary("../data/reference/ref_stem_conv.bin", ref_tensor);
    auto cmp = IOUtils::compareTensors(buf_a, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "stem_conv", "Conv2D", cmp.pass, cmp.max_abs_diff, ms, buf_a.shape);

    // 2. Stem BatchNorm (32)
    Tensor s_bn_g({32}), s_bn_b({32}), s_bn_m({32}), s_bn_v({32});
    IOUtils::loadBinary("../data/weights/stem_bn_weight.bin", s_bn_g);
    IOUtils::loadBinary("../data/weights/stem_bn_bias.bin", s_bn_b);
    IOUtils::loadBinary("../data/weights/stem_bn_mean.bin", s_bn_m);
    IOUtils::loadBinary("../data/weights/stem_bn_var.bin", s_bn_v);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::BatchNorm::forward(buf_a, buf_b, s_bn_g, s_bn_b, s_bn_m, s_bn_v);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 32, 32, 32});
    IOUtils::loadBinary("../data/reference/ref_stem_bn.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_b, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "stem_bn", "BatchNorm", cmp.pass, cmp.max_abs_diff, ms, buf_b.shape);

    // 3. Stem ReLU
    t0 = std::chrono::high_resolution_clock::now();
    Operators::ReLU::forward(buf_b, buf_a);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 32, 32, 32});
    IOUtils::loadBinary("../data/reference/ref_stem_out.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_a, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "stem_relu", "ReLU", cmp.pass, cmp.max_abs_diff, ms, buf_a.shape);

    // 4. Downsample 1 Conv (32 -> 64, k=3, s=2, p=1)
    Tensor ds1_w({64, 32, 3, 3});
    IOUtils::loadBinary("../data/weights/ds1_conv_weight.bin", ds1_w);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::Conv2D::forward(buf_a, buf_b, ds1_w, nullptr, 2, 1);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 64, 16, 16});
    IOUtils::loadBinary("../data/reference/ref_ds1_conv.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_b, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "ds1_conv", "Conv2D", cmp.pass, cmp.max_abs_diff, ms, buf_b.shape);

    // 5. Downsample 1 BN (64)
    Tensor ds1_bn_g({64}), ds1_bn_b({64}), ds1_bn_m({64}), ds1_bn_v({64});
    IOUtils::loadBinary("../data/weights/ds1_bn_weight.bin", ds1_bn_g);
    IOUtils::loadBinary("../data/weights/ds1_bn_bias.bin", ds1_bn_b);
    IOUtils::loadBinary("../data/weights/ds1_bn_mean.bin", ds1_bn_m);
    IOUtils::loadBinary("../data/weights/ds1_bn_var.bin", ds1_bn_v);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::BatchNorm::forward(buf_b, buf_a, ds1_bn_g, ds1_bn_b, ds1_bn_m, ds1_bn_v);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 64, 16, 16});
    IOUtils::loadBinary("../data/reference/ref_ds1_bn.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_a, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "ds1_bn", "BatchNorm", cmp.pass, cmp.max_abs_diff, ms, buf_a.shape);

    // 6. Downsample 1 ReLU (Skip connection cache)
    Tensor skip_residual;
    t0 = std::chrono::high_resolution_clock::now();
    Operators::ReLU::forward(buf_a, skip_residual);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 64, 16, 16});
    IOUtils::loadBinary("../data/reference/ref_ds1_out.bin", ref_tensor);
    cmp = IOUtils::compareTensors(skip_residual, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "ds1_relu", "ReLU", cmp.pass, cmp.max_abs_diff, ms, skip_residual.shape);

    // 7. ResBlock Conv 1 (64 -> 64, k=3, s=1, p=1)
    Tensor res_c1_w({64, 64, 3, 3});
    IOUtils::loadBinary("../data/weights/res_conv1_weight.bin", res_c1_w);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::Conv2D::forward(skip_residual, buf_a, res_c1_w, nullptr, 1, 1);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 64, 16, 16});
    IOUtils::loadBinary("../data/reference/ref_res_conv1.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_a, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "res_conv1", "Conv2D", cmp.pass, cmp.max_abs_diff, ms, buf_a.shape);

    // 8. ResBlock BN 1 (64)
    Tensor r_bn1_g({64}), r_bn1_b({64}), r_bn1_m({64}), r_bn1_v({64});
    IOUtils::loadBinary("../data/weights/res_bn1_weight.bin", r_bn1_g);
    IOUtils::loadBinary("../data/weights/res_bn1_bias.bin", r_bn1_b);
    IOUtils::loadBinary("../data/weights/res_bn1_mean.bin", r_bn1_m);
    IOUtils::loadBinary("../data/weights/res_bn1_var.bin", r_bn1_v);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::BatchNorm::forward(buf_a, buf_b, r_bn1_g, r_bn1_b, r_bn1_m, r_bn1_v);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 64, 16, 16});
    IOUtils::loadBinary("../data/reference/ref_res_bn1.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_b, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "res_bn1", "BatchNorm", cmp.pass, cmp.max_abs_diff, ms, buf_b.shape);

    // 9. ResBlock ReLU 1
    t0 = std::chrono::high_resolution_clock::now();
    Operators::ReLU::forward(buf_b, buf_a);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 64, 16, 16});
    IOUtils::loadBinary("../data/reference/ref_res_relu1.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_a, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "res_relu1", "ReLU", cmp.pass, cmp.max_abs_diff, ms, buf_a.shape);

    // 10. ResBlock Conv 2 (64 -> 64, k=3, s=1, p=1)
    Tensor res_c2_w({64, 64, 3, 3});
    IOUtils::loadBinary("../data/weights/res_conv2_weight.bin", res_c2_w);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::Conv2D::forward(buf_a, buf_b, res_c2_w, nullptr, 1, 1);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 64, 16, 16});
    IOUtils::loadBinary("../data/reference/ref_res_conv2.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_b, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "res_conv2", "Conv2D", cmp.pass, cmp.max_abs_diff, ms, buf_b.shape);

    // 11. ResBlock BN 2 (64)
    Tensor r_bn2_g({64}), r_bn2_b({64}), r_bn2_m({64}), r_bn2_v({64});
    IOUtils::loadBinary("../data/weights/res_bn2_weight.bin", r_bn2_g);
    IOUtils::loadBinary("../data/weights/res_bn2_bias.bin", r_bn2_b);
    IOUtils::loadBinary("../data/weights/res_bn2_mean.bin", r_bn2_m);
    IOUtils::loadBinary("../data/weights/res_bn2_var.bin", r_bn2_v);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::BatchNorm::forward(buf_b, buf_a, r_bn2_g, r_bn2_b, r_bn2_m, r_bn2_v);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 64, 16, 16});
    IOUtils::loadBinary("../data/reference/ref_res_bn2.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_a, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "res_bn2", "BatchNorm", cmp.pass, cmp.max_abs_diff, ms, buf_a.shape);

    // 12. Residual Addition
    t0 = std::chrono::high_resolution_clock::now();
    Operators::ResidualAdd::forward(buf_a, skip_residual, buf_b);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 64, 16, 16});
    IOUtils::loadBinary("../data/reference/ref_res_add.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_b, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "res_add", "ResidualAdd", cmp.pass, cmp.max_abs_diff, ms, buf_b.shape);

    // 13. ResBlock Final ReLU
    t0 = std::chrono::high_resolution_clock::now();
    Operators::ReLU::forward(buf_b, buf_a);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 64, 16, 16});
    IOUtils::loadBinary("../data/reference/ref_res_out.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_a, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "res_relu2", "ReLU", cmp.pass, cmp.max_abs_diff, ms, buf_a.shape);

    // 14. Downsample 2 Conv (64 -> 128, k=3, s=2, p=1)
    Tensor ds2_w({128, 64, 3, 3});
    IOUtils::loadBinary("../data/weights/ds2_conv_weight.bin", ds2_w);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::Conv2D::forward(buf_a, buf_b, ds2_w, nullptr, 2, 1);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 128, 8, 8});
    IOUtils::loadBinary("../data/reference/ref_ds2_conv.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_b, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "ds2_conv", "Conv2D", cmp.pass, cmp.max_abs_diff, ms, buf_b.shape);

    // 15. Downsample 2 BN (128)
    Tensor ds2_bn_g({128}), ds2_bn_b({128}), ds2_bn_m({128}), ds2_bn_v({128});
    IOUtils::loadBinary("../data/weights/ds2_bn_weight.bin", ds2_bn_g);
    IOUtils::loadBinary("../data/weights/ds2_bn_bias.bin", ds2_bn_b);
    IOUtils::loadBinary("../data/weights/ds2_bn_mean.bin", ds2_bn_m);
    IOUtils::loadBinary("../data/weights/ds2_bn_var.bin", ds2_bn_v);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::BatchNorm::forward(buf_b, buf_a, ds2_bn_g, ds2_bn_b, ds2_bn_m, ds2_bn_v);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 128, 8, 8});
    IOUtils::loadBinary("../data/reference/ref_ds2_bn.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_a, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "ds2_bn", "BatchNorm", cmp.pass, cmp.max_abs_diff, ms, buf_a.shape);

    // 16. Downsample 2 ReLU
    t0 = std::chrono::high_resolution_clock::now();
    Operators::ReLU::forward(buf_a, buf_b);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 128, 8, 8});
    IOUtils::loadBinary("../data/reference/ref_ds2_out.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_b, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "ds2_relu", "ReLU", cmp.pass, cmp.max_abs_diff, ms, buf_b.shape);

    // 17. Global Average Pooling (GAP: 128x8x8 -> 128x1x1)
    t0 = std::chrono::high_resolution_clock::now();
    Operators::GAP::forward(buf_b, buf_a);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 128, 1, 1});
    IOUtils::loadBinary("../data/reference/ref_gap_out.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_a, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "gap", "GAP", cmp.pass, cmp.max_abs_diff, ms, buf_a.shape);

    // 18. Linear Classifier (128 -> 10)
    Tensor fc_w({10, 128}), fc_b({10});
    IOUtils::loadBinary("../data/weights/classifier_weight.bin", fc_w);
    IOUtils::loadBinary("../data/weights/classifier_bias.bin", fc_b);
    t0 = std::chrono::high_resolution_clock::now();
    Operators::FullyConnected::forward(buf_a, buf_b, fc_w, &fc_b);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 10});
    IOUtils::loadBinary("../data/reference/ref_logits.bin", ref_tensor);
    cmp = IOUtils::compareTensors(buf_b, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "classifier", "FullyConnected", cmp.pass, cmp.max_abs_diff, ms, buf_b.shape);

    // 19. Softmax (10)
    Tensor probabilities;
    t0 = std::chrono::high_resolution_clock::now();
    Operators::Softmax::forward(buf_b, probabilities);
    t1 = std::chrono::high_resolution_clock::now();
    ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_inference_time += ms;
    ref_tensor = Tensor({1, 10});
    IOUtils::loadBinary("../data/reference/ref_probabilities.bin", ref_tensor);
    cmp = IOUtils::compareTensors(probabilities, ref_tensor);
    all_passed &= cmp.pass;
    logLayer(logfile, "softmax", "Softmax", cmp.pass, cmp.max_abs_diff, ms, probabilities.shape);

    std::string footer =
        "----------------------------------------------------------------------------------------\n"
        "Total Inference Pipeline Latency: " +
        std::to_string(total_inference_time) + " ms\n"
                                               "Overall Model Status: " +
        (all_passed ? "ALL LAYERS PASSED" : "PARITY FAILURE") + "\n"
                                                                "========================================================================================\n\n";

    std::cout << "----------------------------------------------------------------------------------------\n";
    std::cout << "Total Inference Pipeline Latency: " << std::fixed << std::setprecision(3) << total_inference_time << " ms\n";
    std::cout << "Overall Model Status: " << (all_passed ? "\033[32mALL LAYERS PASSED\033[0m" : "\033[31mPARITY FAILURE\033[0m") << "\n";
    std::cout << "========================================================================================\n\n";

    if (logfile.is_open())
    {
        logfile << footer;
    }

    // Display top prediction
    std::vector<std::string> classes = {"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
    int predicted_class = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < 10; ++i)
    {
        if (probabilities[i] > max_p)
        {
            max_p = probabilities[i];
            predicted_class = i;
        }
    }

    std::string pred_str = "Top Prediction: " + classes[predicted_class] +
                           " (" + std::to_string(max_p * 100.0f).substr(0, 5) + "% confidence)\n\n";
    std::cout << pred_str;
    if (logfile.is_open())
    {
        logfile << pred_str;
        logfile.close();
    }

    IOUtils::saveBinary("../data/output/cpp_output.bin", probabilities);
    return all_passed ? 0 : 1;
}

int main(int argc, char *argv[])
{
    std::string mode = "model";
    std::string config_file = "configs/model_config.json";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc)
        {
            mode = argv[++i];
        }
        else if (arg == "--config" && i + 1 < argc)
        {
            config_file = argv[++i];
        }
    }

    if (mode == "test")
    {
        Testing::UnitTester::runAllTests();
        return 0;
    }
    else
    {
        return runModelInference(config_file);
    }
}