# CIFAR-10 Mini-ResNet: Bare-Metal C++ Inference Engine

> **Deep Learning Assignment 2:** Operators & C++ Implementation

A high-performance, bare-metal C++17 inference engine executing a 19-stage Mini-ResNet trained on the CIFAR-10 dataset. Developed completely from scratch with **zero third-party linear algebra or deep learning runtime libraries** (no Eigen, OpenBLAS, or ONNX Runtime). The engine evaluates IEEE-754 FP32 tensors using contiguous 1D flat-memory buffers, zero-allocation ping-pong execution scratchpads, and delivers verified numerical parity against PyTorch.

---

## 1. Project Overview

- **Zero Runtime Allocation:** Intermediate layer evaluations reuse pre-allocated ping-pong memory buffers (`buf_a`, `buf_b`) to eliminate dynamic heap allocation (`malloc`/`free`) latency traps during forward inference.
- **Cache-Locality First:** Contiguous row-major 1D flat-memory layout eliminates pointer-chasing and multi-level indirection overhead across tensor dimensions.
- **Dual Execution Modes:** Fully configurable via command-line arguments to run either isolated single-layer validation (**Unit Test Mode**) or the complete 19-stage forward graph (**Model Mode**).
- **Strict FP32 Parity:** Verified against PyTorch reference binary outputs with maximum absolute differences bounded within $\le 3.4 \times 10^{-6}$.
- **Operator Library Completeness:** Implements all core deep learning operators: Conv2D, BatchNorm, ReLU, ResidualAdd, Global Average Pooling (GAP), FullyConnected, Softmax, and MaxPool2D.

---

## 2. Repository Structure

````text
Scratch-CPP-Deep-Learning-Operator-Engine/
├── CMakeLists.txt              # Unified build configuration (C++17, -O3, -march=native)
├── README.md                   # Formatted Markdown documentation
├── Readme.txt                  # Plain-text reproduction and submission guide
├── .gitignore                  # Repository hygiene rules (ignores binaries and caches)
├── configs/
│   └── model_config.json       # 19-stage layer graph definition and file paths
├── Operators/
│   ├── include/                # Operator declarations (conv2d, batchnorm, relu,
│   │                           # residual_add, gap, fully_connected, softmax, maxpool)
│   └── src/                    # Optimized FP32 operator implementations
├── Test_Operators/
│   ├── include/                # Unit test harness headers (test_suite.hpp)
│   └── src/                    # 40 parameterized test cases & test_main.cpp
├── utilities/
│   ├── include/                # Flat Tensor abstraction and binary I/O utilities
│   └── src/                    # Tensor memory management & dual-tolerance comparator
├── src/
│   └── main.cpp                # Unified CLI runner (--mode test / --mode model)
├── data/
│   ├── input/                  # sample_input.bin (1x3x32x32 FP32 raw tensor)
│   ├── output/                 # cpp_output.bin (Calculated probability distribution)
│   ├── reference/              # Layer-wise PyTorch ground-truth validation binaries
│   └── weights/                # Serialized weights, biases, and BatchNorm statistics
└── report/
    ├── execution_report.log    # Structured runtime execution trace
    └── assignment2_report.pdf  # Comprehensive technical evaluation report
````

---

## 3. Environment & Tooling

* **Operating System:** Windows 11 or Linux (Ubuntu 22.04 LTS)
* **Compiler:** MinGW-w64 GCC 13.x+ or Clang with `-std=c++17 -O3 -march=native`
* **Build System:** CMake 3.16 or higher
* **Reference Runtime:** Python 3.10+ and PyTorch 2.x

## 4. Build Instructions

Run the following commands from the repository root:

```bash
# 1. Create and enter the build directory
mkdir build
cd build

# 2. Generate build files via CMake
cmake ..

# 3. Compile both executables
cmake --build . --target cifar10_engine
cmake --build . --target run_unit_tests
````

On Windows, the generated executables are named `cifar10_engine.exe` and
`run_unit_tests.exe`. On Linux, omit the `.exe` suffix.

## 5. Execution & CLI Modes

The application supports two execution modes:

### Mode A: Unit Test Mode

Runs the isolated test harness across all 8 implemented operators with 5
parameter variations per operator, covering input geometries, strides, kernel
dimensions, paddings, and value ranges.

From the `build` directory, run either the unified application:

```bash
./cifar10_engine.exe --mode test
```

or the dedicated test binary:

```bash
./run_unit_tests.exe
```

### Mode B: Model Mode

Executes the full 19-stage forward inference pipeline using the model
configuration file, validates each stage against PyTorch ground truth with
dual tolerances, and writes the execution trace to disk.

```bash
./cifar10_engine.exe --mode model --config ../configs/model_config.json
```

## 6. Interpreting Results

### Unit Test Verification

The unit tests compare each calculated operator output with an independent
mathematical reference. For the standard operator tolerance, a test passes
when:

$$
\left|y_{\text{calc}} - y_{\text{ref}}\right|
\leq
\max\left(10^{-4},\;10^{-3}\cdot\left|y_{\text{ref}}\right|\right)
$$

Some elementwise operators use the stricter tolerance configured in the test
harness. A successful run reports **40 / 40 PASS** across all operator test
cases.

### Model Inference Trace & Logging

Runtime metrics are printed to standard output and written to
`report/execution_report.log`. Each stage reports its name, operator, pass/fail
status, maximum absolute difference, execution time, and output shape.

```text
stem_conv              Conv2D          PASS      0.0000e+00      5.806 [1, 32, 32, 32]
stem_bn                BatchNorm       PASS      9.5367e-07      0.125 [1, 32, 32, 32]
stem_relu              ReLU            PASS      9.5367e-07      0.035 [1, 32, 32, 32]
...
classifier              FullyConnected  PASS      3.2187e-06      0.052 [1, 10]
softmax                 Softmax         PASS      4.1723e-07      0.036 [1, 10]
```

After inference, the application displays the top CIFAR-10 category and its
confidence score, for example:

```text
Top Prediction: bird (73.35% confidence)
```

The probability tensor is serialized to `data/output/cpp_output.bin`.
