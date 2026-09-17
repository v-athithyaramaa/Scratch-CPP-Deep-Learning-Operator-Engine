#pragma once
#include <vector>
#include <numeric>
#include <stdexcept>
#include <iostream>

class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;

    Tensor() = default;

    explicit Tensor(const std::vector<int>& dims, float init_val = 0.0f) 
        : shape(dims) {
        size_t total_elements = total_size();
        data.assign(total_elements, init_val);
    }

    size_t total_size() const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    }

    // 4D element accessor: [N, C, H, W]
    inline float& at4D(int n, int c, int h, int w) {
        size_t idx = ((static_cast<size_t>(n) * shape[1] + c) * shape[2] + h) * shape[3] + w;
        return data[idx];
    }

    inline const float& at4D(int n, int c, int h, int w) const {
        size_t idx = ((static_cast<size_t>(n) * shape[1] + c) * shape[2] + h) * shape[3] + w;
        return data[idx];
    }

    // 2D element accessor: [Row, Col]
    inline float& at2D(int r, int c) {
        return data[static_cast<size_t>(r) * shape[1] + c];
    }

    inline const float& at2D(int r, int c) const {
        return data[static_cast<size_t>(r) * shape[1] + c];
    }

    // Flat index accessor
    inline float& operator[](size_t idx) { return data[idx]; }
    inline const float& operator[](size_t idx) const { return data[idx]; }
};