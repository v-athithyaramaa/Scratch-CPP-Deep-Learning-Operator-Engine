#include "io_utils.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>

namespace IOUtils {

bool loadBinary(const std::string& filepath, Tensor& tensor) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[!] Failed to open file: " << filepath << std::endl;
        return false;
    }
    
    size_t elements = tensor.total_size();
    file.read(reinterpret_cast<char*>(tensor.data.data()), elements * sizeof(float));
    
    if (!file) {
        std::cerr << "[!] Incomplete binary read for: " << filepath << std::endl;
        return false;
    }
    return true;
}

bool saveBinary(const std::string& filepath, const Tensor& tensor) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[!] Failed to open file for write: " << filepath << std::endl;
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(tensor.data.data()), tensor.data.size() * sizeof(float));
    return true;
}

ComparisonResult compareTensors(
    const Tensor& calc, 
    const Tensor& ref, 
    float abs_tol, 
    float rel_tol
) {
    ComparisonResult res{true, 0.0f, 0.0f, 0};
    
    if (calc.total_size() != ref.total_size()) {
        std::cerr << "[!] Tensor size mismatch: " << calc.total_size() 
                  << " vs " << ref.total_size() << std::endl;
        res.pass = false;
        return res;
    }

    for (size_t i = 0; i < calc.total_size(); ++i) {
        float a = calc[i];
        float b = ref[i];
        float abs_diff = std::fabs(a - b);
        float rel_diff = abs_diff / (std::fabs(b) + 1e-7f);

        res.max_abs_diff = std::max(res.max_abs_diff, abs_diff);
        res.max_rel_diff = std::max(res.max_rel_diff, rel_diff);

        // Mismatch flag triggers only if BOTH tolerances fail
        if (abs_diff > abs_tol && rel_diff > rel_tol) {
            res.pass = false;
            res.mismatch_count++;
        }
    }
    return res;
}

}