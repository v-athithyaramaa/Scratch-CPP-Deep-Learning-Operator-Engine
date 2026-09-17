#pragma once

#include "tensor.hpp"
#include <string>

namespace IOUtils
{

    // Reads raw contiguous float32 data from disk into a Tensor
    bool loadBinary(const std::string &filepath, Tensor &tensor);

    // Saves Tensor data to a binary file
    bool saveBinary(const std::string &filepath, const Tensor &tensor);

    struct ComparisonResult
    {
        bool pass;
        float max_abs_diff;
        float max_rel_diff;
        size_t mismatch_count;
    };

    // Evaluates both absolute and relative difference to eliminate float noise
    ComparisonResult compareTensors(
        const Tensor &calculated,
        const Tensor &reference,
        float abs_tol = 1e-4f,
        float rel_tol = 1e-3f);

} 