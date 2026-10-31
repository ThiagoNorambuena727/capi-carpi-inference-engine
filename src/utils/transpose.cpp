// Archivo: src/utils/transpose.cpp
#include "transpose.h"
#include <omp.h>

extern "C" {

GCAPI_EXPORT void transpose_matrix(const float* input, float* output, int rows, int cols) {
    if (!input || !output || rows <= 0 || cols <= 0) return;

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[static_cast<size_t>(j) * rows + i] = input[static_cast<size_t>(i) * cols + j];
        }
    }
}

} // extern "C"