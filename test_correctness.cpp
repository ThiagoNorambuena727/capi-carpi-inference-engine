// Archivo: test_correctness.cpp
// Verifica la exactitud numerica de los kernels AVX2 contra referencias escalares puras.
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include "src/attention/softmax.h"
#include "src/avx/avx_ops.hpp"
#include "src/avx/ternary_ops.h"

using namespace gcapi_carpi;

bool test_softmax() {
    const int N = 67; // Dimension no multiplo de 8 para probar bucle AVX2 + cola escalar
    std::vector<float> avx_data(N), ref_data(N);
    for (int i = 0; i < N; ++i) {
        float val = std::sin(i * 0.37f) * 12.0f;
        avx_data[i] = val;
        ref_data[i] = val;
    }

    float max_v = ref_data[0];
    for (float v : ref_data) if (v > max_v) max_v = v;
    float sum = 0.0f;
    for (float& v : ref_data) { v = std::exp(v - max_v); sum += v; }
    for (float& v : ref_data) v /= sum;

    softmax_avx2(avx_data.data(), 1, N);

    float max_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        max_err = std::max(max_err, std::abs(avx_data[i] - ref_data[i]));
    }
    std::cout << "[TEST 1] Softmax AVX2 vs std::exp        | Max Error: " << std::scientific << max_err
              << (max_err < 1e-5f ? " -> [PASS]\n" : " -> [FAIL]\n");
    return max_err < 1e-5f;
}

bool test_matmul_fastpath() {
    const int K = 136, N = 64;
    std::vector<float> A(K), B(N * K), C_avx(N, 0.0f), C_ref(N, 0.0f);
    for (int i = 0; i < K; ++i) A[i] = (i % 7 - 3) * 0.15f;
    for (int i = 0; i < N * K; ++i) B[i] = (i % 11 - 5) * 0.08f;

    for (int j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[k] * B[j * K + k];
        C_ref[j] = acc;
    }

    matrix_multiply(A.data(), B.data(), C_avx.data(), 1, K, N);

    float max_err = 0.0f;
    for (int j = 0; j < N; ++j) {
        max_err = std::max(max_err, std::abs(C_avx[j] - C_ref[j]));
    }
    std::cout << "[TEST 2] MatMul AVX2 Fast-Path (M=1)     | Max Error: " << std::scientific << max_err
              << (max_err < 1e-4f ? " -> [PASS]\n" : " -> [FAIL]\n");
    return max_err < 1e-4f;
}

bool test_ternary_linear() {
    const int in_dim = 64, out_dim = 16;
    const int blocks_per_row = in_dim / 32;
    std::vector<float> in_vec(in_dim), out_avx(out_dim, 0.0f), out_ref(out_dim, 0.0f);
    std::vector<TernaryWeightBlock> weights(out_dim * blocks_per_row);

    for (int i = 0; i < in_dim; ++i) in_vec[i] = (i + 1) * 0.05f;

    for (auto& blk : weights) {
        for (int b = 0; b < 8; ++b) blk.packed_weights[b] = static_cast<uint8_t>((b * 37 + 19) & 0xAA);
    }

    for (int r = 0; r < out_dim; ++r) {
        float acc = 0.0f;
        for (int c = 0; c < in_dim; ++c) {
            int blk_idx = c / 32;
            int byte_idx = (c % 32) / 4;
            int pair_idx = c % 4;
            uint8_t byte_val = weights[r * blocks_per_row + blk_idx].packed_weights[byte_idx];
            int8_t w = map_2bit_to_ternary_value(get_2bit_value(byte_val, pair_idx));
            acc += w * in_vec[c];
        }
        out_ref[r] = acc;
    }

    ternary_linear_forward(in_vec.data(), weights.data(), out_avx.data(), in_dim, out_dim);

    float max_err = 0.0f;
    for (int r = 0; r < out_dim; ++r) {
        max_err = std::max(max_err, std::abs(out_avx[r] - out_ref[r]));
    }
    std::cout << "[TEST 3] Ternary 1.58b AVX2 (No-Mul)     | Max Error: " << std::scientific << max_err
              << (max_err < 1e-5f ? " -> [PASS]\n" : " -> [FAIL]\n");
    return max_err < 1e-5f;
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  SUITE DE VERIFICACION MATEMATICA (AVX2 SIMD vs ESCALAR)\n";
    std::cout << "=================================================================\n";
    bool ok = test_softmax() && test_matmul_fastpath() && test_ternary_linear();
    std::cout << "=================================================================\n";
    return ok ? 0 : 1;
}
