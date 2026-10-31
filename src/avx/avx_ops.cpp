// Archivo: src/avx/avx_ops.cpp
#include "avx_ops.hpp"
#include "../utils/transpose.h"
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <new>

namespace {

inline float horizontal_sum_m256(__m256 v) {
    __m128 v128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    __m128 v64  = _mm_add_ps(v128, _mm_shuffle_ps(v128, v128, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128 v32  = _mm_add_ps(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(v32);
}

} // namespace

extern "C" {

GCAPI_EXPORT void gcapi_carpi::matrix_multiply(const float* A, const float* B, float* C, int M, int K, int N) {
    if (!A || !B || !C || M <= 0 || K <= 0 || N <= 0) return;

    const int K_aligned = (K / 8) * 8;

    // =========================================================================
    // FAST-PATH ZERO-ALLOCATION PARA INFERENCIA AUTORREGRESIVA (M == 1)
    // Evita reservar memoria con _mm_malloc y transponer B en cada token.
    // El vector A (d_model) permanece en Cache L1 mientras los hilos de OpenMP
    // calculan productos punto FMA sobre las filas contiguas de B.
    // =========================================================================
    if (M == 1) {
        #pragma omp parallel for
        for (int j = 0; j < N; ++j) {
            const float* row_B = &B[static_cast<size_t>(j) * K];
            __m256 sum_vec = _mm256_setzero_ps();

            for (int k = 0; k < K_aligned; k += 8) {
                __m256 a_v = _mm256_loadu_ps(&A[k]);
                __m256 b_v = _mm256_loadu_ps(&row_B[k]);
                sum_vec = _mm256_fmadd_ps(a_v, b_v, sum_vec);
            }

            float c_val = horizontal_sum_m256(sum_vec);
            for (int k = K_aligned; k < K; ++k) {
                c_val += A[k] * row_B[k];
            }
            C[j] = c_val;
        }
        return;
    }

    // =========================================================================
    // CASO GENERAL PARA PROCESAMIENTO POR LOTES (M > 1)
    // Transpone B (K x N -> N x K) para garantizar accesos secuenciales a memoria
    // durante el bucle interno vectorizado.
    // =========================================================================
    float* B_T = static_cast<float*>(
        _mm_malloc(static_cast<size_t>(N) * K * sizeof(float), 32)
    );
    if (!B_T) throw std::bad_alloc();

    #pragma omp parallel for
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            B_T[static_cast<size_t>(j) * K + k] = B[static_cast<size_t>(k) * N + j];
        }
    }

    // Se aplana el espacio bidimensional M * N en un unico bucle para mantener
    // compatibilidad total con OpenMP 2.0 en compiladores de Windows.
    const long long total_cells = static_cast<long long>(M) * N;

    #pragma omp parallel for
    for (long long idx = 0; idx < total_cells; ++idx) {
        const int i = static_cast<int>(idx / N);
        const int j = static_cast<int>(idx % N);
        const float* row_A   = &A[static_cast<size_t>(i) * K];
        const float* row_B_T = &B_T[static_cast<size_t>(j) * K];

        __m256 sum_vec = _mm256_setzero_ps();
        for (int k = 0; k < K_aligned; k += 8) {
            __m256 a_v = _mm256_loadu_ps(&row_A[k]);
            __m256 b_v = _mm256_loadu_ps(&row_B_T[k]);
            sum_vec = _mm256_fmadd_ps(a_v, b_v, sum_vec);
        }

        float c_val = horizontal_sum_m256(sum_vec);
        for (int k = K_aligned; k < K; ++k) {
            c_val += row_A[k] * row_B_T[k];
        }
        C[static_cast<size_t>(i) * N + j] = c_val;
    }

    _mm_free(B_T);
}

GCAPI_EXPORT void gcapi_carpi::matrix_multiply_backward(const float* A, const float* B, const float* dC,
                                                        float* dA, float* dB,
                                                        int M, int K, int N) {
    if (!A || !B || !dC || !dA || !dB || M <= 0 || K <= 0 || N <= 0) return;

    float* B_T_data = static_cast<float*>(
        _mm_malloc(static_cast<size_t>(N) * K * sizeof(float), 32)
    );
    if (!B_T_data) throw std::bad_alloc();

    try {
        transpose_matrix(B, B_T_data, K, N);
        gcapi_carpi::matrix_multiply(dC, B_T_data, dA, M, N, K);
    } catch (...) {
        _mm_free(B_T_data);
        throw;
    }
    _mm_free(B_T_data);

    float* A_T_data = static_cast<float*>(
        _mm_malloc(static_cast<size_t>(K) * M * sizeof(float), 32)
    );
    if (!A_T_data) throw std::bad_alloc();

    try {
        transpose_matrix(A, A_T_data, M, K);
        gcapi_carpi::matrix_multiply(A_T_data, dC, dB, K, M, N);
    } catch (...) {
        _mm_free(A_T_data);
        throw;
    }
    _mm_free(A_T_data);
}

} // extern "C"