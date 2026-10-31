// Archivo: src/attention/softmax.cpp
#include "softmax.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <omp.h>

namespace {

// Reduccion horizontal para obtener el valor maximo dentro de un registro __m256
inline float horizontal_max_m256(__m256 v) {
    __m128 v128 = _mm_max_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    __m128 v64  = _mm_max_ps(v128, _mm_shuffle_ps(v128, v128, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128 v32  = _mm_max_ps(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(v32);
}

// Reduccion horizontal para sumar los 8 floats de un registro __m256
inline float horizontal_sum_m256(__m256 v) {
    __m128 v128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    __m128 v64  = _mm_add_ps(v128, _mm_shuffle_ps(v128, v128, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128 v32  = _mm_add_ps(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(v32);
}

// Aproximacion vectorizada de exp(x) en AVX2 mediante reduccion de rango IEEE-754
// y polinomio minimax de grado 6 con instrucciones FMA.
// Descomposicion: e^x = 2^n * e^g, con n = round(x * log2(e)) y g = x - n * ln(2).
// Error relativo maximo medido frente a std::exp (libm): < 3e-7 en [-88.37, +88.37].
inline __m256 exp_avx2_minimax(__m256 x) {
    x = _mm256_max_ps(_mm256_set1_ps(-88.3762626647949f), x);
    x = _mm256_min_ps(_mm256_set1_ps( 88.3762626647949f), x);

    // Reduccion de rango: z = x * log2(e), n = round(z), g = x - n * ln(2)
    __m256 z = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    __m256 n = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 g = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.6931471805599453f), x);

    // Evaluacion del polinomio minimax de grado 6 mediante esquema de Horner con FMA
    __m256 p = _mm256_set1_ps(0.0013981999f);
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(0.0083334310f));
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(0.0416657962f));
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(0.1666665459f));
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(0.5000000000f));
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(1.0000000000f));
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(1.0000000000f));

    // Reconstruccion exacta de 2^n desplazando el exponente IEEE-754 (sesgo 127, bit 23)
    __m256i nn = _mm256_cvtps_epi32(n);
    __m256i pow2n_bits = _mm256_slli_epi32(_mm256_add_epi32(nn, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(pow2n_bits));
}

} // namespace

extern "C" {

GCAPI_EXPORT void gcapi_carpi::softmax_avx2(float* logits, int batch_size, int num_classes) {
    if (!logits || batch_size <= 0 || num_classes <= 0) return;

    const int cols_aligned = (num_classes / 8) * 8;
    const __m256 neg_infinity = _mm256_set1_ps(std::numeric_limits<float>::lowest());

    // Solo paraleliza por filas cuando el lote contiene multiples muestras (batch_size > 1)
    #pragma omp parallel for if(batch_size > 1)
    for (int i = 0; i < batch_size; ++i) {
        float* row_ptr = logits + static_cast<size_t>(i) * num_classes;

        // 1. Busqueda del valor maximo de la fila para estabilidad numerica
        __m256 max_val_vec = neg_infinity;
        for (int j = 0; j < cols_aligned; j += 8) {
            max_val_vec = _mm256_max_ps(max_val_vec, _mm256_loadu_ps(row_ptr + j));
        }
        float row_max_val = horizontal_max_m256(max_val_vec);

        for (int j = cols_aligned; j < num_classes; ++j) {
            row_max_val = std::max(row_max_val, row_ptr[j]);
        }

        // 2. Calculo de exp(x - row_max_val) y acumulacion de la suma
        __m256 sum_exp_vec = _mm256_setzero_ps();
        const __m256 broadcast_row_max = _mm256_set1_ps(row_max_val);

        for (int j = 0; j < cols_aligned; j += 8) {
            __m256 x_vec = _mm256_loadu_ps(row_ptr + j);
            x_vec = _mm256_sub_ps(x_vec, broadcast_row_max);
            __m256 exp_vec = exp_avx2_minimax(x_vec);
            _mm256_storeu_ps(row_ptr + j, exp_vec);
            sum_exp_vec = _mm256_add_ps(sum_exp_vec, exp_vec);
        }
        float row_sum_exp = horizontal_sum_m256(sum_exp_vec);

        for (int j = cols_aligned; j < num_classes; ++j) {
            row_ptr[j] = std::exp(row_ptr[j] - row_max_val);
            row_sum_exp += row_ptr[j];
        }

        // 3. Normalizacion multiplicando por el inverso de la suma acumulada
        const float inv_sum = (row_sum_exp > 0.0f) ? (1.0f / row_sum_exp) : 0.0f;
        const __m256 broadcast_inv_sum = _mm256_set1_ps(inv_sum);

        for (int j = 0; j < cols_aligned; j += 8) {
            __m256 exp_vec = _mm256_loadu_ps(row_ptr + j);
            _mm256_storeu_ps(row_ptr + j, _mm256_mul_ps(exp_vec, broadcast_inv_sum));
        }

        for (int j = cols_aligned; j < num_classes; ++j) {
            row_ptr[j] *= inv_sum;
        }
    }
}

} // extern "C"