// Archivo: src/attention/ssm_forward.cpp
#include "ssm_forward.h"
#include "../avx/ternary_ops.h"
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>
#include <new>
#include <omp.h>

namespace {

// =============================================================================
// NUCLEOS MATEMATICOS AVX2 (Reduccion de Rango IEEE-754 + Minimax Grado 6)
// =============================================================================

inline __m256 exp_avx2(__m256 x) {
    // 1. Limitar rango para evitar overflow/underflow en punto flotante de 32 bits
    x = _mm256_max_ps(_mm256_set1_ps(-88.3762626647949f), x);
    x = _mm256_min_ps(_mm256_set1_ps( 88.3762626647949f), x);

    // 2. Reduccion de rango: z = x * log2(e), n = round(z), g = x - n * ln(2)
    __m256 z = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    __m256 n = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 g = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.6931471805599453f), x);

    // 3. Polinomio minimax de grado 6 para e^g en [-ln(2)/2, +ln(2)/2]
    __m256 p = _mm256_set1_ps(0.0013981999f);
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(0.0083334310f));
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(0.0416657962f));
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(0.1666665459f));
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(0.5000000000f));
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(1.0000000000f));
    p = _mm256_fmadd_ps(p, g, _mm256_set1_ps(1.0000000000f));

    // 4. Reconstruccion de 2^n sobre el campo de exponente IEEE-754
    __m256i nn = _mm256_cvtps_epi32(n);
    __m256i p_int = _mm256_slli_epi32(_mm256_add_epi32(nn, _mm256_set1_epi32(127)), 23);
    __m256 pow2n = _mm256_castsi256_ps(p_int);

    return _mm256_mul_ps(p, pow2n);
}

// Funcion de activacion Sigmoide vectorizada: 1.0f / (1.0f + exp(-x))
inline __m256 sigmoid_avx2(__m256 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_nx = exp_avx2(neg_x);
    __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_nx));
}

// Funcion de activacion Tanh vectorizada: (exp(2x) - 1.0f) / (exp(2x) + 1.0f)
inline __m256 tanh_avx2(__m256 x) {
    __m256 x2 = _mm256_mul_ps(x, _mm256_set1_ps(2.0f));
    __m256 exp2x = exp_avx2(x2);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 num = _mm256_sub_ps(exp2x, one);
    __m256 den = _mm256_add_ps(exp2x, one);
    return _mm256_div_ps(num, den);
}

} // namespace

extern "C" {

GCAPI_EXPORT void ssm_forward_avx2(const float* input_t_original,
                                   const float* h_prev_data,
                                   float* h_out_data,
                                   const gcapi_carpi::TernaryWeightBlock* decay_weights,
                                   const gcapi_carpi::TernaryWeightBlock* receptivity_weights,
                                   const float* input_t_projected,
                                   int32_t d_model,
                                   int32_t d_state) {
    // 1. Validacion de punteros en la frontera ABI
    if (!input_t_original || !h_prev_data || !h_out_data ||
        !decay_weights || !receptivity_weights || !input_t_projected) {
        throw std::runtime_error("ssm_forward_avx2: Null pointer detected in critical arguments.");
    }
    if (d_model <= 0 || d_state <= 0) return;

    // 2. Reserva de buffers intermedios alineados a 32 bytes para cargas _mm256_load_ps
    float* temp_decay = static_cast<float*>(
        _mm_malloc(static_cast<size_t>(d_state) * sizeof(float), 32)
    );
    float* temp_receptivity = static_cast<float*>(
        _mm_malloc(static_cast<size_t>(d_state) * sizeof(float), 32)
    );

    if (!temp_decay || !temp_receptivity) {
        if (temp_decay) _mm_free(temp_decay);
        if (temp_receptivity) _mm_free(temp_receptivity);
        throw std::bad_alloc();
    }

    try {
        // 3. Proyecciones lineales ternarias (1.58-bit, sin multiplicaciones en punto flotante)
        ternary_linear_forward(h_prev_data, decay_weights, temp_decay, d_model, d_state);
        ternary_linear_forward(input_t_projected, receptivity_weights, temp_receptivity, d_model, d_state);

        // 4. Actualizacion de estado recurrente vectorizada con AVX2 + FMA:
        //    h_out = tanh(temp_decay) + sigmoid(temp_receptivity) * input_t_original
        const int32_t d_state_aligned = (d_state / 8) * 8;

        #pragma omp parallel for if(d_state >= 1024)
        for (int32_t i = 0; i < d_state_aligned; i += 8) {
            __m256 dec_vec = _mm256_load_ps(&temp_decay[i]);
            __m256 rec_vec = _mm256_load_ps(&temp_receptivity[i]);
            __m256 in_vec  = _mm256_loadu_ps(&input_t_original[i]);

            __m256 tanh_dec = tanh_avx2(dec_vec);
            __m256 sig_rec  = sigmoid_avx2(rec_vec);

            __m256 h_out_vec = _mm256_fmadd_ps(sig_rec, in_vec, tanh_dec);
            _mm256_storeu_ps(&h_out_data[i], h_out_vec);
        }

        // 5. Procesamiento de la cola residual manteniendo el mismo nucleo matematico
        for (int32_t i = d_state_aligned; i < d_state; ++i) {
            __m256 rec_vec = _mm256_set1_ps(temp_receptivity[i]);
            __m256 dec_vec = _mm256_set1_ps(temp_decay[i]);

            float sigmoid_rec = _mm_cvtss_f32(_mm256_castps256_ps128(sigmoid_avx2(rec_vec)));
            float tanh_dec    = _mm_cvtss_f32(_mm256_castps256_ps128(tanh_avx2(dec_vec)));

            h_out_data[i] = tanh_dec + (sigmoid_rec * input_t_original[i]);
        }
    } catch (...) {
        _mm_free(temp_decay);
        _mm_free(temp_receptivity);
        throw;
    }

    // 6. Liberacion determinista de memoria temporal
    _mm_free(temp_decay);
    _mm_free(temp_receptivity);
}

} // extern "C"