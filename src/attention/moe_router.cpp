// Archivo: src/attention/moe_router.cpp
#include "moe_router.h"
#include "../avx/avx_ops.hpp"
#include "softmax.h"
#include <immintrin.h>
#include <algorithm>
#include <stdexcept>
#include <new>
#include <omp.h>
#include <cstdint>

namespace gcapi_carpi {

    struct ExpertScore {
        float probability;
        int32_t idx;
    };

} // namespace gcapi_carpi

GCAPI_EXPORT void gcapi_carpi::router_forward_avx2(const float* input_vector,
                                                   const float* router_weights,
                                                   int32_t num_experts,
                                                   int32_t d_model,
                                                   int32_t top_k,
                                                   int32_t* out_expert_indices,
                                                   float* out_expert_probabilities) {
    // 1. Validacion de punteros y rangos dimensionales en la frontera de ejecucion
    if (!input_vector || !router_weights || !out_expert_indices || !out_expert_probabilities) {
        throw std::runtime_error("router_forward_avx2: Null pointer detected in arguments.");
    }
    if (num_experts <= 0 || d_model <= 0 || top_k <= 0 || top_k > num_experts) {
        throw std::runtime_error("router_forward_avx2: Dimensional parameters out of valid range.");
    }

    // 2. Reserva de memoria alineada a 32 bytes para operaciones vectoriales AVX2
    float* expert_probabilities_buffer = static_cast<float*>(
        _mm_malloc(static_cast<size_t>(num_experts) * sizeof(float), 32)
    );
    gcapi_carpi::ExpertScore* scores = static_cast<gcapi_carpi::ExpertScore*>(
        _mm_malloc(static_cast<size_t>(num_experts) * sizeof(gcapi_carpi::ExpertScore), 32)
    );

    if (!expert_probabilities_buffer || !scores) {
        if (expert_probabilities_buffer) _mm_free(expert_probabilities_buffer);
        if (scores) _mm_free(scores);
        throw std::bad_alloc();
    }

    try {
        // 3. Proyeccion lineal vectorizada del router (M = 1, K = d_model, N = num_experts)
        gcapi_carpi::matrix_multiply(
            input_vector, router_weights, expert_probabilities_buffer, 1, d_model, num_experts
        );

        // 4. Normalizacion Softmax in-place vectorizada con AVX2
        gcapi_carpi::softmax_avx2(expert_probabilities_buffer, 1, num_experts);

        // 5. Construccion de pares (probabilidad, indice).
        // Solo se paraleliza si num_experts >= 1024 para evitar el overhead de sincronizacion de hilos.
        #pragma omp parallel for if(num_experts >= 1024)
        for (int32_t e = 0; e < num_experts; ++e) {
            scores[e].probability = expert_probabilities_buffer[e];
            scores[e].idx = e;
        }

        // 6. Seleccion Top-K descendente en tiempo O(N log K) mediante std::partial_sort
        std::partial_sort(
            scores,
            scores + top_k,
            scores + num_experts,
            [](const gcapi_carpi::ExpertScore& a, const gcapi_carpi::ExpertScore& b) {
                return a.probability > b.probability;
            }
        );

        // 7. Escritura de los Top-K expertos seleccionados en los buffers de salida
        for (int32_t k = 0; k < top_k; ++k) {
            out_expert_probabilities[k] = scores[k].probability;
            out_expert_indices[k] = scores[k].idx;
        }
    } catch (...) {
        _mm_free(expert_probabilities_buffer);
        _mm_free(scores);
        throw;
    }

    // 8. Liberacion determinista de buffers temporales alineados
    _mm_free(expert_probabilities_buffer);
    _mm_free(scores);
}