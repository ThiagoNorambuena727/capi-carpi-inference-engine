// Archivo: src/attention/moe_router.h
#pragma once

#include <cstdint>

#ifdef _WIN32
    #define GCAPI_EXPORT __declspec(dllexport)
#else
    #define GCAPI_EXPORT
#endif

namespace gcapi_carpi {

extern "C" {
    /**
     * @brief Calcula las puntuaciones del enrutador Sparse MoE, aplica Softmax y selecciona los Top-K expertos.
     *
     * @param input_vector Vector de entrada del token actual (float32, dimension d_model).
     * @param router_weights Matriz de pesos del enrutador (float32, dimension num_experts * d_model).
     * @param num_experts Cantidad total de expertos disponibles en la capa.
     * @param d_model Dimension del espacio oculto (embedding).
     * @param top_k Cantidad de expertos a activar por token (1 <= top_k <= num_experts).
     * @param out_expert_indices Buffer de salida con los indices de los Top-K expertos seleccionados.
     * @param out_expert_probabilities Buffer de salida con las probabilidades normalizadas de los Top-K expertos.
     */
    GCAPI_EXPORT void router_forward_avx2(const float* input_vector,
                                          const float* router_weights,
                                          int32_t num_experts,
                                          int32_t d_model,
                                          int32_t top_k,
                                          int32_t* out_expert_indices,
                                          float* out_expert_probabilities);
}

} // namespace gcapi_carpi