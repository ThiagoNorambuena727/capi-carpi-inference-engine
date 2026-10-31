// Archivo: src/attention/ssm_forward.h
#pragma once

#include <cstdint>
#include <cstddef>
#include "../tensor/ternary_tensor.h"

#ifdef _WIN32
    #define GCAPI_EXPORT __declspec(dllexport)
#else
    #define GCAPI_EXPORT
#endif

extern "C" {
    /**
     * @brief Ejecuta el paso de actualizacion recurrente hacia adelante del State Space Model (SSM).
     *
     * Calcula las proyecciones de estado utilizando bloques de pesos ternarios empaquetados (1.58-bit)
     * y actualiza el estado oculto mediante funciones de activacion vectorizadas en AVX2:
     * h_out = tanh(W_decay * h_prev) + sigmoid(W_receptivity * input_proj) * input_original
     *
     * @param input_t_original Vector de entrada del token actual (float32, dimension d_model).
     * @param h_prev_data Estado oculto del paso temporal previo (float32, dimension d_state).
     * @param h_out_data Buffer de salida para el nuevo estado oculto (float32, dimension d_state).
     * @param decay_weights Pesos ternarios empaquetados para la proyeccion de decaimiento.
     * @param receptivity_weights Pesos ternarios empaquetados para la compuerta de receptividad.
     * @param input_t_projected Vector de entrada proyectado linealmente (float32, dimension d_model).
     * @param d_model Dimension de entrada del modelo.
     * @param d_state Dimension del estado recurrente interno.
     */
    GCAPI_EXPORT void ssm_forward_avx2(const float* input_t_original,
                                       const float* h_prev_data,
                                       float* h_out_data,
                                       const gcapi_carpi::TernaryWeightBlock* decay_weights,
                                       const gcapi_carpi::TernaryWeightBlock* receptivity_weights,
                                       const float* input_t_projected,
                                       int32_t d_model,
                                       int32_t d_state);
}