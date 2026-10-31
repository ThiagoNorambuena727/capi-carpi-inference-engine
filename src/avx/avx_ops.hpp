// Archivo: src/avx/avx_ops.hpp
#pragma once

#include <cstddef>

#ifdef _WIN32
    #define GCAPI_EXPORT __declspec(dllexport)
#else
    #define GCAPI_EXPORT
#endif

namespace gcapi_carpi {

extern "C" {
    /**
     * @brief Ejecuta la multiplicacion matricial vectorizada con AVX2, FMA y OpenMP.
     *
     * Incluye un Fast-Path libre de asignaciones dinamicas (Zero-Allocation) para
     * inferencia autorregresiva (M == 1), manteniendo el vector de entrada en Cache L1
     * mientras lee las filas contiguas de B directamente desde el mapeo mmap.
     *
     * @param A Puntero a la matriz/vector de entrada A (dimension M x K).
     * @param B Puntero a la matriz de pesos B (almacenada como N x K para M == 1, o K x N para M > 1).
     * @param C Puntero al buffer de salida C (dimension M x N).
     * @param M Numero de filas de A y C (M == 1 durante inferencia token a token).
     * @param K Dimension interna compartida (d_model).
     * @param N Numero de columnas de salida (num_experts, d_model o vocab_size).
     */
    GCAPI_EXPORT void matrix_multiply(const float* A, const float* B, float* C, int M, int K, int N);

    /**
     * @brief Calcula los gradientes con respecto a la entrada (dA) y a los pesos (dB).
     *
     * @param A Matriz de entrada original (M x K).
     * @param B Matriz de pesos original (K x N).
     * @param dC Gradiente entrante respecto a la salida C (M x N).
     * @param dA Buffer de salida para el gradiente respecto a A (M x K).
     * @param dB Buffer de salida para el gradiente respecto a B (K x N).
     * @param M Numero de filas de A y dC.
     * @param K Numero de columnas de A y filas de B.
     * @param N Numero de columnas de B y dC.
     */
    GCAPI_EXPORT void matrix_multiply_backward(const float* A, const float* B, const float* dC,
                                               float* dA, float* dB,
                                               int M, int K, int N);
}

} // namespace gcapi_carpi