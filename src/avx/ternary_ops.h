// Archivo: src/avx/ternary_ops.h
#pragma once

#include <cstddef>
#include <cstdint>
#include "../tensor/ternary_tensor.h"

#ifdef _WIN32
    #define GCAPI_EXPORT __declspec(dllexport)
#else
    #define GCAPI_EXPORT
#endif

extern "C" {

/**
 * @brief Ejecuta una proyeccion lineal ternaria (1.58-bit) vectorizada con AVX2 sin multiplicaciones.
 *
 * Los pesos ternarios {-1, 0, +1} se leen empaquetados a 2 bits por peso (4 pesos por byte)
 * desde bloques contiguos TernaryWeightBlock. En lugar de utilizar instrucciones de
 * multiplicacion en punto flotante (_mm256_mul_ps), desempaqueta los pares de bits en
 * registros AVX2 mediante desplazamientos vectoriales (_mm256_srlv_epi32) y aplica
 * mascaras condicionales (_mm256_blendv_ps) para acumular sumas y restas puras.
 *
 * @param input_vec Vector de entrada en punto flotante (dimension: input_dim).
 * @param weight_matrix Matriz de bloques ternarios (dimension: output_dim * blocks_per_row).
 * @param output_vec Vector de salida en punto flotante (dimension: output_dim).
 * @param input_dim Cantidad de columnas de entrada.
 * @param output_dim Cantidad de neuronas (filas) de salida.
 */
GCAPI_EXPORT void ternary_linear_forward(const float* input_vec,
                                         const gcapi_carpi::TernaryWeightBlock* weight_matrix,
                                         float* output_vec,
                                         int input_dim,
                                         int output_dim);

/**
 * @brief Multiplicacion matriz-vector sobre un flujo de bytes empaquetados (4 pesos por byte).
 *
 * Variante compacta para tensores binarios planos donde cada par de bits codifica
 * directamente el peso desplazado en +1 (00 -> -1, 01 -> 0, 10 -> +1).
 *
 * @param packed_matrix Puntero a la matriz empaquetada en bytes (filas * (d_model / 4)).
 * @param input_vec Vector de entrada float32 (dimension: d_model).
 * @param output_vec Vector de salida float32 (dimension: rows).
 * @param rows Numero de filas de salida.
 * @param d_model Dimension del vector de entrada (multiplo de 4).
 */
GCAPI_EXPORT void packed_ternary_matmul(const uint8_t* packed_matrix,
                                        const float* input_vec,
                                        float* output_vec,
                                        int rows,
                                        int d_model);

} // extern "C"