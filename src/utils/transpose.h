// Archivo: src/utils/transpose.h
#pragma once

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
    #define GCAPI_EXPORT __declspec(dllexport)
#else
    #define GCAPI_EXPORT
#endif

extern "C" {

/**
 * @brief Transpone una matriz en punto flotante de dimensiones (rows x cols) a (cols x rows).
 *
 * Operacion paralelizada mediante OpenMP para acelerar la reorganizacion de memoria
 * en el paso de retropropagacion y multiplicacion por lotes.
 *
 * @param input Puntero a la matriz de entrada (dimension: rows * cols).
 * @param output Puntero al buffer de salida (dimension: cols * rows).
 * @param rows Numero de filas de la matriz de entrada.
 * @param cols Numero de columnas de la matriz de entrada.
 */
GCAPI_EXPORT void transpose_matrix(const float* input, float* output, int rows, int cols);

} // extern "C"