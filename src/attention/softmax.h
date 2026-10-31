// Archivo: src/attention/softmax.h
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
     * @brief Aplica la funcion de activacion Softmax in-place vectorizada con AVX2.
     *
     * Transforma cada fila de logits en una distribucion de probabilidad valida.
     * Resta el valor maximo de cada fila antes de exponenciar para garantizar
     * estabilidad numerica estricta y prevenir desbordamientos (overflow).
     *
     * @param logits Puntero al buffer contiguo de logits (dimension: batch_size * num_classes).
     * @param batch_size Cantidad de filas independientes a procesar.
     * @param num_classes Cantidad de columnas (clases o expertos) por fila.
     */
    GCAPI_EXPORT void softmax_avx2(float* logits, int batch_size, int num_classes);
}

} // namespace gcapi_carpi