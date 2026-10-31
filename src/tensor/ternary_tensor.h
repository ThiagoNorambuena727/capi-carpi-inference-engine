// Archivo: src/tensor/ternary_tensor.h
#pragma once

#include <cstdint>
#include <cstddef>

namespace gcapi_carpi {

/**
 * @brief Cantidad de pesos ternarios {-1, 0, +1} almacenados por bloque (2 bits por peso).
 */
static constexpr size_t BLOCK_SIZE_WEIGHTS = 32;

/**
 * @brief Cantidad de bytes por bloque (32 pesos * 2 bits / 8 bits = 8 bytes exactos).
 */
static constexpr size_t BLOCK_SIZE_BYTES = (BLOCK_SIZE_WEIGHTS * 2) / 8;

/**
 * @brief Bloque compacto de 8 bytes que almacena 32 pesos ternarios empaquetados (1.58-bit).
 *
 * NOTA DE LAYOUT DE MEMORIA: Esta estructura no debe declarar alignas(32).
 * En C++, sizeof(T) es siempre multiplo de alignof(T); aplicar alignas(32) agregaria
 * 24 bytes de padding silencioso (llevando sizeof a 32 bytes en lugar de 8 bytes)
 * y desfasaria por un factor de 4x toda la aritmetica de punteros sobre el archivo .bin.
 */
struct TernaryWeightBlock {
    uint8_t packed_weights[BLOCK_SIZE_BYTES];
};

static_assert(sizeof(TernaryWeightBlock) == BLOCK_SIZE_BYTES,
              "TernaryWeightBlock must measure exactly 8 bytes to match the binary file stride.");

/**
 * @brief Extrae el par de bits (0..3) ubicado en la posicion bit_pair_index (0..3) dentro de un byte.
 */
inline uint8_t get_2bit_value(uint8_t byte, int bit_pair_index) noexcept {
    return (byte >> (bit_pair_index * 2)) & 0x03;
}

/**
 * @brief Decodifica un par de bits a su valor ternario {-1, 0, +1} mediante tabla de consulta O(1).
 *
 * Codificacion binaria:
 *   0b00 (0) -> -1
 *   0b01 (1) ->  0
 *   0b10 (2) -> +1
 *   0b11 (3) ->  0 (estado neutro de seguridad)
 */
inline int8_t map_2bit_to_ternary_value(uint8_t two_bits) noexcept {
    static constexpr int8_t lookup_table[4] = {-1, 0, 1, 0};
    return lookup_table[two_bits & 0x03];
}

} // namespace gcapi_carpi