// Archivo: src/model/model_header.h
#pragma once

#include <cstdint>

namespace gcapi_carpi {

#pragma pack(push, 1) // Asegura que C++ no agregue bytes de relleno (padding)

/**
 * @brief Estructura de la cabecera del archivo model.bin
 * ALINEACIÓN ESTRICTA: Debe coincidir exactamente con el struct.pack("<4i 5Q") de Python
 */
struct ModelHeader {
    int32_t d_model;                    // 4 bytes
    int32_t num_experts;                // 4 bytes
    int32_t top_k;                      // 4 bytes
    int32_t vocab_size;                 // 4 bytes
    uint64_t embedding_offset;          // 8 bytes
    uint64_t router_weights_offset;     // 8 bytes
    uint64_t lm_head_offset;            // 8 bytes
    uint64_t expert_weights_base_offset;// 8 bytes
    uint64_t expert_weight_block_size_bytes; // 8 bytes
};

#pragma pack(pop)

// Archivo: src/model/model_header.h
#pragma once

#include <cstdint>

namespace gcapi_carpi {

#pragma pack(push, 1)

/**
 * @brief Estructura binaria de la cabecera del modelo (56 bytes exactos).
 *
 * Define la topologia de la red y los desplazamientos (offsets) absolutos
 * en bytes dentro del archivo mapeado por mmap. Coincide bit a bit con el
 * formato struct.pack("<4i5Q") de la interfaz Python.
 */
struct ModelHeader {
    int32_t  d_model;                        // 4 bytes: Dimension del espacio oculto
    int32_t  num_experts;                    // 4 bytes: Cantidad total de expertos SSM
    int32_t  top_k;                          // 4 bytes: Expertos activos por token
    int32_t  vocab_size;                     // 4 bytes: Tamano del vocabulario
    uint64_t embedding_offset;               // 8 bytes: Offset de la tabla de embeddings
    uint64_t router_weights_offset;          // 8 bytes: Offset de la matriz del router MoE
    uint64_t lm_head_offset;                 // 8 bytes: Offset de la proyeccion de salida
    uint64_t expert_weights_base_offset;     // 8 bytes: Offset base del array de expertos
    uint64_t expert_weight_block_size_bytes; // 8 bytes: Tamano en bytes de cada experto
};

#pragma pack(pop)

static_assert(sizeof(ModelHeader) == 56,
              "ModelHeader must measure exactly 56 bytes (4 int32 + 5 uint64) without padding.");

} // namespace gcapi_carpi