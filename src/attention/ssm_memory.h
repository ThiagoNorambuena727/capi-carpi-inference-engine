// Archivo: src/attention/ssm_memory.h
#pragma once

#include <cstddef>
#include <immintrin.h>
#include <stdexcept>
#include <new>
#include <algorithm>

namespace gcapi_carpi {

/**
 * @brief Gestiona el buffer de estado oculto recurrente de tamano fijo en State Space Models (SSM).
 *
 * La memoria se reserva una unica vez alineada a 32 bytes para operaciones SIMD AVX2.
 * Deshabilita copias implicitas (RAII) para garantizar complejidad espacial O(1) durante la inferencia.
 */
class SSMMemoryState {
public:
    /**
     * @brief Reserva e inicializa a 0.0f el vector de estado oculto alineado a 32 bytes.
     * @param d_model Dimension del estado oculto.
     * @throws std::bad_alloc Si falla la reserva de memoria alineada.
     */
    explicit SSMMemoryState(int d_model);

    /**
     * @brief Libera el buffer alineado si la instancia posee la propiedad del puntero.
     */
    ~SSMMemoryState();

    // Prohibir copias implicitas para evitar duplicacion accidental de buffers
    SSMMemoryState(const SSMMemoryState&) = delete;
    SSMMemoryState& operator=(const SSMMemoryState&) = delete;

    // Permitir semantica de movimiento (Move Semantics) sin costo de copia
    SSMMemoryState(SSMMemoryState&& other) noexcept;
    SSMMemoryState& operator=(SSMMemoryState&& other) noexcept;

    /**
     * @brief Reinicia todos los elementos del estado oculto a 0.0f.
     */
    void reset() noexcept;

    /**
     * @brief Retorna el puntero al buffer alineado a 32 bytes.
     */
    float* data() const noexcept { return state_h_; }

    /**
     * @brief Retorna la dimension del estado oculto.
     */
    int dim() const noexcept { return d_model_; }

private:
    float* state_h_;
    int d_model_;
    bool owner_;
};

} // namespace gcapi_carpi