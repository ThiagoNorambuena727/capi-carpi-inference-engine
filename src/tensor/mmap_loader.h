// Archivo: src/tensor/mmap_loader.h
#pragma once

#include <string>
#include <cstdint>
#include <cstddef>

namespace gcapi_carpi {

/**
 * @brief Cargador Zero-Copy multiplataforma basado en memoria virtual (mmap / FileMapping).
 *
 * Mapea el archivo binario de pesos directamente al espacio de direcciones virtuales
 * del proceso en modo de solo lectura, permitiendo acceder a tensores desde unidades
 * NVMe/SSD sin realizar copias intermedias en el heap.
 */
class MmapWeightsLoader {
public:
    explicit MmapWeightsLoader(const std::string& file_path);
    ~MmapWeightsLoader();

    // Deshabilitar copia y movimiento para evitar liberaciones dobles del descriptor del SO
    MmapWeightsLoader(const MmapWeightsLoader&) = delete;
    MmapWeightsLoader& operator=(const MmapWeightsLoader&) = delete;
    MmapWeightsLoader(MmapWeightsLoader&&) = delete;
    MmapWeightsLoader& operator=(MmapWeightsLoader&&) = delete;

    /**
     * @brief Retorna la direccion base del archivo mapeado en memoria virtual.
     */
    const uint8_t* get_base_address() const noexcept { return mapped_data_; }

    /**
     * @brief Retorna un puntero constante desplazado dentro del archivo con verificacion de limites.
     * @param offset Desplazamiento en bytes desde el inicio del archivo.
     * @throws std::out_of_range Si el offset excede el tamano fisico del archivo.
     */
    const uint8_t* get_pointer_at_offset(size_t offset) const;

    /**
     * @brief Retorna el tamano total en bytes del archivo mapeado.
     */
    size_t size() const noexcept { return file_size_; }

private:
    std::string file_path_;
    size_t file_size_;
    const uint8_t* mapped_data_;

#ifdef _WIN32
    void* file_handle_;
    void* mapping_handle_;
#else
    int fd_;
#endif
};

} // namespace gcapi_carpi