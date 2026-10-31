// Archivo: src/tensor/mmap_loader.cpp
#include "mmap_loader.h"
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace gcapi_carpi {

MmapWeightsLoader::MmapWeightsLoader(const std::string& file_path)
    : file_path_(file_path),
      file_size_(0),
      mapped_data_(nullptr)
#ifdef _WIN32
    , file_handle_(INVALID_HANDLE_VALUE),
      mapping_handle_(NULL)
#else
    , fd_(-1)
#endif
{
#ifdef _WIN32
    // Apertura en modo de solo lectura (GENERIC_READ) para habilitar optimizaciones del Page Cache
    file_handle_ = CreateFileA(
        file_path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("MmapWeightsLoader: Failed to open model file: " + file_path);
    }

    LARGE_INTEGER size{};
    if (!GetFileSizeEx(file_handle_, &size) || size.QuadPart <= 0) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("MmapWeightsLoader: Invalid or empty model file.");
    }
    file_size_ = static_cast<size_t>(size.QuadPart);

    // PAGE_READONLY asegura que el sistema operativo comparta paginas fisicas sin bloqueo
    mapping_handle_ = CreateFileMappingA(file_handle_, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mapping_handle_ == NULL) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("MmapWeightsLoader: CreateFileMappingA failed.");
    }

    // FILE_MAP_READ proyecta el archivo completo en el espacio de direcciones virtuales
    mapped_data_ = static_cast<const uint8_t*>(MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0));
    if (mapped_data_ == nullptr) {
        CloseHandle(mapping_handle_);
        CloseHandle(file_handle_);
        mapping_handle_ = NULL;
        file_handle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("MmapWeightsLoader: MapViewOfFile failed (insufficient virtual address space).");
    }
#else
    fd_ = open(file_path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("MmapWeightsLoader: POSIX open failed for file: " + file_path);
    }

    struct stat sb{};
    if (fstat(fd_, &sb) == -1 || sb.st_size <= 0) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("MmapWeightsLoader: POSIX fstat failed or file is empty.");
    }
    file_size_ = static_cast<size_t>(sb.st_size);

    void* map_ptr = mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ptr == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("MmapWeightsLoader: POSIX mmap failed.");
    }
    mapped_data_ = static_cast<const uint8_t*>(map_ptr);
#endif
}

MmapWeightsLoader::~MmapWeightsLoader() {
#ifdef _WIN32
    if (mapped_data_) {
        UnmapViewOfFile(mapped_data_);
        mapped_data_ = nullptr;
    }
    if (mapping_handle_ != NULL) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = NULL;
    }
    if (file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (mapped_data_ && mapped_data_ != reinterpret_cast<const uint8_t*>(MAP_FAILED)) {
        munmap(const_cast<uint8_t*>(mapped_data_), file_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
#endif
}

const uint8_t* MmapWeightsLoader::get_pointer_at_offset(size_t offset) const {
    if (offset >= file_size_) {
        throw std::out_of_range("MmapWeightsLoader: Requested byte offset exceeds mapped file size.");
    }
    return mapped_data_ + offset;
}

} // namespace gcapi_carpi