// Archivo: src/attention/ssm_memory.cpp
#include "ssm_memory.h"
#include <stdexcept>
#include <new>
#include <algorithm>

namespace gcapi_carpi {

SSMMemoryState::SSMMemoryState(int d_model)
    : state_h_(nullptr), d_model_(d_model), owner_(true) {
    if (d_model_ <= 0) {
        d_model_ = 0;
        return;
    }

    state_h_ = static_cast<float*>(
        _mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32)
    );
    if (!state_h_) {
        throw std::bad_alloc();
    }
    std::fill(state_h_, state_h_ + d_model_, 0.0f);
}

SSMMemoryState::~SSMMemoryState() {
    if (owner_ && state_h_) {
        _mm_free(state_h_);
        state_h_ = nullptr;
    }
}

SSMMemoryState::SSMMemoryState(SSMMemoryState&& other) noexcept
    : state_h_(other.state_h_), d_model_(other.d_model_), owner_(other.owner_) {
    other.state_h_ = nullptr;
    other.d_model_ = 0;
    other.owner_ = false;
}

SSMMemoryState& SSMMemoryState::operator=(SSMMemoryState&& other) noexcept {
    if (this != &other) {
        if (owner_ && state_h_) {
            _mm_free(state_h_);
        }

        state_h_ = other.state_h_;
        d_model_ = other.d_model_;
        owner_ = other.owner_;

        other.state_h_ = nullptr;
        other.d_model_ = 0;
        other.owner_ = false;
    }
    return *this;
}

void SSMMemoryState::reset() noexcept {
    if (state_h_ && d_model_ > 0) {
        std::fill(state_h_, state_h_ + d_model_, 0.0f);
    }
}

} // namespace gcapi_carpi