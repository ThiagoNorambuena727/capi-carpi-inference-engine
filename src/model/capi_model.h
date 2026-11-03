// Archivo: src/model/capi_model.h
#pragma once

#include <string>
#include <cstddef>
#include <cstdint>
#include "../tensor/mmap_loader.h"
#include "../tensor/ternary_tensor.h"
#include "model_header.h"

#ifdef _WIN32
    #define GCAPI_EXPORT __declspec(dllexport)
#else
    #define GCAPI_EXPORT
#endif

namespace gcapi_carpi {

struct ExpertWeights {
    const TernaryWeightBlock* decay_weights;
    const TernaryWeightBlock* receptivity_weights;
    const float* input_projection_weights;
};

class CapiModel {
public:
    explicit CapiModel(const std::string& model_file_path);
    ~CapiModel();

    CapiModel(const CapiModel&) = delete;
    CapiModel& operator=(const CapiModel&) = delete;
    CapiModel(CapiModel&&) = delete;
    CapiModel& operator=(CapiModel&&) = delete;

    float* forward_step(int32_t token_id);
    float* forward_step_mezo(int32_t token_id, uint32_t seed, float epsilon);
    float compute_cross_entropy_loss(const float* logits, int32_t target_token) const;
    void clear_sequence_states() noexcept;

    int32_t get_d_model() const noexcept { return d_model_; }
    int32_t get_vocab_size() const noexcept { return vocab_size_; }
    int32_t get_num_experts() const noexcept { return num_experts_; }
    int32_t get_top_k() const noexcept { return top_k_; }

private:
    MmapWeightsLoader model_weights_loader_;
    int32_t d_model_;
    int32_t num_experts_;
    int32_t top_k_;
    int32_t vocab_size_;
    size_t expert_weight_block_size_bytes_;

    uint64_t embedding_offset_;
    uint64_t router_weights_offset_;
    uint64_t lm_head_offset_;
    uint64_t expert_weights_base_offset_;

    const uint8_t* router_weights_ptr_;
    const uint8_t* lm_head_weights_ptr_;

    float* temp_embedding_vec_;
    float* temp_ssm_h_out_per_expert_;
    float* temp_combined_expert_output_;
    float* temp_logits_output_;
    int32_t* temp_expert_indices_;
    float* temp_expert_probabilities_;
    float* temp_input_projected_for_ssm_;
    float* ssm_hardware_state_buffer_;
    bool* expert_active_in_batch_;

    void release_buffers() noexcept;
    ExpertWeights get_expert_weights(int32_t expert_idx) const;
};

} // namespace gcapi_carpi

extern "C" {
    GCAPI_EXPORT bool init_model(const char* path);
    GCAPI_EXPORT float* forward_step(int token);
    GCAPI_EXPORT float* forward_step_mezo(int token, uint32_t seed, float epsilon);
    GCAPI_EXPORT void limpiar_memoria_secuencia();
}
