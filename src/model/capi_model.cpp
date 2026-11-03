// Archivo: src/model/capi_model.cpp
#include "capi_model.h"
#include <immintrin.h>
#include <stdexcept>
#include <new>
#include <cmath>
#include <iostream>
#include <algorithm>

extern "C" {
    void router_forward_avx2(const float* input, const float* router_weights, int32_t exp, int32_t d, int32_t k, int32_t* out_idx, float* out_prob);
    void ssm_forward_avx2(const float* in_t, const float* h_prev, float* h_out, const gcapi_carpi::TernaryWeightBlock* decay, const gcapi_carpi::TernaryWeightBlock* recept, const float* in_proj, int32_t d_mod, int32_t d_state);
    void matrix_multiply(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
}

namespace {

inline float deterministic_noise(uint32_t seed, int64_t index) noexcept {
    uint32_t hash = seed + static_cast<uint32_t>(index) + static_cast<uint32_t>(index >> 32);
    hash ^= hash << 13;
    hash ^= hash >> 17;
    hash ^= hash << 5;
    return (static_cast<float>(hash) / 4294967295.0f) * 2.0f - 1.0f;
}

} // namespace

namespace gcapi_carpi {

CapiModel::CapiModel(const std::string& model_file_path)
    : model_weights_loader_(model_file_path),
      d_model_(0),
      num_experts_(0),
      top_k_(0),
      vocab_size_(0),
      expert_weight_block_size_bytes_(0),
      embedding_offset_(0),
      router_weights_offset_(0),
      lm_head_offset_(0),
      expert_weights_base_offset_(0),
      router_weights_ptr_(nullptr),
      lm_head_weights_ptr_(nullptr),
      temp_embedding_vec_(nullptr),
      temp_ssm_h_out_per_expert_(nullptr),
      temp_combined_expert_output_(nullptr),
      temp_logits_output_(nullptr),
      temp_expert_indices_(nullptr),
      temp_expert_probabilities_(nullptr),
      temp_input_projected_for_ssm_(nullptr),
      ssm_hardware_state_buffer_(nullptr),
      expert_active_in_batch_(nullptr) {

    const uint8_t* base_ptr = model_weights_loader_.get_base_address();
    const ModelHeader* header = reinterpret_cast<const ModelHeader*>(base_ptr);

    if (header->d_model > 0 && header->d_model <= 32768 &&
        header->vocab_size > 0 && header->num_experts > 0 &&
        header->top_k > 0 && header->top_k <= header->num_experts &&
        header->embedding_offset == sizeof(ModelHeader)) {

        d_model_                        = header->d_model;
        vocab_size_                     = header->vocab_size;
        num_experts_                    = header->num_experts;
        top_k_                          = header->top_k;
        embedding_offset_               = header->embedding_offset;
        router_weights_offset_          = header->router_weights_offset;
        lm_head_offset_                 = header->lm_head_offset;
        expert_weights_base_offset_     = header->expert_weights_base_offset;
        expert_weight_block_size_bytes_ = static_cast<size_t>(header->expert_weight_block_size_bytes);

        std::cout << "[ENGINE] Header verificado | d_model: " << d_model_
                  << " | vocab: " << vocab_size_
                  << " | experts: " << num_experts_ << " (top-" << top_k_ << ")\n";
    } else {
        throw std::runtime_error("Invalid model header. Generate a valid binary using model_builder.");
    }

    const size_t total_memory_required =
        static_cast<size_t>(expert_weights_base_offset_) +
        (static_cast<size_t>(num_experts_) * expert_weight_block_size_bytes_);

    model_weights_loader_.get_pointer_at_offset(total_memory_required - 1);

    volatile uint8_t page_touch_acc = 0;
    for (size_t i = 0; i < total_memory_required; i += 4096) {
        page_touch_acc ^= base_ptr[i];
    }
    (void)page_touch_acc;

    router_weights_ptr_  = model_weights_loader_.get_pointer_at_offset(router_weights_offset_);
    lm_head_weights_ptr_ = model_weights_loader_.get_pointer_at_offset(lm_head_offset_);

    temp_embedding_vec_          = static_cast<float*>(_mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32));
    temp_ssm_h_out_per_expert_   = static_cast<float*>(_mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32));
    temp_combined_expert_output_ = static_cast<float*>(_mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32));
    temp_logits_output_          = static_cast<float*>(_mm_malloc(static_cast<size_t>(vocab_size_) * sizeof(float), 32));
    temp_expert_indices_         = static_cast<int32_t*>(_mm_malloc(static_cast<size_t>(top_k_) * sizeof(int32_t), 32));
    temp_expert_probabilities_   = static_cast<float*>(_mm_malloc(static_cast<size_t>(top_k_) * sizeof(float), 32));
    temp_input_projected_for_ssm_= static_cast<float*>(_mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32));
    ssm_hardware_state_buffer_   = static_cast<float*>(_mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32));
    expert_active_in_batch_      = static_cast<bool*>(_mm_malloc(static_cast<size_t>(num_experts_) * sizeof(bool), 32));

    if (!temp_embedding_vec_ || !temp_ssm_h_out_per_expert_ || !temp_combined_expert_output_ ||
        !temp_logits_output_ || !temp_expert_indices_ || !temp_expert_probabilities_ ||
        !temp_input_projected_for_ssm_ || !ssm_hardware_state_buffer_ || !expert_active_in_batch_) {
        release_buffers();
        throw std::bad_alloc();
    }

    clear_sequence_states();
}

CapiModel::~CapiModel() {
    release_buffers();
}

void CapiModel::release_buffers() noexcept {
    if (temp_embedding_vec_)          { _mm_free(temp_embedding_vec_);          temp_embedding_vec_ = nullptr; }
    if (temp_ssm_h_out_per_expert_)   { _mm_free(temp_ssm_h_out_per_expert_);   temp_ssm_h_out_per_expert_ = nullptr; }
    if (temp_combined_expert_output_) { _mm_free(temp_combined_expert_output_); temp_combined_expert_output_ = nullptr; }
    if (temp_logits_output_)          { _mm_free(temp_logits_output_);          temp_logits_output_ = nullptr; }
    if (temp_expert_indices_)         { _mm_free(temp_expert_indices_);         temp_expert_indices_ = nullptr; }
    if (temp_expert_probabilities_)   { _mm_free(temp_expert_probabilities_);   temp_expert_probabilities_ = nullptr; }
    if (temp_input_projected_for_ssm_){ _mm_free(temp_input_projected_for_ssm_);temp_input_projected_for_ssm_ = nullptr; }
    if (ssm_hardware_state_buffer_)   { _mm_free(ssm_hardware_state_buffer_);   ssm_hardware_state_buffer_ = nullptr; }
    if (expert_active_in_batch_)      { _mm_free(expert_active_in_batch_);      expert_active_in_batch_ = nullptr; }
}

float* CapiModel::forward_step(int32_t token_id) {
    return forward_step_mezo(token_id, 0, 0.0f);
}

float* CapiModel::forward_step_mezo(int32_t token_id, uint32_t seed, float epsilon) {
    if (token_id < 0 || token_id >= vocab_size_) token_id = 0;

    const size_t emb_offset =
        static_cast<size_t>(embedding_offset_) +
        (static_cast<size_t>(token_id) * d_model_ * sizeof(float));
    const float* emb_ptr = reinterpret_cast<const float*>(
        model_weights_loader_.get_pointer_at_offset(emb_offset)
    );
    std::copy(emb_ptr, emb_ptr + d_model_, temp_embedding_vec_);

    if (epsilon != 0.0f) {
        const int64_t base_idx = static_cast<int64_t>(token_id) * d_model_;
        for (int32_t i = 0; i < d_model_; ++i) {
            temp_embedding_vec_[i] += epsilon * deterministic_noise(seed, base_idx + i);
        }
    }

    const float* router_float_ptr = reinterpret_cast<const float*>(router_weights_ptr_);
    router_forward_avx2(
        temp_embedding_vec_,
        router_float_ptr,
        num_experts_,
        d_model_,
        top_k_,
        temp_expert_indices_,
        temp_expert_probabilities_
    );

    std::fill(temp_combined_expert_output_, temp_combined_expert_output_ + d_model_, 0.0f);

    for (int32_t k = 0; k < top_k_; ++k) {
        int32_t exp_idx = temp_expert_indices_[k];
        if (exp_idx < 0 || exp_idx >= num_experts_) exp_idx = 0;

        expert_active_in_batch_[exp_idx] = true;
        ExpertWeights w = get_expert_weights(exp_idx);

        matrix_multiply(
            temp_embedding_vec_,
            w.input_projection_weights,
            temp_input_projected_for_ssm_,
            1,
            d_model_,
            d_model_
        );

        ssm_forward_avx2(
            temp_embedding_vec_,
            ssm_hardware_state_buffer_,
            temp_ssm_h_out_per_expert_,
            w.decay_weights,
            w.receptivity_weights,
            temp_input_projected_for_ssm_,
            d_model_,
            d_model_
        );

        const float prob = temp_expert_probabilities_[k];
        for (int32_t i = 0; i < d_model_; ++i) {
            temp_combined_expert_output_[i] += temp_ssm_h_out_per_expert_[i] * prob;
        }
    }

    for (int32_t i = 0; i < d_model_; ++i) {
        float val = temp_combined_expert_output_[i];
        if (std::isnan(val)) val = 0.0f;
        ssm_hardware_state_buffer_[i] = std::max(-10.0f, std::min(10.0f, val));
    }

    const float* lm_head_float_ptr = reinterpret_cast<const float*>(lm_head_weights_ptr_);
    matrix_multiply(
        ssm_hardware_state_buffer_,
        lm_head_float_ptr,
        temp_logits_output_,
        1,
        d_model_,
        vocab_size_
    );

    if (epsilon != 0.0f) {
        const int64_t head_base = static_cast<int64_t>(vocab_size_ + num_experts_) * d_model_;
        for (int32_t j = 0; j < vocab_size_; ++j) {
            temp_logits_output_[j] += epsilon * deterministic_noise(seed, head_base + j);
        }
    }

    return temp_logits_output_;
}

float CapiModel::compute_cross_entropy_loss(const float* logits, int32_t target_token) const {
    if (!logits || target_token < 0 || target_token >= vocab_size_) return 0.0f;

    float max_logit = logits[0];
    for (int32_t i = 1; i < vocab_size_; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    for (int32_t i = 0; i < vocab_size_; ++i) {
        sum_exp += std::exp(logits[i] - max_logit);
    }

    const float target_prob = std::exp(logits[target_token] - max_logit) / (sum_exp + 1e-9f);
    return -std::log(target_prob + 1e-9f);
}

ExpertWeights CapiModel::get_expert_weights(int32_t idx) const {
    const size_t off =
        static_cast<size_t>(expert_weights_base_offset_) +
        (static_cast<size_t>(idx) * expert_weight_block_size_bytes_);
    const size_t ternary_bytes = (static_cast<size_t>(d_model_) * d_model_ + 3) / 4;

    return {
        reinterpret_cast<const TernaryWeightBlock*>(model_weights_loader_.get_pointer_at_offset(off)),
        reinterpret_cast<const TernaryWeightBlock*>(model_weights_loader_.get_pointer_at_offset(off + ternary_bytes)),
        reinterpret_cast<const float*>(model_weights_loader_.get_pointer_at_offset(off + 2 * ternary_bytes))
    };
}

void CapiModel::clear_sequence_states() noexcept {
    if (ssm_hardware_state_buffer_) {
        std::fill(ssm_hardware_state_buffer_, ssm_hardware_state_buffer_ + d_model_, 0.0f);
    }
    if (expert_active_in_batch_) {
        std::fill(expert_active_in_batch_, expert_active_in_batch_ + num_experts_, false);
    }
}

} // namespace gcapi_carpi

extern "C" {
    static gcapi_carpi::CapiModel* global_model_instance = nullptr;

    GCAPI_EXPORT bool init_model(const char* path) {
        try {
            if (global_model_instance) {
                delete global_model_instance;
                global_model_instance = nullptr;
            }
            if (!path) return false;
            global_model_instance = new gcapi_carpi::CapiModel(path);
            return true;
        } catch (...) {
            return false;
        }
    }

    GCAPI_EXPORT float* forward_step(int token) {
        if (!global_model_instance) return nullptr;
        return global_model_instance->forward_step(token);
    }

    GCAPI_EXPORT float* forward_step_mezo(int token, uint32_t seed, float epsilon) {
        if (!global_model_instance) return nullptr;
        return global_model_instance->forward_step_mezo(token, seed, epsilon);
    }

    GCAPI_EXPORT void limpiar_memoria_secuencia() {
        if (global_model_instance) {
            global_model_instance->clear_sequence_states();
        }
    }
}
