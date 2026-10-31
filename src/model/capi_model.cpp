// Archivo: src/model/capi_model.cpp
#include "capi_model.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

extern "C" {
    void router_forward_avx2(const float* input, const float* router_weights, int32_t exp, int32_t d, int32_t k, int32_t* out_idx, float* out_prob);
    void ssm_forward_avx2(const float* in_t, const float* h_prev, float* h_out, const gcapi_carpi::TernaryWeightBlock* decay, const gcapi_carpi::TernaryWeightBlock* recept, const float* in_proj, int32_t d_mod, int32_t d_state);
    void matrix_multiply(const float* A, const float* B, float* C, int32_t rowsA, int32_t colsA, int32_t colsB);
}

namespace gcapi_carpi {

void CapiModel::aplicar_amnesia_topologica() {}

CapiModel::CapiModel(const std::string& model_file_path) : model_weights_loader_(model_file_path) {
    const uint8_t* base_ptr = model_weights_loader_.get_base_address();
    const ModelHeader* header = reinterpret_cast<const ModelHeader*>(base_ptr);

    if (header->d_model > 0 && header->d_model <= 32768 && header->embedding_offset == sizeof(ModelHeader)) {
        d_model_ = header->d_model;
        vocab_size_ = header->vocab_size;
        num_experts_ = header->num_experts;
        top_k_ = header->top_k;
        embedding_offset_ = header->embedding_offset;
        router_weights_offset_ = header->router_weights_offset;
        lm_head_offset_ = header->lm_head_offset;
        expert_weights_base_offset_ = header->expert_weights_base_offset;
        expert_weight_block_size_bytes_ = header->expert_weight_block_size_bytes;
        std::cout << "[ENGINE] Header verificado | d_model: " << d_model_
                  << " | vocab: " << vocab_size_
                  << " | experts: " << num_experts_ << " (top-" << top_k_ << ")\n";
    } else {
        throw std::runtime_error("Header de modelo invalido. Genera el binario con forjador_titanico.");
    }

    size_t total_memory_required = expert_weights_base_offset_ + (static_cast<size_t>(num_experts_) * expert_weight_block_size_bytes_);

    // Pre-calentamiento de páginas de memoria (Zero-Copy Page Touch)
    volatile uint8_t acumulador_memoria = 0;
    for (size_t i = 0; i < total_memory_required; i += 4096) {
        acumulador_memoria ^= base_ptr[i];
    }

    router_weights_ptr_ = reinterpret_cast<const uint8_t*>(model_weights_loader_.get_pointer_at_offset(router_weights_offset_));
    lm_head_weights_ptr_ = reinterpret_cast<const uint8_t*>(model_weights_loader_.get_pointer_at_offset(lm_head_offset_));

    temp_embedding_vec_ = static_cast<float*>(_mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32));
    temp_ssm_h_out_per_expert_ = static_cast<float*>(_mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32));
    temp_combined_expert_output_ = static_cast<float*>(_mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32));
    temp_logits_output_ = static_cast<float*>(_mm_malloc(static_cast<size_t>(vocab_size_) * sizeof(float), 32));
    temp_expert_indices_ = static_cast<int32_t*>(_mm_malloc(static_cast<size_t>(top_k_) * sizeof(int32_t), 32));
    temp_expert_probabilities_ = static_cast<float*>(_mm_malloc(static_cast<size_t>(top_k_) * sizeof(float), 32));
    temp_input_projected_for_ssm_ = static_cast<float*>(_mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32));
    ssm_hardware_state_buffer_ = static_cast<float*>(_mm_malloc(static_cast<size_t>(d_model_) * sizeof(float), 32));

    std::fill(ssm_hardware_state_buffer_, ssm_hardware_state_buffer_ + d_model_, 0.0f);

    expert_active_in_batch_ = static_cast<bool*>(_mm_malloc(static_cast<size_t>(num_experts_) * sizeof(bool), 32));
    std::fill(expert_active_in_batch_, expert_active_in_batch_ + num_experts_, false);
}

CapiModel::~CapiModel() {
    _mm_free(temp_embedding_vec_);
    _mm_free(temp_ssm_h_out_per_expert_);
    _mm_free(temp_combined_expert_output_);
    _mm_free(temp_logits_output_);
    _mm_free(temp_expert_indices_);
    _mm_free(temp_expert_probabilities_);
    _mm_free(temp_input_projected_for_ssm_);
    _mm_free(ssm_hardware_state_buffer_);
    _mm_free(expert_active_in_batch_);
}

float* CapiModel::forward_step(int32_t token_id) {
    return forward_step_mezo(token_id, 0, 0.0f);
}

float* CapiModel::forward_step_mezo(int32_t token_id, uint32_t seed, float epsilon) {
    if (token_id < 0 || token_id >= vocab_size_) token_id = 0;

    // 1. Extracción Zero-Copy del Embedding
    size_t offset = embedding_offset_ + (static_cast<size_t>(token_id) * d_model_ * sizeof(float));
    const float* emb_ptr = reinterpret_cast<const float*>(model_weights_loader_.get_pointer_at_offset(offset));
    std::copy(emb_ptr, emb_ptr + d_model_, temp_embedding_vec_);

    // 2. Router MoE AVX2
    const float* router_float_ptr = reinterpret_cast<const float*>(router_weights_ptr_);
    router_forward_avx2(temp_embedding_vec_, router_float_ptr, num_experts_, d_model_, top_k_, temp_expert_indices_, temp_expert_probabilities_);

    // 3. Ejecución Esparsa Top-K de Expertos SSM (1.58-bit Ternary + AVX2)
    std::fill(temp_combined_expert_output_, temp_combined_expert_output_ + d_model_, 0.0f);

    for (int32_t k = 0; k < top_k_; ++k) {
        int exp_idx = temp_expert_indices_[k];
        if (exp_idx < 0 || exp_idx >= num_experts_) exp_idx = 0;

        expert_active_in_batch_[exp_idx] = true;
        ExpertWeights w = get_expert_weights(exp_idx);

        matrix_multiply(temp_embedding_vec_, w.input_projection_weights, temp_input_projected_for_ssm_, 1, d_model_, d_model_);

        ssm_forward_avx2(temp_embedding_vec_, ssm_hardware_state_buffer_, temp_ssm_h_out_per_expert_,
                         w.decay_weights, w.receptivity_weights, temp_input_projected_for_ssm_, d_model_, d_model_);

        const float prob = temp_expert_probabilities_[k];
        for (int i = 0; i < d_model_; ++i) {
            temp_combined_expert_output_[i] += temp_ssm_h_out_per_expert_[i] * prob;
        }
    }

    for (int i = 0; i < d_model_; ++i) {
        float val = temp_combined_expert_output_[i];
        if (std::isnan(val)) val = 0.0f;
        ssm_hardware_state_buffer_[i] = std::max(-10.0f, std::min(10.0f, val));
    }

    // 4. Proyección de Salida (LM Head) AVX2 + OpenMP
    const float* lm_head_float_ptr = reinterpret_cast<const float*>(lm_head_weights_ptr_);
    matrix_multiply(ssm_hardware_state_buffer_, lm_head_float_ptr, temp_logits_output_, 1, d_model_, vocab_size_);

    return temp_logits_output_;
}

float* CapiModel::forward_step_multimodal(const float* input_embedding, uint32_t seed, float epsilon) { return temp_logits_output_; }
float CapiModel::calcular_loss_interno(const float* logits, int32_t target_token) { return 0.0f; }
void CapiModel::entrenar_lote_cmezo(const int32_t* tokens, int32_t seq_len, uint32_t semilla, float epsilon, float learning_rate, float* out_loss, float* out_diff) {}
void CapiModel::actualizar_pesos_mezo(uint32_t semilla, float epsilon, float factor_escala) {}
void CapiModel::mutar_pesos_mezo(uint32_t semilla, float epsilon, int32_t signo) {}
void CapiModel::backward_step(float* gradientes_logits, float learning_rate) {}

ExpertWeights CapiModel::get_expert_weights(int32_t idx) const {
    size_t off = expert_weights_base_offset_ + (static_cast<size_t>(idx) * expert_weight_block_size_bytes_);
    size_t ternary_bytes = (static_cast<size_t>(d_model_) * d_model_ + 3) / 4;
    return {
        reinterpret_cast<const TernaryWeightBlock*>(model_weights_loader_.get_pointer_at_offset(off)),
        reinterpret_cast<const TernaryWeightBlock*>(model_weights_loader_.get_pointer_at_offset(off + ternary_bytes)),
        reinterpret_cast<const float*>(model_weights_loader_.get_pointer_at_offset(off + 2 * ternary_bytes))
    };
}

void CapiModel::clear_sequence_states() {
    std::fill(ssm_hardware_state_buffer_, ssm_hardware_state_buffer_ + d_model_, 0.0f);
    std::fill(expert_active_in_batch_, expert_active_in_batch_ + num_experts_, false);
}

} // namespace gcapi_carpi

extern "C" {
    static gcapi_carpi::CapiModel* global_model_instance = nullptr;

    GCAPI_EXPORT bool init_model(const char* path) {
        try {
            if (global_model_instance) delete global_model_instance;
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

    GCAPI_EXPORT void limpiar_memoria_secuencia() {
        if (global_model_instance) global_model_instance->clear_sequence_states();
    }
}