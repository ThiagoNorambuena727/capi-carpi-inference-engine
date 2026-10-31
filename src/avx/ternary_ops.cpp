// Archivo: src/avx/ternary_ops.cpp
#include "ternary_ops.h"
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <omp.h>

namespace {

inline float horizontal_sum_m256(__m256 v) {
    __m128 v128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
    __m128 v64  = _mm_add_ps(v128, _mm_shuffle_ps(v128, v128, _MM_SHUFFLE(1, 0, 3, 2)));
    __m128 v32  = _mm_add_ps(v64, _mm_shuffle_ps(v64, v64, _MM_SHUFFLE(0, 0, 0, 1)));
    return _mm_cvtss_f32(v32);
}

// Constantes SIMD para desempaquetar 8 pesos de 2 bits (contenidos en un uint16_t)
// hacia 8 carriles enteros de 32 bits dentro de un registro __m256i.
static const __m256i shifts_for_2bit_unpack   = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);
static const __m256i two_bits_mask            = _mm256_set1_epi32(0x03);
static const __m256i ternary_one_val_i        = _mm256_set1_epi32(1);
static const __m256i ternary_neg_one_val_i    = _mm256_set1_epi32(-1);
static const __m256i two_bits_pattern_neg_one = _mm256_set1_epi32(0b00);
static const __m256i two_bits_pattern_pos_one = _mm256_set1_epi32(0b10);

} // namespace

extern "C" {

GCAPI_EXPORT void ternary_linear_forward(const float* input_vec,
                                         const gcapi_carpi::TernaryWeightBlock* weight_matrix,
                                         float* output_vec,
                                         int input_dim,
                                         int output_dim) {
    if (!input_vec || !weight_matrix || !output_vec || input_dim <= 0 || output_dim <= 0) return;

    const size_t num_blocks_per_row =
        (static_cast<size_t>(input_dim) + gcapi_carpi::BLOCK_SIZE_WEIGHTS - 1) /
        gcapi_carpi::BLOCK_SIZE_WEIGHTS;
    const int input_dim_aligned = (input_dim / 8) * 8;

    #pragma omp parallel for
    for (int i = 0; i < output_dim; ++i) {
        __m256 row_acc = _mm256_setzero_ps();
        float scalar_acc = 0.0f;

        const gcapi_carpi::TernaryWeightBlock* current_row_weights =
            weight_matrix + static_cast<size_t>(i) * num_blocks_per_row;

        for (int j = 0; j < input_dim_aligned; j += 8) {
            __m256 input_chunk = _mm256_loadu_ps(input_vec + j);

            const size_t current_weight_idx_start = static_cast<size_t>(j);
            const size_t block_offset        = current_weight_idx_start / gcapi_carpi::BLOCK_SIZE_WEIGHTS;
            const size_t weight_in_block_idx = current_weight_idx_start % gcapi_carpi::BLOCK_SIZE_WEIGHTS;
            const size_t byte_in_block_idx   = weight_in_block_idx / 4;

            // Lectura de 16 bits (2 bytes = 8 pesos ternarios de 2 bits)
            const uint16_t two_packed_bytes = *reinterpret_cast<const uint16_t*>(
                &current_row_weights[block_offset].packed_weights[byte_in_block_idx]
            );
            __m256i packed_bytes_256 = _mm256_set1_epi32(static_cast<int>(two_packed_bytes));

            // Desempaquetado paralelo a 8 enteros de 2 bits en cada carril de 32 bits
            __m256i shifted_values = _mm256_srlv_epi32(packed_bytes_256, shifts_for_2bit_unpack);
            __m256i two_bit_values = _mm256_and_si256(shifted_values, two_bits_mask);

            // Generacion de mascaras para patrones -1 (0b00) y +1 (0b10)
            __m256i mask_neg_one_pattern = _mm256_cmpeq_epi32(two_bit_values, two_bits_pattern_neg_one);
            __m256i mask_pos_one_pattern = _mm256_cmpeq_epi32(two_bit_values, two_bits_pattern_pos_one);

            __m256i unpacked_weights_i = _mm256_setzero_si256();
            unpacked_weights_i = _mm256_blendv_epi8(unpacked_weights_i, ternary_neg_one_val_i, mask_neg_one_pattern);
            unpacked_weights_i = _mm256_blendv_epi8(unpacked_weights_i, ternary_one_val_i, mask_pos_one_pattern);

            __m256i mask_is_pos_one_i = _mm256_cmpeq_epi32(unpacked_weights_i, ternary_one_val_i);
            __m256i mask_is_neg_one_i = _mm256_cmpeq_epi32(unpacked_weights_i, ternary_neg_one_val_i);

            __m256 mask_is_pos_one_f = _mm256_castsi256_ps(mask_is_pos_one_i);
            __m256 mask_is_neg_one_f = _mm256_castsi256_ps(mask_is_neg_one_i);

            __m256 neg_input_chunk = _mm256_sub_ps(_mm256_setzero_ps(), input_chunk);

            // Seleccion sin multiplicacion mediante mezcla condicional (_mm256_blendv_ps)
            __m256 term_to_add = _mm256_setzero_ps();
            term_to_add = _mm256_blendv_ps(term_to_add, input_chunk, mask_is_pos_one_f);
            term_to_add = _mm256_blendv_ps(term_to_add, neg_input_chunk, mask_is_neg_one_f);

            row_acc = _mm256_add_ps(row_acc, term_to_add);
        }

        scalar_acc += horizontal_sum_m256(row_acc);

        // Cola escalar para dimensiones no multiplos de 8
        for (int j = input_dim_aligned; j < input_dim; ++j) {
            const size_t block_offset        = static_cast<size_t>(j) / gcapi_carpi::BLOCK_SIZE_WEIGHTS;
            const size_t weight_in_block_idx = static_cast<size_t>(j) % gcapi_carpi::BLOCK_SIZE_WEIGHTS;
            const size_t byte_in_block_idx   = weight_in_block_idx / 4;
            const int bit_pair_in_byte_idx   = static_cast<int>(weight_in_block_idx % 4);

            const uint8_t packed_byte = current_row_weights[block_offset].packed_weights[byte_in_block_idx];
            const uint8_t two_bits    = gcapi_carpi::get_2bit_value(packed_byte, bit_pair_in_byte_idx);
            const int8_t weight_val   = gcapi_carpi::map_2bit_to_ternary_value(two_bits);

            if (weight_val == 1) {
                scalar_acc += input_vec[j];
            } else if (weight_val == -1) {
                scalar_acc -= input_vec[j];
            }
        }
        output_vec[i] = scalar_acc;
    }
}

GCAPI_EXPORT void packed_ternary_matmul(const uint8_t* packed_matrix,
                                        const float* input_vec,
                                        float* output_vec,
                                        int rows,
                                        int d_model) {
    if (!packed_matrix || !input_vec || !output_vec || rows <= 0 || d_model <= 0) return;

    const int packed_cols = d_model / 4;

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        float acc = 0.0f;
        const size_t row_offset = static_cast<size_t>(i) * packed_cols;

        for (int j = 0; j < packed_cols; ++j) {
            const uint8_t packed_byte = packed_matrix[row_offset + j];

            const int8_t w0 = static_cast<int8_t>((packed_byte >> 6) & 0x03) - 1;
            const int8_t w1 = static_cast<int8_t>((packed_byte >> 4) & 0x03) - 1;
            const int8_t w2 = static_cast<int8_t>((packed_byte >> 2) & 0x03) - 1;
            const int8_t w3 = static_cast<int8_t>(packed_byte & 0x03) - 1;

            const int idx = j * 4;
            acc += (w0 * input_vec[idx]) +
                   (w1 * input_vec[idx + 1]) +
                   (w2 * input_vec[idx + 2]) +
                   (w3 * input_vec[idx + 3]);
        }
        output_vec[i] = acc;
    }
}

} // extern "C"