// Archivo: src/tools/model_builder.cpp
//
// Generador CLI parametrizado y verificador de integridad estructural para archivos
// de pesos binarios de Capi-Carpi (State Space Model + Sparse MoE 1.58-bit).
//
// Ejemplos de uso:
//   model_builder --d-model 512 --vocab 16384 --experts 8 --top-k 2 --out capi_test.bin --yes
//   model_builder --verify capi_test.bin

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <string>
#include <algorithm>
#include "../model/model_header.h"

using gcapi_carpi::ModelHeader;

// Inicializacion uniforme de Xavier/Glorot para mantener la varianza de las activaciones
static void xavier_init(std::vector<float>& buffer, float fan_in, float fan_out, std::mt19937& rng) {
    const float limit = std::sqrt(6.0f / (fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& w : buffer) {
        w = dist(rng);
    }
}

// Inicializacion ternaria equilibrada (4 pesos de 2 bits por cada byte).
// Codificacion (ver src/tensor/ternary_tensor.h): 00 -> -1, 01 -> 0, 10 -> +1.
static void random_ternary_pack(std::vector<uint8_t>& buffer, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 2);
    static constexpr uint8_t pattern[3] = {0x00, 0x01, 0x02};
    for (uint8_t& byte_val : buffer) {
        byte_val = static_cast<uint8_t>(
              pattern[dist(rng)]
            | (pattern[dist(rng)] << 2)
            | (pattern[dist(rng)] << 4)
            | (pattern[dist(rng)] << 6)
        );
    }
}

static int verify_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] No se pudo abrir el archivo: " << path << "\n";
        return 1;
    }

    ModelHeader h{};
    file.read(reinterpret_cast<char*>(&h), sizeof(h));
    file.seekg(0, std::ios::end);
    const uint64_t actual_size = static_cast<uint64_t>(file.tellg());

    const uint64_t emb_bytes  = static_cast<uint64_t>(h.vocab_size) * h.d_model * sizeof(float);
    const uint64_t rt_bytes   = static_cast<uint64_t>(h.num_experts) * h.d_model * sizeof(float);
    const uint64_t head_bytes = static_cast<uint64_t>(h.vocab_size) * h.d_model * sizeof(float);
    const uint64_t expected_size =
        sizeof(ModelHeader) + emb_bytes + rt_bytes + head_bytes +
        static_cast<uint64_t>(h.num_experts) * h.expert_weight_block_size_bytes;

    std::cout << "== " << path << " ==\n"
              << "  d_model          : " << h.d_model << "\n"
              << "  vocab_size       : " << h.vocab_size << "\n"
              << "  num_experts      : " << h.num_experts << "\n"
              << "  top_k            : " << h.top_k << "\n"
              << "  offsets          : emb=" << h.embedding_offset
              << " router=" << h.router_weights_offset
              << " head=" << h.lm_head_offset
              << " experts=" << h.expert_weights_base_offset << "\n"
              << "  bytes / experto  : " << h.expert_weight_block_size_bytes << " bytes\n"
              << "  tamano esperado  : " << expected_size << " bytes\n"
              << "  tamano real      : " << actual_size << " bytes\n";

    const uint64_t total_params =
        static_cast<uint64_t>(h.vocab_size) * h.d_model * 2ULL +
        static_cast<uint64_t>(h.num_experts) * h.d_model +
        static_cast<uint64_t>(h.num_experts) * 3ULL * h.d_model * h.d_model;

    std::cout << "  parametros       : " << total_params << " (" << (total_params / 1e6) << " M)\n";

    if (expected_size != actual_size) {
        std::cerr << "  [FAIL] El tamano fisico del archivo no coincide con la cabecera.\n";
        return 1;
    }
    std::cout << "  [OK] Estructura binaria coherente.\n";
    return 0;
}

int main(int argc, char** argv) {
    int32_t d_model = 512, num_experts = 8, top_k = 2, vocab_size = 16384;
    uint32_t seed = 42;
    std::string out_path = "capi_test.bin";
    bool auto_confirm = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next_val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };

        if      (arg == "--d-model") d_model     = std::stoi(next_val());
        else if (arg == "--vocab")   vocab_size  = std::stoi(next_val());
        else if (arg == "--experts") num_experts = std::stoi(next_val());
        else if (arg == "--top-k")   top_k       = std::stoi(next_val());
        else if (arg == "--seed")    seed        = static_cast<uint32_t>(std::stoul(next_val()));
        else if (arg == "--out")     out_path    = next_val();
        else if (arg == "--yes" || arg == "-y") auto_confirm = true;
        else if (arg == "--verify")  return verify_binary(next_val());
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Uso: model_builder [--d-model N] [--vocab N] [--experts N] [--top-k N]\n"
                         "                   [--seed N] [--out archivo.bin] [--yes]\n"
                         "     model_builder --verify archivo.bin\n";
            return 0;
        } else {
            std::cerr << "[ERROR] Argumento desconocido: " << arg << "\n";
            return 1;
        }
    }

    if (d_model <= 0 || vocab_size <= 0 || num_experts <= 0 || top_k <= 0) {
        std::cerr << "[ERROR] Todas las dimensiones deben ser enteras positivas.\n";
        return 1;
    }
    if (top_k > num_experts) {
        std::cerr << "[ERROR] top_k no puede superar num_experts.\n";
        return 1;
    }
    if (d_model % 8 != 0) {
        std::cerr << "[WARN] d_model no es multiplo de 8; los kernels AVX2 utilizaran la cola escalar.\n";
    }

    const uint64_t emb_bytes   = static_cast<uint64_t>(vocab_size) * d_model * sizeof(float);
    const uint64_t rt_bytes    = static_cast<uint64_t>(num_experts) * d_model * sizeof(float);
    const uint64_t head_bytes  = static_cast<uint64_t>(vocab_size) * d_model * sizeof(float);
    const uint64_t exp_weights = static_cast<uint64_t>(d_model) * d_model;
    const uint64_t tern_bytes  = (exp_weights + 3) / 4;
    const uint64_t exp_block   = (tern_bytes * 2) + (exp_weights * sizeof(float));

    ModelHeader h{};
    h.d_model                        = d_model;
    h.num_experts                    = num_experts;
    h.top_k                          = top_k;
    h.vocab_size                     = vocab_size;
    h.embedding_offset               = sizeof(ModelHeader);
    h.router_weights_offset          = h.embedding_offset + emb_bytes;
    h.lm_head_offset                 = h.router_weights_offset + rt_bytes;
    h.expert_weights_base_offset     = h.lm_head_offset + head_bytes;
    h.expert_weight_block_size_bytes = exp_block;

    const uint64_t total_bytes =
        sizeof(ModelHeader) + emb_bytes + rt_bytes + head_bytes + (exp_block * num_experts);
    const uint64_t total_params =
        static_cast<uint64_t>(vocab_size) * d_model * 2ULL +
        static_cast<uint64_t>(num_experts) * d_model +
        static_cast<uint64_t>(num_experts) * 3ULL * d_model * d_model;
    const uint64_t active_params =
        static_cast<uint64_t>(vocab_size) * d_model +
        static_cast<uint64_t>(num_experts) * d_model +
        static_cast<uint64_t>(top_k) * 3ULL * d_model * d_model;

    std::cout << "==================================================\n"
              << "  CAPI-CARPI MODEL BUILDER CLI (SSM + MoE 1.58b)\n"
              << "==================================================\n"
              << "  d_model     : " << d_model << "\n"
              << "  vocab_size  : " << vocab_size << "\n"
              << "  num_experts : " << num_experts << " (top-" << top_k << ")\n"
              << "  parametros  : " << (total_params / 1e6) << " M (activos/token: "
              << (active_params / 1e6) << " M)\n"
              << "  archivo     : " << out_path << " -> "
              << (total_bytes / (1024.0 * 1024.0)) << " MB\n"
              << "  semilla     : " << seed << "\n\n";

    if (!auto_confirm) {
        std::cout << "Generar archivo binario? (s/n): ";
        char response = 'n';
        std::cin >> response;
        if (response != 's' && response != 'S') return 0;
    }

    std::ofstream out_file(out_path, std::ios::binary | std::ios::trunc);
    if (!out_file) {
        std::cerr << "[ERROR] No se pudo crear el archivo de salida: " << out_path << "\n";
        return 1;
    }

    std::mt19937 rng(seed);
    out_file.write(reinterpret_cast<const char*>(&h), sizeof(h));

    static constexpr size_t CHUNK_SIZE = 262144;
    std::vector<float>   float_buf;
    std::vector<uint8_t> ternary_buf;

    auto write_float_section = [&](uint64_t count, float fan_in, float fan_out, const char* label) {
        std::cout << "  -> " << label << " ..." << std::flush;
        while (count > 0) {
            const size_t k = static_cast<size_t>(std::min<uint64_t>(CHUNK_SIZE, count));
            float_buf.resize(k);
            xavier_init(float_buf, fan_in, fan_out, rng);
            out_file.write(reinterpret_cast<const char*>(float_buf.data()), k * sizeof(float));
            count -= k;
        }
        std::cout << " listo\n";
    };

    write_float_section(static_cast<uint64_t>(vocab_size) * d_model, static_cast<float>(vocab_size), static_cast<float>(d_model), "embeddings");
    write_float_section(static_cast<uint64_t>(num_experts) * d_model, static_cast<float>(d_model), static_cast<float>(num_experts), "router");
    write_float_section(static_cast<uint64_t>(vocab_size) * d_model, static_cast<float>(d_model), static_cast<float>(vocab_size), "lm_head");

    std::cout << "  -> " << num_experts << " expertos SSM ..." << std::flush;
    for (int32_t e = 0; e < num_experts; ++e) {
        for (int m = 0; m < 2; ++m) {
            uint64_t remaining = tern_bytes;
            while (remaining > 0) {
                const size_t k = static_cast<size_t>(std::min<uint64_t>(CHUNK_SIZE, remaining));
                ternary_buf.resize(k);
                random_ternary_pack(ternary_buf, rng);
                out_file.write(reinterpret_cast<const char*>(ternary_buf.data()), k);
                remaining -= k;
            }
        }
        uint64_t remaining = exp_weights;
        while (remaining > 0) {
            const size_t k = static_cast<size_t>(std::min<uint64_t>(CHUNK_SIZE, remaining));
            float_buf.resize(k);
            xavier_init(float_buf, static_cast<float>(d_model), static_cast<float>(d_model), rng);
            out_file.write(reinterpret_cast<const char*>(float_buf.data()), k * sizeof(float));
            remaining -= k;
        }
        if (num_experts >= 16 && (e + 1) % (num_experts / 8) == 0) {
            std::cout << "." << std::flush;
        }
    }
    std::cout << " listo\n";
    out_file.close();

    std::cout << "\nARCHIVO BINARIO GENERADO EXITOSAMENTE.\n\n";
    return verify_binary(out_path);
}