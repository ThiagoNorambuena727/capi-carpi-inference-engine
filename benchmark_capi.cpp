// Archivo: benchmark_capi.cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <string>
#include "src/model/capi_model.h"

using namespace gcapi_carpi;

int main(int argc, char** argv) {
    const std::string model_path = (argc > 1) ? argv[1] : "capi_test.bin";
    const int num_tokens = (argc > 2) ? std::stoi(argv[2]) : 200;

    std::cout << "=================================================================\n";
    std::cout << "  CAPI-CARPI AVX2 INFERENCE ENGINE - BENCHMARK DE RENDIMIENTO\n";
    std::cout << "  Arquitectura: State Space Model (SSM) + Sparse MoE (1.58-bit)\n";
    std::cout << "=================================================================\n\n";

    try {
        const auto t_load_start = std::chrono::high_resolution_clock::now();
        CapiModel engine(model_path);
        const auto t_load_end = std::chrono::high_resolution_clock::now();
        const double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();

        std::cout << "[OK] Mapeo Zero-Copy (mmap) completado en: "
                  << std::fixed << std::setprecision(2) << load_ms << " ms\n";
        std::cout << "[>>] Ejecutando inferencia continua de " << num_tokens << " tokens...\n\n";

        engine.clear_sequence_states();
        int32_t current_token = 1;
        const int32_t vocab_size = engine.get_vocab_size();

        const auto t_inf_start = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < num_tokens; ++step) {
            const float* logits = engine.forward_step(current_token);

            int32_t best_tok = 0;
            float max_val = logits[0];
            for (int32_t v = 1; v < vocab_size; ++v) {
                if (logits[v] > max_val) {
                    max_val = logits[v];
                    best_tok = v;
                }
            }
            current_token = best_tok;
        }
        const auto t_inf_end = std::chrono::high_resolution_clock::now();

        const double total_sec = std::chrono::duration<double>(t_inf_end - t_inf_start).count();
        const double ms_per_token = (total_sec * 1000.0) / num_tokens;
        const double tokens_per_sec = num_tokens / total_sec;

        std::cout << "-----------------------------------------------------------------\n";
        std::cout << "  RESULTADOS DEL BENCHMARK (CPU AVX2 + FMA + OpenMP)\n";
        std::cout << "-----------------------------------------------------------------\n";
        std::cout << "  Tokens procesados : " << num_tokens << "\n";
        std::cout << "  Tiempo total      : " << std::setprecision(4) << total_sec << " s\n";
        std::cout << "  Latencia / Token  : " << std::setprecision(2) << ms_per_token << " ms/tok\n";
        std::cout << "  Throughput        : " << std::setprecision(2) << tokens_per_sec << " tokens/seg\n";
        std::cout << "=================================================================\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}