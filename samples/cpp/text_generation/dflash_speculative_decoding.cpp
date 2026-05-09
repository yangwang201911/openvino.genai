// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 4) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> <DFLASH_DRAFT_MODEL_DIR> '<PROMPT>' [DEVICE]");
    }

    std::string main_model_path = argv[1];
    std::string draft_model_path = argv[2];
    std::string prompt = argv[3];
    std::string device = argc > 4 ? argv[4] : "CPU";

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    config.num_assistant_tokens = 16;

    ov::genai::LLMPipeline pipe(
        main_model_path,
        device,
        ov::genai::draft_model(draft_model_path, device));

    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    auto result = pipe.generate(prompt, config, streamer);
    auto sd_perf_metrics = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(result.extended_perf_metrics);

    if (sd_perf_metrics) {
        auto main_model_metrics = sd_perf_metrics->main_model_metrics;
        std::cout << "\nMAIN MODEL (Target)" << std::endl;
        std::cout << "  Generate time: " << main_model_metrics.get_generate_duration().mean << " ms" << std::endl;
        std::cout << "  TTFT: " << main_model_metrics.get_ttft().mean  << " ms" << std::endl;
        std::cout << "  TPOT: " << main_model_metrics.get_tpot().mean  << " ms/iteration " << std::endl;
        std::cout << "  Num generated token: " << main_model_metrics.get_num_generated_tokens() << " tokens" << std::endl;
        std::cout << "  Num accepted token: " << sd_perf_metrics->get_num_accepted_tokens() << " tokens" << std::endl;

        auto draft_model_metrics = sd_perf_metrics->draft_model_metrics;
        std::cout << "\nDRAFT MODEL (DFlash)" << std::endl;
        std::cout << "  Generate time: " << draft_model_metrics.get_generate_duration().mean << " ms" << std::endl;
        std::cout << "  Num generated token: " << draft_model_metrics.get_num_generated_tokens() << " tokens" << std::endl;
    }
    std::cout << std::endl;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
