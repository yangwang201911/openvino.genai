// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "logger.hpp"

namespace ov::genai {

namespace {
// Log non-KV-cache model inputs for diagnosis (skip key_cache.*, value_cache.* to reduce noise)
void log_model_io(const std::string& label, const std::shared_ptr<ov::Model>& model) {
    GENAI_INFO("[DFlash] " + label + " inputs:");
    for (const auto& input : model->inputs()) {
        const std::string name = input.get_any_name();
        if (name.find("key_cache") != std::string::npos || name.find("value_cache") != std::string::npos)
            continue;
        GENAI_INFO("[DFlash]   " + name + ": " + input.get_partial_shape().to_string());
    }
    size_t kv_count = 0;
    for (const auto& input : model->inputs()) {
        const std::string name = input.get_any_name();
        if (name.find("key_cache") != std::string::npos || name.find("value_cache") != std::string::npos)
            ++kv_count;
    }
    if (kv_count > 0)
        GENAI_INFO("[DFlash]   ... and " + std::to_string(kv_count) + " KV cache inputs");

    GENAI_INFO("[DFlash] " + label + " outputs:");
    for (const auto& output : model->outputs()) {
        std::string name;
        try { name = output.get_any_name(); } catch (...) { name = "(unnamed)"; }
        GENAI_INFO("[DFlash]   " + name + ": " + output.get_partial_shape().to_string());
    }
}

std::string format_layers(const std::vector<int32_t>& layers) {
    std::string s = "[";
    for (size_t i = 0; i < layers.size(); ++i) {
        if (i > 0) s += ",";
        s += std::to_string(layers[i]);
    }
    return s + "]";
}
}  // namespace

ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashDecodingImpl(
    const ov::genai::ModelDesc& main_model_desc,
    const ov::genai::ModelDesc& draft_model_desc,
    const utils::dflash::DFlashRTInfo& dflash_info)
    : m_block_size(dflash_info.block_size),
      m_mask_token_id(dflash_info.mask_token_id) {
    GENAI_INFO("[DFlash] ========== DFlash Pipeline Initialization ==========");
    GENAI_INFO("[DFlash] Config: block_size=" + std::to_string(m_block_size) +
               ", mask_token_id=" + std::to_string(m_mask_token_id) +
               ", target_layers=" + format_layers(dflash_info.hidden_layers_list));

    auto scheduler_configs = init_speculative_models(main_model_desc, draft_model_desc);
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(main_model && draft_model);

    // Log model topology for diagnosis
    log_model_io("Main model (before transform)", main_model);
    log_model_io("Draft model (before transform)", draft_model);

    // Check if draft model has standard 'input_ids' input required by the CB pipeline
    bool draft_has_input_ids = false;
    for (const auto& input : draft_model->inputs()) {
        if (input.get_any_name().find("input_ids") != std::string::npos) {
            draft_has_input_ids = true;
            break;
        }
    }
    if (!draft_has_input_ids) {
        GENAI_WARN("[DFlash] Draft model has NO 'input_ids' input! "
                   "This is a native DFlash draft model (expects noise_embedding + target_hidden). "
                   "The GenAI CB pipeline model runner requires 'input_ids' as input. "
                   "Pipeline CANNOT work correctly with this draft model.");
    }

    auto main_device = main_model_desc.device;
    std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;
    ov::AnyMap draft_properties =
        draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;

    const Tokenizer& main_model_tokenizer = main_model_desc.tokenizer;
    const Tokenizer& draft_model_tokenizer = draft_model_desc.tokenizer;
    m_tokenizer = main_model_tokenizer;

    // Step 1: Share embedding weights between main and draft models
    GENAI_INFO("[DFlash] Step 1/4: Sharing vocabulary (embedding weights) main -> draft...");
    utils::eagle3::share_vocabulary(main_model, draft_model);

    // Step 2: Transform main model to export hidden states from selected layers
    GENAI_INFO("[DFlash] Step 2/4: Adding hidden state export from " +
               std::to_string(dflash_info.hidden_layers_list.size()) + " main model layers: " +
               format_layers(dflash_info.hidden_layers_list));
    utils::eagle3::transform_hidden_state(main_model, dflash_info.hidden_layers_list);

    // Step 3: Try to move FC layer from draft to main
    bool has_eagle3_draft_structure = false;
    GENAI_INFO("[DFlash] Step 3/4: Attempting to extract FC layer from draft model...");
    try {
        utils::eagle3::move_fc_from_draft_to_main(draft_model, main_model);
        utils::eagle3::transform_hidden_state(draft_model, {-1});
        has_eagle3_draft_structure = true;
        GENAI_INFO("[DFlash] Step 3/4: SUCCESS - Draft has EAGLE3-compatible FC layer and hidden state structure.");
    } catch (const ov::Exception& e) {
        GENAI_WARN("[DFlash] Step 3/4: FAILED - " + std::string(e.what()));
        GENAI_WARN("[DFlash] FALLBACK to standard speculative decoding:");
        GENAI_WARN("[DFlash]   - Draft model runs as independent autoregressive model (full Qwen3-8B)");
        GENAI_WARN("[DFlash]   - NO hidden state transfer between main and draft");
        GENAI_WARN("[DFlash]   - This is NOT true DFlash block diffusion");
        GENAI_WARN("[DFlash]   - Both main and draft are full-size models -> performance will be WORSE than no SD");
    }

    // Log model topology after transformation
    log_model_io("Main model (after transform)", main_model);
    log_model_io("Draft model (after transform)", draft_model);

    // Step 4: Create pipelines
    if (has_eagle3_draft_structure) {
        GENAI_INFO("[DFlash] Step 4/4: Creating EAGLE3-style pipelines (hidden state export/import enabled)...");
        m_main_pipeline = std::make_shared<ContinuousBatchingForEagle3DecodingImpl>(
            main_model, main_model_tokenizer, main_model_desc.generation_config,
            scheduler_configs.first, main_device, main_model_desc.properties, true);

        m_draft_pipeline = std::make_shared<ContinuousBatchingForEagle3DecodingImpl>(
            draft_model, draft_model_tokenizer, draft_model_desc.generation_config,
            scheduler_configs.second, draft_device, draft_properties, false);

        update_dflash_pipeline_params();
    } else {
        GENAI_INFO("[DFlash] Step 4/4: Creating standard SD pipelines (ContinuousBatchingForSpeculativeDecodingImpl)...");
        m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
            main_model, main_model_tokenizer, main_model_desc.generation_config,
            scheduler_configs.first, main_device, main_model_desc.properties, true);

        m_draft_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
            draft_model, draft_model_tokenizer, draft_model_desc.generation_config,
            scheduler_configs.second, draft_device, draft_properties, false);
    }

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    m_draft_pipeline->raw_perf_metrics.m_inference_durations = {{ MicroSeconds(0.0f) }};

    GENAI_INFO("[DFlash] ========== Pipeline Ready ==========");
    GENAI_INFO("[DFlash] pipeline_type=" + std::string(has_eagle3_draft_structure ? "eagle3-hidden-state" : "standard-SD-fallback") +
               ", main_device=" + main_device + ", draft_device=" + draft_device);
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::update_dflash_pipeline_params() {
    auto m_main_dflash_pipeline = std::dynamic_pointer_cast<ContinuousBatchingForEagle3DecodingImpl>(m_main_pipeline);
    auto m_draft_dflash_pipeline = std::dynamic_pointer_cast<ContinuousBatchingForEagle3DecodingImpl>(m_draft_pipeline);

    if (m_main_dflash_pipeline) {
        m_main_dflash_pipeline->set_hidden_state_export_needed(true);
        GENAI_INFO("[DFlash] Main pipeline: hidden state EXPORT enabled");
    }
    if (m_draft_dflash_pipeline) {
        m_draft_dflash_pipeline->set_hidden_state_export_needed(true);
        m_draft_dflash_pipeline->set_hidden_state_import_needed(true);
        m_draft_dflash_pipeline->set_hidden_state_internal_needed(true);
        GENAI_INFO("[DFlash] Draft pipeline: hidden state EXPORT + IMPORT + INTERNAL enabled");
    }
}

GenerationHandle
ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(uint64_t request_id,
                                                             const ov::Tensor& input_ids,
                                                             const ov::genai::GenerationConfig& sampling_params,
                                                             std::optional<ov::Tensor> token_type_ids,
                                                             std::optional<ov::Tensor> prompt_ids,
                                                             std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};

    GENAI_INFO("[DFlash] add_request: id=" + std::to_string(request_id) +
               ", input_ids_len=" + std::to_string(input_ids.get_shape().back()) +
               ", num_assistant_tokens=" + std::to_string(sampling_params.num_assistant_tokens) +
               ", draft_input=SAME_AS_MAIN (no shift)");

    m_draft_generations.insert({request_id,
        m_draft_pipeline->add_request(request_id, input_ids, draft_sampling_params, token_type_ids, prompt_ids, lm_extra_inputs)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params, token_type_ids, prompt_ids, lm_extra_inputs);
}

GenerationHandle
ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(uint64_t request_id,
                                                             const std::string& prompt,
                                                             const ov::genai::GenerationConfig& sampling_params) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};

    GENAI_INFO("[DFlash] add_request: id=" + std::to_string(request_id) +
               ", prompt_len=" + std::to_string(prompt.size()) +
               ", num_assistant_tokens=" + std::to_string(sampling_params.num_assistant_tokens) +
               ", draft_input=SAME_AS_MAIN (no shift)");

    m_draft_generations.insert({request_id,
        m_draft_pipeline->add_request(request_id, prompt, draft_sampling_params)});
    return m_main_pipeline->add_request(request_id, prompt, sampling_params);
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::DFlashDecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    const std::optional<std::vector<ov::Tensor>>& token_type_ids,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids,
    const std::optional<std::vector<ov::Tensor>>& prompt_ids,
    const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list
) {
    GENAI_INFO("[DFlash] generate() called: " + std::to_string(input_ids.size()) + " request(s), "
               "block_size=" + std::to_string(m_block_size));

    GenerateStrategy strategy;
    strategy.prepare_request = [this](size_t idx,
                                      const ov::Tensor& in_ids,
                                      GenerationConfig& main_cfg,
                                      GenerationConfig& draft_cfg,
                                      ov::Tensor& main_in,
                                      ov::Tensor& draft_in) {
        if (main_cfg.num_assistant_tokens == 0) {
            main_cfg.num_assistant_tokens = static_cast<int64_t>(m_block_size);
            draft_cfg.num_assistant_tokens = main_cfg.num_assistant_tokens;
        }
        draft_cfg.ignore_eos = true;
        draft_cfg.stop_strings = {};
        main_in = in_ids;
        draft_in = in_ids;

        GENAI_INFO("[DFlash] prepare_request[" + std::to_string(idx) + "]: "
                   "input_len=" + std::to_string(in_ids.get_shape().back()) +
                   ", num_assistant_tokens=" + std::to_string(main_cfg.num_assistant_tokens));
    };

    strategy.check_streaming = [](const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr,
                                  const std::vector<ov::Tensor>& input_ids,
                                  const std::vector<GenerationConfig>& sampling_params) {
        OPENVINO_ASSERT(!streamer_ptr->has_callback() ||
                            (input_ids.size() == 1 &&
                             (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial())),
                        "DFlash: streaming is only supported for single-request greedy or multinomial decoding");
    };

    strategy.start_timer = []() -> TimePoint {
        return std::chrono::steady_clock::now();
    };

    strategy.stop_timer = [](TimePoint start) -> uint64_t {
        auto stop = std::chrono::steady_clock::now();
        return PerfMetrics::get_microsec(stop - start);
    };

    return generate_common(this, input_ids, sampling_params, streamer, 
                          token_type_ids.has_value() ? std::optional<std::vector<ov::Tensor>>(token_type_ids.value()) : std::nullopt,
                          prompt_ids.has_value() ? std::optional<std::vector<ov::Tensor>>(prompt_ids.value()) : std::nullopt,
                          strategy);
}

}  // namespace ov::genai
