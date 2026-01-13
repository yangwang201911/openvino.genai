// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderMiniCPM : public VisionEncoder {
    // A resampler model to resample image embeddings.
    // [N, H*W, old_hidden_size] is the input shape.
    // [N, query_num, hidden_size] is the output shape.
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_resampler;
    // Precomputed positional embeddings for the resampler.
    // [70, 70, hidden_size]. 70 is the initial guess of the image
    // height and width after dividing by patch_size.
    ov::Tensor m_pos_embed_cache;
    // VLM config
    VLMConfig m_vlm_config;

    ov::Tensor resample(const ov::Tensor& encoded_image, const ImageSize& target_size, size_t pad_to_max);

    ResampledImage resample_encoded_image(const EncodedImage& image, const ov::Tensor& slices, const ImageSize& target_sizes);
public:
    VisionEncoderMiniCPM(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);


    VisionEncoderMiniCPM(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);
    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderMiniCPM : public InputsEmbedder::IInputsEmbedder {

public:
    InputsEmbedderMiniCPM(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);
    
    InputsEmbedderMiniCPM(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

private:
    /**
     * @brief Prepare and merge all vision embeddings from encoded images.
     * Collects all vision features (resampled sources and slices) into a single tensor.
     * @param images Vector of encoded images
     * @param images_sequence Sequence of image indices
     * @return Merged vision embeddings tensor [1, total_tokens, hidden_size]
     */
    ov::Tensor prepare_vision_embeddings(
        const std::vector<EncodedImage>& images,
        const std::vector<size_t>& images_sequence,
        std::vector<size_t>& tokens_per_image,
        std::vector<std::array<size_t, 3>>& images_grid_thw);

    /**
     * @brief Merge text embeddings with vision embeddings.
     * Similar to qwen2vl_utils::merge_text_and_video_image_embeddings but adapted for MiniCPM.
     * @param input_ids Input token IDs
     * @param text_embeds Text embeddings
     * @param vision_embeddings Vision embeddings (can be pruned)
     * @param images Vector of encoded images
     * @param images_sequence Sequence of image indices
     * @param im_start_id Image start token ID
     * @param im_end_id Image end token ID
     * @param slice_start_id Slice start token ID
     * @param slice_end_id Slice end token ID
     * @return Merged embeddings tensor
     */
    ov::Tensor merge_text_and_vision_embeddings(
        const ov::Tensor& input_ids,
        const ov::Tensor& text_embeds,
        const ov::Tensor& vision_embeddings,
        const std::vector<EncodedImage>& images,
        const std::vector<size_t>& tokens_per_image,
        const std::vector<size_t>& images_sequence,
        int64_t im_start_id,
        int64_t im_end_id,
        int64_t slice_start_id,
        int64_t slice_end_id);
};

} // namespace ov::genai
