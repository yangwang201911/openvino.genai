// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/tensor.hpp"
#include <vector>
#include <string>

namespace ov::genai {
namespace cdpruner {

/**
 * @brief CDPruner (Conditional Determinantal Point Process) for visual token pruning
 * 
 * This class implements the CDPruner method from the paper:
 * "Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs"
 * 
 * The method reduces visual token count while maintaining both diversity and quality
 * by using conditional DPP to select the subset of tokens to keep.
 */
class CDPruner {
public:
    /**
     * @brief Constructor
     * @param target_token_count Number of visual tokens to retain after pruning
     * @param relevance_weight Weight for relevance score influence (theta parameter)
     */
    CDPruner(size_t target_token_count = 32, float relevance_weight = 0.5f);

    /**
     * @brief Apply CDPruner to reduce visual tokens
     * 
     * @param image_features Vision features tensor [N, hidden_size]
     * @param text_query Text query for conditional relevance computation
     * @param text_encoder_fn Function to encode text query into embeddings
     * @return Tuple of (pruned_image_features, selected_indices)
     */
    std::pair<ov::Tensor, std::vector<size_t>> prune_visual_tokens(
        const ov::Tensor& image_features,
        const std::string& text_query,
        std::function<ov::Tensor(const std::string&)> text_encoder_fn
    );

    /**
     * @brief Apply CDPruner to reduce visual tokens (using pre-computed text embeddings)
     * 
     * @param image_features Vision features tensor [N, hidden_size]
     * @param text_embeddings Pre-computed text embeddings tensor [M, hidden_size]
     * @return Tuple of (pruned_image_features, selected_indices)
     */
    std::pair<ov::Tensor, std::vector<size_t>> prune_visual_tokens(
        const ov::Tensor& image_features,
        const ov::Tensor& text_embeddings
    );

    /**
     * @brief Set the number of tokens to retain
     * @param count Target token count
     */
    void set_target_token_count(size_t count) { m_target_token_count = count; }

    /**
     * @brief Set relevance weight parameter
     * @param weight Relevance weight (theta parameter from paper)
     */
    void set_relevance_weight(float weight) { m_relevance_weight = weight; }

private:
    /**
     * @brief Compute cosine similarity matrix between image features
     * @param features Image features tensor [N, hidden_size]
     * @return Similarity matrix [N, N]
     */
    ov::Tensor compute_similarity_matrix(const ov::Tensor& features);

    /**
     * @brief Compute relevance scores for image features conditioned on text
     * @param image_features Image features tensor [N, hidden_size]
     * @param text_features Text features tensor [M, hidden_size]
     * @return Relevance scores [N]
     */
    ov::Tensor compute_conditional_relevance(
        const ov::Tensor& image_features,
        const ov::Tensor& text_features
    );

    /**
     * @brief Construct conditional DPP kernel matrix
     * @param similarity_matrix Similarity matrix [N, N]
     * @param relevance_scores Relevance scores [N]
     * @return Kernel matrix [N, N]
     */
    ov::Tensor construct_kernel_matrix(
        const ov::Tensor& similarity_matrix,
        const ov::Tensor& relevance_scores
    );

    /**
     * @brief Fast MAP inference for conditional DPP
     * @param kernel_matrix Kernel matrix [N, N]
     * @param target_count Number of tokens to select
     * @return Selected token indices
     */
    std::vector<size_t> fast_map_inference(
        const ov::Tensor& kernel_matrix,
        size_t target_count
    );

    /**
     * @brief Normalize tensor features
     * @param features Input features tensor
     * @return L2-normalized features
     */
    ov::Tensor normalize_features(const ov::Tensor& features);

    size_t m_target_token_count;
    float m_relevance_weight;
};

} // namespace cdpruner
} // namespace ov::genai
