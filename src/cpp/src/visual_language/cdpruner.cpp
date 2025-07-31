// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cdpruner.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace ov::genai {
namespace cdpruner {

CDPruner::CDPruner(size_t target_token_count, float relevance_weight)
    : m_target_token_count(target_token_count), m_relevance_weight(relevance_weight) {}

std::pair<ov::Tensor, std::vector<size_t>> CDPruner::prune_visual_tokens(
    const ov::Tensor& image_features,
    const std::string& text_query,
    std::function<ov::Tensor(const std::string&)> text_encoder_fn) {
    
    auto shape = image_features.get_shape();
    size_t num_tokens = shape[0];
    size_t hidden_size = shape[1];
    
    // If we already have fewer tokens than target, return as is
    if (num_tokens <= m_target_token_count) {
        std::vector<size_t> indices(num_tokens);
        std::iota(indices.begin(), indices.end(), 0);
        return {image_features, indices};
    }
    
    // Step 1: Encode text query
    ov::Tensor text_features = text_encoder_fn(text_query);
    
    // Step 2: Normalize features
    ov::Tensor norm_image_features = normalize_features(image_features);
    ov::Tensor norm_text_features = normalize_features(text_features);
    
    // Step 3: Compute similarity matrix
    ov::Tensor similarity_matrix = compute_similarity_matrix(norm_image_features);
    
    // Step 4: Compute conditional relevance
    ov::Tensor relevance_scores = compute_conditional_relevance(norm_image_features, norm_text_features);
    
    // Step 5: Construct kernel matrix
    ov::Tensor kernel_matrix = construct_kernel_matrix(similarity_matrix, relevance_scores);
    
    // Step 6: Fast MAP inference
    std::vector<size_t> selected_indices = fast_map_inference(kernel_matrix, m_target_token_count);
    
    // Step 7: Create pruned features tensor
    ov::Shape pruned_shape = {m_target_token_count, hidden_size};
    ov::Tensor pruned_features(image_features.get_element_type(), pruned_shape);
    
    const float* input_data = image_features.data<const float>();
    float* output_data = pruned_features.data<float>();
    
    for (size_t i = 0; i < m_target_token_count; ++i) {
        size_t src_idx = selected_indices[i];
        std::copy_n(
            input_data + src_idx * hidden_size,
            hidden_size,
            output_data + i * hidden_size
        );
    }
    
    return {pruned_features, selected_indices};
}

std::pair<ov::Tensor, std::vector<size_t>> CDPruner::prune_visual_tokens(
    const ov::Tensor& image_features,
    const ov::Tensor& text_embeddings) {
    
    auto shape = image_features.get_shape();
    size_t num_tokens = shape[0];
    size_t hidden_size = shape[1];
    
    // If we already have fewer tokens than target, return as is
    if (num_tokens <= m_target_token_count) {
        std::vector<size_t> indices(num_tokens);
        std::iota(indices.begin(), indices.end(), 0);
        return {image_features, indices};
    }
    
    // Step 1: Normalize features
    ov::Tensor norm_image_features = normalize_features(image_features);
    ov::Tensor norm_text_features = normalize_features(text_embeddings);
    
    // Step 2: Compute similarity matrix
    ov::Tensor similarity_matrix = compute_similarity_matrix(norm_image_features);
    
    // Step 3: Compute conditional relevance
    ov::Tensor relevance_scores = compute_conditional_relevance(norm_image_features, norm_text_features);
    
    // Step 4: Construct kernel matrix
    ov::Tensor kernel_matrix = construct_kernel_matrix(similarity_matrix, relevance_scores);
    
    // Step 5: Fast MAP inference
    std::vector<size_t> selected_indices = fast_map_inference(kernel_matrix, m_target_token_count);
    
    // Step 6: Create pruned features tensor
    ov::Shape pruned_shape = {m_target_token_count, hidden_size};
    ov::Tensor pruned_features(image_features.get_element_type(), pruned_shape);
    
    const float* input_data = image_features.data<const float>();
    float* output_data = pruned_features.data<float>();
    
    for (size_t i = 0; i < m_target_token_count; ++i) {
        size_t src_idx = selected_indices[i];
        std::copy_n(
            input_data + src_idx * hidden_size,
            hidden_size,
            output_data + i * hidden_size
        );
    }
    
    return {pruned_features, selected_indices};
}

ov::Tensor CDPruner::compute_similarity_matrix(const ov::Tensor& features) {
    auto shape = features.get_shape();
    size_t num_tokens = shape[0];
    size_t hidden_size = shape[1];
    
    ov::Shape sim_shape = {num_tokens, num_tokens};
    ov::Tensor similarity_matrix(ov::element::f32, sim_shape);
    
    const float* feat_data = features.data<const float>();
    float* sim_data = similarity_matrix.data<float>();
    
    // Compute cosine similarity: normalized features dot product
    for (size_t i = 0; i < num_tokens; ++i) {
        for (size_t j = 0; j < num_tokens; ++j) {
            float dot_product = 0.0f;
            for (size_t k = 0; k < hidden_size; ++k) {
                dot_product += feat_data[i * hidden_size + k] * feat_data[j * hidden_size + k];
            }
            sim_data[i * num_tokens + j] = dot_product;
        }
    }
    
    return similarity_matrix;
}

ov::Tensor CDPruner::compute_conditional_relevance(
    const ov::Tensor& image_features,
    const ov::Tensor& text_features) {
    
    auto img_shape = image_features.get_shape();
    auto txt_shape = text_features.get_shape();
    size_t num_img_tokens = img_shape[0];
    size_t num_txt_tokens = txt_shape[0];
    size_t hidden_size = img_shape[1];
    
    ov::Shape relevance_shape = {num_img_tokens};
    ov::Tensor relevance_scores(ov::element::f32, relevance_shape);
    
    const float* img_data = image_features.data<const float>();
    const float* txt_data = text_features.data<const float>();
    float* rel_data = relevance_scores.data<float>();
    
    // Compute relevance as average similarity to all text tokens
    for (size_t i = 0; i < num_img_tokens; ++i) {
        float total_similarity = 0.0f;
        
        for (size_t j = 0; j < num_txt_tokens; ++j) {
            float dot_product = 0.0f;
            for (size_t k = 0; k < hidden_size; ++k) {
                dot_product += img_data[i * hidden_size + k] * txt_data[j * hidden_size + k];
            }
            total_similarity += dot_product;
        }
        
        // Negative average similarity (as in CDPruner paper)
        rel_data[i] = -total_similarity / static_cast<float>(num_txt_tokens);
    }
    
    // Normalize relevance scores to [0, 1] range
    float min_rel = *std::min_element(rel_data, rel_data + num_img_tokens);
    float max_rel = *std::max_element(rel_data, rel_data + num_img_tokens);
    float range = max_rel - min_rel + 1e-6f;  // Add small epsilon to avoid division by zero
    
    for (size_t i = 0; i < num_img_tokens; ++i) {
        rel_data[i] = (rel_data[i] - min_rel) / range;
    }
    
    return relevance_scores;
}

ov::Tensor CDPruner::construct_kernel_matrix(
    const ov::Tensor& similarity_matrix,
    const ov::Tensor& relevance_scores) {
    
    auto sim_shape = similarity_matrix.get_shape();
    size_t num_tokens = sim_shape[0];
    
    ov::Tensor kernel_matrix(ov::element::f32, sim_shape);
    
    const float* sim_data = similarity_matrix.data<const float>();
    const float* rel_data = relevance_scores.data<const float>();
    float* kernel_data = kernel_matrix.data<float>();
    
    // Optional: Apply exponential scaling to relevance scores
    // float alpha = m_relevance_weight / (2.0f * (1.0f - m_relevance_weight));
    
    // Construct kernel: relevance[i] * similarity[i,j] * relevance[j]
    for (size_t i = 0; i < num_tokens; ++i) {
        for (size_t j = 0; j < num_tokens; ++j) {
            // Use exponential scaling if desired:
            // float rel_i = std::exp(alpha * rel_data[i]);
            // float rel_j = std::exp(alpha * rel_data[j]);
            float rel_i = rel_data[i];
            float rel_j = rel_data[j];
            
            kernel_data[i * num_tokens + j] = rel_i * sim_data[i * num_tokens + j] * rel_j;
        }
    }
    
    return kernel_matrix;
}

std::vector<size_t> CDPruner::fast_map_inference(
    const ov::Tensor& kernel_matrix,
    size_t target_count) {
    
    auto shape = kernel_matrix.get_shape();
    size_t num_tokens = shape[0];
    
    if (target_count >= num_tokens) {
        std::vector<size_t> indices(num_tokens);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }
    
    const float* kernel_data = kernel_matrix.data<const float>();
    
    // Fast MAP inference algorithm from CDPruner paper
    std::vector<std::vector<float>> cis(target_count, std::vector<float>(num_tokens, 0.0f));
    std::vector<float> diagonal(num_tokens);
    std::vector<size_t> selected_indices(target_count);
    
    // Initialize diagonal
    for (size_t i = 0; i < num_tokens; ++i) {
        diagonal[i] = kernel_data[i * num_tokens + i];
    }
    
    // Iterative selection process
    for (size_t t = 0; t < target_count; ++t) {
        // Find token with maximum diagonal value
        size_t best_idx = std::distance(diagonal.begin(), 
            std::max_element(diagonal.begin(), diagonal.end()));
        selected_indices[t] = best_idx;
        
        // Update cis vectors
        float sqrt_diag = std::sqrt(diagonal[best_idx]);
        for (size_t i = 0; i < num_tokens; ++i) {
            float dot_product = 0.0f;
            for (size_t prev_t = 0; prev_t < t; ++prev_t) {
                dot_product += cis[prev_t][best_idx] * cis[prev_t][i];
            }
            cis[t][i] = (kernel_data[best_idx * num_tokens + i] - dot_product) / sqrt_diag;
        }
        
        // Update diagonal values
        for (size_t i = 0; i < num_tokens; ++i) {
            diagonal[i] -= cis[t][i] * cis[t][i];
        }
        diagonal[best_idx] = -std::numeric_limits<float>::infinity();
    }
    
    // Sort selected indices to maintain order
    std::sort(selected_indices.begin(), selected_indices.end());
    
    return selected_indices;
}

ov::Tensor CDPruner::normalize_features(const ov::Tensor& features) {
    auto shape = features.get_shape();
    size_t num_tokens = shape[0];
    size_t hidden_size = shape[1];
    
    ov::Tensor normalized_features(features.get_element_type(), shape);
    
    const float* input_data = features.data<const float>();
    float* output_data = normalized_features.data<float>();
    
    for (size_t i = 0; i < num_tokens; ++i) {
        // Compute L2 norm
        float norm = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            float val = input_data[i * hidden_size + j];
            norm += val * val;
        }
        norm = std::sqrt(norm) + 1e-8f;  // Add small epsilon for numerical stability
        
        // Normalize
        for (size_t j = 0; j < hidden_size; ++j) {
            output_data[i * hidden_size + j] = input_data[i * hidden_size + j] / norm;
        }
    }
    
    return normalized_features;
}

} // namespace cdpruner
} // namespace ov::genai
