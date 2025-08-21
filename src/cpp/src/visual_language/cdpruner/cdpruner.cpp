// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cdpruner.hpp"
#include "openvino/openvino.hpp"
#include <stdexcept>
#include <chrono>
#include <iomanip>

namespace ov::genai::cdpruner {

CDPruner::CDPruner(const Config& config) 
    : m_config(config)
    , m_relevance_calc(config)
    , m_kernel_builder(config)
    , m_dpp_selector(config) {
    
    // Validate configuration
    validate_config(config);

    if (m_config.pruning_debug_mode) {
        std::cout << "CDPruner initialized with configuration:" << std::endl;
        std::cout << "  visual_tokens_retain_percentage: " << m_config.visual_tokens_retain_percentage << "%" << std::endl;
        std::cout << "  relevance_weight: " << m_config.relevance_weight << std::endl;
        std::cout << "  use_negative_relevance: " << (m_config.use_negative_relevance ? "true" : "false") << std::endl;
        std::cout << "  enable_pruning: " << (m_config.enable_pruning ? "true" : "false") << std::endl;
        std::cout << "  use_ops_model: " << (m_config.use_ops_model ? "true (OpenVINO ops)" : "false (traditional)")
                  << std::endl;
        std::cout << "  use_dpp_ops_model: " << (m_config.use_dpp_ops_model ? "true (OpenVINO DPP ops)" : "false (traditional DPP)")
                  << std::endl;
        std::cout << "  device: " << m_config.device << std::endl;
    }
}

std::vector<std::vector<size_t>> CDPruner::select_tokens(const ov::Tensor& visual_features, 
                                                        const ov::Tensor& text_features) {
    // Input validation
    if (!m_config.enable_pruning) {
        // If pruning is disabled, return all tokens
        if (m_config.pruning_debug_mode) {
            std::cout << "Pruning is disabled. Returning all tokens." << std::endl;
        }
        return create_all_tokens_selection(visual_features);
    }

    // Calculate actual number of tokens to keep based on percentage
    size_t total_tokens = visual_features.get_shape()[1];
    size_t num_tokens_to_keep = static_cast<size_t>(std::round(total_tokens * m_config.visual_tokens_retain_percentage / 100.0));
    
    if (m_config.visual_tokens_retain_percentage == 0 || num_tokens_to_keep >= total_tokens) {
        if (m_config.pruning_debug_mode) {
            std::cout << "Warning: visual_tokens_retain_percentage is 0 or results in keeping all tokens. "
                      << "Returning all tokens without pruning." << std::endl;
        }
        return create_all_tokens_selection(visual_features);
    }

    validate_input_tensors(visual_features, text_features);
    
    // Performance timing setup
    auto overall_start = std::chrono::high_resolution_clock::now();
    
    // Get input dimensions for context
    auto visual_shape = visual_features.get_shape();
    auto text_shape = text_features.get_shape();

    try {
        std::vector<std::vector<size_t>> selected_tokens;

        if (m_config.use_ops_model) {
            // New OpenVINO ops model approach
            if (m_config.pruning_debug_mode) {
                std::cout << "Step 1-2: Computing relevance scores and kernel matrix using ov model on device: "
                          << m_config.device << "..." << std::endl;
            }
            auto computation_start = std::chrono::high_resolution_clock::now();
            auto kernel_matrix = m_kernel_builder.build(visual_features, text_features);
            auto computation_end = std::chrono::high_resolution_clock::now();

            auto computation_duration =
                std::chrono::duration_cast<std::chrono::microseconds>(computation_end - computation_start);

            if (m_config.pruning_debug_mode) {
                std::cout << "  Kernel building via ov model took: " << computation_duration.count() << " us"
                          << std::endl;
            }

            // Step 3: Select tokens using fast greedy DPP
            if (m_config.pruning_debug_mode) {
                std::cout << "Step 3: Selecting tokens using DPP..." << std::endl;
            }
            auto dpp_start = std::chrono::high_resolution_clock::now();

            if (m_config.use_dpp_ops_model) {
                selected_tokens = m_dpp_selector.select_with_ops_model(kernel_matrix, num_tokens_to_keep);
            } else {
                selected_tokens = m_dpp_selector.select(kernel_matrix, num_tokens_to_keep);
            }
            
            auto dpp_end = std::chrono::high_resolution_clock::now();

            auto dpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpp_end - dpp_start);

            if (m_config.pruning_debug_mode) {
                std::cout << "  DPP selection took: " << dpp_duration.count() << " us" << std::endl;
            }
        } else {
            // Original step-by-step approach
            if (m_config.pruning_debug_mode) {
                std::cout << "Step 1: Computing relevance scores..." << std::endl;
            }
            auto relevance_start = std::chrono::high_resolution_clock::now();
            ov::Tensor relevance_scores = m_relevance_calc.compute(visual_features, text_features);
            auto relevance_end = std::chrono::high_resolution_clock::now();
            
            auto relevance_duration = std::chrono::duration_cast<std::chrono::microseconds>(relevance_end - relevance_start);
            
            if (m_config.pruning_debug_mode) {
                std::cout << "  Relevance computation took: " << relevance_duration.count() << " us" << std::endl;
            }
            
            // Step 2: Build conditional kernel matrix
            if (m_config.pruning_debug_mode) {
                std::cout << "Step 2: Building conditional kernel matrix..." << std::endl;
            }
            auto kernel_start = std::chrono::high_resolution_clock::now();
            ov::Tensor kernel_matrix = m_kernel_builder.build(visual_features, relevance_scores);
            auto kernel_end = std::chrono::high_resolution_clock::now();
            
            auto kernel_duration = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start);

            if (m_config.pruning_debug_mode) {
                std::cout << "  Kernel building took: " << kernel_duration.count() << " us" << std::endl;
            }

            // Step 3: Select tokens using fast greedy DPP
            if (m_config.pruning_debug_mode) {
                std::cout << "Step 3: Selecting tokens using DPP..." << std::endl;
            }
            auto dpp_start = std::chrono::high_resolution_clock::now();
            
            if (m_config.use_dpp_ops_model) {
                selected_tokens = m_dpp_selector.select_with_ops_model(kernel_matrix, num_tokens_to_keep);
            } else {
                selected_tokens = m_dpp_selector.select(kernel_matrix, num_tokens_to_keep);
            }
            
            auto dpp_end = std::chrono::high_resolution_clock::now();

            auto dpp_duration = std::chrono::duration_cast<std::chrono::microseconds>(dpp_end - dpp_start);
            
            if (m_config.pruning_debug_mode) {
                std::cout << "  DPP selection took: " << dpp_duration.count() << " us" << std::endl;
            }
        }

        // Overall timing summary
        auto overall_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start);

        std::cout << "\n==== Performance Summary ====" << std::endl;
        std::cout << "Computation mode: "
                  << (m_config.use_ops_model ? (std::string("OV Model by ") + m_config.device)
                                             : "Traditional Step-by-Step by CPU")
                  << std::endl;
        std::cout << "Total processing time: " << total_duration.count() << " us (" << (total_duration.count() / 1000.0)
                  << " ms)" << std::endl;

        // Performance metrics
        size_t total_input_tokens = visual_shape[0] * visual_shape[1];
        size_t total_output_tokens = visual_shape[0] * num_tokens_to_keep;
        std::cout << "\nPerformance Metrics:" << std::endl;
        std::cout << "  Overall throughput: " << (static_cast<double>(total_input_tokens) / total_duration.count() * 1000000) << " input tokens/sec" << std::endl;
        std::cout << "  Pruning efficiency: " << (static_cast<double>(total_output_tokens) / total_duration.count() * 1000000) << " output tokens/sec" << std::endl;
        std::cout << "  Pruning ratio: " << (1.0 - static_cast<double>(num_tokens_to_keep) / visual_shape[1]) * 100 << "%" << std::endl;
        
        if (m_config.pruning_debug_mode) {
            std::cout << "CDPruner total processing time: " << total_duration.count() << " us" << std::endl;
            print_selection_statistics(visual_features, selected_tokens);
        }
        
        std::cout << "================================\n" << std::endl;
        
        return selected_tokens;

    } catch (const std::exception& e) {
        throw std::runtime_error("CDPruner::select_tokens failed: " + std::string(e.what()));
    }
}

std::vector<bool> CDPruner::create_pruning_mask(const ov::Tensor& visual_features, 
                                               const ov::Tensor& text_features) {
    auto selected_tokens = select_tokens(visual_features, text_features);
    
    auto visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t total_tokens = visual_shape[1];
    
    return FastGreedyDPP::create_mask(selected_tokens, total_tokens);
}

ov::Tensor CDPruner::apply_pruning(const ov::Tensor& visual_features, 
                                 const ov::Tensor& text_features) {
    auto visual_shape = visual_features.get_shape();
    size_t batch_size = visual_shape[0];
    size_t total_tokens = visual_shape[1];
    size_t feature_dim = visual_shape[2];
    
    // Calculate actual number of tokens to keep based on percentage
    size_t num_tokens_to_keep = static_cast<size_t>(std::round(total_tokens * m_config.visual_tokens_retain_percentage / 100.0));
    
    auto text_shape = text_features.get_shape();
    size_t text_tokens = text_shape[0];
    size_t text_feature_dim = text_shape[1];
    
    // Print detailed token statistics
    std::cout << "\n==== CDPruner Token Statistics ====" << std::endl;
    std::cout << "Input Information:" << std::endl;
    std::cout << "  Vision tokens (before pruning): " << total_tokens << std::endl;
    std::cout << "  Text tokens: " << text_tokens << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Vision feature dimension: " << feature_dim << std::endl;
    std::cout << "  Text feature dimension: " << text_feature_dim << std::endl;
    
    std::cout << "Pruning Configuration:" << std::endl;
    std::cout << "  Visual tokens percentage: " << m_config.visual_tokens_retain_percentage << "%" << std::endl;
    std::cout << "  Target vision tokens (after pruning): " << num_tokens_to_keep << std::endl;
    std::cout << "  Relevance weight: " << m_config.relevance_weight << std::endl;
    std::cout << "  Use negative relevance: " << (m_config.use_negative_relevance ? "true (CLIP/LLaVA mode)" : "false (standard mode)") << std::endl;
    
    // Calculate pruning statistics
    float pruning_ratio = 1.0f - static_cast<float>(num_tokens_to_keep) / static_cast<float>(total_tokens);
    float reduction_percentage = pruning_ratio * 100.0f;
    size_t tokens_removed = total_tokens - num_tokens_to_keep;
    
    std::cout << "Pruning Impact:" << std::endl;
    std::cout << "  Tokens to be removed: " << tokens_removed << " (" << reduction_percentage << "%)" << std::endl;
    std::cout << "  Tokens to be kept: " << num_tokens_to_keep << " (" << (100.0f - reduction_percentage) << "%)" << std::endl;
    std::cout << "  Memory reduction ratio: " << pruning_ratio << std::endl;
    
    auto selected_tokens = select_tokens(visual_features, text_features);
    
    // Create output tensor with selected tokens only
    ov::Tensor pruned_features(visual_features.get_element_type(), 
                              {batch_size, num_tokens_to_keep, feature_dim});
    
    const float* input_data = visual_features.data<const float>();
    float* output_data = pruned_features.data<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& batch_selected = selected_tokens[b];
        
        for (size_t t = 0; t < batch_selected.size(); ++t) {
            size_t src_token_idx = batch_selected[t];
            
            // Copy features for this token
            for (size_t f = 0; f < feature_dim; ++f) {
                size_t src_idx = b * total_tokens * feature_dim + src_token_idx * feature_dim + f;
                size_t dst_idx = b * num_tokens_to_keep * feature_dim + t * feature_dim + f;
                output_data[dst_idx] = input_data[src_idx];
            }
        }
    }
    
    // Print pruning completion statistics
    std::cout << "\nPruning Completed Successfully!" << std::endl;
    std::cout << "Final Results:" << std::endl;
    std::cout << "  Original vision tokens: " << total_tokens << std::endl;
    std::cout << "  Pruned vision tokens: " << num_tokens_to_keep << std::endl;
    std::cout << "  Actual pruning ratio: " << pruning_ratio << " (" << reduction_percentage << "% reduction)" << std::endl;
    std::cout << "  Output tensor shape: [" << batch_size << ", " << num_tokens_to_keep << ", " << feature_dim << "]" << std::endl;

    // Update statistics
    m_last_statistics.total_tokens = total_tokens;
    m_last_statistics.selected_tokens = num_tokens_to_keep;
    m_last_statistics.pruning_ratio = 1.0f - static_cast<float>(num_tokens_to_keep) / total_tokens;
    m_last_statistics.batch_size = batch_size;

    // Show selected token indices for first batch (for debugging)
    if (!selected_tokens.empty() && m_config.pruning_debug_mode) {
        std::cout << "Selected token indices (batch 0): [";
        const auto& first_batch_tokens = selected_tokens[0];
        for (size_t i = 0; i < std::min(static_cast<size_t>(10), first_batch_tokens.size()); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << first_batch_tokens[i];
        }
        if (first_batch_tokens.size() > 10) {
            std::cout << ", ... (+" << (first_batch_tokens.size() - 10) << " more)";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "================================\n" << std::endl;
    
    return pruned_features;
}

float CDPruner::compute_pruning_ratio() const {
    // Return the percentage as a ratio (30% -> 0.30)
    return m_config.visual_tokens_retain_percentage / 100.0f;
}

size_t CDPruner::get_default_token_count() const {
    // LLaVA typical token count (can be made configurable)
    return 576; // 24x24 patches for most LLaVA configurations
}

PruningStatistics CDPruner::get_last_pruning_statistics() const {
    return m_last_statistics;
}

void CDPruner::validate_config(const Config& config) {
    if (!config.enable_pruning)
        return;

    if (config.visual_tokens_retain_percentage < 0 || config.visual_tokens_retain_percentage > 100) {
        throw std::invalid_argument("visual_tokens_retain_percentage must be between 1 and 100");
    }
    
    if (config.relevance_weight < 0.0f || config.relevance_weight > 1.0f) {
        throw std::invalid_argument("relevance_weight must be in range [0.0, 1.0]");
    }
    
    if (config.numerical_threshold < 0.0f) {
        throw std::invalid_argument("numerical_threshold must be positive");
    }
    
    if (config.device.empty()) {
        throw std::invalid_argument("device cannot be empty");
    }
}

bool CDPruner::update_config(const Config& new_config) {
    try {
        // Validate the new configuration first
        validate_config(new_config);

        // Update the configuration
        m_config = new_config;

        // Reinitialize components with new config
        m_relevance_calc = RelevanceCalculator(new_config);
        m_kernel_builder = ConditionalKernelBuilder(new_config);
        m_dpp_selector = FastGreedyDPP(new_config);

        if (m_config.pruning_debug_mode) {
            std::cout << "CDPruner configuration updated successfully:" << std::endl;
            std::cout << "  visual_tokens_retain_percentage: " << m_config.visual_tokens_retain_percentage << "%" << std::endl;
            std::cout << "  relevance_weight: " << m_config.relevance_weight << std::endl;
            std::cout << "  enable_pruning: " << (m_config.enable_pruning ? "true" : "false") << std::endl;
        }

        return true;
    } catch (const std::exception& e) {
        if (m_config.pruning_debug_mode) {
            std::cerr << "Failed to update CDPruner configuration: " << e.what() << std::endl;
        }
        return false;
    }
}

void CDPruner::validate_input_tensors(const ov::Tensor& visual_features, 
                                    const ov::Tensor& text_features) {
    // Validate visual features
    if (visual_features.get_shape().size() != 3) {
        throw std::invalid_argument("Visual features must be 3D tensor [B, N, D]");
    }
    
    // Validate text features
    if (text_features.get_shape().size() != 2) {
        throw std::invalid_argument("Text features must be 2D tensor [M, D]");
    }
    
    auto visual_shape = visual_features.get_shape();
    auto text_shape = text_features.get_shape();
    
    // Check feature dimension consistency
    if (visual_shape[2] != text_shape[1]) {
        throw std::invalid_argument("Visual and text features must have same feature dimension");
    }
    
    // Calculate actual token count based on percentage
    size_t num_tokens_to_keep = static_cast<size_t>(std::round(visual_shape[1] * m_config.visual_tokens_retain_percentage / 100.0));
    
    // Check if percentage would result in zero tokens
    if (num_tokens_to_keep == 0) {
        throw std::invalid_argument("Percentage is too small, would result in zero tokens");
    }
    
    // Check tensor data types
    if (visual_features.get_element_type() != ov::element::f32 || 
        text_features.get_element_type() != ov::element::f32) {
        throw std::invalid_argument("Input tensors must be float32 type");
    }
}

std::vector<std::vector<size_t>> CDPruner::create_all_tokens_selection(const ov::Tensor& visual_features) {
    auto shape = visual_features.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens = shape[1];
    
    std::vector<std::vector<size_t>> all_tokens(batch_size);
    
    for (size_t b = 0; b < batch_size; ++b) {
        all_tokens[b].reserve(total_tokens);
        for (size_t t = 0; t < total_tokens; ++t) {
            all_tokens[b].push_back(t);
        }
    }
    
    return all_tokens;
}

void CDPruner::print_selection_statistics(const ov::Tensor& visual_features, 
                                        const std::vector<std::vector<size_t>>& selected_tokens) {
    auto shape = visual_features.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens = shape[1];
    size_t selected_token_count = static_cast<size_t>(std::round(total_tokens * m_config.visual_tokens_retain_percentage / 100.0));
    
    std::cout << "Selection Statistics:" << std::endl;
    std::cout << "  Total tokens: " << total_tokens << std::endl;
    std::cout << "  Selected tokens: " << selected_token_count << " (" << m_config.visual_tokens_retain_percentage << "%)" << std::endl;
    std::cout << "  Pruning ratio: " << (1.0f - static_cast<float>(selected_token_count) / total_tokens) * 100.0f << "%" << std::endl;
    
    for (size_t b = 0; b < batch_size && b < 3; ++b) { // Show first 3 batches max
        std::cout << "  Batch " << b << " selected indices: [";
        for (size_t i = 0; i < selected_tokens[b].size() && i < 10; ++i) { // Show first 10 indices
            if (i > 0) std::cout << ", ";
            std::cout << selected_tokens[b][i];
        }
        if (selected_tokens[b].size() > 10) {
            std::cout << ", ...";
        }
        std::cout << "]" << std::endl;
    }
    
    // Update statistics
    m_last_statistics.total_tokens = total_tokens;
    m_last_statistics.selected_tokens = selected_token_count;
    m_last_statistics.pruning_ratio = 1.0f - static_cast<float>(selected_token_count) / total_tokens;
    m_last_statistics.batch_size = batch_size;
}

} // namespace ov::genai::cdpruner 