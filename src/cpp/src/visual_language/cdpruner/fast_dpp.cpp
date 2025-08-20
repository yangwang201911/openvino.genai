// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fast_dpp.hpp"
#include "openvino/openvino.hpp"
#include "openvino/op/ops.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace ov::genai::cdpruner {

FastGreedyDPP::FastGreedyDPP(const Config& config) : m_config(config) {
    // Constructor implementation
}

std::vector<std::vector<size_t>> FastGreedyDPP::select(const ov::Tensor& kernel, size_t num_tokens) {
    // Input validation
    if (kernel.get_shape().size() != 3) {
        throw std::invalid_argument("Kernel must be 3D tensor [B, N, N]");
    }
    
    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens = shape[1];
    
    if (shape[1] != shape[2]) {
        throw std::invalid_argument("Kernel matrix must be square [B, N, N]");
    }
    
    if (num_tokens > total_tokens) {
        throw std::invalid_argument("Cannot select more tokens than available");
    }
    
    std::vector<std::vector<size_t>> batch_results(batch_size);
    
    // Process each batch independently
    for (size_t b = 0; b < batch_size; ++b) {
        batch_results[b] = select_single_batch(kernel, b, num_tokens);
    }
    
    return batch_results;
}

std::vector<size_t> FastGreedyDPP::select_single_batch(const ov::Tensor& kernel, size_t batch_idx, size_t num_tokens) {
    auto shape = kernel.get_shape();
    size_t total_tokens = shape[1];
    
    // Initialize working tensors for this batch
    // cis: Orthogonalized vectors [T, N] where T is the number of selected tokens
    ov::Tensor cis(ov::element::f32, {num_tokens, total_tokens});
    
    // di2s: Diagonal elements (marginal gains) [N]
    ov::Tensor di2s(ov::element::f32, {total_tokens});
    
    // Copy diagonal elements from kernel for this batch
    const float* kernel_data = kernel.data<const float>();
    float* di2s_data = di2s.data<float>();
    
    for (size_t i = 0; i < total_tokens; ++i) {
        size_t diag_idx = batch_idx * total_tokens * total_tokens + i * total_tokens + i;
        di2s_data[i] = kernel_data[diag_idx];
    }
    
    std::vector<size_t> selected_indices;
    selected_indices.reserve(num_tokens);
    
    float* cis_data = cis.data<float>();
    std::memset(cis_data, 0, cis.get_byte_size());
    
    // Greedy selection loop - this is the core DPP algorithm
    for (size_t t = 0; t < num_tokens; ++t) {
        // Find the token with maximum marginal gain
        size_t best_idx = argmax(di2s);
        selected_indices.push_back(best_idx);
        
        // Compute the new orthogonalized vector e_i
        // eis = (kernel[batch, best_idx] - sum(cis[:t] * cis[:t, best_idx])) / sqrt(di2s[best_idx])
        update_orthogonal_vector(kernel, batch_idx, best_idx, t, cis, di2s);
        
        // Update marginal gains by subtracting the squared new orthogonal vector
        // di2s -= square(eis)
        update_marginal_gains(t, best_idx, cis, di2s);

        // Debug output: print cis matrix content
        if (m_config.pruning_debug_mode && t < 10) {
            std::cout << "=== CIS Matrix Content after iteration " << t << " ===" << std::endl;
            std::cout << "CIS matrix shape: [" << (t+1) << ", " << total_tokens << "]" << std::endl;
            
            const float* cis_data_debug = cis.data<const float>();
            size_t print_tokens = std::min(total_tokens, static_cast<size_t>(10));
            
            // Print each orthogonal vector (each row of cis) - only first 10 elements
            for (size_t row = 0; row <= t; ++row) {
                std::cout << "cis[" << row << "] (orthogonal vector for selected token " 
                          << selected_indices[row] << "): [";
                
                for (size_t col = 0; col < print_tokens; ++col) {
                    if (col > 0) std::cout << ", ";
                    size_t idx = row * total_tokens + col;
                    std::cout << std::fixed << std::setprecision(4) << cis_data_debug[idx];
                }
                
                if (total_tokens > 10) {
                    std::cout << ", ... (" << (total_tokens - 10) << " more)";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << std::endl;
        }

        // Debug output: print updated conditional kernel matrix after each selection
        if (m_config.pruning_debug_mode && t < 10) {
            // Print current selected indices
            std::cout << "Selected tokens so far: [";
            for (size_t i = 0; i < selected_indices.size(); ++i) {
                if (i > 0)
                    std::cout << ", ";
                std::cout << selected_indices[i];
            }
            std::cout << "]" << std::endl;

            // Print current marginal gains (di2s) - limited to first 10 elements
            std::cout << "Current marginal gains: [";
            const float* di2s_data_debug = di2s.data<const float>();
            size_t print_gains_size = std::min(total_tokens, static_cast<size_t>(10));
            
            for (size_t i = 0; i < print_gains_size; ++i) {
                if (i > 0)
                    std::cout << ", ";
                if (di2s_data_debug[i] == -std::numeric_limits<float>::infinity()) {
                    std::cout << "-inf";
                } else {
                    std::cout << std::fixed << std::setprecision(4) << di2s_data_debug[i];
                }
            }
            if (total_tokens > 10) {
                std::cout << ", ... (" << (total_tokens - 10) << " more elements)";
            }
            std::cout << "]" << std::endl << std::endl;
        }

        // Set the selected token's gain to negative infinity to prevent re-selection
        di2s_data[best_idx] = -std::numeric_limits<float>::infinity();
    }
    
    // Sort the selected indices for deterministic output
    std::sort(selected_indices.begin(), selected_indices.end());
    
    return selected_indices;
}

size_t FastGreedyDPP::argmax(const ov::Tensor& scores) {
    const float* data = scores.data<const float>();
    size_t size = scores.get_size();
    
    if (size == 0) {
        throw std::invalid_argument("Cannot find argmax of empty tensor");
    }
    
    size_t best_idx = 0;
    float best_value = data[0];
    
    for (size_t i = 1; i < size; ++i) {
        if (data[i] > best_value) {
            best_value = data[i];
            best_idx = i;
        }
    }
    
    return best_idx;
}

void FastGreedyDPP::update_orthogonal_vector(const ov::Tensor& kernel, size_t batch_idx, size_t selected_idx, 
                                           size_t iteration, ov::Tensor& cis, const ov::Tensor& di2s) {
    // This implements the key DPP orthogonalization step:
    // eis = (kernel[batch, selected_idx] - sum(cis[:iteration] * cis[:iteration, selected_idx])) / sqrt(di2s[selected_idx])
    
    auto kernel_shape = kernel.get_shape();
    size_t total_tokens = kernel_shape[1];
    
    const float* kernel_data = kernel.data<const float>();
    const float* di2s_data = di2s.data<const float>();
    float* cis_data = cis.data<float>();
    
    // Get the normalization factor
    float norm_factor = std::sqrt(di2s_data[selected_idx] + m_config.numerical_threshold);
    
    // Compute the new orthogonal vector for each token
    for (size_t j = 0; j < total_tokens; ++j) {
        // Get kernel[batch_idx, selected_idx, j]
        size_t kernel_idx = batch_idx * total_tokens * total_tokens + selected_idx * total_tokens + j;
        float kernel_val = kernel_data[kernel_idx];
        
        // Subtract the projection onto previously selected vectors
        // sum(cis[:iteration, selected_idx] * cis[:iteration, j])
        float projection = 0.0f;
        for (size_t prev_t = 0; prev_t < iteration; ++prev_t) {
            size_t cis_selected_idx = prev_t * total_tokens + selected_idx;
            size_t cis_j_idx = prev_t * total_tokens + j;
            projection += cis_data[cis_selected_idx] * cis_data[cis_j_idx];
        }
        
        // Store the orthogonalized vector element
        size_t cis_current_idx = iteration * total_tokens + j;
        cis_data[cis_current_idx] = (kernel_val - projection) / norm_factor;
    }
}

void FastGreedyDPP::update_marginal_gains(size_t iteration, size_t selected_idx, 
                                        const ov::Tensor& cis, ov::Tensor& di2s) {
    // This implements: di2s -= square(eis)
    // where eis is the newly computed orthogonal vector cis[iteration, :]
    
    auto cis_shape = cis.get_shape();
    size_t total_tokens = cis_shape[1];
    
    const float* cis_data = cis.data<const float>();
    float* di2s_data = di2s.data<float>();
    
    // Update marginal gains for all tokens
    for (size_t j = 0; j < total_tokens; ++j) {
        // Skip updating if this token is already selected (marked as negative infinity)
        if (di2s_data[j] == -std::numeric_limits<float>::infinity()) {
            continue;
        }
        
        size_t cis_idx = iteration * total_tokens + j;
        float eis_j = cis_data[cis_idx];
        
        // Subtract the squared orthogonal component
        di2s_data[j] -= eis_j * eis_j;
    }
}

std::vector<bool> FastGreedyDPP::create_mask(const std::vector<std::vector<size_t>>& selected_indices, 
                                           size_t total_tokens) {
    if (selected_indices.empty()) {
        return std::vector<bool>(total_tokens, false);
    }
    
    size_t batch_size = selected_indices.size();
    std::vector<bool> mask(batch_size * total_tokens, false);
    
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t idx : selected_indices[b]) {
            if (idx < total_tokens) {
                mask[b * total_tokens + idx] = true;
            }
        }
    }
    
    return mask;
}

float FastGreedyDPP::compute_determinant_approximation(const ov::Tensor& kernel, 
                                                     const std::vector<size_t>& selected_indices) {
    // This is a simplified approximation for validation purposes
    // In practice, the greedy algorithm approximates the determinant maximization
    
    if (selected_indices.empty()) {
        return 0.0f;
    }
    
    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];
    
    if (batch_size != 1) {
        throw std::invalid_argument("Determinant approximation only supports single batch");
    }
    
    const float* kernel_data = kernel.data<const float>();
    size_t total_tokens = shape[1];
    
    // Compute the product of diagonal elements of selected tokens as approximation
    float det_approx = 1.0f;
    for (size_t idx : selected_indices) {
        size_t diag_idx = idx * total_tokens + idx;
        det_approx *= kernel_data[diag_idx];
    }
    
    return det_approx;
}

std::vector<std::vector<size_t>> FastGreedyDPP::select_with_ops_model(const ov::Tensor& kernel, size_t num_tokens) {
    // Input validation
    if (kernel.get_shape().size() != 3) {
        throw std::invalid_argument("Kernel must be 3D tensor [B, N, N]");
    }
    
    auto shape = kernel.get_shape();
    size_t batch_size = shape[0];
    size_t total_tokens = shape[1];
    
    if (shape[1] != shape[2]) {
        throw std::invalid_argument("Kernel matrix must be square [B, N, N]");
    }
    
    if (num_tokens > total_tokens) {
        throw std::invalid_argument("Cannot select more tokens than available");
    }

    if (m_config.pruning_debug_mode) {
        std::cout << "Using OpenVINO ops-based DPP model for token selection" << std::endl;
        std::cout << "  Input kernel shape: [" << batch_size << ", " << total_tokens << ", " << total_tokens << "]" << std::endl;
        std::cout << "  Target tokens to select: " << num_tokens << std::endl;
    }

    // Create DPP model
    auto dpp_model = create_dpp_selection_model(num_tokens, total_tokens);
    
    // Compile model
    ov::Core core;
    auto compiled_model = core.compile_model(dpp_model, m_config.device);
    auto infer_request = compiled_model.create_infer_request();

    std::vector<std::vector<size_t>> batch_results(batch_size);

    // Process each batch
    for (size_t b = 0; b < batch_size; ++b) {
        // Extract single batch kernel [N, N] (no batch dimension)
        ov::Tensor batch_kernel(kernel.get_element_type(), {total_tokens, total_tokens});
        const float* kernel_data = kernel.data<const float>();
        float* batch_data = batch_kernel.data<float>();
        
        // Copy batch data - extract the specific batch slice
        const float* batch_source = kernel_data + b * total_tokens * total_tokens;
        std::memcpy(batch_data, batch_source, total_tokens * total_tokens * sizeof(float));

        // Set input tensor
        infer_request.set_input_tensor(batch_kernel);
        
        // Run inference
        infer_request.infer();
        
        // Get selected indices
        auto output = infer_request.get_output_tensor();
        const int64_t* selected_data = output.data<const int64_t>();
        
        std::vector<size_t> selected_indices;
        for (size_t i = 0; i < num_tokens; ++i) {
            selected_indices.push_back(static_cast<size_t>(selected_data[i]));
        }
        
        // Sort indices for deterministic output
        std::sort(selected_indices.begin(), selected_indices.end());
        batch_results[b] = selected_indices;

        if (m_config.pruning_debug_mode) {
            std::cout << "Batch " << b << " selected indices: [";
            for (size_t i = 0; i < selected_indices.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << selected_indices[i];
            }
            std::cout << "]" << std::endl;
        }
    }

    return batch_results;
}

std::shared_ptr<ov::Model> FastGreedyDPP::create_dpp_selection_model(size_t num_tokens, size_t total_tokens) {
    // Input parameter: kernel matrix [N, N] (no batch dimension)
    auto kernel_input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, 
        ov::PartialShape{static_cast<int64_t>(total_tokens), static_cast<int64_t>(total_tokens)}
    );
    
    auto kernel_matrix = kernel_input;
    
    // Extract diagonal elements as initial marginal gains [N]
    std::vector<int64_t> diag_indices_data;
    for (size_t i = 0; i < total_tokens; ++i) {
        diag_indices_data.push_back(static_cast<int64_t>(i));
        diag_indices_data.push_back(static_cast<int64_t>(i));
    }
    
    auto diag_indices = ov::op::v0::Constant::create(
        ov::element::i64, 
        {static_cast<int64_t>(total_tokens), 2}, 
        diag_indices_data
    );
    
    auto diagonal_values = std::make_shared<ov::op::v8::GatherND>(kernel_matrix, diag_indices);
    
    // Initialize selected indices vector and orthogonal vectors
    std::vector<std::shared_ptr<ov::Node>> selected_indices_nodes;
    std::vector<std::shared_ptr<ov::Node>> orthogonal_vectors; // Store orthogonalized vectors
    std::shared_ptr<ov::Node> current_gains = diagonal_values;
    
    // Complete DPP Greedy Selection with Gram-Schmidt Orthogonalization
    for (size_t t = 0; t < num_tokens; ++t) {
        // 1. Find the token with maximum marginal gain
        auto k_one = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
        auto topk = std::make_shared<ov::op::v11::TopK>(
            current_gains,
            k_one,
            0,
            ov::op::v11::TopK::Mode::MAX,
            ov::op::v11::TopK::SortType::NONE,
            ov::element::i64
        );
        
        auto selected_idx = std::make_shared<ov::op::v0::Squeeze>(topk->output(1));
        selected_indices_nodes.push_back(selected_idx);
        
        // 2. Extract the selected token's row from kernel matrix
        auto selected_row = std::make_shared<ov::op::v8::Gather>(
            kernel_matrix,
            selected_idx,
            ov::op::v0::Constant::create(ov::element::i64, {}, {0})
        );
        
        // 3. Gram-Schmidt Orthogonalization Process
        std::shared_ptr<ov::Node> orthogonalized_vector = selected_row;
        
        // Orthogonalize against all previously selected vectors
        for (size_t j = 0; j < orthogonal_vectors.size(); ++j) {
            // Compute dot product: <selected_row, orthogonal_vectors[j]>
            auto dot_product = std::make_shared<ov::op::v1::ReduceSum>(
                std::make_shared<ov::op::v1::Multiply>(orthogonalized_vector, orthogonal_vectors[j]),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                false
            );
            
            // Compute norm squared of orthogonal_vectors[j]
            auto norm_squared = std::make_shared<ov::op::v1::ReduceSum>(
                std::make_shared<ov::op::v1::Multiply>(orthogonal_vectors[j], orthogonal_vectors[j]),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                false
            );
            
            // Add small epsilon for numerical stability
            auto epsilon = ov::op::v0::Constant::create(ov::element::f32, {}, {1e-8f});
            std::shared_ptr<ov::Node> norm_squared_stable = std::make_shared<ov::op::v1::Add>(norm_squared, epsilon);
            
            // Compute projection coefficient: dot_product / norm_squared
            auto projection_coeff = std::make_shared<ov::op::v1::Divide>(dot_product, norm_squared_stable);
            
            // Compute projection: projection_coeff * orthogonal_vectors[j]
            auto projection = std::make_shared<ov::op::v1::Multiply>(
                std::make_shared<ov::op::v0::Unsqueeze>(
                    projection_coeff,
                    ov::op::v0::Constant::create(ov::element::i64, {1}, {0})
                ),
                orthogonal_vectors[j]
            );
            
            // Subtract projection: orthogonalized_vector -= projection
            orthogonalized_vector = std::make_shared<ov::op::v1::Subtract>(
                orthogonalized_vector, 
                projection
            );
        }
        
        // Store the orthogonalized vector for future iterations
        orthogonal_vectors.push_back(orthogonalized_vector);
        
        // 4. Update marginal gains for remaining tokens
        if (t < num_tokens - 1) {
            // Compute squared norm of orthogonalized vector
            auto orth_norm_squared = std::make_shared<ov::op::v1::ReduceSum>(
                std::make_shared<ov::op::v1::Multiply>(orthogonalized_vector, orthogonalized_vector),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                false
            );
            
            // Add epsilon for numerical stability
            auto epsilon = ov::op::v0::Constant::create(ov::element::f32, {}, {1e-8f});
            std::shared_ptr<ov::Node> orth_norm_squared_stable = std::make_shared<ov::op::v1::Add>(orth_norm_squared, epsilon);
            
            // Create a vector of updated gains for all tokens
            std::vector<std::shared_ptr<ov::Node>> updated_gains_per_token;
            
            for (size_t i = 0; i < total_tokens; ++i) {
                // Get the i-th row of kernel matrix
                auto token_idx = ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(i)});
                auto token_row = std::make_shared<ov::op::v8::Gather>(
                    kernel_matrix,
                    token_idx,
                    ov::op::v0::Constant::create(ov::element::i64, {}, {0})
                );
                
                // Compute dot product with orthogonalized vector
                auto dot_prod = std::make_shared<ov::op::v1::ReduceSum>(
                    std::make_shared<ov::op::v1::Multiply>(token_row, orthogonalized_vector),
                    ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                    false
                );
                
                // Compute squared dot product
                auto dot_prod_squared = std::make_shared<ov::op::v1::Multiply>(dot_prod, dot_prod);
                
                // Compute gain reduction: dot_prod_squared / orth_norm_squared
                auto gain_reduction = std::make_shared<ov::op::v1::Divide>(
                    dot_prod_squared, 
                    orth_norm_squared_stable
                );
                
                // Get current gain for token i
                auto current_gain_i = std::make_shared<ov::op::v8::Gather>(
                    current_gains,
                    token_idx,
                    ov::op::v0::Constant::create(ov::element::i64, {}, {0})
                );
                
                // Update gain: current_gain - gain_reduction
                auto updated_gain = std::make_shared<ov::op::v1::Subtract>(
                    current_gain_i,
                    gain_reduction
                );
                
                updated_gains_per_token.push_back(updated_gain);
            }
            
            // Stack all updated gains into a vector
            std::vector<std::shared_ptr<ov::Node>> unsqueezed_gains;
            for (auto& gain : updated_gains_per_token) {
                auto unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(
                    gain,
                    ov::op::v0::Constant::create(ov::element::i64, {1}, {0})
                );
                unsqueezed_gains.push_back(unsqueezed);
            }
            
            current_gains = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector(unsqueezed_gains.begin(), unsqueezed_gains.end()),
                0
            );
            
            // Set selected position to -inf to prevent re-selection
            auto neg_inf_scalar = ov::op::v0::Constant::create(
                ov::element::f32, {1}, {-std::numeric_limits<float>::infinity()}
            );
            
            // Convert scalar index to 1D tensor for ScatterUpdate
            auto selected_idx_1d = std::make_shared<ov::op::v0::Unsqueeze>(
                selected_idx,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0})
            );
            
            current_gains = std::make_shared<ov::op::v3::ScatterUpdate>(
                current_gains,
                selected_idx_1d,
                neg_inf_scalar,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0})
            );
        }
    }
    
    // Convert each scalar index to 1D tensor for concatenation
    std::vector<std::shared_ptr<ov::Node>> unsqueezed_indices;
    for (auto& idx_node : selected_indices_nodes) {
        auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(idx_node, unsqueeze_axis);
        unsqueezed_indices.push_back(unsqueezed);
    }
    
    // Stack selected indices into a single tensor [num_tokens]
    auto selected_indices_concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector(unsqueezed_indices.begin(), unsqueezed_indices.end()),
        0
    );
    
    // Create result
    auto result = std::make_shared<ov::op::v0::Result>(selected_indices_concat);
    
    return std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{kernel_input},
        "CompleteGreedyDPP_Selection_Model"
    );
}

std::shared_ptr<ov::Node> FastGreedyDPP::create_argmax_ops(std::shared_ptr<ov::Node> input, int64_t axis) {
    auto k_const = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
    auto topk = std::make_shared<ov::op::v11::TopK>(input, 
                                                   k_const, 
                                                   axis, 
                                                   ov::op::v11::TopK::Mode::MAX, 
                                                   ov::op::v11::TopK::SortType::NONE,
                                                   ov::element::i64);
    
    // Get the indices output (output 1) and squeeze to remove the K dimension
    auto indices = topk->output(1);
    return std::make_shared<ov::op::v0::Squeeze>(indices);
}

std::shared_ptr<ov::Node> FastGreedyDPP::create_orthogonalization_ops(
    std::shared_ptr<ov::Node> kernel_row,
    std::shared_ptr<ov::Node> cis_matrix,
    std::shared_ptr<ov::Node> selected_idx,
    size_t iteration,
    std::shared_ptr<ov::Node> norm_factor) {
    
    // For the simplified version, just return the normalized kernel row
    // In a full implementation, this would handle the complete orthogonalization
    return std::make_shared<ov::op::v1::Divide>(kernel_row, norm_factor);
}

} // namespace ov::genai::cdpruner 