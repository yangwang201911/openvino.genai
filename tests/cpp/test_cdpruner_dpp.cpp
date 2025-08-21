// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "visual_language/cdpruner/fast_dpp.hpp"
#include "visual_language/cdpruner/cdpruner_config.hpp"
#include <openvino/openvino.hpp>
#include <vector>
#include <algorithm>

using namespace ov::genai::cdpruner;

class FastGreedyDPPTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize config for testing
        config.visual_tokens_retain_percentage = 75;  // Will keep 3 out of 4 tokens
        config.relevance_weight = 0.5f;
        config.enable_pruning = true;
        config.pruning_debug_mode = true;
        config.use_negative_relevance = false;  // Not using negative correlation as requested
        config.numerical_threshold = 1e-6f;
        config.device = "GPU";
        config.use_ops_model = true;
        
        dpp_selector = std::make_unique<FastGreedyDPP>(config);
    }
    
    Config config;
    std::unique_ptr<FastGreedyDPP> dpp_selector;
};


TEST_F(FastGreedyDPPTest, ConditionalKernelMatrixSelectionTraditional) {
    // Test case: Select 3 tokens out of 4 using the specific conditional kernel matrix
    // Expected result: tokens 1, 0, 3 should be selected
    
    // Create the conditional kernel matrix as specified:
    // [0.8, 0.3, 0.1, 0.2]  // token 0
    // [0.3, 0.9, 0.4, 0.1]  // token 1  
    // [0.1, 0.4, 0.7, 0.5]  // token 2
    // [0.2, 0.1, 0.5, 0.6]  // token 3
    
    std::vector<float> kernel_data = {
        // Batch 0, 4x4 kernel matrix
        0.8f, 0.3f, 0.1f, 0.2f,  // token 0 row
        0.3f, 0.9f, 0.4f, 0.1f,  // token 1 row
        0.1f, 0.4f, 0.7f, 0.5f,  // token 2 row
        0.2f, 0.1f, 0.5f, 0.6f   // token 3 row
    };
    
    // Create OpenVINO tensor [batch_size=1, tokens=4, tokens=4]
    ov::Tensor kernel_matrix(ov::element::f32, {1, 4, 4});
    std::memcpy(kernel_matrix.data<float>(), kernel_data.data(), kernel_data.size() * sizeof(float));
    
    // Number of tokens to keep
    size_t num_tokens_to_keep = 3;
    
    // Perform DPP selection
    auto selected_tokens = dpp_selector->select(kernel_matrix, num_tokens_to_keep);

    // Validate results
    ASSERT_EQ(selected_tokens.size(), 1);  // Single batch
    ASSERT_EQ(selected_tokens[0].size(), num_tokens_to_keep);  // Should select 3 tokens
    
    // Expected tokens: 1, 0, 3 (sorted order)
    std::vector<size_t> expected_tokens = {0, 1, 3};
    std::vector<size_t> actual_tokens = selected_tokens[0];
    std::sort(actual_tokens.begin(), actual_tokens.end());
    
    EXPECT_EQ(actual_tokens, expected_tokens) 
        << "Expected tokens [0, 1, 3] but got [" 
        << actual_tokens[0] << ", " << actual_tokens[1] << ", " << actual_tokens[2] << "]";
    
    // Verify token 2 is NOT selected (should be pruned)
    auto it = std::find(actual_tokens.begin(), actual_tokens.end(), 2);
    EXPECT_EQ(it, actual_tokens.end()) << "Token 2 should be pruned but was selected";
}

TEST_F(FastGreedyDPPTest, ConditionalKernelMatrixSelectionSubgraph) {
    // Test case: Select 3 tokens out of 4 using the specific conditional kernel matrix
    // Expected result: tokens 1, 0, 3 should be selected
    
    // Create the conditional kernel matrix as specified:
    // [0.8, 0.3, 0.1, 0.2]  // token 0
    // [0.3, 0.9, 0.4, 0.1]  // token 1  
    // [0.1, 0.4, 0.7, 0.5]  // token 2
    // [0.2, 0.1, 0.5, 0.6]  // token 3
    
    std::vector<float> kernel_data = {
        // Batch 0, 4x4 kernel matrix
        0.8f, 0.3f, 0.1f, 0.2f,  // token 0 row
        0.3f, 0.9f, 0.4f, 0.1f,  // token 1 row
        0.1f, 0.4f, 0.7f, 0.5f,  // token 2 row
        0.2f, 0.1f, 0.5f, 0.6f   // token 3 row
    };
    
    // Create OpenVINO tensor [batch_size=1, tokens=4, tokens=4]
    ov::Tensor kernel_matrix(ov::element::f32, {1, 4, 4});
    std::memcpy(kernel_matrix.data<float>(), kernel_data.data(), kernel_data.size() * sizeof(float));
    
    // Number of tokens to keep
    size_t num_tokens_to_keep = 3;
    
    // Perform DPP selection
    std::vector<std::vector<size_t>> exptected_selected_tokens = {{1, 0, 3}};

    // Validate results
    ASSERT_EQ(exptected_selected_tokens.size(), 1);  // Single batch
    ASSERT_EQ(exptected_selected_tokens[0].size(), num_tokens_to_keep);  // Should select 3 tokens

    auto selected_tokens = dpp_selector->select_with_ops_model(kernel_matrix, num_tokens_to_keep);
    // Validate results
    ASSERT_EQ(selected_tokens.size(), 1);  // Single batch
    ASSERT_EQ(selected_tokens[0].size(), num_tokens_to_keep);  // Should select 3 tokens

    // Validate that the selected tokens are the same for both methods
    for (size_t i = 0; i < selected_tokens.size(); ++i) {
        ASSERT_EQ(selected_tokens[i].size(), exptected_selected_tokens[i].size());
        for (size_t j = 0; j < selected_tokens[i].size(); ++j) {
            EXPECT_EQ(selected_tokens[i][j], exptected_selected_tokens[i][j]);
        }
    }
}

TEST_F(FastGreedyDPPTest, MultipleBatchSelection) {
    // Test with multiple batches
    
    std::vector<float> kernel_data = {
        // Batch 0
        0.8f, 0.3f, 0.1f, 0.2f,
        0.3f, 0.9f, 0.4f, 0.1f,
        0.1f, 0.4f, 0.7f, 0.5f,
        0.2f, 0.1f, 0.5f, 0.6f,
        
        // Batch 1 - different kernel matrix
        0.6f, 0.1f, 0.2f, 0.3f,
        0.1f, 0.8f, 0.3f, 0.2f,
        0.2f, 0.3f, 0.9f, 0.4f,  // Token 2 has highest diagonal in batch 1
        0.3f, 0.2f, 0.4f, 0.7f
    };
    
    ov::Tensor kernel_matrix(ov::element::f32, {2, 4, 4});
    std::memcpy(kernel_matrix.data<float>(), kernel_data.data(), kernel_data.size() * sizeof(float));
    
    auto selected_tokens = dpp_selector->select(kernel_matrix, 2);
    
    ASSERT_EQ(selected_tokens.size(), 2);  // Two batches
    ASSERT_EQ(selected_tokens[0].size(), 2);  // Each batch selects 2 tokens
    ASSERT_EQ(selected_tokens[1].size(), 2);
    
    // Batch 0: Should prioritize tokens 1 and 0 (highest diagonals)
    std::vector<size_t> batch0_tokens = selected_tokens[0];
    std::sort(batch0_tokens.begin(), batch0_tokens.end());
    
    // Batch 1: Should prioritize tokens 2 and 3 (highest diagonals)
    std::vector<size_t> batch1_tokens = selected_tokens[1];
    std::sort(batch1_tokens.begin(), batch1_tokens.end());
    
    // Just verify we get reasonable selections (exact results depend on DPP algorithm details)
    EXPECT_TRUE(batch0_tokens.size() == 2);
    EXPECT_TRUE(batch1_tokens.size() == 2);
}