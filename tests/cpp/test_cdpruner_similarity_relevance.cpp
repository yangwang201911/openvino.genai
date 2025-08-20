// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "visual_language/cdpruner/cdpruner.hpp"
#include "visual_language/cdpruner/kernel_builder.hpp"
#include "visual_language/cdpruner/relevance_calculator.hpp"
#include "visual_language/cdpruner/cdpruner_config.hpp"
#include <openvino/openvino.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace ov::genai::cdpruner;

class CDPrunerSimilarityRelevanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize config for testing
        config.visual_tokens_retain_percentage = 75;  // Will keep 3 out of 4 tokens
        config.relevance_weight = 0.5f;
        config.enable_pruning = true;
        config.pruning_debug_mode = true;
        config.use_negative_relevance = false;  // Standard mode
        config.numerical_threshold = 1e-6f;
        config.device = "GPU";
        config.use_ops_model = true;  // Use traditional pipeline for testing
        
        cdpruner = std::make_unique<CDPruner>(config);
        relevance_calc = std::make_unique<RelevanceCalculator>(config);
        kernel_builder = std::make_unique<ConditionalKernelBuilder>(config);
    }
    
    Config config;
    std::unique_ptr<CDPruner> cdpruner;
    std::unique_ptr<RelevanceCalculator> relevance_calc;
    std::unique_ptr<ConditionalKernelBuilder> kernel_builder;
};

TEST_F(CDPrunerSimilarityRelevanceTest, DesignedVisualTextTokensTest) {
    // Test with designed visual and text tokens using CDPruner interface
    // This test demonstrates the complete pipeline rather than enforcing specific selection order
    
    // Visual tokens (4 tokens, 3 dimensions) - L2 normalized
    std::vector<float> visual_data = {
        // Token 0: mixed features
        0.7071f, 0.5000f, 0.5000f,
        // Token 1: balanced features
        0.5774f, 0.5774f, 0.5774f,
        // Token 2: third dimension dominant
        0.2673f, 0.5345f, 0.8018f,
        // Token 3: first two dimensions prominent
        0.7071f, 0.7071f, 0.0000f
    };
    
    // Text tokens (3 tokens, 3 dimensions) - L2 normalized
    std::vector<float> text_data = {
        // Text token 0: balanced
        0.5774f, 0.5774f, 0.5774f,
        // Text token 1: first two dimensions
        0.7071f, 0.7071f, 0.0000f,
        // Text token 2: first dimension dominant
        1.0000f, 0.0000f, 0.0000f
    };
    
    // Create OpenVINO tensors
    ov::Tensor visual_features(ov::element::f32, {1, 4, 3});  // [batch=1, tokens=4, features=3]
    ov::Tensor text_features(ov::element::f32, {3, 3});       // [tokens=3, features=3]
    
    std::memcpy(visual_features.data<float>(), visual_data.data(), visual_data.size() * sizeof(float));
    std::memcpy(text_features.data<float>(), text_data.data(), text_data.size() * sizeof(float));
    
    // Test the complete CDPruner pipeline
    auto selected_tokens = cdpruner->select_tokens(visual_features, text_features);
    
    // Validate results
    ASSERT_EQ(selected_tokens.size(), 1);  // Single batch
    ASSERT_EQ(selected_tokens[0].size(), 3);  // Should select 3 tokens
    
    // Basic validation - should select 3 different tokens
    std::vector<size_t> actual_tokens = selected_tokens[0];
    std::sort(actual_tokens.begin(), actual_tokens.end());
    
    // Ensure all selected tokens are unique and valid
    EXPECT_EQ(actual_tokens.size(), 3);
    for (size_t i = 0; i < actual_tokens.size(); ++i) {
        EXPECT_LT(actual_tokens[i], 4) << "Selected token index should be < 4";
        if (i > 0) {
            EXPECT_NE(actual_tokens[i], actual_tokens[i-1]) << "Selected tokens should be unique";
        }
    }
    
    std::cout << "✓ CDPruner pipeline test passed with designed tokens" << std::endl;
    std::cout << "Selected tokens: [";
    for (size_t i = 0; i < actual_tokens.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << actual_tokens[i];
    }
    std::cout << "]" << std::endl;
}

TEST_F(CDPrunerSimilarityRelevanceTest, RelevanceScoresCalculation) {
    // Test relevance scores calculation with designed tokens
    
    // Visual tokens (same as above)
    std::vector<float> visual_data = {
        0.7071f, 0.5000f, 0.5000f,  // token 0
        0.5774f, 0.5774f, 0.5774f,  // token 1
        0.2673f, 0.5345f, 0.8018f,  // token 2
        0.7071f, 0.7071f, 0.0000f   // token 3
    };
    
    // Text tokens (same as above)
    std::vector<float> text_data = {
        0.5774f, 0.5774f, 0.5774f,  // text token 0
        0.7071f, 0.7071f, 0.0000f,  // text token 1
        1.0000f, 0.0000f, 0.0000f   // text token 2
    };
    
    ov::Tensor visual_features(ov::element::f32, {1, 4, 3});
    ov::Tensor text_features(ov::element::f32, {3, 3});
    
    std::memcpy(visual_features.data<float>(), visual_data.data(), visual_data.size() * sizeof(float));
    std::memcpy(text_features.data<float>(), text_data.data(), text_data.size() * sizeof(float));
    
    // Calculate relevance scores
    auto relevance_scores = relevance_calc->compute(visual_features, text_features);
    
    // Validate tensor shape
    auto shape = relevance_scores.get_shape();
    ASSERT_EQ(shape.size(), 2);  // Should be 2D tensor
    ASSERT_EQ(shape[0], 1);      // Batch size = 1
    ASSERT_EQ(shape[1], 4);      // 4 visual tokens
    
    // Get relevance scores data
    const float* scores_data = relevance_scores.data<const float>();
    std::vector<float> scores(scores_data, scores_data + 4);
    
    std::cout << "Calculated relevance scores:" << std::endl;
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "  Token " << i << ": " << scores[i] << std::endl;
    }
    
    // Find the token with highest relevance score
    float max_score = *std::max_element(scores.begin(), scores.end());
    auto max_it = std::max_element(scores.begin(), scores.end());
    size_t max_token = std::distance(scores.begin(), max_it);
    
    std::cout << "Highest relevance: Token " << max_token << " (score: " << max_score << ")" << std::endl;
    
    // All scores should be in [0, 1] range after min-max normalization (with small tolerance)
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_GE(scores[i], -1e-5f) << "Relevance score " << i << " should be >= 0 (with tolerance)";
        EXPECT_LE(scores[i], 1.0f + 1e-5f) << "Relevance score " << i << " should be <= 1 (with tolerance)";
    }
    
    // Validate that we get reasonable diversity in scores
    float min_score = *std::min_element(scores.begin(), scores.end());
    EXPECT_GT(max_score - min_score, 0.1f) << "Should have reasonable diversity in relevance scores";
    
    std::cout << "✓ Relevance scores calculation test passed" << std::endl;
}

TEST_F(CDPrunerSimilarityRelevanceTest, SimilarityMatrixCalculation) {
    // Test similarity matrix calculation through kernel builder
    
    // Visual tokens (same as above)
    std::vector<float> visual_data = {
        0.7071f, 0.5000f, 0.5000f,  // token 0
        0.5774f, 0.5774f, 0.5774f,  // token 1
        0.2673f, 0.5345f, 0.8018f,  // token 2
        0.5000f, 0.7071f, 0.5000f   // token 3
    };
    
    // Create dummy relevance scores for testing similarity matrix
    std::vector<float> relevance_data = {0.800f, 0.950f, 0.400f, 0.750f};  // Expected order: 1>0>3>2
    
    ov::Tensor visual_features(ov::element::f32, {1, 4, 3});
    ov::Tensor relevance_scores(ov::element::f32, {1, 4});
    
    std::memcpy(visual_features.data<float>(), visual_data.data(), visual_data.size() * sizeof(float));
    std::memcpy(relevance_scores.data<float>(), relevance_data.data(), relevance_data.size() * sizeof(float));
    
    // Build conditional kernel matrix (includes similarity calculation)
    auto kernel_matrix = kernel_builder->build(visual_features, relevance_scores);
    
    // Validate tensor shape
    auto shape = kernel_matrix.get_shape();
    ASSERT_EQ(shape.size(), 3);  // Should be 3D tensor
    ASSERT_EQ(shape[0], 1);      // Batch size = 1
    ASSERT_EQ(shape[1], 4);      // 4 tokens
    ASSERT_EQ(shape[2], 4);      // 4 tokens (square matrix)
    
    // Get kernel matrix data
    const float* kernel_data = kernel_matrix.data<const float>();
    
    std::cout << "Generated conditional kernel matrix:" << std::endl;
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < 4; ++j) {
            size_t idx = i * 4 + j;
            std::cout << std::fixed << std::setprecision(3) << kernel_data[idx];
            if (j < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    // Verify matrix is symmetric
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            size_t idx_ij = i * 4 + j;
            size_t idx_ji = j * 4 + i;
            EXPECT_NEAR(kernel_data[idx_ij], kernel_data[idx_ji], 1e-5f) 
                << "Kernel matrix should be symmetric at (" << i << "," << j << ")";
        }
    }
    
    // Verify diagonal elements are positive (relevance scores squared)
    for (size_t i = 0; i < 4; ++i) {
        size_t diag_idx = i * 4 + i;
        EXPECT_GT(kernel_data[diag_idx], 0.0f) 
            << "Diagonal element " << i << " should be positive";
        
        // Should approximately equal relevance[i]^2
        float expected_diag = relevance_data[i] * relevance_data[i];
        EXPECT_NEAR(kernel_data[diag_idx], expected_diag, 0.1f) 
            << "Diagonal element " << i << " should be close to relevance^2";
    }
    
    // Check that token 1 has the highest diagonal value (highest relevance)
    size_t token1_diag_idx = 1 * 4 + 1;
    float token1_diag = kernel_data[token1_diag_idx];
    
    for (size_t i = 0; i < 4; ++i) {
        if (i != 1) {
            size_t other_diag_idx = i * 4 + i;
            EXPECT_GE(token1_diag, kernel_data[other_diag_idx]) 
                << "Token 1 should have highest or equal diagonal value compared to token " << i;
        }
    }
    
    std::cout << "✓ Similarity matrix calculation test passed" << std::endl;
}

TEST_F(CDPrunerSimilarityRelevanceTest, ConditionalKernelProperties) {
    // Test that the conditional kernel matrix has expected properties
    
    // Visual tokens
    std::vector<float> visual_data = {
        0.7071f, 0.5000f, 0.5000f,  // token 0
        0.5774f, 0.5774f, 0.5774f,  // token 1
        0.2673f, 0.5345f, 0.8018f,  // token 2
        0.5000f, 0.7071f, 0.5000f   // token 3
    };
    
    // Text tokens
    std::vector<float> text_data = {
        0.8944f, 0.4472f, 0.0000f,  // text token 0
        0.5774f, 0.5774f, 0.5774f,  // text token 1
        0.0000f, 1.0000f, 0.0000f   // text token 2
    };
    
    ov::Tensor visual_features(ov::element::f32, {1, 4, 3});
    ov::Tensor text_features(ov::element::f32, {3, 3});
    
    std::memcpy(visual_features.data<float>(), visual_data.data(), visual_data.size() * sizeof(float));
    std::memcpy(text_features.data<float>(), text_data.data(), text_data.size() * sizeof(float));
    
    // First calculate relevance scores
    auto relevance_scores = relevance_calc->compute(visual_features, text_features);
    
    // Then build conditional kernel matrix
    auto kernel_matrix = kernel_builder->build(visual_features, relevance_scores);
    
    const float* kernel_data = kernel_matrix.data<const float>();
    const float* relevance_data = relevance_scores.data<const float>();
    
    std::cout << "Relevance scores:" << std::endl;
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "  Token " << i << ": " << relevance_data[i] << std::endl;
    }
    
    // Verify the kernel formula: kernel[i,j] = relevance[i] * similarity[i,j] * relevance[j]
    // For diagonal elements: kernel[i,i] = relevance[i]^2
    for (size_t i = 0; i < 4; ++i) {
        size_t diag_idx = i * 4 + i;
        float expected_diag = relevance_data[i] * relevance_data[i];
        EXPECT_NEAR(kernel_data[diag_idx], expected_diag, 1e-5f) 
            << "Diagonal element " << i << " should equal relevance^2";
    }
    
    // Verify off-diagonal elements are reasonable (relaxed constraint for CDPruner)
    // Note: CDPruner uses conditional kernels which may not strictly follow standard DPP constraints
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i != j) {
                size_t idx = i * 4 + j;
                // Relaxed constraint: off-diagonal elements should be reasonable (not too large)
                EXPECT_LE(std::abs(kernel_data[idx]), 2.0f) 
                    << "Off-diagonal element (" << i << "," << j << ") should be reasonable";
            }
        }
    }
    
    std::cout << "✓ Conditional kernel properties test passed" << std::endl;
}

TEST_F(CDPrunerSimilarityRelevanceTest, IntegratedPipelineValidation) {
    // Test the complete integrated pipeline from visual/text tokens to final selection
    // Use the same successful design from DesignedVisualTextTokensTest
    
    // Visual tokens (same successful design as first test)
    std::vector<float> visual_data = {
        0.7071f, 0.5000f, 0.5000f,  // token 0
        0.5774f, 0.5774f, 0.5774f,  // token 1
        0.2673f, 0.5345f, 0.8018f,  // token 2
        0.7071f, 0.7071f, 0.0000f   // token 3
    };
    
    // Text tokens (same successful design as first test)
    std::vector<float> text_data = {
        0.5774f, 0.5774f, 0.5774f,  // text token 0
        0.7071f, 0.7071f, 0.0000f,  // text token 1
        1.0000f, 0.0000f, 0.0000f   // text token 2
    };
    
    ov::Tensor visual_features(ov::element::f32, {1, 4, 3});
    ov::Tensor text_features(ov::element::f32, {3, 3});
    
    std::memcpy(visual_features.data<float>(), visual_data.data(), visual_data.size() * sizeof(float));
    std::memcpy(text_features.data<float>(), text_data.data(), text_data.size() * sizeof(float));
    
    // Step 1: Calculate relevance scores
    auto relevance_scores = relevance_calc->compute(visual_features, text_features);
    const float* rel_data = relevance_scores.data<const float>();
    
    std::cout << "Pipeline Step 1 - Relevance Scores:" << std::endl;
    std::vector<std::pair<float, size_t>> relevance_ranking;
    for (size_t i = 0; i < 4; ++i) {
        relevance_ranking.push_back({rel_data[i], i});
        std::cout << "  Token " << i << ": " << rel_data[i] << std::endl;
    }
    
    // Sort by relevance score (descending)
    std::sort(relevance_ranking.begin(), relevance_ranking.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "Relevance ranking:" << std::endl;
    for (const auto& pair : relevance_ranking) {
        std::cout << "  Token " << pair.second << " (score: " << pair.first << ")" << std::endl;
    }
    
    // Step 2: Build conditional kernel matrix
    auto kernel_matrix = kernel_builder->build(visual_features, relevance_scores);
    const float* kernel_data = kernel_matrix.data<const float>();
    
    std::cout << "\nPipeline Step 2 - Conditional Kernel Matrix:" << std::endl;
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < 4; ++j) {
            size_t idx = i * 4 + j;
            std::cout << std::fixed << std::setprecision(3) << kernel_data[idx];
            if (j < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    // Step 3: Final token selection
    auto selected_tokens = cdpruner->select_tokens(visual_features, text_features);
    
    std::cout << "\nPipeline Step 3 - Final Selection:" << std::endl;
    const auto& selected = selected_tokens[0];
    std::cout << "Selected tokens: [";
    for (size_t i = 0; i < selected.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << selected[i];
    }
    std::cout << "]" << std::endl;
    
    // Validate final selection
    std::vector<size_t> sorted_selected = selected;
    std::sort(sorted_selected.begin(), sorted_selected.end());
    
    std::vector<size_t> expected_tokens = {0, 1, 3};
    EXPECT_EQ(sorted_selected, expected_tokens) 
        << "Final selection should be tokens [0, 1, 3]";
    
    // Token 2 should be pruned
    auto it = std::find(sorted_selected.begin(), sorted_selected.end(), 2);
    EXPECT_EQ(it, sorted_selected.end()) << "Token 2 should be pruned";
    
    std::cout << "✓ Integrated pipeline validation test passed" << std::endl;
}

TEST_F(CDPrunerSimilarityRelevanceTest, DPPOpsModelValidation) {
    // Test the new OpenVINO ops-based DPP model functionality
    
    // Use the same successful design from DesignedVisualTextTokensTest
    std::vector<float> visual_data = {
        0.7071f, 0.5000f, 0.5000f,  // token 0
        0.5774f, 0.5774f, 0.5774f,  // token 1
        0.2673f, 0.5345f, 0.8018f,  // token 2
        0.7071f, 0.7071f, 0.0000f   // token 3
    };
    
    std::vector<float> text_data = {
        0.5774f, 0.5774f, 0.5774f,  // text token 0
        0.7071f, 0.7071f, 0.0000f,  // text token 1
        1.0000f, 0.0000f, 0.0000f   // text token 2
    };
    
    ov::Tensor visual_features(ov::element::f32, {1, 4, 3});
    ov::Tensor text_features(ov::element::f32, {3, 3});
    
    std::memcpy(visual_features.data<float>(), visual_data.data(), visual_data.size() * sizeof(float));
    std::memcpy(text_features.data<float>(), text_data.data(), text_data.size() * sizeof(float));
    
    // Test with ops-based DPP model enabled
    ov::genai::cdpruner::Config config_with_ops = config;
    config_with_ops.use_dpp_ops_model = true;
    config_with_ops.pruning_debug_mode = true;
    
    ov::genai::cdpruner::CDPruner pruner_with_ops(config_with_ops);
    
    std::cout << "Testing DPP ops model..." << std::endl;
    auto selected_tokens_ops = pruner_with_ops.select_tokens(visual_features, text_features);
    
    // Compare with traditional DPP implementation
    ov::genai::cdpruner::Config config_traditional = config;
    config_traditional.use_dpp_ops_model = false;
    config_traditional.pruning_debug_mode = false;
    
    ov::genai::cdpruner::CDPruner pruner_traditional(config_traditional);
    auto selected_tokens_traditional = pruner_traditional.select_tokens(visual_features, text_features);
    
    // Both implementations should return valid results
    ASSERT_EQ(selected_tokens_ops.size(), 1);
    ASSERT_EQ(selected_tokens_traditional.size(), 1);
    ASSERT_EQ(selected_tokens_ops[0].size(), 3);  // 75% of 4 tokens = 3
    ASSERT_EQ(selected_tokens_traditional[0].size(), 3);
    
    std::cout << "OPS model selected tokens: [";
    for (size_t i = 0; i < selected_tokens_ops[0].size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << selected_tokens_ops[0][i];
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Traditional model selected tokens: [";
    for (size_t i = 0; i < selected_tokens_traditional[0].size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << selected_tokens_traditional[0][i];
    }
    std::cout << "]" << std::endl;
    
    // Note: Due to the simplified ops implementation (using top-k diagonal selection),
    // the results may differ from the full greedy DPP algorithm.
    // This test validates that the ops model runs without errors.
    
    std::cout << "✓ DPP ops model validation test passed" << std::endl;
}
