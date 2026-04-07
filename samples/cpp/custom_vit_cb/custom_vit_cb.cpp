// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/runtime/tensor.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DATA_DIR> [DEVICE]");
    }
    const std::filesystem::path model_dir = argv[1];
    const std::string data_dir = std::string(argv[2]) + "/";
    const std::string device = (argc >= 4) ? argv[3] : "CPU";

    auto shape_to_string = [](const ov::Shape& shape) -> std::string {
        std::string s = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(shape[i]);
        }
        return s + "]";
    };

    auto load_tensor = [&](const std::string& filename) -> ov::Tensor {
        auto trim = [](std::string value) {
            value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
                            return !std::isspace(ch);
                        }));
            value.erase(std::find_if(value.rbegin(),
                                     value.rend(),
                                     [](unsigned char ch) {
                                         return !std::isspace(ch);
                                     })
                            .base(),
                        value.end());
            return value;
        };
        // load shape and type.
        std::string meta_file = data_dir + filename + "_0_meta.txt";
        // Reference meta file content:
        // shape: 1 1 2560
        // element_type: f32
        std::ifstream meta_ifs(meta_file);
        if (!meta_ifs.is_open()) {
            throw std::runtime_error("Failed to open meta file: " + meta_file);
        }
        std::string line;
        ov::Shape shape;
        ov::element::Type type;
        while (std::getline(meta_ifs, line)) {
            if (line.find("shape:") == 0) {
                std::istringstream iss(line.substr(6));
                size_t dim;
                while (iss >> dim) {
                    shape.push_back(dim);
                }
            } else if (line.find("element_type:") == 0) {
                std::string type_str = trim(line.substr(13));
                type = ov::element::Type(type_str);
            }
        }

        // based shape size and type, load data.
        ov::Tensor tensor(type, shape);
        std::string data_file = data_dir + filename + "_0_data.txt";
        std::ifstream data_ifs(data_file);
        if (!data_ifs.is_open()) {
            throw std::runtime_error("Failed to open data file: " + data_file);
        }
        if (type == ov::element::f32) {
            auto* data_ptr = tensor.data<float>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data_ifs >> data_ptr[i];
            }
        } else if (type == ov::element::i64) {
            auto* data_ptr = tensor.data<int64_t>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data_ifs >> data_ptr[i];
            }
        } else if (type == ov::element::i32) {
            auto* data_ptr = tensor.data<int32_t>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data_ifs >> data_ptr[i];
            }
        } else if (type == ov::element::boolean) {
            auto* data_ptr = tensor.data<bool>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                int val;
                data_ifs >> val;
                data_ptr[i] = static_cast<bool>(val);
            }
        } else {
            throw std::runtime_error("Unsupported element type in meta file: " + type.to_string());
        }
        std::cerr << "[DEBUG] Loaded tensor '" << filename << "': shape=" << shape_to_string(tensor.get_shape())
                  << " type=" << tensor.get_element_type() << std::endl;
        return tensor;
    };

    ov::Tensor inputs_embeds = load_tensor("inputs_embeds");
    ov::Tensor visual_pos_masks = load_tensor("visual_pos_mask");
    ov::Tensor deepstack_embeds_0 = load_tensor("deepstack_embeds_0");
    ov::Tensor deepstack_embeds_1 = load_tensor("deepstack_embeds_1");
    ov::Tensor deepstack_embeds_2 = load_tensor("deepstack_embeds_2");
    ov::Tensor position_ids = load_tensor("position_ids");

    // Combine per-layer deepstack tensors into a single [layers, vision_tokens, hidden_size] tensor.
    // Each individual tensor is expected to have shape [1, vision_tokens, hidden_size].
    {
        const std::vector<ov::Tensor> layers = {deepstack_embeds_0, deepstack_embeds_1, deepstack_embeds_2};
        const size_t num_layers = layers.size();
        const auto& first_shape = layers[0].get_shape();
        OPENVINO_ASSERT(first_shape.size() == 3 && first_shape[0] == 1,
            "Expected deepstack layer tensor shape [1, tokens, hidden], got ", shape_to_string(first_shape));
        const size_t vision_tokens = first_shape[1];
        const size_t hidden_size = first_shape[2];
        ov::Tensor deepstack_visual_embeds(layers[0].get_element_type(),
                                           {num_layers, vision_tokens, hidden_size});
        const size_t layer_elems = vision_tokens * hidden_size;
        auto* dst = deepstack_visual_embeds.data<float>();
        for (size_t i = 0; i < num_layers; ++i) {
            OPENVINO_ASSERT(layers[i].get_shape() == first_shape,
                "Deepstack layer ", i, " shape mismatch: ", shape_to_string(layers[i].get_shape()),
                " vs ", shape_to_string(first_shape));
            std::memcpy(dst + i * layer_elems, layers[i].data<float>(), layer_elems * sizeof(float));
        }
        std::cerr << "[DEBUG] Combined deepstack_visual_embeds: shape="
                  << shape_to_string(deepstack_visual_embeds.get_shape()) << std::endl;

        // Build extra_inputs with the key names expected by SequenceGroup:
        //   "deepstack_visual_embeds" — combined [layers, tokens, hidden] tensor
        //   "visual_pos_masks"        — boolean mask tensor
        // NOTE: attention_mask and beam_idx are managed internally by the CB pipeline.
        std::unordered_map<std::string, ov::Tensor> extra_inputs;
        extra_inputs["deepstack_visual_embeds"] = deepstack_visual_embeds;
        extra_inputs["visual_pos_masks"] = visual_pos_masks;

        std::cerr << "[DEBUG] extra_inputs keys: ";
        for (const auto& [k, v] : extra_inputs) {
            std::cerr << k << "=" << shape_to_string(v.get_shape()) << " ";
        }
        std::cerr << std::endl;

        std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>> extra_inputs_list =
            std::vector<std::unordered_map<std::string, ov::Tensor>>{std::move(extra_inputs)};

        std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>> position_ids_list =
            std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>{{position_ids, std::nullopt}};

        ov::genai::SchedulerConfig scheduler_config;
        scheduler_config.enable_prefix_caching = false;
        // Use a bounded token budget to avoid excessive KV cache pre-allocation on GPU.
        // Must be >= prompt length (1721 in this test case).
        scheduler_config.max_num_batched_tokens = 4096;
        scheduler_config.max_num_seqs = 2;
        // Limit KV cache size to 2 GB to keep within GPU memory limits.
        scheduler_config.cache_size = 2;

        std::cerr << "[DEBUG] Creating ContinuousBatchingPipeline with model_dir=" << model_dir
                  << " device=" << device << std::endl;

        const ov::AnyMap props;
        auto m_pipeline =
            std::make_unique<ov::genai::ContinuousBatchingPipeline>(model_dir, scheduler_config, device, props);

        ov::genai::GenerationConfig generation_config;
        generation_config.max_new_tokens = 64;

        std::cerr << "[DEBUG] inputs_embeds shape=" << shape_to_string(inputs_embeds.get_shape())
                  << " position_ids shape=" << shape_to_string(position_ids.get_shape()) << std::endl;
        std::cerr << "[DEBUG] Calling generate()..." << std::endl;

        const auto results = m_pipeline->generate({inputs_embeds},
                                                  {generation_config},
                                                  std::monostate(),
                                                  std::nullopt,
                                                  position_ids_list,
                                                  extra_inputs_list);

        std::string text;
        if (!results.empty() && !results[0].m_generation_ids.empty() && !results[0].m_generation_ids[0].empty()) {
            ov::genai::Tokenizer tokenizer = m_pipeline->get_tokenizer();
            text = tokenizer.decode(results[0].m_generation_ids[0]);
        }
        std::cout << "Generated text: " << text << std::endl;
    }

    return 0;
}