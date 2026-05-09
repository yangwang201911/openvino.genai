// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_model_transforms.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "logger.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config, const std::filesystem::path& models_path) {
    DFlashRTInfo dflash_rt_info;
    if (config.find("dflash_mode") == config.end()) {
        return dflash_rt_info;
    }

    dflash_rt_info.dflash_mode = config.at("dflash_mode").as<bool>();
    config.erase("dflash_mode");

    auto block_it = config.find("dflash_block_size");
    if (block_it != config.end()) {
        dflash_rt_info.block_size = block_it->second.as<size_t>();
        config.erase("dflash_block_size");
    }

    auto mask_it = config.find("dflash_mask_token_id");
    if (mask_it != config.end()) {
        dflash_rt_info.mask_token_id = mask_it->second.as<int64_t>();
        config.erase("dflash_mask_token_id");
    }

    auto layers_it = config.find("hidden_layers_list");
    if (layers_it != config.end()) {
        OPENVINO_ASSERT(layers_it->second.is<std::vector<int32_t>>(),
                        "hidden_layers_list must be a vector of int32_t values");
        dflash_rt_info.hidden_layers_list = layers_it->second.as<std::vector<int32_t>>();
        config.erase("hidden_layers_list");
    } else if (!models_path.empty()) {
        auto config_file_path = models_path / "config.json";
        if (std::filesystem::exists(config_file_path)) {
            std::ifstream file(config_file_path);
            nlohmann::json data = nlohmann::json::parse(file);
            using ov::genai::utils::read_json_param;
            int num_decoder_layers = 0;
            read_json_param(data, "num_hidden_layers", num_decoder_layers);

            OPENVINO_ASSERT(
                num_decoder_layers >= 4,
                "num_decoder_layers must be at least 4 for automatic hidden layer selection, got: ",
                num_decoder_layers);

            if (num_decoder_layers >= 10) {
                dflash_rt_info.hidden_layers_list = { 2, num_decoder_layers / 2, num_decoder_layers - 3 };
            } else {
                dflash_rt_info.hidden_layers_list = { 0, num_decoder_layers / 2, num_decoder_layers - 1 };
            }

            if (data.contains("dflash_config")) {
                const auto& dflash_cfg = data["dflash_config"];
                if (dflash_cfg.contains("target_layer_ids")) {
                    dflash_rt_info.hidden_layers_list.clear();
                    for (const auto& id : dflash_cfg["target_layer_ids"]) {
                        dflash_rt_info.hidden_layers_list.push_back(id.get<int32_t>());
                    }
                }
                if (dflash_cfg.contains("mask_token_id") && dflash_rt_info.mask_token_id == -1) {
                    dflash_rt_info.mask_token_id = dflash_cfg["mask_token_id"].get<int64_t>();
                }
            }
            if (data.contains("block_size") && dflash_rt_info.block_size == 16) {
                dflash_rt_info.block_size = data["block_size"].get<size_t>();
            }
        }
    }

    return dflash_rt_info;
}

void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties) {
    if (model->has_rt_info("dflash_mode") && model->get_rt_info<bool>("dflash_mode")) {
        properties["dflash_mode"] = true;
        if (model->has_rt_info("dflash_block_size")) {
            properties["dflash_block_size"] = model->get_rt_info<size_t>("dflash_block_size");
        }
        if (model->has_rt_info("dflash_mask_token_id")) {
            properties["dflash_mask_token_id"] = model->get_rt_info<int64_t>("dflash_mask_token_id");
        }
        if (model->has_rt_info("hidden_layers_list")) {
            properties["hidden_layers_list"] = model->get_rt_info<std::vector<int>>("hidden_layers_list");
        }
    }
}

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
