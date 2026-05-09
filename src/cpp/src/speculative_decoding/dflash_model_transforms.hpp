// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/core.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace dflash {

struct DFlashRTInfo {
    bool dflash_mode = false;
    size_t block_size = 16;
    int64_t mask_token_id = -1;
    std::vector<int32_t> hidden_layers_list;
};

DFlashRTInfo extract_dflash_info_from_config(ov::AnyMap& config, const std::filesystem::path& models_path = {});
void apply_dflash_rt_info(std::shared_ptr<ov::Model>& model, ov::AnyMap& properties);

}  // namespace dflash
}  // namespace utils
}  // namespace genai
}  // namespace ov
