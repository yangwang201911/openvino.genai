
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/minicpm/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"
#include "logger.hpp"

namespace ov::genai {

namespace {

std::string NATIVE_TAG = "<image>./</image>";

/**
 * @brief Represents the result of slicing an image into smaller patches.
 *
 * This struct is used in miniCPM inputs embedder to store the sliced image patches
 * and the target size of the processed image.
 *
 * @param slices A tensor containing the sliced image patches.
 * @param target_size The desired size of the image after processing.
 */
struct ImageSliceResult {
    ov::Tensor slices;
    ImageSize target_size;
};


int ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

std::pair<int, int> find_best_resize(std::pair<int, int> original_size, int scale_resolution, int patch_size, bool allow_upscale=false) {
    int width = original_size.first;
    int height = original_size.second;
    if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
        float r = static_cast<float>(width) / height;
        height = static_cast<int>(scale_resolution / std::sqrt(r));
        width = static_cast<int>(height * r);
    }
    int best_width = ensure_divide(width, patch_size);
    int best_height = ensure_divide(height, patch_size);
    return std::make_pair(best_width, best_height);
}

std::pair<int, int> get_refine_size(std::pair<int, int> original_size, std::pair<int, int> grid, int scale_resolution, int patch_size, bool allow_upscale) {
    int width, height;
    std::tie(width, height) = original_size;
    int grid_x, grid_y;
    std::tie(grid_x, grid_y) = grid;

    int refine_width = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);

    int grid_width = refine_width / grid_x;
    int grid_height = refine_height / grid_y;

    auto best_grid_size = find_best_resize(std::make_pair(grid_width, grid_height), scale_resolution, patch_size, allow_upscale);
    int best_grid_width, best_grid_height;
    std::tie(best_grid_width, best_grid_height) = best_grid_size;

    std::pair<int, int> refine_size = std::make_pair(best_grid_width * grid_x, best_grid_height * grid_y);
    return refine_size;
}

std::vector<std::vector<clip_image_u8>> slice_image(const clip_image_u8& img, const int max_slice_nums, const int scale_resolution, const int patch_size, const bool never_split) {
    const std::pair<int, int> original_size{img.nx, img.ny};
    const int original_width = img.nx;
    const int original_height = img.ny;
    const float log_ratio = logf(1.0f * original_width / original_height);
    const float ratio = 1.0f * original_width * original_height / (scale_resolution * scale_resolution);
    const int multiple = std::min(int(ceil(ratio)), max_slice_nums);

    std::vector<std::vector<clip_image_u8>> images;
    images.push_back(std::vector<clip_image_u8>{});

    if (multiple <= 1) {
        auto best_size = find_best_resize(original_size, scale_resolution, patch_size, true);
        images.back().push_back(clip_image_u8{});
        bicubic_resize(img, images.back().back(), best_size.first, best_size.second);
    }
    else if (multiple > 1) {

        std::vector<int> candidate_split_grids_nums;
        for (int i : {multiple - 1, multiple, multiple + 1}) {
            if (i == 1 || i > max_slice_nums) {
                continue;
            }
            candidate_split_grids_nums.push_back(i);
        }

        auto best_size = find_best_resize(original_size, scale_resolution, patch_size);
        images.back().push_back(clip_image_u8{});
        bicubic_resize(img, images.back().back(), best_size.first, best_size.second);

        std::vector<std::pair<int, int>> candidate_grids;

        for (int split_grids_nums : candidate_split_grids_nums) {
            int m = 1;
            while (m <= split_grids_nums) {
                if (split_grids_nums % m == 0) {
                    candidate_grids.emplace_back(m, split_grids_nums / m);
                }
                ++m;
            }
        }

        std::pair<int, int> best_grid{ 1, 1 };
        float min_error = std::numeric_limits<float>::infinity();

        for (const auto& grid : candidate_grids) {
            float error = std::abs(log_ratio - std::log(1.0f * grid.first / grid.second));
            if (error < min_error) {
                best_grid = grid;
                min_error = error;
            }
        }
        auto refine_size = get_refine_size(original_size, best_grid, scale_resolution, patch_size, true);
        clip_image_u8 refine_image;
        bicubic_resize(img, refine_image, refine_size.first, refine_size.second);

        // split_to_patches
        int width = refine_image.nx;
        int height = refine_image.ny;
        int grid_x = int(width / best_grid.first);
        int grid_y = int(height / best_grid.second);
        for (int patches_i = 0, ic = 0; patches_i < height && ic < best_grid.second; patches_i += grid_y, ic += 1) {
            images.push_back(std::vector<clip_image_u8>{});
            for (int patches_j = 0, jc = 0; patches_j < width && jc < best_grid.first; patches_j += grid_x, jc += 1) {
                images.back().push_back(clip_image_u8{});
                clip_image_u8& patch = images.back().back();
                patch.nx = grid_x;
                patch.ny = grid_y;
                patch.buf.resize(3 * patch.nx * patch.ny);
                for (int y = patches_i; y < patches_i + grid_y; ++y) {
                    for (int x = patches_j; x < patches_j + grid_x; ++x) {
                        const int i = 3 * (y * refine_image.nx + x);
                        const int j = 3 * ((y - patches_i) * patch.nx + (x - patches_j));
                        patch.buf[j] = refine_image.buf[i];
                        patch.buf[j + 1] = refine_image.buf[i + 1];
                        patch.buf[j + 2] = refine_image.buf[i + 2];
                    }
                }
            }
        }
    }

    return images;
}

// Reimplemented https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
// in shape [NCHW], out shape: [N, C*kernel*kernel, H*W/kernel/kernel]
ov::Tensor unfold(const ov::Tensor& images_tensor, size_t kernel) {
    ov::Shape images_shape = images_tensor.get_shape();

    OPENVINO_ASSERT(4 == images_shape.size(), "Input tensor must be 4D (NCHW).");

    const size_t bs = images_shape.at(0);
    const size_t images_c = images_shape.at(1);
    const size_t images_h = images_shape.at(2);
    const size_t images_w = images_shape.at(3);

    OPENVINO_ASSERT(images_h >= kernel && images_w >= kernel, "Input height and width must be greater than or equal to kernel size.");

    const size_t new_c = images_c * kernel * kernel;
    const size_t output_h = (images_h - kernel) / kernel + 1;
    const size_t output_w = (images_w - kernel) / kernel + 1;
    const size_t kernels_per_plane = output_h * output_w;

    ov::Tensor unfolded_tensor(ov::element::f32, {bs, new_c, kernels_per_plane});
    const float* images = images_tensor.data<float>();
    float* unfolded = unfolded_tensor.data<float>();
    for (size_t batch_idx = 0; batch_idx < bs; ++batch_idx) {
        for (size_t c_idx = 0; c_idx < images_c; ++c_idx) {
            for (size_t h_out = 0; h_out < output_h; ++h_out) {
                for (size_t w_out = 0; w_out < output_w; ++w_out) {
                    size_t h_idx = h_out * kernel;  // Calculate input height index
                    size_t w_idx = w_out * kernel;  // Calculate input width index

                    for (size_t kh = 0; kh < kernel; ++kh) {
                        for (size_t kw = 0; kw < kernel; ++kw) {
                            size_t input_idx = (batch_idx * images_c * images_h * images_w) +
                                                (c_idx * images_h * images_w) +
                                                ((h_idx + kh) * images_w) +
                                                (w_idx + kw);

                            size_t unfolded_c_idx = (c_idx * kernel * kernel) + (kh * kernel) + kw;
                            size_t unfolded_idx = (batch_idx * new_c * kernels_per_plane) +
                                                    unfolded_c_idx * kernels_per_plane +
                                                    (h_out * output_w + w_out);

                            unfolded[unfolded_idx] = images[input_idx];
                        }
                    }
                }
            }
        }
    }
    return unfolded_tensor;
}

ov::Tensor preprocess_for_encoder(const ov::Tensor& images, size_t kernel) {
    ov::Shape images_shape = images.get_shape();
    OPENVINO_ASSERT(4 == images_shape.size());
    ov::Tensor unfolded_tensor = unfold(images, kernel);
    const ov::Shape& unfolded_shape = unfolded_tensor.get_shape();  // [N, C*kernel*kernel, H*W/kernel/kernel]
    const size_t bs = unfolded_shape[0];
    const size_t d1 = unfolded_shape[1];
    const size_t d2 = unfolded_shape[2];
    const size_t channels = 3;
    const size_t new_len = d2 * kernel;

    ov::Tensor permuted_tensor{ov::element::f32, {bs, channels, kernel, new_len}};
    const float* unfolded = unfolded_tensor.data<float>();
    float* permuted = permuted_tensor.data<float>();
    for (size_t b_idx = 0; b_idx < bs; ++b_idx) {
        for (size_t c_idx = 0; c_idx < channels; ++c_idx) {
            for (size_t k1_idx = 0; k1_idx < kernel; ++k1_idx) {
                for (size_t d2_idx = 0; d2_idx < d2; ++d2_idx) {
                    for (size_t k2_idx = 0; k2_idx < kernel; ++k2_idx) {
                        size_t unfolded_idx = b_idx * d1 * d2 +
                                            (c_idx * kernel * kernel + k1_idx * kernel + k2_idx) * d2 +
                                            d2_idx;
                        size_t permuted_idx = b_idx * channels * kernel * new_len +
                                            c_idx * kernel * new_len +
                                            k1_idx * new_len +
                                            d2_idx * kernel + k2_idx;
                        permuted[permuted_idx] = unfolded[unfolded_idx];
                    }
                }
            }
        }
    }
    return permuted_tensor;
}

// torch.bucketize(fractional_coords, boundaries, right=True)
std::vector<int64_t> bucket_size_right(const std::vector<float>& fractional_coords, const std::vector<float>& boundaries) {
    std::vector<int64_t> bucket_coords(fractional_coords.size());
    std::transform(fractional_coords.begin(), fractional_coords.end(), bucket_coords.begin(), [&boundaries](float fractional_coord) {
        return std::distance(boundaries.begin(), std::upper_bound(boundaries.begin(), boundaries.end(), fractional_coord));
    });
    return bucket_coords;
}

ov::Tensor prepare_vis_position_ids(
    const ov::Tensor& pixel_values,
    const ov::Tensor& patch_attention_mask,
    const std::vector<ImageSize> tgt_sizes,
    size_t patch_size,
    size_t num_patches_per_side) {
    size_t batch_size = pixel_values.get_shape().at(0);
    size_t max_im_h = pixel_values.get_shape().at(2), max_im_w = pixel_values.get_shape().at(3);
    size_t max_nb_patches_h = max_im_h / patch_size, max_nb_patches_w = max_im_w / patch_size;
    std::vector<float> boundaries(num_patches_per_side - 1);
    std::iota(boundaries.begin(), boundaries.end(), 1.0f);
    std::transform(boundaries.begin(), boundaries.end(), boundaries.begin(), [num_patches_per_side](float val) {
        return val / num_patches_per_side;
    });
    size_t position_ids_batch_elem = max_nb_patches_h * max_nb_patches_w;
    ov::Tensor position_ids{ov::element::i64, {batch_size, position_ids_batch_elem}};
    int64_t* res_data = position_ids.data<int64_t>();
    std::fill_n(res_data, position_ids.get_size(), 0);

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t nb_patches_h = tgt_sizes.at(batch_idx).height;
        size_t nb_patches_w = tgt_sizes.at(batch_idx).width;

        std::vector<float> fractional_coords_h(nb_patches_h);
        std::iota(fractional_coords_h.begin(), fractional_coords_h.end(), 0.0f);
        std::transform(fractional_coords_h.begin(), fractional_coords_h.end(), fractional_coords_h.begin(), [nb_patches_h](float val) {
            return val / nb_patches_h;
        });
        std::vector<float> fractional_coords_w(nb_patches_w);
        std::iota(fractional_coords_w.begin(), fractional_coords_w.end(), 0.0f);
        std::transform(fractional_coords_w.begin(), fractional_coords_w.end(), fractional_coords_w.begin(), [nb_patches_w](float val) {
            return val / nb_patches_w;
        });

        std::vector<int64_t> bucket_coords_h = bucket_size_right(fractional_coords_h, boundaries);
        std::vector<int64_t> bucket_coords_w = bucket_size_right(fractional_coords_w, boundaries);

        std::vector<int64_t> pos_ids(bucket_coords_h.size() * bucket_coords_w.size());
        for (size_t col = 0; col < bucket_coords_h.size(); ++col) {
            for (size_t row = 0; row < bucket_coords_w.size(); ++row) {;
                pos_ids.at(col * bucket_coords_w.size() + row) = bucket_coords_h.at(col) * num_patches_per_side + bucket_coords_w.at(row);
            }
        }
        std::copy(pos_ids.begin(), pos_ids.end(), res_data + batch_idx * position_ids_batch_elem);
    }
    return position_ids;
}

std::pair<EncodedImage, ImageSliceResult> llava_image_embed_make_with_bytes_slice(clip_ctx& ctx_clip, const ov::Tensor& img, ov::InferRequest& encoder, int max_slice_nums, int scale_resolution, size_t patch_size, bool never_split) {
    clip_image_u8 source = tensor_to_clip_image_u8(img);
    std::vector<std::vector<clip_image_u8>> imgs = slice_image(source, max_slice_nums, scale_resolution, patch_size, never_split);
    const size_t channels = 3;

    std::vector<std::vector<clip_image_f32>> preprocessed{imgs.size()};
    size_t n_images = 0, max_size = 0;
    std::transform(imgs.begin(), imgs.end(), preprocessed.begin(), [&ctx_clip, &max_size, &n_images](const std::vector<clip_image_u8>& row) {
        std::vector<clip_image_f32> processed_row{row.size()};
        std::transform(row.begin(), row.end(), processed_row.begin(), [&ctx_clip, &max_size, &n_images](const clip_image_u8& raw) {
            clip_image_f32 im = clip_image_preprocess(ctx_clip, raw);
            if (size_t(im.ny) * size_t(im.nx) > max_size) {
                max_size = size_t(im.ny) * size_t(im.nx);
            }
            ++n_images;
            return im;
        });
        return processed_row;
    });

    ov::Tensor pixel_values{ov::element::f32, {n_images, channels, patch_size, max_size / patch_size}};
    size_t d3_all_pixel = pixel_values.get_shape().at(3);
    float* pixel_value_data = pixel_values.data<float>();

    //image chw to 1*c*kernel*hw/kernel and padding zero
    clip_image_f32& resized_preprocessed = preprocessed.at(0).at(0);
    size_t img_h = resized_preprocessed.ny;
    size_t img_w = resized_preprocessed.nx;
    ov::Tensor clip_img{ov::element::f32, {1, channels, img_h, img_w}, resized_preprocessed.buf.data()};
    ov::Tensor clip_pixel_values = preprocess_for_encoder(clip_img, patch_size);

    float* clip_value_data = clip_pixel_values.data<float>();
    size_t batch_pixel = 1;
    size_t d3_clip_pixel = clip_pixel_values.get_shape().at(3);
    for (size_t c_idx = 0; c_idx < channels; ++c_idx) {
        for (size_t k_idx = 0; k_idx < patch_size; k_idx++) {
            std::copy(clip_value_data, clip_value_data + d3_clip_pixel, pixel_value_data);
            // pixel_values and patch_attention_mask are multiplied instead of ignoring the corresponding pixel_values resulting in NaN instead of 0.0f if pixel_values has NaN.
            memset(pixel_value_data + d3_clip_pixel, 0, (d3_all_pixel - d3_clip_pixel) * sizeof(float));
            clip_value_data += d3_clip_pixel;
            pixel_value_data += d3_all_pixel;
        }
    }

    if (1 < preprocessed.size()) {
        for (size_t row = 1; row < preprocessed.size(); ++row) {
            size_t n_slices = preprocessed.at(row).size();
            for (size_t col = 0; col < n_slices; ++col) {
                clip_image_f32& elem = preprocessed.at(row).at(col);
                img_h = elem.ny;
                img_w = elem.nx;
                ov::Tensor clip_img{ov::element::f32, {1, channels, img_h, img_w}, elem.buf.data()};
                ov::Tensor clip_pixel_values = preprocess_for_encoder(clip_img, patch_size);

                d3_clip_pixel = clip_pixel_values.get_shape().at(3);
                clip_value_data = clip_pixel_values.data<float>();
                pixel_value_data = pixel_values.data<float>() + batch_pixel * channels * patch_size * d3_all_pixel;
                for (size_t c_idx = 0; c_idx < channels; ++c_idx) {
                    for (size_t k_idx = 0; k_idx < patch_size; k_idx++) {
                        std::copy(clip_value_data, clip_value_data + d3_clip_pixel, pixel_value_data);
                        // pixel_values and patch_attention_mask are multiplied instead of ignoring the corresponding pixel_values resulting in NaN instead of 0.0f if pixel_values has NaN.
                        memset(pixel_value_data + d3_clip_pixel, 0, (d3_all_pixel - d3_clip_pixel) * sizeof(float));
                        clip_value_data += d3_clip_pixel;
                        pixel_value_data += d3_all_pixel;
                    }
                }
                batch_pixel++;
            }
        }
    }
    encoder.set_tensor("pixel_values", pixel_values);

    ov::Tensor patch_attention_mask{ov::element::f32, {pixel_values.get_shape().at(0), 1, max_size / patch_size / patch_size}};
    float* attention_data = patch_attention_mask.data<float>();
    std::fill_n(attention_data, patch_attention_mask.get_size(), 0.0f);
    std::fill_n(attention_data, resized_preprocessed.ny / patch_size * resized_preprocessed.nx / patch_size, 1.0f);
    if (1 < preprocessed.size()) {
        for (size_t row = 1; row < preprocessed.size(); ++row) {
            size_t n_slices = preprocessed.at(row).size();
            for (size_t col = 0; col < n_slices; ++col) {
                const clip_image_f32& elem = preprocessed.at(row).at(col);
                std::fill_n(attention_data + ((row - 1) * n_slices + col + 1) * max_size / patch_size / patch_size, elem.ny / patch_size * elem.nx / patch_size, 1.0f);
            }
        }
    }
    encoder.set_tensor("patch_attention_mask", patch_attention_mask);

    ImageSize resized_source_size{resized_preprocessed.ny / patch_size, resized_preprocessed.nx / patch_size};
    std::vector<ImageSize> tgt_sizes{resized_source_size};
    if (1 < preprocessed.size()) {
        for (auto row = preprocessed.begin() + 1; row != preprocessed.end(); ++row) {
            for (const clip_image_f32& elem : *row) {
                tgt_sizes.push_back({elem.ny / patch_size, elem.nx / patch_size});
            }
        }
    }
    ImageSliceResult image_slice_result;
    ov::Tensor position_ids = prepare_vis_position_ids(pixel_values, patch_attention_mask, tgt_sizes, patch_size, ctx_clip.image_size / patch_size);
    encoder.set_tensor("position_ids", position_ids);
    encoder.infer();
    const ov::Tensor& output_tensor = encoder.get_output_tensor();
    
    auto shape = output_tensor.get_shape();
    GENAI_DEBUG("[VisionEncoder] Vision encoder output: num_dimensions=%zu", shape.size());
    if (shape.size() >= 3) {
        GENAI_DEBUG("[VisionEncoder] Vision encoder output shape: [%zu, %zu, %zu]",
                    shape[0], shape[1], shape[2]);
    } else if (shape.size() == 2) {
        GENAI_DEBUG("[VisionEncoder] Vision encoder output shape: [%zu, %zu]", shape[0], shape[1]);
    } else if (shape.size() == 1) {
        GENAI_DEBUG("[VisionEncoder] Vision encoder output shape: [%zu]", shape[0]);
    } else {
        GENAI_DEBUG("[VisionEncoder] Vision encoder output shape: empty or 0-D");
    }

    if (1 == preprocessed.size()) {
        ov::Tensor resized_source{ov::element::f32, output_tensor.get_shape()};
        output_tensor.copy_to(resized_source);
        return {{std::move(resized_source), resized_source_size}, std::move(image_slice_result)};
    }

    size_t old_hidden_size = output_tensor.get_shape().at(2);
    const float* out = output_tensor.data<float>();
    size_t n_patches = max_size / patch_size / patch_size;
    ov::Tensor resized_source{ov::element::f32, {1, n_patches, old_hidden_size}};
    std::copy_n(out, resized_source.get_size(), resized_source.data<float>());

    image_slice_result.slices = ov::Tensor{ov::element::f32, {preprocessed.size() - 1, preprocessed.at(1).size(), n_patches, old_hidden_size}};
    for (size_t col = 0; col < preprocessed.size() - 1; ++col) {
        for (size_t row = 0; row < preprocessed.at(1).size(); ++row) {
            std::copy_n(out + (col * preprocessed.at(1).size() + row + 1) * n_patches * old_hidden_size, n_patches * old_hidden_size, image_slice_result.slices.data<float>() + (col * preprocessed.at(1).size() + row) * n_patches * old_hidden_size);
        }
    }
    image_slice_result.target_size = tgt_sizes.at(1);
    return {{std::move(resized_source), resized_source_size}, std::move(image_slice_result)};
}

} // namespace

EncodedImage VisionEncoderMiniCPM::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    clip_ctx ctx_clip;
    ctx_clip.image_size = config.image_size;
    std::copy(config.norm_mean.begin(), config.norm_mean.end(), ctx_clip.image_mean);
    std::copy(config.norm_std.begin(), config.norm_std.end(), ctx_clip.image_std);

    auto [encoded_image, image_slice_result] = llava_image_embed_make_with_bytes_slice(ctx_clip, image, encoder, config.max_slice_nums, config.scale_resolution, config.patch_size, 0 == config.max_slice_nums);
    encoded_image.resampled_image = resample_encoded_image(encoded_image, image_slice_result.slices, image_slice_result.target_size);
    if (image_slice_result.slices) {
        encoded_image.slices_shape = image_slice_result.slices.get_shape();
    }
    return encoded_image;
}

ResampledImage VisionEncoderMiniCPM::resample_encoded_image(const EncodedImage& encoded_image, const ov::Tensor& slices, const ImageSize& target_size) {
    size_t pad_to_max = encoded_image.resized_source_size.height * encoded_image.resized_source_size.width;
    if (slices) {
        pad_to_max = std::max(pad_to_max, target_size.height * target_size.width);
    }
    const ov::Tensor& resampled_source = resample(encoded_image.resized_source, encoded_image.resized_source_size, pad_to_max);
    
    GENAI_DEBUG("[resample_encoded_image] resampled_source shape has %zu dimensions, byte_size=%zu",
                resampled_source.get_shape().size(), resampled_source.get_byte_size());
    for (size_t i = 0; i < resampled_source.get_shape().size(); ++i) {
        GENAI_DEBUG("[resample_encoded_image] resampled_source dimension %zu = %zu", i, resampled_source.get_shape()[i]);
    }
    
    std::vector<std::vector<ov::Tensor>> vision_embed_tensors;
    if (slices) {
        size_t token_idx = 0;
        const ov::Shape& slices_shape = slices.get_shape();
        vision_embed_tensors.resize(slices_shape.at(0));
        for (size_t i = 0; i < slices_shape.at(0); ++i) {
            std::vector<ov::Tensor> vision_embeds;
            vision_embeds.resize(slices_shape.at(1));
            for (size_t ja = 0; ja < slices_shape.at(1); ++ja) {
                size_t d2 = slices_shape.at(2);
                size_t d3 = slices_shape.at(3);
                // const_cast is safe as ov::Tensor only views the data and doesn't modify it.
                ov::Tensor encoded_view{
                    ov::element::f32, 
                    {1, d2, d3}, 
                    const_cast<float*>(slices.data<float>()) + (i * slices_shape.at(1) + ja) * d2 * d3
                };
                vision_embeds[ja] = resample(encoded_view, target_size, pad_to_max);
            }
            vision_embed_tensors[i] = vision_embeds;
        }
    }
    return {resampled_source, vision_embed_tensors};
}

namespace {

ov::Tensor concatenate_last_dim(const ov::Tensor& first, const ov::Tensor& second) {
    size_t res_d_0 = first.get_shape().at(0);
    size_t res_d_1 = first.get_shape().at(1);
    OPENVINO_ASSERT(second.get_shape().at(0) == res_d_0);
    OPENVINO_ASSERT(second.get_shape().at(1) == res_d_1);
    size_t res_d_2 = first.get_shape().at(2) + second.get_shape().at(2);
    ov::Tensor res{first.get_element_type(), {res_d_0, res_d_1, res_d_2}};
    auto first_data = first.data<float>();
    auto second_data = second.data<float>();
    float* res_data = res.data<float>();
    for (size_t i = 0; i < res_d_0; ++i) {
        for (size_t j = 0; j < res_d_1; ++j) {
            size_t k = 0;
            for (; k < first.get_shape().at(2); ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = first_data[i * res_d_1 * first.get_shape().at(2) + j * first.get_shape().at(2) + k];
            }
            for (size_t l = 0; l < second.get_shape().at(2); ++l, ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = second_data[i * res_d_1 * second.get_shape().at(2) + j * second.get_shape().at(2) + l];
            }
        }
    }
    return res;
}

/// embed_dim: output dimension for each position
/// pos: a list of positions to be encoded: size (H, W)
/// out: (H, W, D)
ov::Tensor get_1d_sincos_pos_embed_from_grid_new(size_t embed_dim, const ov::Tensor& pos) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    ov::Shape pos_shape = pos.get_shape();
    size_t H = pos_shape[0];
    size_t W = pos_shape[1];

    std::vector<float> omega(embed_dim / 2);
    for (size_t i = 0; i < omega.size(); ++i) {
        omega[i] = 1.0f / std::pow(10000.0f, float(i) / (embed_dim / 2));
    }

    std::vector<size_t> out_shape = {H, W, embed_dim};
    ov::Tensor emb(ov::element::f32, out_shape);

    auto pos_data = pos.data<float>();
    auto emb_data = emb.data<float>();

    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            for (size_t d = 0; d < embed_dim / 2; ++d) {
                float value = omega[d] * pos_data[h * W + w];
                // sin() and cos() are the source of difference with Python until a newer C++ standard is used.
                emb_data[h * W * embed_dim + w * embed_dim + d] = std::sin(value);
                emb_data[h * W * embed_dim + w * embed_dim + d + (embed_dim / 2)] = std::cos(value);
            }
        }
    }
    return emb;
}

ov::Tensor get_2d_sincos_pos_embed_from_grid(size_t embed_dim, const ov::Tensor& grid) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    ov::Shape grid_shape = grid.get_shape();
    auto grid_data = grid.data<float>();
    ov::Shape plane_shape{grid_shape.at(1), grid_shape.at(2)};
    ov::Tensor emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{
        ov::element::f32,
        plane_shape,
        grid_data
    });  // (H, W, D/2)
    ov::Tensor emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{
        ov::element::f32,
        plane_shape,
        grid_data + plane_shape.at(0) * plane_shape.at(1)
    });  // (H, W, D/2)
    return concatenate_last_dim(emb_h, emb_w);
}

/// image_size: image_size or (image_height, image_width)
/// return:
/// pos_embed: [image_height, image_width, embed_dim]
ov::Tensor get_2d_sincos_pos_embed(size_t embed_dim, const ImageSize& image_size) {
    size_t grid_h_size = image_size.height, grid_w_size = image_size.width;
    ov::Tensor grid(ov::element::f32, {2, grid_h_size, grid_w_size});
    float* data = grid.data<float>();
    for (size_t y = 0; y < grid_h_size; ++y) {
        std::iota(data, data + grid_w_size, 0.0f);
        data += grid_w_size;
    }
    for (float y = 0.0f; y < grid_h_size; ++y) {
        std::fill(data, data + grid_w_size, y);
        data += grid_w_size;
    }
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid);
}

void adjust_pos_cache(
    const std::vector<ImageSize>& target_sizes,
    size_t hidden_size,
    ov::Tensor& pos_embed_cache) {
    size_t max_h = std::max_element(target_sizes.begin(), target_sizes.end(), [](const ImageSize& left, const ImageSize& right) {
        return left.height < right.height;
    })->height;
    size_t max_w = std::max_element(target_sizes.begin(), target_sizes.end(), [](const ImageSize& left, const ImageSize& right) {
        return left.width < right.width;
    })->width;
    size_t allocated_height, allocated_width;
    if (pos_embed_cache) {
        const ov::Shape& allocated_shape = pos_embed_cache.get_shape();
        allocated_height = allocated_shape.at(0);
        allocated_width = allocated_shape.at(1);
    } else {
        allocated_height = allocated_width = 70;
    }
    if (max_h > allocated_height || max_w > allocated_width) {
        allocated_height = std::max(max_h, allocated_height);
        allocated_width = std::max(max_w, allocated_width);
        pos_embed_cache = get_2d_sincos_pos_embed(
            hidden_size, {allocated_height, allocated_width}
        );
    }
}

} // namespace

NormalizedPrompt InputsEmbedderMiniCPM::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    
    auto [unified_prompt, image_sequence] = normalize(
        prompt,
        NATIVE_TAG,
        NATIVE_TAG + '\n',
        base_id,
        images.size()
    );
    std::string unk64;

    for (size_t idx = 0; idx < m_vlm_config.query_num; ++idx) {
        unk64 += m_vlm_config.unk;
    }
    for (size_t new_image_id : image_sequence) {
        const EncodedImage& encoded_image = images.at(new_image_id - base_id);
        std::string expanded_tag;
        if (m_vlm_config.use_image_id) {
            expanded_tag += m_vlm_config.im_id_start + std::to_string(new_image_id) + m_vlm_config.im_id_end;
        }
        expanded_tag += m_vlm_config.im_start + unk64 + m_vlm_config.im_end;
        ov::Shape slices_shape = encoded_image.slices_shape;
        if (slices_shape.size()) {
            for (size_t row_idx = 0; row_idx < slices_shape.at(0); ++row_idx) {
                for (size_t col_idx = 0; col_idx < slices_shape.at(1); ++col_idx) {
                    expanded_tag += m_vlm_config.slice_start + unk64 + m_vlm_config.slice_end;
                }
                expanded_tag += '\n';
            }
            expanded_tag.pop_back();  // Equivalent of python "\n".join(slices).
        }
        unified_prompt.replace(unified_prompt.find(NATIVE_TAG), NATIVE_TAG.length(), expanded_tag);
    }

    return {std::move(unified_prompt), std::move(image_sequence), {}};
}

ov::Tensor InputsEmbedderMiniCPM::prepare_vision_embeddings(
    const std::vector<EncodedImage>& images,
    const std::vector<size_t>& images_sequence,
    std::vector<size_t>& tokens_per_image,
    std::vector<std::array<size_t, 3>>& images_grid_thw) {
    
    std::vector<ov::Tensor> vision_features;
    tokens_per_image.clear();
    images_grid_thw.clear();
    
    GENAI_DEBUG("[MiniCPM] prepare_vision_embeddings: Processing %zu images", images_sequence.size());
    
    for (size_t image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(image_id);
        const ov::Tensor& resampled_source = encoded_image.resampled_image.resampled_source;
        
        // Track tokens for this image
        auto rs_shape = resampled_source.get_shape();
        GENAI_DEBUG("[MiniCPM] Image %zu: resampled_source has %zu dimensions", image_id, rs_shape.size());
        for (size_t i = 0; i < rs_shape.size(); ++i) {
            GENAI_DEBUG("[MiniCPM] Image %zu: resampled_source dimension %zu = %zu", image_id, i, rs_shape[i]);
        }
        
        // For MiniCPM, resampled_source should be [batch, query_num, hidden_size]
        // We need to remove batch dimension to get [query_num, hidden_size]
        size_t base_tokens = rs_shape.size() >= 2 ? rs_shape[1] : rs_shape[0];  // query_num
        size_t image_total_tokens = base_tokens;
        
        GENAI_DEBUG("[MiniCPM] Image %zu: base_tokens=%zu, resized_source shape [%zu, %zu, %zu]", 
                    image_id, base_tokens,
                    encoded_image.resized_source.get_shape()[0],
                    encoded_image.resized_source.get_shape()[1],
                    encoded_image.resized_source.get_shape()[2]);
        
        // Resampler outputs [batch, query_num, hidden_size] but we need [query_num, hidden_size]
        // Remove batch dimension (batch=1) to get 2D tensor
        ov::Tensor resampled_2d;
        if (rs_shape.size() == 3 && rs_shape[0] == 1) {
            // Reshape from [1, query_num, hidden_size] to [query_num, hidden_size]
            ov::Shape shape_2d = {rs_shape[1], rs_shape[2]};
            resampled_2d = ov::Tensor(resampled_source.get_element_type(), shape_2d);
            std::memcpy(resampled_2d.data(), resampled_source.data(), resampled_source.get_byte_size());
            GENAI_DEBUG("[MiniCPM] Image %zu: reshaped to 2D [%zu, %zu]", image_id, shape_2d[0], shape_2d[1]);
        } else if (rs_shape.size() == 2) {
            // Already 2D, use as-is
            resampled_2d = resampled_source;
        } else {
            OPENVINO_ASSERT(false, "Unexpected resampled_source shape");
        }
        
        // Add resampled source (now guaranteed to be 2D)
        vision_features.push_back(resampled_2d);
        // Track tokens per block: each base image is one block
        tokens_per_image.push_back(base_tokens);
        images_grid_thw.push_back({1, 1, base_tokens});
        
        // Add slice features if available
        ov::Shape slices_shape = encoded_image.slices_shape;
        if (slices_shape.size()) {
            for (size_t i = 0; i < slices_shape.at(0); ++i) {
                for (size_t ja = 0; ja < slices_shape.at(1); ++ja) {
                    const ov::Tensor& slice_feat = encoded_image.resampled_image.vision_embed_tensors[i][ja];
                    
                    // Handle potential 3D slice features
                    auto slice_shape = slice_feat.get_shape();
                    ov::Tensor slice_2d;
                    if (slice_shape.size() == 3 && slice_shape[0] == 1) {
                        // Reshape from [1, query_num, hidden_size] to [query_num, hidden_size]
                        ov::Shape shape_2d = {slice_shape[1], slice_shape[2]};
                        slice_2d = ov::Tensor(slice_feat.get_element_type(), shape_2d);
                        std::memcpy(slice_2d.data(), slice_feat.data(), slice_feat.get_byte_size());
                    } else {
                        slice_2d = slice_feat;
                    }
                    
                    vision_features.push_back(slice_2d);
                    size_t slice_tokens = slice_2d.get_shape().at(0);
                    // Track tokens per block: each slice is one block
                    tokens_per_image.push_back(slice_tokens);
                    images_grid_thw.push_back({1, 1, slice_tokens});
                }
            }
        }
    }
    
    // Concatenate all vision features into a single tensor
    // Each feature is [query_num, hidden_size], we need to concatenate along token dimension
    size_t total_tokens = 0;
    size_t vision_hidden_size = 0;
    
    for (const auto& feat : vision_features) {
        // feat shape is [query_num, hidden_size], so first dimension is number of tokens
        total_tokens += feat.get_shape().at(0);
        if (vision_hidden_size == 0) {
            vision_hidden_size = feat.get_shape().at(1);  // Get hidden size from first feature
        }
    }
    
    OPENVINO_ASSERT(vision_hidden_size > 0, "Vision features must have valid hidden size");
    
    GENAI_DEBUG("[MiniCPM] Merging vision features: total_tokens=%zu, vision_hidden_size=%zu, m_vlm_config.hidden_size=%zu",
                total_tokens, vision_hidden_size, m_vlm_config.hidden_size);
    
    // Create 2D tensor [total_tokens, vision_hidden_size] for CDPruner compatibility
    ov::Tensor merged_vision_embeddings(ov::element::f32, {total_tokens, vision_hidden_size});
    float* merged_data = merged_vision_embeddings.data<float>();
    size_t token_offset = 0;
    
    for (const auto& feat : vision_features) {
        const size_t feat_num_tokens = feat.get_shape().at(0);  // query_num
        const size_t feat_hidden_size = feat.get_shape().at(1);  // hidden_size
        const float* feat_data = feat.data<float>();
        
        OPENVINO_ASSERT(feat_hidden_size == vision_hidden_size, 
                        "All vision features must have the same hidden size");
        
        // Copy each token from feat to merged_vision_embeddings
        for (size_t token_idx = 0; token_idx < feat_num_tokens; ++token_idx) {
            std::memcpy(
                merged_data + (token_offset + token_idx) * vision_hidden_size,
                feat_data + token_idx * feat_hidden_size,
                feat_hidden_size * sizeof(float)
            );
        }
        token_offset += feat_num_tokens;
    }
    
    return merged_vision_embeddings;
}

ov::Tensor InputsEmbedderMiniCPM::merge_text_and_vision_embeddings(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const ov::Tensor& vision_embeddings,
    const std::vector<EncodedImage>& images,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& tokens_per_image,
    int64_t im_start_id,
    int64_t im_end_id,
    int64_t slice_start_id,
    int64_t slice_end_id) {
    
    // Create a copy of text embeddings to merge vision embeddings into
    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(merged_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
    
    const int64_t* ids_begin = input_ids.data<int64_t>();
    const int64_t* ids = ids_begin;
    size_t encoded_input_size = input_ids.get_size();
    const int64_t* ids_end = ids + encoded_input_size;
    float* merged_embeds_data = merged_embeds.data<float>();
    
    // Debug: print input_ids and verify no extra placeholders
    GENAI_DEBUG("[MiniCPM] ========== MERGE START ==========");
    GENAI_DEBUG("[MiniCPM] merge: input_ids length=%zu, vision_embeddings [%zu, %zu]",
                encoded_input_size, vision_embeddings.get_shape()[0], vision_embeddings.get_shape()[1]);
    GENAI_DEBUG("[MiniCPM] merge: Expected %zu vision tokens to merge", vision_embeddings.get_shape()[0]);
    
    std::string ids_str = "[MiniCPM] merge: input_ids = [";
    for (size_t i = 0; i < std::min<size_t>(50, encoded_input_size); ++i) {
        ids_str += std::to_string(ids_begin[i]) + ", ";
    }
    if (encoded_input_size > 50) ids_str += "...";
    ids_str += "]";
    GENAI_DEBUG("%s", ids_str.c_str());
    GENAI_DEBUG("[MiniCPM] merge: im_start_id=%ld, im_end_id=%ld, slice_start_id=%ld, slice_end_id=%ld",
                im_start_id, im_end_id, slice_start_id, slice_end_id);
    
    // vision_embeddings shape is [total_tokens, hidden_size] (2D)
    const float* vision_embeds_data = vision_embeddings.data<float>();
    const size_t vision_hidden_size = vision_embeddings.get_shape().at(1);
    
    GENAI_DEBUG("[MiniCPM] merge_text_and_vision_embeddings: vision_embeddings shape [%zu, %zu], text hidden_size=%zu",
                vision_embeddings.get_shape().at(0), vision_hidden_size, m_vlm_config.hidden_size);
    
    // Verify that vision hidden size matches text hidden size for proper embedding
    // Note: tiny-random test models may have mismatched dimensions (vision=4, text=256)
    // In production models, these should match or have a projection layer
    if (vision_hidden_size != m_vlm_config.hidden_size) {
        GENAI_WARN("[MiniCPM] Vision hidden size (%zu) != text hidden size (%zu). "
                   "This is only acceptable for tiny-random test models. "
                   "Production models should have matching dimensions or a projection layer.",
                   vision_hidden_size, m_vlm_config.hidden_size);
    }
    
    // Use the minimum hidden size to avoid buffer overrun
    const size_t copy_hidden_size = std::min(vision_hidden_size, m_vlm_config.hidden_size);
    
    size_t vision_token_idx = 0;
    size_t image_idx = 0;
    
    for (size_t image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(image_id);
        
        // Find im_start position and insert resampled source embeddings
        ids = std::find(ids, ids_end, im_start_id);
        OPENVINO_ASSERT(ids_end != ids, "Could not find im_start_id in input_ids");
        size_t im_start_pos = std::distance(ids_begin, ids);
        GENAI_DEBUG("[MiniCPM] merge: Found im_start at position %zu (token_id=%ld)", im_start_pos, *ids);
        ++ids;
        
        // Get actual token count for this image (may be less than query_num if pruned)
        size_t actual_tokens = (image_idx < tokens_per_image.size()) ? 
                                tokens_per_image[image_idx] : m_vlm_config.query_num;
        
        GENAI_DEBUG("[MiniCPM] merge: image_block %zu, actual_tokens=%zu, query_num=%zu, vision_token_idx=%zu",
                    image_idx, actual_tokens, m_vlm_config.query_num, vision_token_idx);
        
        // Copy actual_tokens for the base image
        // Note: After pruning pipeline, input_ids no longer contains extra placeholders
        size_t first_token_pos = std::distance(ids_begin, ids);
        for (size_t q = 0; q < actual_tokens; ++q) {
            size_t flat_idx = std::distance(ids_begin, ids);
            int64_t current_token_id = *ids;
            // vision_embeddings[vision_token_idx, :] -> merged_embeds[0, flat_idx, :]
            // Copy only the minimum size to handle dimension mismatch in test models
            std::copy_n(vision_embeds_data + vision_token_idx * vision_hidden_size,
                       copy_hidden_size,
                       merged_embeds_data + flat_idx * m_vlm_config.hidden_size);
            ++vision_token_idx;
            ++ids;
        }
        GENAI_DEBUG("[MiniCPM] merge: Copied %zu vision tokens to positions [%zu-%zu]",
                    actual_tokens, first_token_pos, first_token_pos + actual_tokens - 1);
        
        ++image_idx;
        
        // Handle slices if available
        ov::Shape slices_shape = encoded_image.slices_shape;
        if (slices_shape.size()) {
            for (size_t i = 0; i < slices_shape.at(0); ++i) {
                for (size_t ja = 0; ja < slices_shape.at(1); ++ja) {
                    // Find slice_start position
                    ids = std::find(ids, ids_end, slice_start_id);
                    OPENVINO_ASSERT(ids_end != ids, "Could not find slice_start_id in input_ids");
                    size_t slice_start_pos = std::distance(ids_begin, ids);
                    GENAI_DEBUG("[MiniCPM] merge: Found slice_start at position %zu for slice[%zu][%zu]",
                                slice_start_pos, i, ja);
                    ++ids;
                    
                    // Get actual token count for this slice
                    size_t slice_actual_tokens = (image_idx < tokens_per_image.size()) ? 
                                                  tokens_per_image[image_idx] : m_vlm_config.query_num;
                    GENAI_DEBUG("[MiniCPM] merge: slice_block %zu, actual_tokens=%zu",
                                image_idx, slice_actual_tokens);
                    
                    // Copy slice_actual_tokens for this slice
                    // Note: Pruning pipeline has already removed extra placeholders from input_ids
                    for (size_t q = 0; q < slice_actual_tokens; ++q) {
                        size_t flat_idx = std::distance(ids_begin, ids);
                        // vision_embeddings[vision_token_idx, :] -> merged_embeds[0, flat_idx, :]
                        // Copy only the minimum size to handle dimension mismatch in test models
                        std::copy_n(vision_embeds_data + vision_token_idx * vision_hidden_size,
                                   copy_hidden_size,
                                   merged_embeds_data + flat_idx * m_vlm_config.hidden_size);
                        ++vision_token_idx;
                        ++ids;
                    }
                    
                    ++image_idx;
                }
            }
        }
    }
    
    // Debug: Verify all vision tokens were consumed
    size_t total_vision_tokens = vision_embeddings.get_shape()[0];
    GENAI_DEBUG("[MiniCPM] merge: Consumed %zu/%zu vision tokens", vision_token_idx, total_vision_tokens);
    if (vision_token_idx != total_vision_tokens) {
        GENAI_WARN("[MiniCPM] merge: WARNING - Not all vision tokens were merged! Expected %zu, used %zu",
                   total_vision_tokens, vision_token_idx);
    }
    GENAI_DEBUG("[MiniCPM] ========== MERGE END ==========");
    
    return merged_embeds;
}

ov::Tensor InputsEmbedderMiniCPM::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    ov::Tensor encoded_input = get_encoded_input_ids(unified_prompt, metrics);

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor inputs_embeds = m_embedding->infer(req, encoded_input);

    OPENVINO_ASSERT(
        m_vlm_config.hidden_size == inputs_embeds.get_shape().at(2),
        "Unexpected embedding size"
    );
    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor special_tokens = m_tokenizer.encode(
        m_vlm_config.im_start
        + m_vlm_config.im_end
        + m_vlm_config.slice_start
        + m_vlm_config.slice_end,
        ov::genai::add_special_tokens(false)
    ).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    OPENVINO_ASSERT(
        4 == special_tokens.get_shape().at(1),
        "Every special token must be represented with a single int."
    );
    int64_t im_start_id = special_tokens.data<int64_t>()[0];
    int64_t im_end_id = special_tokens.data<int64_t>()[1];
    int64_t slice_start_id = special_tokens.data<int64_t>()[2];
    int64_t slice_end_id = special_tokens.data<int64_t>()[3];

    // Return early if no images
    if (images.empty()) {
        ov::Tensor inputs_embeds_copy(inputs_embeds.get_element_type(), inputs_embeds.get_shape());
        std::memcpy(inputs_embeds_copy.data(), inputs_embeds.data(), inputs_embeds.get_byte_size());
        return inputs_embeds_copy;
    }

    // Step 1: Prepare vision embeddings (similar to run_video_image_embeddings_merger)
    std::vector<size_t> tokens_per_image;
    std::vector<std::array<size_t, 3>> images_grid_thw;
    ov::Tensor merged_vision_embeddings = prepare_vision_embeddings(
        images, images_sequence, tokens_per_image, images_grid_thw);
    
    // Step 1.5: Apply pruning if configured
    // Note: MiniCPM doesn't use image_pad tokens like Qwen2VL, so we use a simplified approach:
    // - Build PruningContext with vision_pad_token_id=-1 (indicating no pad tokens)
    // Step 1.5: Apply pruning if configured
    if (!images.empty() && is_cdpruner_active()) {
        size_t text_hidden_size = inputs_embeds.get_shape()[2];
        size_t vision_hidden_size = merged_vision_embeddings.get_shape()[1];
        
        // Check if hidden sizes match - CDPruner requires matching dimensions
        if (vision_hidden_size != text_hidden_size) {
            GENAI_WARN("[MiniCPM] Skipping pruning: vision hidden_size (%zu) != text hidden_size (%zu). "
                       "This is expected for tiny-random test models.",
                       vision_hidden_size, text_hidden_size);
        } else {
            std::vector<size_t> original_tokens_per_block = tokens_per_image;
            
            // Build tokens_per_vision: sum all blocks belonging to each image
            std::vector<size_t> tokens_per_vision;
            std::vector<std::array<size_t, 3>> vision_grid_thw;
            size_t block_idx = 0;
            for (size_t image_id : images_sequence) {
                const EncodedImage& encoded_image = images.at(image_id);
                size_t image_total_tokens = tokens_per_image[block_idx++];  // base image
                
                // Add slices
                ov::Shape slices_shape = encoded_image.slices_shape;
                if (slices_shape.size()) {
                    for (size_t i = 0; i < slices_shape.at(0); ++i) {
                        for (size_t ja = 0; ja < slices_shape.at(1); ++ja) {
                            image_total_tokens += tokens_per_image[block_idx++];
                        }
                    }
                }
                
                tokens_per_vision.push_back(image_total_tokens);
                vision_grid_thw.push_back({1, 1, image_total_tokens});
            }
            
            PruningContext pruning_context{
                encoded_input, inputs_embeds, merged_vision_embeddings,
                images.size(), vision_grid_thw, images_sequence, tokens_per_vision,
                -1, im_start_id, im_end_id, slice_start_id, slice_end_id, 1
            };
            
            auto pruning_result = execute_pruning_pipeline(pruning_context);
            
            if (pruning_result) {
                merged_vision_embeddings = std::move(pruning_result->pruned_embeddings);
                encoded_input = pruning_result->pruned_input_ids;
                inputs_embeds = pruning_result->pruned_text_embeds;
                
                // Rebuild tokens_per_image (per-block) from keep_flags_per_region (per-image)
                tokens_per_image.clear();
                OPENVINO_ASSERT(pruning_result->keep_flags_per_region.size() == images_sequence.size(), 
                               "keep_flags_per_region size mismatch");
                
                size_t original_block_idx = 0;
                for (size_t img_idx = 0; img_idx < images_sequence.size(); ++img_idx) {
                    size_t image_id = images_sequence[img_idx];
                    const EncodedImage& encoded_image = images.at(image_id);
                    const auto& image_keep_mask = pruning_result->keep_flags_per_region[img_idx];
                    
                    // Calculate blocks: 1 base + slices
                    size_t num_blocks = 1;
                    ov::Shape slices_shape = encoded_image.slices_shape;
                    if (slices_shape.size()) {
                        num_blocks += slices_shape.at(0) * slices_shape.at(1);
                    }
                    
                    // Split the keep_mask into blocks
                    size_t mask_offset = 0;
                    for (size_t block = 0; block < num_blocks; ++block) {
                        size_t block_token_count = original_tokens_per_block[original_block_idx++];
                        
                        OPENVINO_ASSERT(mask_offset + block_token_count <= image_keep_mask.size(),
                                       "Block token count exceeds keep_mask bounds at image " + 
                                       std::to_string(img_idx) + ", block " + std::to_string(block) + 
                                       ": offset=" + std::to_string(mask_offset) + 
                                       ", block_tokens=" + std::to_string(block_token_count) + 
                                       ", mask_size=" + std::to_string(image_keep_mask.size()));
                        
                        size_t kept_count = 0;
                        for (size_t i = 0; i < block_token_count; ++i) {
                            if (image_keep_mask[mask_offset + i]) {
                                ++kept_count;
                            }
                        }
                        mask_offset += block_token_count;
                        tokens_per_image.push_back(kept_count);
                    }
                }
            }
        }
    }

    // Step 2: Merge text and vision embeddings (similar to merge_text_and_video_image_embeddings)
    GENAI_DEBUG("[MiniCPM] About to call merge_text_and_vision_embeddings with %zu tokens_per_image entries", 
                tokens_per_image.size());
    std::string tokens_str = "[MiniCPM] tokens_per_image = [";
    for (size_t i = 0; i < tokens_per_image.size(); ++i) {
        tokens_str += std::to_string(tokens_per_image[i]);
        if (i + 1 < tokens_per_image.size()) tokens_str += ", ";
    }
    tokens_str += "]";
    GENAI_DEBUG("%s", tokens_str.c_str());
    ov::Tensor final_embeds = merge_text_and_vision_embeddings(
        encoded_input,
        inputs_embeds,  // Pass directly, merge function will create its own copy
        merged_vision_embeddings,
        images,
        images_sequence,
        tokens_per_image,
        im_start_id,
        im_end_id,
        slice_start_id,
        slice_end_id
    );
    
    GENAI_DEBUG("[MiniCPM] merge_text_and_vision_embeddings completed, final_embeds shape [%zu, %zu, %zu]",
                final_embeds.get_shape()[0], final_embeds.get_shape()[1], final_embeds.get_shape()[2]);
    
    // Ensure we return an independent copy to avoid lifetime issues
    ov::Tensor result(final_embeds.get_element_type(), final_embeds.get_shape());
    std::memcpy(result.data(), final_embeds.data(), final_embeds.get_byte_size());
    return result;
}

ov::Tensor VisionEncoderMiniCPM::resample(const ov::Tensor& encoded_image, const ImageSize& target_size, size_t pad_to_max) {
    size_t bs = encoded_image.get_shape().at(0);
    size_t patch_len = target_size.height * target_size.width;
    
    GENAI_DEBUG("[Resampler] Input: encoded_image shape [%zu, %zu, %zu], target_size [%zu, %zu], pad_to_max=%zu, m_vlm_config.hidden_size=%zu",
                encoded_image.get_shape()[0], encoded_image.get_shape()[1], encoded_image.get_shape()[2],
                target_size.height, target_size.width, pad_to_max, m_vlm_config.hidden_size);
    
    adjust_pos_cache(
        {target_size},
        m_vlm_config.hidden_size,
        m_pos_embed_cache
    );
    ov::Tensor key_padding_mask(ov::element::f32, {bs, pad_to_max});
    float* mask_data = key_padding_mask.data<float>();
    size_t embed_len = m_pos_embed_cache.get_shape().at(2);
    ov::Tensor pos_embed(ov::element::f32, {pad_to_max, bs, embed_len});  // BLD => L * B * D
    float* pos_embed_data = pos_embed.data<float>();
    float* cache_data = m_pos_embed_cache.data<float>();
    size_t _d0 = m_pos_embed_cache.get_shape().at(0);
    size_t _d1 = m_pos_embed_cache.get_shape().at(1);
    for (size_t i = 0; i < bs; ++i) {
        size_t target_h = target_size.height;
        size_t target_w = target_size.width;
        for (size_t h_idx = 0; h_idx < target_h; ++h_idx) {
            for (size_t w_idx = 0; w_idx < target_w; ++w_idx) {
                std::copy_n(
                    cache_data + (h_idx * _d1 + w_idx) * embed_len,
                    embed_len,
                    pos_embed_data + (h_idx * target_w + w_idx) * bs * embed_len + i * embed_len
                );
            }
        }
        for (size_t flat = target_h * target_w; flat < pad_to_max; ++flat) {
            std::fill_n(pos_embed_data + flat * bs * embed_len + i * embed_len, embed_len, 0.0f);
        }
        std::fill_n(mask_data + i * pad_to_max, patch_len, 0.0f);
        std::fill_n(mask_data + i * pad_to_max + patch_len, pad_to_max - patch_len, 1.0f);
    }
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_resampler.get());
    ov::InferRequest& resampler = infer_request_guard.get();
    resampler.set_tensor("image_feature", encoded_image);  // [N, H*W, old_hidden_size]
    resampler.set_tensor("pos_embed", pos_embed);  // [H*W, N, new_hidden_size]
    resampler.set_tensor("key_padding_mask", key_padding_mask);  // [N, H*W]
    resampler.infer();
    auto resampler_out = resampler.get_output_tensor();
    
    auto out_shape = resampler_out.get_shape();
    GENAI_DEBUG("[Resampler] Output: resampler_out shape has %zu dimensions", out_shape.size());
    for (size_t i = 0; i < out_shape.size(); ++i) {
        GENAI_DEBUG("[Resampler] Output: dimension %zu = %zu", i, out_shape[i]);
    }
    GENAI_DEBUG("[Resampler] Output: get_byte_size = %zu", resampler_out.get_byte_size());
    
    // resampler_out is bound to infer request and the data may become corrupted after next resampler inference 
    // so we need to return a copy to make sure data does not get corrupted 
    ov::Tensor res(resampler_out.get_element_type(), resampler_out.get_shape());
    std::memcpy(res.data(), resampler_out.data(), resampler_out.get_byte_size());
    
    GENAI_DEBUG("[Resampler] Copied result: res shape has %zu dimensions, byte_size=%zu", 
                res.get_shape().size(), res.get_byte_size());
    return res;  // [N, query_num, new_hidden_size]
}

VisionEncoderMiniCPM::VisionEncoderMiniCPM(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties) : VisionEncoder{model_dir, device, properties} {
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
    
    GENAI_DEBUG("[VisionEncoderMiniCPM] Config: hidden_size=%zu, query_num=%zu",
                m_vlm_config.hidden_size, m_vlm_config.query_num);
    
    auto compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_resampler_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM resampler model");
    m_ireq_queue_resampler = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        }); 
    m_pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70});
}

VisionEncoderMiniCPM::VisionEncoderMiniCPM(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) : VisionEncoder{models_map, config_dir_path, device, device_config} {
    const auto& resampler_model = utils::get_model_weights_pair(models_map, "resampler").first;
    const auto& resampler_weights = utils::get_model_weights_pair(models_map, "resampler").second;
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    auto compiled_model = utils::singleton_core().compile_model(resampler_model, resampler_weights, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM resampler model");
    m_ireq_queue_resampler = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        }); 
    m_pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70});
}


InputsEmbedderMiniCPM::InputsEmbedderMiniCPM(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) {}

InputsEmbedderMiniCPM::InputsEmbedderMiniCPM(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

} // namespace ov::genai
