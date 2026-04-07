# Extend CB for custom VIT inputs_embeds

## Description

扩展 `ContinuousBatchingPipeline`，支持客户自定义 VIT 产出的 `inputs_embeds`，并传入额外 LM 输入（如 `deepstack_embeds.*`、`attention_mask`、`visual_pos_mask`、`beam_idx`）。

## Build

```bash
./build.sh
```

## Run sample

```bash
# 使用默认路径（远程机器 xiping@odt-huyuan-openvino-ci-74）
./run_sample.sh

# 或显式指定参数
./run_sample.sh <MODEL_DIR> <DATA_DIR> [DEVICE]

# 直接运行二进制
./build/samples/cpp/custom_vit_cb/custom_vit_cb <MODEL_DIR> <DATA_DIR> [DEVICE]
```

- `MODEL_DIR` — 模型目录，`config.json` 须含 `"model_type": "modeling_vl"`
- `DATA_DIR` — tensor 数据目录（含 `inputs_embeds`、`attention_mask`、`position_ids`、`deepstack_embeds_*` 等）
- `DEVICE` — 推理设备（默认 `CPU`）

输出文本可读即可，语义与 sample 内 `expected_text` 类似即可。

## Implementation summary (current)

### 1) Dummy ModelingVL classes
当 VIT 已由客户处理、直接传递 `inputs_embeds` 时，仍然走 VLM pipeline。提供 dummy 版本以满足接口：

- [src/cpp/src/visual_language/modeling_vl/classes.hpp](src/cpp/src/visual_language/modeling_vl/classes.hpp)
- [src/cpp/src/visual_language/modeling_vl/classes.cpp](src/cpp/src/visual_language/modeling_vl/classes.cpp)

### 2) ModelingVL model_type 支持
新增 `model_type: "modeling_vl"` 以绕开已有模型类型约束：

- `VLMModelType` 新增 `MODELING_VL`（[vlm_config.hpp](src/cpp/src/visual_language/vlm_config.hpp)）
- `vlm_config.cpp` 中加入 `{"modeling_vl", VLMModelType::MODELING_VL}`（[vlm_config.cpp](src/cpp/src/visual_language/vlm_config.cpp)）

### 3) VisionEncoder/InputsEmbedder 工厂路由
`MODELING_VL` 分支在工厂中创建对应的 dummy 实例：

- [src/cpp/src/visual_language/vision_encoder.cpp](src/cpp/src/visual_language/vision_encoder.cpp)
- [src/cpp/src/visual_language/inputs_embedder.cpp](src/cpp/src/visual_language/inputs_embedder.cpp)

### 4) ContinuousBatchingPipeline 公共 API 扩展
新增两个 `generate()` 重载，支持传入 `token_type_ids`、`position_ids`、`lm_extra_inputs_list`：

- [src/cpp/include/openvino/genai/continuous_batching_pipeline.hpp](src/cpp/include/openvino/genai/continuous_batching_pipeline.hpp)
- [src/cpp/src/continuous_batching/pipeline.cpp](src/cpp/src/continuous_batching/pipeline.cpp)
- [src/cpp/src/continuous_batching/pipeline_impl.cpp](src/cpp/src/continuous_batching/pipeline_impl.cpp) — 增加了 `m_inputs_embedder` 空指针保护

### 5) Sample
`samples/cpp/custom_vit_cb/custom_vit_cb.cpp` 通过命令行参数接收 `MODEL_DIR`、`DATA_DIR`、`DEVICE`，从文件加载 tensor，调用 `generate()` 传入：

- `inputs_embeds`
- `position_ids`（可选）
- `extra_inputs`（如 `deepstack_embeds.*`, `attention_mask`, `visual_pos_mask`, `beam_idx`）

### 6) 配置要求
`config.json` 必须包含 `model_type: "modeling_vl"`，以避免 `Unsupported 'xxx' VLM model type`。
