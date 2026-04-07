# C++ custom ViT + continuous batching (inputs_embeds)

This sample shows how to pass user-produced `inputs_embeds` directly into
`ov::genai::ContinuousBatchingPipeline` for inference. The ViT (or any custom
visual encoder) is expected to be implemented by the caller; the sample only
demonstrates how to feed its output into continuous batching with extra LM
inputs such as `deepstack_embeds.*`, `attention_mask`, `visual_pos_mask`, and
`beam_idx`.

## Build

```bash
# From the openvino.genai root:
./build.sh
```

## Run

```bash
./run_sample.sh [MODEL_DIR] [DATA_DIR] [DEVICE]
```

Or directly:

```bash
./build/samples/cpp/custom_vit_cb/custom_vit_cb <MODEL_DIR> <DATA_DIR> [DEVICE]
```

- `MODEL_DIR` — path to a model directory whose `config.json` contains
  `"model_type": "modeling_vl"` (e.g. Qwen3-Omni-4B-Instruct-multilingual-int4).
- `DATA_DIR` — path to a directory containing pre-exported tensor files
  (`inputs_embeds`, `attention_mask`, `position_ids`, `deepstack_embeds_*`, etc.).
- `DEVICE` — inference device (default: `CPU`).
