# DFlash Continuous Batching Pipeline 集成文档

## 目录

- [1. 背景知识](#1-背景知识)
  - [1.1 EAGLE3 简介](#11-eagle3-简介)
  - [1.2 DFlash 简介](#12-dflash-简介)
  - [1.3 EAGLE3 与 DFlash 对比](#13-eagle3-与-dflash-对比)
- [2. GenAI 中的 Speculative Decoding 架构](#2-genai-中的-speculative-decoding-架构)
- [3. DFlash Pipeline 实现细节](#3-dflash-pipeline-实现细节)
  - [3.1 文件结构](#31-文件结构)
  - [3.2 核心类设计](#32-核心类设计)
  - [3.3 Pipeline 工厂链](#33-pipeline-工厂链)
  - [3.4 模型变换流程](#34-模型变换流程)
  - [3.5 Generate 流程](#35-generate-流程)
- [4. 模型准备](#4-模型准备)
- [5. Sample 验证](#5-sample-验证)
  - [5.1 编译](#51-编译)
  - [5.2 运行 Sample](#52-运行-sample)
  - [5.3 预期输出](#53-预期输出)
- [6. Test Case](#6-test-case)
  - [6.1 测试设计思路](#61-测试设计思路)
  - [6.2 运行测试](#62-运行测试)
  - [6.3 预期测试结果](#63-预期测试结果)
- [7. 故障排查](#7-故障排查)

---

## 1. 背景知识

### 1.1 EAGLE3 简介

**EAGLE3**（Efficient Augmented LLM via Extrapolation of Attention for Generation Enhancement, 第三代）是一种基于自回归草稿模型（autoregressive draft model）的推测解码（speculative decoding）算法。

**核心原理：**
- 目标模型（target model）在推理过程中，从选定的隐藏层（如 early/middle/late 层）提取 hidden states
- 这些 hidden states 经过 FC（全连接层）投影后，传递给一个小型草稿模型
- 草稿模型使用这些特征逐 token 地自回归生成候选 token 序列
- 目标模型一次性验证所有候选 token，接受正确的 token

**EAGLE3 在 GenAI 中的实现特点：**
- 草稿模型有独立的架构，包含 `midlayer` 节点和 FC 权重
- 使用 `eagle3_mode` rt_info 标记草稿模型
- 主模型和草稿模型之间通过 hidden state export/import 机制通信
- 草稿模型逐 token 生成（每步 `num_assistant_tokens` 个候选）
- 典型模型对：`Qwen3-1.7B`（目标）+ `AngelSlim/Qwen3-1.7B_eagle3`（草稿）

### 1.2 DFlash 简介

**DFlash**（Diffusion Flash）是一种基于块扩散模型（block diffusion model）的推测解码算法，来源于论文 [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) 和开源仓库 [z-lab/dflash](https://github.com/z-lab/dflash)。

**核心原理：**
- 与 EAGLE3 逐 token 生成不同，DFlash 在**单次前向传播**中生成整个 block 的草稿 token
- 目标模型从选定隐藏层提取特征 → 拼接 → FC 投影 → DFlash 草稿模型以此为条件，一次性生成 `block_size` 个候选 token
- 目标模型再一次性验证这些候选 token

**关键参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `block_size` | 每次草稿生成的 token 数量 | 16 |
| `mask_token_id` | 掩码 token 的 ID | -1 |
| `target_layer_ids` | 目标模型中用于提取 hidden state 的层索引 | 自动选择 |

**与 EAGLE3 的关键区别：**
- DFlash 的主模型和草稿模型使用**相同的 input_ids**（不做 token 移位）
- 草稿生成是单步块级操作，而非自回归逐 token 生成

### 1.3 EAGLE3 与 DFlash 对比

| 特性 | EAGLE3 | DFlash |
|------|--------|--------|
| 草稿生成方式 | 自回归逐 token | 单步块级扩散 |
| 生成一轮候选的前向次数 | `num_assistant_tokens` 次 | 1 次 |
| input_ids 处理 | 草稿去掉首 token（移位） | 主模型与草稿使用相同 input_ids |
| 草稿模型架构 | 独立架构（含 midlayer + FC） | 可复用目标模型或自定义架构 |
| hidden state 传递 | 必需（export → FC → import） | 可选（有 FC 时使用，无 FC 时退化为标准 SD） |
| 标记方式 | `eagle3_mode` rt_info | `dflash_mode` rt_info |

---

## 2. GenAI 中的 Speculative Decoding 架构

GenAI 的 Continuous Batching 管线使用以下类层次结构：

```
IContinuousBatchingPipeline
  └── SpeculativeDecodingImpl          (基类，持有 m_main_pipeline + m_draft_pipeline)
        ├── Eagle3DecodingImpl         (EAGLE3 推测解码)
        ├── DFlashDecodingImpl         (DFlash 推测解码)  ← 新增
        └── PromptLookupImpl          (Prompt Lookup 推测解码)
```

内部子管线类型：
```
ContinuousBatchingImpl
  └── ContinuousBatchingForSpeculativeDecodingImpl    (标准 SD 子管线)
        └── ContinuousBatchingForEagle3DecodingImpl   (带 hidden state export/import)
```

Pipeline 工厂选择链（在 `pipeline.cpp` 中）：
```
prompt_lookup → eagle3 → dflash → regular_speculative_decoding → plain_continuous_batching
```

---

## 3. DFlash Pipeline 实现细节

### 3.1 文件结构

```
src/cpp/
├── include/openvino/genai/
│   └── continuous_batching_pipeline.hpp          # 添加 DFlashDecodingImpl 前向声明 + friend
├── src/
│   ├── continuous_batching/
│   │   └── pipeline.cpp                          # 工厂链中添加 DFlash 分支
│   ├── llm/
│   │   └── pipeline.cpp                          # draft_model() 中调用 apply_dflash_rt_info()
│   └── speculative_decoding/
│       ├── dflash_model_transforms.hpp           # DFlashRTInfo 结构体 + 函数声明
│       ├── dflash_model_transforms.cpp           # 配置提取 + rt_info 应用实现
│       └── continuous_batching/
│           ├── dflash_strategy.hpp               # DFlashDecodingImpl 类声明
│           └── dflash_strategy.cpp               # 完整构造逻辑 + generate 实现
samples/cpp/text_generation/
├── CMakeLists.txt                                # 添加 dflash_speculative_decoding 目标
└── dflash_speculative_decoding.cpp               # C++ sample
tests/python_tests/samples/
└── test_dflash_speculative_decoding_lm.py        # Pytest test case
tools/dflash/
└── prepare_dflash_draft_model.py                 # 模型转换脚本
```

### 3.2 核心类设计

#### `DFlashRTInfo` 结构体

```cpp
struct DFlashRTInfo {
    bool dflash_mode = false;
    size_t block_size = 16;
    int64_t mask_token_id = -1;
    std::vector<int32_t> hidden_layers_list;
};
```

配置读取优先级：
1. 用户通过 `ov::AnyMap` 显式传入 `dflash_mode`、`dflash_block_size` 等
2. 从草稿模型的 `config.json` 中读取 `dflash_config` 段落
3. 根据 `num_hidden_layers` 自动选择 `target_layer_ids`（≥10 层：`[2, n/2, n-3]`；4-9 层：`[0, n/2, n-1]`）

#### `DFlashDecodingImpl` 类

继承自 `SpeculativeDecodingImpl`，重写：
- **构造函数**：初始化模型变换 + 子管线创建
- **`add_request()`**：主模型和草稿模型使用相同 input_ids（区别于 EAGLE3 的 token 移位）
- **`generate()`**：使用 `GenerateStrategy` + `generate_common` 模板

### 3.3 Pipeline 工厂链

在 `pipeline.cpp` 的 4 个构造函数重载中，DFlash 分支插入在 EAGLE3 和标准 SD 之间：

```cpp
auto dflash_rt_info = utils::dflash::extract_dflash_info_from_config(
    properties_without_draft_model_without_gguf, draft_models_path);

if (eagle_rt_info.eagle3_mode) {
    // ... Eagle3 分支
} else if (dflash_rt_info.dflash_mode) {
    return std::make_shared<DFlashDecodingImpl>(
        ModelDesc{main_model, ...},
        ModelDesc{draft_model, ...},
        dflash_rt_info);
} else {
    // ... 标准 SD 分支
}
```

### 3.4 模型变换流程

`DFlashDecodingImpl` 构造函数中的模型变换步骤：

```
1. share_vocabulary(main, draft)
   → 将主模型的 embedding 权重拷贝到草稿模型（确保词表一致）

2. transform_hidden_state(main, [2, 15, 27])
   → 在主模型中找到指定层的残差 Add 节点
   → 将这3层的 hidden states 沿 last dim 拼接（Concat）
   → 添加为新输出 "last_hidden_state"

3. 尝试 move_fc_from_draft_to_main(draft, main)
   → 如果草稿模型有 FC 层（自定义训练的 DFlash 模型）：
     - 从草稿中提取 FC 权重，移入主模型
     - 使用 ContinuousBatchingForEagle3DecodingImpl（支持 hidden state import）
   → 如果没有 FC 层（标准模型作为草稿）：
     - 退化为 ContinuousBatchingForSpeculativeDecodingImpl
     - 功能等同于标准推测解码，但走 DFlash 的参数控制逻辑
```

### 3.5 Generate 流程

```
generate() 入口
  │
  ├── 构造 GenerateStrategy:
  │     prepare_request: 设置 num_assistant_tokens = block_size，
  │                      main_in = draft_in = input_ids（无移位）
  │     check_streaming: 仅支持单请求 greedy/multinomial
  │
  └── generate_common(this, ...)
        │
        ├── 添加请求到 main_pipeline + draft_pipeline
        │
        └── while (has_non_finished_requests):
              step()  ← SpeculativeDecodingImpl::step()
              │
              ├── draft_pipeline->multistep()
              │     → 生成 num_assistant_tokens 个候选 token
              │
              ├── main_pipeline->update_request(candidates)
              │     → 将候选写入主模型的 KV cache
              │
              ├── main_pipeline->step()
              │     → 验证候选，接受正确 token，生成 bonus token
              │
              └── draft_pipeline->update_request(verified)
                    → 同步验证结果，裁剪/重置草稿状态
```

---

## 4. 模型准备

### 4.1 导出目标模型

使用 `optimum-cli` 将 HuggingFace 模型导出为 OpenVINO IR 格式：

```bash
pip install optimum-intel openvino
optimum-cli export openvino \
    --model HuggingFaceTB/SmolLM2-135M \
    --task text-generation-with-past \
    --trust-remote-code \
    ~/models/smollm2-135m-ov
```

### 4.2 准备 DFlash 草稿模型

使用 `prepare_dflash_draft_model.py` 脚本将标准模型转换为 DFlash 草稿模型：

```bash
python tools/dflash/prepare_dflash_draft_model.py \
    --source-model-dir ~/models/smollm2-135m-ov \
    --output-dir ~/models/smollm2-135m-dflash-draft \
    --block-size 4
```

该脚本执行以下操作：
1. 拷贝源模型所有文件到输出目录
2. 在 OpenVINO 模型的 rt_info 中添加 `dflash_mode=true`、`dflash_block_size`、`dflash_mask_token_id`、`hidden_layers_list`
3. 在 `config.json` 中添加 `dflash_config` 段落

预期输出：
```
DFlash draft model prepared at: /home/ywang2/models/smollm2-135m-dflash-draft
  block_size: 4
  mask_token_id: -1
  target_layer_ids: [2, 15, 27]
```

---

## 5. Sample 验证

### 5.1 编译

使用 VS Code CMake Tools 编译：
- 目标：`openvino_genai` + `dflash_speculative_decoding`

或命令行（确保 OpenVINO 环境已初始化）：
```bash
cd build
cmake --build . --target openvino_genai dflash_speculative_decoding -j$(nproc)
```

### 5.2 运行 Sample

**CPU 设备：**
```bash
cd /home/ywang2/openvino.genai/build
LD_LIBRARY_PATH=$(pwd)/openvino_genai:$LD_LIBRARY_PATH \
  ./bin/dflash_speculative_decoding \
  ~/models/smollm2-135m-ov \
  ~/models/smollm2-135m-dflash-draft \
  "Alan Turing was a" \
  CPU
```

**GPU 设备：**
```bash
cd /home/ywang2/openvino.genai/build
LD_LIBRARY_PATH=$(pwd)/openvino_genai:$LD_LIBRARY_PATH \
  ./bin/dflash_speculative_decoding \
  ~/models/smollm2-135m-ov \
  ~/models/smollm2-135m-dflash-draft \
  "Alan Turing was a" \
  GPU
```

**Sample 用法：**
```
./bin/dflash_speculative_decoding <MODEL_DIR> <DFLASH_DRAFT_MODEL_DIR> '<PROMPT>' [DEVICE]
```

### 5.3 预期输出

**CPU 运行结果：**
```
 British mathematician and computer scientist who is best known for his work
 on the theory of computation. He is also known for his work on the theory of
 computation and the theory of computation and computation theory. Turing was
 born on 23rd January 1912 in London, England. He was the son of a school
 teacher. Turing was educated at the University of Cambridge and at the
 University of London. He was a mathematician and computer scientist. He was
 awarded the Turing Award in

MAIN MODEL (Target)
  Generate time: ~218 ms
  TTFT: ~48 ms
  TPOT: ~1.9 ms/iteration
  Num generated token: 100 tokens
  Num accepted token: 92 tokens

DRAFT MODEL (DFlash)
  Generate time: ~957 ms
  Num generated token: 108 tokens
```

**GPU 运行结果：**
```
 British mathematician and computer scientist who is best known for his work
 on the theory of computation. ...

MAIN MODEL (Target)
  Generate time: ~317 ms
  TTFT: ~94 ms
  TPOT: ~2.0 ms/iteration
  Num generated token: 100 tokens
  Num accepted token: 93 tokens

DRAFT MODEL (DFlash)
  Generate time: ~1262 ms
  Num generated token: 99 tokens
```

**验证成功标准：**
1. 程序以 exit code 0 退出
2. 生成文本非空
3. 输出与 `greedy_causal_lm` 基线一致（使用相同目标模型时）
4. `Num accepted token > 0`，表明推测解码机制生效

**对比验证（greedy baseline）：**
```bash
LD_LIBRARY_PATH=$(pwd)/openvino_genai:$LD_LIBRARY_PATH \
  ./bin/greedy_causal_lm ~/models/smollm2-135m-ov "Alan Turing was a"
```

输出应与 DFlash sample 生成的文本完全一致。

---

## 6. Test Case

### 6.1 测试设计思路

测试文件：`tests/python_tests/samples/test_dflash_speculative_decoding_lm.py`

**测试类：`TestDFlashSpeculativeDecodingLM`**

测试流程：
1. 通过 `conftest.py` 的 `convert_model` fixture 将 `SmolLM2-135M` 导出为 OpenVINO IR
2. 调用 `_prepare_dflash_draft()` 辅助函数，拷贝模型并添加 `dflash_mode` rt_info
3. 运行 C++ `dflash_speculative_decoding` sample
4. 验证 exit code = 0，输出非空
5. 运行 `greedy_causal_lm` baseline，验证输出一致

**标记：** `@pytest.mark.dflash_decoding`

**参数化：**
| 目标模型 | Prompt |
|----------|--------|
| SmolLM2-135M | "Alan Turing was a" |

### 6.2 运行测试

```bash
# 确保环境变量已设置
source ~/setup_dev_env.sh
cd /home/ywang2/openvino.genai

# 运行 DFlash 测试
SAMPLES_CPP_DIR=build/bin \
  pytest tests/python_tests/samples/test_dflash_speculative_decoding_lm.py \
  -m dflash_decoding \
  -v \
  --tb=short
```

### 6.3 预期测试结果

```
tests/python_tests/samples/test_dflash_speculative_decoding_lm.py::TestDFlashSpeculativeDecodingLM::test_dflash_speculative_decoding_lm[SmolLM2-135M-Alan Turing was a] PASSED
```

测试通过条件：
- C++ sample 返回码为 0
- 输出文本非空
- 输出文本包含 greedy baseline 的完整输出

---

## 7. 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `Failed to locate FC weights in eagle3 draft model` | 草稿模型没有 EAGLE3 兼容的 FC 层 | 正常行为，DFlash 自动退化为标准 SD 模式 |
| `stored_seq_len == total_num_tokens` assertion | 使用了 `ContinuousBatchingForEagle3DecodingImpl` 但模型不支持 hidden state import | 确认 dflash_strategy.cpp 中有 try/catch 逻辑，退化到 `ContinuousBatchingForSpeculativeDecodingImpl` |
| `dflash_mode` 未被识别 | 草稿模型 rt_info 中未设置 `dflash_mode` | 使用 `prepare_dflash_draft_model.py` 重新准备草稿模型 |
| 编译错误：`qualified name does not name a class` | `continuous_batching_pipeline.hpp` 缺少前向声明 | 确认 `class DFlashDecodingImpl;` 和 `friend class DFlashDecodingImpl;` 已添加 |
| 输出与 greedy baseline 不一致 | `num_assistant_tokens` 设置过大导致验证行为差异 | 尝试减小 `block_size`（如 4）或使用 `do_sample=False` |

---

## 8. Qwen3-8B 实验

### 8.1 模型准备

**目标模型（Target）：** Qwen3-8B OV IR（已预转换）
```bash
# 从 /mnt/bell/Qwen3-8B 复制到本地
mkdir -p ~/models/intel/Qwen3-8B
cp /mnt/bell/Qwen3-8B/openvino_model.* \
   /mnt/bell/Qwen3-8B/config.json \
   /mnt/bell/Qwen3-8B/generation_config.json \
   /mnt/bell/Qwen3-8B/openvino_tokenizer.* \
   /mnt/bell/Qwen3-8B/openvino_detokenizer.* \
   /mnt/bell/Qwen3-8B/tokenizer*.json \
   /mnt/bell/Qwen3-8B/special_tokens_map.json \
   /mnt/bell/Qwen3-8B/merges.txt \
   /mnt/bell/Qwen3-8B/vocab.json \
   ~/models/intel/Qwen3-8B/
```

**草稿模型（Draft）：** 从 Qwen3-8B 目标模型准备
```bash
cd /home/ywang2/openvino.genai
python3 tools/dflash/prepare_dflash_draft_model.py \
  --source-model-dir ~/models/intel/Qwen3-8B \
  --output-dir ~/models/intel/Qwen3-8B-dflash-draft \
  --block-size 16 \
  --mask-token-id 151669 \
  --target-layer-ids 1 9 17 25 33
```

**DFlash 真正的草稿模型（z-lab/Qwen3-8B-DFlash-b16）：**
```bash
# 使用国内镜像下载
HF_ENDPOINT=https://hf-mirror.com python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('z-lab/Qwen3-8B-DFlash-b16',
                  local_dir='~/models/intel/Qwen3-8B-DFlash-b16')
"
```

> **注意：** z-lab/Qwen3-8B-DFlash-b16 是一个自定义架构的草稿模型，其输入接口（`noise_embedding`, `target_hidden`）与 GenAI CB 管线的标准 `input_ids` 接口不兼容。当前实现使用目标模型副本作为草稿模型，通过标准推测解码路径运行。完整的原生 DFlash 草稿模型集成需要模型运行器（model runner）的重大架构改动。

### 8.2 运行命令

```bash
source ~/setup_dev_env.sh
cd /home/ywang2/openvino.genai/build

# DFlash 推测解码
LD_LIBRARY_PATH=$(pwd)/openvino_genai:$LD_LIBRARY_PATH \
  ./bin/dflash_speculative_decoding \
  ~/models/intel/Qwen3-8B \
  ~/models/intel/Qwen3-8B-dflash-draft \
  "What is the capital of France?" \
  CPU

# Greedy baseline（对比）
LD_LIBRARY_PATH=$(pwd)/openvino_genai:$LD_LIBRARY_PATH \
  ./bin/greedy_causal_lm \
  ~/models/intel/Qwen3-8B \
  "What is the capital of France?"
```

### 8.3 实验结果

| 指标 | DFlash 推测解码 | 说明 |
|------|----------------|------|
| 总生成时间 | 1459.17 ms | 目标模型生成时间 |
| 首 token 时间 (TTFT) | 570.081 ms | |
| 每 token 时间 (TPOT) | 8.97 ms/iteration | |
| 生成 token 数 | 100 tokens | |
| 接受 token 数 | 93/100 (93%) | 高接受率 |
| 草稿模型生成时间 | 7394.68 ms | |

**输出一致性：** DFlash 输出与 greedy baseline 输出一致，均正确回答"巴黎是法国首都"。

### 8.4 结果分析

- **高接受率（93%）：** 由于当前使用目标模型自身作为草稿模型，草稿质量极高，几乎所有草稿 token 都被接受
- **草稿模型开销较大：** 使用完整 Qwen3-8B 作为草稿模型导致草稿生成时间（7.4s）远超目标模型验证时间（1.5s）
- **真正的 DFlash 草稿模型（5层）预期可显著降低草稿生成开销**，但需要 GenAI 管线架构适配其自定义输入接口

---

## 9. DFlash 工作原理详解

### 9.1 核心思想

DFlash（Block Diffusion Speculative Decoding）是一种基于扩散模型思想的推测解码方法。与传统的自回归推测解码（如 EAGLE3）不同，DFlash 通过**单次前向传播**同时生成整个 block 的草稿 token。

**关键区别：**

| 特性 | 传统推测解码（EAGLE3） | DFlash |
|------|----------------------|--------|
| 草稿生成方式 | 自回归，逐 token 生成 | 单次前向传播，一次生成整个 block |
| 前向传播次数 | block_size 次 | 1 次 |
| 注意力机制 | 因果注意力（causal） | 双向注意力（bidirectional） |
| 输入 | input_ids | noise_embedding + target_hidden |
| KV-cache | 需要管理 | 不需要（无自回归） |

### 9.2 工作流程（以 block_size=4 为例）

#### 步骤 1：目标模型前向传播

输入 prompt: `"The capital of France is"`

目标模型在生成每个 token 时，会在指定层（如 layer 1, 9, 17, 25, 33）输出隐藏状态（hidden states）。

```
Target model forward pass:
  Layer 1  hidden: h1 ∈ R^(seq_len × 4096)
  Layer 9  hidden: h9 ∈ R^(seq_len × 4096)
  Layer 17 hidden: h17 ∈ R^(seq_len × 4096)
  Layer 25 hidden: h25 ∈ R^(seq_len × 4096)
  Layer 33 hidden: h33 ∈ R^(seq_len × 4096)
```

#### 步骤 2：隐藏状态拼接与 FC 投影

将 5 层隐藏状态在特征维度拼接，通过 FC 层投影回模型维度：

```
target_hidden = FC(concat(h1, h9, h17, h25, h33))
             = FC(R^(seq_len × 20480))
             → R^(seq_len × 4096)
```

#### 步骤 3：噪声嵌入准备

使用 mask token 的 embedding 作为初始噪声：

```
mask_token_id = 151669
noise_embedding = embed_tokens(mask_token_id).repeat(block_size)
                → R^(4 × 4096)  # 4 个位置的噪声嵌入
```

#### 步骤 4：DFlash 草稿模型前向传播

草稿模型接收 `noise_embedding` 和 `target_hidden`，通过**双向注意力**一次性生成所有草稿 token 的 logits：

```
DFlash draft model forward:
  Input:
    noise_embedding: R^(4 × 4096)    # 4 个草稿位置
    target_hidden:   R^(4 × 4096)    # 最近 4 个上下文位置
  
  Attention (non-causal):
    Q = q_proj(noise_embedding)       # 仅 noise 参与 query
    K = concat(k_proj(target_hidden), k_proj(noise_embedding))  # 上下文 + noise
    V = concat(v_proj(target_hidden), v_proj(noise_embedding))
    # 每个草稿 token 可以看到所有上下文和所有其他草稿 token
  
  Output: logits ∈ R^(4 × vocab_size)  # 4 个草稿 token 的概率分布
```

#### 步骤 5：采样与验证

从 logits 中采样得到草稿 token：
```
draft_tokens = [argmax(logits[0]), argmax(logits[1]), argmax(logits[2]), argmax(logits[3])]
             = ["Paris", ",", "known", "for"]
```

目标模型验证这些草稿 token：
```
Target model verify:
  Input: original_prompt + draft_tokens
  Accept: "Paris" ✓, "," ✓, "known" ✓, "for" ✗ (target says "a")
  
  Result: 接受 3/4 个草稿 token，从 "for" 位置重新开始
```

### 9.3 对应代码逻辑

#### 管线创建（pipeline.cpp）

```cpp
// 工厂模式：检测 dflash_mode 并创建 DFlash 管线
if (is_dflash_enabled(properties)) {
    return std::make_shared<DFlashDecodingImpl>(...);
}
```
对应文件：[src/cpp/src/continuous_batching/pipeline.cpp](../src/cpp/src/continuous_batching/pipeline.cpp)

#### DFlash 策略（dflash_strategy.cpp）

```cpp
// 构造函数：初始化并尝试 EAGLE3 风格的隐藏状态提取
DFlashDecodingImpl::DFlashDecodingImpl(...) {
    // 1. 从 config 提取 DFlash 参数
    auto dflash_info = extract_dflash_info_from_config(draft_properties);
    m_block_size = dflash_info.block_size;
    
    // 2. 对目标模型应用隐藏状态提取变换
    //    在 target_layer_ids 指定的层插入 hidden state 输出
    transform_hidden_state(main_model, target_layer_ids);
    
    // 3. 尝试从草稿模型移动 FC 层到主模型
    try {
        move_fc_from_draft_to_main();  // EAGLE3 兼容路径
    } catch (...) {
        // 非 EAGLE3 草稿模型，使用标准 SD 管线
    }
}
```
对应文件：[src/cpp/src/speculative_decoding/continuous_batching/dflash_strategy.cpp](../src/cpp/src/speculative_decoding/continuous_batching/dflash_strategy.cpp)

#### 隐藏状态提取（eagle3_model_transforms.cpp）

```cpp
// 在目标模型的指定层插入 Result 节点以导出隐藏状态
void transform_hidden_state(Model& model, vector<int32_t>& layers) {
    // 对 layers=[1,9,17,25,33] 中的每一层：
    //   找到 "layers.{idx}/" 对应的 Add 节点
    //   插入额外的 Result 节点导出该层的隐藏状态
}
```
对应文件：[src/cpp/src/speculative_decoding/eagle3_model_transforms.cpp](../src/cpp/src/speculative_decoding/eagle3_model_transforms.cpp)

### 9.4 未来工作：原生 DFlash 草稿模型集成

当前实现使用目标模型副本作为草稿模型，走标准推测解码路径。要支持真正的 DFlash 草稿模型（如 z-lab/Qwen3-8B-DFlash-b16），需要：

1. **自定义模型运行器（Model Runner）：** 支持 `noise_embedding` 和 `target_hidden` 输入（替代 `input_ids`）
2. **嵌入层共享：** 草稿模型复用目标模型的 `embed_tokens` 和 `lm_head`
3. **非因果注意力支持：** 移除 causal mask，改为全注意力
4. **无 KV-cache 推理：** DFlash 草稿每次都是完整前向传播，不需要增量 KV-cache
5. **隐藏状态传递管道：** 目标模型 → FC 投影 → 草稿模型的 `target_hidden` 输入
