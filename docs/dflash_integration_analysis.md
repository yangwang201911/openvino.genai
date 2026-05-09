# DFlash 真正草稿模型集成分析

## 1. 当前实现的问题

### 问题总结

当前 DFlash 管线实现存在以下根本性问题，导致 **并非真正的 DFlash 推测解码**：

| 问题 | 详情 | 表现 |
|------|------|------|
| ❌ 草稿模型是目标模型的拷贝 | `prepare_dflash_draft_model.py` 只是复制了完整的 Qwen3-8B 并添加 rt_info | 两个 8B 模型跑 SD，性能比不用 SD 还差 |
| ❌ Fallback 到标准 SD | `move_fc_from_draft_to_main()` 失败 → 退化为 `ContinuousBatchingForSpeculativeDecodingImpl` | 无 hidden state 传递，非 DFlash |
| ❌ 真正 DFlash 模型缺少 `input_ids` | 导出的 OV IR 输入是 `noise_embedding` + `target_hidden`，无 `input_ids` | model_runner 无法设置输入 |
| ❌ 无 embed_tokens 和 lm_head | DFlash 草稿模型借用目标模型的嵌入层和输出头 | 无法独立产生 logits |
| ❌ 导出时无 KV-cache | 当前导出的 OV IR 是无状态的（0 个 state variables） | SDPAToPagedAttention 无法运行 |
| ❌ 非因果注意力 | DFlash 使用双向注意力（is_causal=False） | SDPAToPagedAttention 假设因果掩码 |

### 详细对比

```
=== 已导出的 DFlash 草稿模型 (Qwen3-8B-DFlash-b16-ov) ===
Parameters: 3 (position_ids, noise_embedding, target_hidden)
Results: 1 (unnamed [?,?,4096])  ← 原始隐藏状态，非 logits
State variables: 0              ← 无 KV-cache
SDPA nodes: 0                   ← 导出时用 eager attention

=== GenAI CB 管线要求的模型接口 ===
Parameters: input_ids [?], position_ids [?], attention_mask [?], beam_idx [?]
Results: logits [?,1,vocab_size]
State variables: 2*num_layers   ← 用于 SDPAToPagedAttention 转换
SDPA nodes: num_layers           ← SDPAToPagedAttention 需要
```

### DFlash 注意力机制（核心差异）

```python
# DFlash 注意力 (非因果，双向)
Q = q_proj(hidden_states)                              # noise only [B, q_len, ...]
K = concat(k_proj(target_hidden), k_proj(hidden_states))  # ctx + noise [B, ctx_len+q_len, ...]
V = concat(v_proj(target_hidden), v_proj(hidden_states))  # ctx + noise [B, ctx_len+q_len, ...]
# 每个 noise token 可以看到 所有 context + 所有 noise tokens

# 标准因果注意力
Q = q_proj(hidden_states)
K = k_proj(hidden_states)
V = v_proj(hidden_states)
# 每个 token 只能看到自身和之前的 tokens
```

### FC 层位置差异

```
EAGLE3 架构:
  main model outputs → concat hidden states → [FC in main] → hidden_states → draft model
  
DFlash 架构:
  main model outputs → concat hidden states (20480-dim) → [FC in draft] → 4096-dim → decoder layers
  FC 是草稿模型的一部分，不在主模型中
```

在 DFlash OV IR 中确认了 FC 层：
```
MatMul: __module.model.fc/aten::linear/MatMul
  in0: target_hidden (Parameter) shape=[?,?,?]
  in1: self.model.fc.weight (Convert) shape=[4096,20480]
```

## 2. 解决方案分析

### 方案 A：重新导出 + GenAI 自定义草稿运行器（不修改 OV Runtime）

**核心思路：** 重新导出 DFlash 模型使其具有标准接口，但草稿模型不使用 PagedAttention。

**步骤：**

1. **重新导出 DFlash 草稿模型**：
   - 将目标模型的 `embed_tokens` 权重嵌入草稿模型（input_ids → embed_tokens → noise_embedding）
   - 将目标模型的 `lm_head` 权重嵌入草稿模型（hidden → lm_head → logits）
   - 使用 `use_cache=True` 导出（生成 KV-cache 状态变量）
   - 重命名：`target_hidden` → `hidden_states`

2. **GenAI 侧修改**：
   - DFlash 草稿模型 **跳过 SDPAToPagedAttention**（保持原始 SDPA + stateful KV-cache）
   - 创建自定义 DFlash 模型运行器（DFlashModelRunner），直接管理 stateful InferRequest
   - 隐藏状态传递：主模型 `last_hidden_state` 输出 → 草稿模型 `hidden_states` 输入
   - FC 层保留在草稿模型内部（不需要 move_fc_from_draft_to_main）

3. **草稿模型运行方式**：
   ```
   主模型 (PagedAttention) → hidden states (20480-dim)
          ↓
   草稿模型 (标准 SDPA, stateful):
     input_ids → embed_tokens → noise_embedding
     hidden_states (20480-dim) → FC (in draft) → target_hidden (4096-dim)
     5层 decoder (non-causal SDPA) → lm_head → logits
   ```

**优点：**
- 不需要修改 OV Runtime
- 草稿模型仅 5 层，不使用 PA 影响可忽略
- 非因果注意力自然工作（SDPA 原生支持 is_causal=False）

**缺点：**
- 需要自定义模型运行器（不用 PagedAttention scheduler 管理草稿 KV-cache）
- 草稿模型的 KV-cache 管理需要手动处理（crop/reset 等）

**工作量：** ~3-4 个关键文件修改
- 重新导出脚本（Python）
- DFlashModelRunner（新增 C++ 类）
- dflash_strategy.cpp（适配新的模型运行器）

---

### 方案 B：重新导出 + 修改 OV Runtime PagedAttention（完整 PA 集成）

**核心思路：** 在 OV Runtime 的 PagedAttention op 和 SDPAToPagedAttention pass 中添加非因果注意力支持。

**步骤：**

1. **重新导出 DFlash 草稿模型**（同方案 A）

2. **修改 OV Runtime**：
   - `PagedAttention` op 添加 `is_causal` 属性（默认 true，向后兼容）
   - `SDPAToPagedAttention` pass 传递 SDPA 节点的 is_causal 标志
   - PA kernel 支持非因果注意力（移除因果掩码）

3. **修改 GenAI**：
   - SDPAToPagedAttention 正常运行在草稿模型上
   - 标准 model_runner 处理草稿模型
   - FC 保留在草稿模型内部

4. **草稿模型运行方式**：
   ```
   主模型 (PagedAttention, causal) → hidden states (20480-dim)
          ↓
   草稿模型 (PagedAttention, non-causal):
     input_ids → embed_tokens → noise_embedding
     hidden_states (20480-dim) → FC (in draft) → target_hidden (4096-dim)
     5层 decoder (non-causal PA) → lm_head → logits
   ```

**优点：**
- 完整的 PA 集成，复用已有的 scheduler 和 model_runner
- KV-cache 由 scheduler 统一管理（paged blocks）
- 代码改动集中在 OV Runtime 层，GenAI 改动最小

**缺点：**
- 需要修改 OV Runtime（影响面较大）
- PA kernel 需要支持两种掩码模式
- 5 层模型用 PA 收益有限

**工作量：** ~5-6 个关键文件修改
- 重新导出脚本（Python）
- OV: PagedAttention op 定义
- OV: SDPAToPagedAttention pass
- OV: PA kernel（CPU/GPU）
- GenAI: dflash_strategy.cpp

---

### 方案对比

| 维度 | 方案 A (GenAI-only) | 方案 B (OV Runtime) |
|------|---------------------|---------------------|
| OV Runtime 修改 | ❌ 无 | ✅ 需要 |
| GenAI 修改量 | 中等（自定义 runner） | 较小（复用已有 runner） |
| 非因果注意力 | SDPA 原生支持 | PA 需要新增支持 |
| KV-cache 管理 | 手动（stateful model） | 自动（scheduler） |
| 性能 | 草稿仅 5 层，差异可忽略 | 最优 |
| 风险 | 低 | 中（OV 改动影响面大） |
| 可复用性 | 仅限 DFlash | 其他非因果模型也可受益 |

## 3. 两个方案共同的前置步骤：重新导出 DFlash 草稿模型

无论选择哪个方案，都需要重新导出 DFlash 草稿模型：

```python
# 伪代码：重新导出 DFlash 草稿模型
import torch
from transformers import AutoModelForCausalLM

# 加载目标模型（获取 embed_tokens 和 lm_head）
target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")

# 加载 DFlash 草稿模型
draft = DFlashDraftModel.from_pretrained("z-lab/Qwen3-8B-DFlash-b16")

# 创建包装模型：添加 embed_tokens + lm_head
class DFlashWrapper(nn.Module):
    def __init__(self, draft_model, embed_tokens, lm_head):
        self.draft = draft_model
        self.embed_tokens = embed_tokens  # 从 target 复制
        self.lm_head = lm_head            # 从 target 复制
    
    def forward(self, input_ids, hidden_states, position_ids, past_key_values):
        noise_embedding = self.embed_tokens(input_ids)
        hidden = self.draft(
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,   # 来自主模型的 hidden states
            position_ids=position_ids,
            past_key_values=past_key_values,
            is_causal=False,
        )
        return self.lm_head(hidden)  # 输出 logits

# 导出为 OV IR（with KV-cache）
```
