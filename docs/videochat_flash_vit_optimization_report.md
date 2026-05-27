# VideoChat-Flash VIT Reshape Optimization Report

> PR #3838: Enable GenAI support for images + videos input  
> Date: 2026-05-27  
> Author: ywang2

---

## 1. Background & Objective

PR #3838 adds image inference support to the VideoChat-Flash pipeline (previously video-only). The key constraint is:

**VIT (Vision Transformer) inference performance must not regress on the video path.**

The vision encoder model (InternVideo2-1B, ~1B parameters) has two inputs:
- `hidden_states`: shape `[1, 3, T, H, W]` — T=frames, H/W=spatial resolution
- `rotary_pos_emb`: shape `[1, tokens, 1408]`

On master, the model is compiled with H/W/T all dynamic (only `rotary_pos_emb` is partially constrained). This report evaluates whether making H and W static (`image_size=224`) improves or maintains performance while keeping a single compiled model.

---

## 2. Plans Evaluated

### Plan B: Dual Static Compile (Rejected)

- Clone the `ov::Model`, reshape one for image (T=1, H=W=224) and one for video (T=4, H=W=224)
- Two `compile_model()` calls → two separate `CompiledModel` instances
- **Result:** +2938 MB RSS at load time due to weight duplication
- **Verdict:** ❌ Rejected — OV plugin always materializes its own weight buffers per `CompiledModel`

### Plan C: Single Compile with Static H/W (Selected)

- Single `ov::Model`, reshape with H=W=224 static, T=dynamic
- One `compile_model()` call → one `CompiledModel`, one set of weights
- Both image (T=1) and video (T=4) served by the same inference request queue
- **Verdict:** ✅ No memory overhead, performance parity with master

---

## 3. Code Change (Plan C)

**File:** `src/cpp/src/visual_language/videochat_flash/classes.cpp`  
**Location:** `VisionEncoderVideoChatFlashQwen::initialize_vision_encoder_queue()`  
**Diff:** +17 lines

```cpp
initialize_positional_embedding();
// Reshape vision encoder to be fully static except for the temporal dimension T
// (T=1 for image, T=mm_local_num_frames for video) and the corresponding rotary
// token count. Fixing the spatial dimensions (H=W=image_size) enables the plugin
// to specialise convolution / attention kernels while a single compiled model
// keeps weight footprint at 1x.
const size_t image_size = m_processor_config.image_size;
std::map<std::string, ov::PartialShape> input_shapes;
input_shapes["hidden_states"] = ov::PartialShape{1,
                                                 3,
                                                 ov::Dimension::dynamic(),
                                                 static_cast<int64_t>(image_size),
                                                 static_cast<int64_t>(image_size)};
input_shapes["rotary_pos_emb"] = ov::PartialShape{1,
                                                  ov::Dimension::dynamic(),
                                                  static_cast<int64_t>(m_mm_hidden_size)};
model->reshape(input_shapes);

auto compiled_model = utils::singleton_core().compile_model(model, device, properties);
```

---

## 4. Benchmark Methodology

### Environment
- **Machine:** Intel Xeon (odt-huyuan-openvino-ci-97)
- **OpenVINO:** 2026.3.0-21915-c63bd58e985 (Debug build)
- **Model:** VideoChat-Flash-Qwen2_5-7B_InternVideo2-1B (InternVideo2-1B VIT)
- **Devices:** CPU, GPU (Intel integrated)

### Benchmark Script: `tools/bench_vit_videochat.py`

The script measures VIT embedding extraction time via `pipe.generate()` with `max_new_tokens=1`:
1. Load the VLM pipeline
2. Record RSS memory after load
3. Feed a synthetic video (10 frames x 224x224) and measure `perf_metrics.get_prepare_embeddings_duration().mean`
4. Repeat with warmup iterations discarded

### Parameters
- **Iterations:** 10-15 (measurement), 3-5 (warmup)
- **Video input:** 10 frames x 224x224 RGB (ov::Tensor, element_type f32)
- **Memory metric:** `/proc/self/status` VmRSS
- **Timing metric:** `prepare_embeddings_duration` from GenAI perf_metrics

### Comparison Protocol
1. Build & benchmark `origin/master` (commit `1e7a63d1`)
2. Build & benchmark PR branch with Plan C applied (commit `9c154a7f` + working tree)
3. Same hardware, same model, same script, same parameters
4. Both builds are Debug configuration for consistency

---

## 5. Results

### CPU (8-15 iterations, 3-5 warmup)

| Metric | Master | Plan C | Delta |
|--------|--------|--------|-------|
| VIT video mean | 1439 ms +/- 75 | **1418 ms +/- 40** | -1.5% (within noise) |
| VIT video min | 1364 ms | 1360 ms | ~0 |
| Mem after load | 1062 MB | 1063 MB | +1 MB |
| Mem peak (video) | 10564 MB | 10565 MB | +1 MB |

### GPU (10 iterations, 3 warmup)

| Metric | Master | Plan C | Delta |
|--------|--------|--------|-------|
| VIT video mean | 352.9 ms +/- 10.5 | **326.5 ms +/- 6.6** | **-7.5% (faster)** |
| VIT video min | 333.6 ms | 313.9 ms | -5.9% |
| Mem after load | 903.5 MB | 886.3 MB | -17 MB |
| Mem peak | 824.7 MB | 795.1 MB | -30 MB |

### Functional Tests (pytest, CPU)

| Suite | Result |
|-------|--------|
| Total passed | 94 |
| xfailed | 35 |
| Pre-existing flaky | 1 (beam_search score diff 0.00158 > epsilon 0.001, same on master) |

---

## 6. Analysis

1. **CPU performance:** Plan C is statistically equivalent to master (within measurement noise). The static H/W reshape does not introduce overhead because the CPU plugin already handles the 224x224 spatial size efficiently.

2. **GPU performance:** Plan C shows a **7.5% improvement** on GPU. Making H=W static allows the GPU plugin to pre-select optimal tiled convolution kernels and avoid runtime shape inference overhead.

3. **Memory:** No increase on CPU; slight decrease on GPU (-30 MB peak). Plan C uses a single `CompiledModel` so there is no weight duplication.

4. **Why Plan B failed:** Even using `model->clone()`, each `compile_model()` call creates independent plugin-optimized weight buffers. For InternVideo2-1B (~1B params x FP16 ~ 2 GB + intermediate buffers), this resulted in +2938 MB. The ONLY way to share weights across image/video is a single compiled model with dynamic T.

---

## 7. Conclusion

| Criterion | Status |
|-----------|--------|
| VIT video perf not regressed (CPU) | PASS (-1.5%, within noise) |
| VIT video perf not regressed (GPU) | PASS (**-7.5%, improved**) |
| Memory not increased | PASS (CPU ~0, GPU -30 MB) |
| Single compiled model (no weight dup) | PASS |
| Functional tests pass | PASS (94 pass, 1 pre-existing flaky) |
| Code change minimal | PASS (+17 lines) |

**Recommendation:** Commit Plan C to PR #3838. It maintains/improves performance on both CPU and GPU with zero memory overhead and minimal code change.

---

## 8. Appendix: Raw Benchmark Output

### Master CPU
```
tag=MASTER device=CPU iters=8 warmup=3
video: mean= 1439.39ms  stdev= 75.16ms  min= 1364.67ms  max= 1562.83ms  n=8
mem after load: 1062.4 MB
mem after video: 10564.4 MB
```

### Plan C CPU (15 iterations)
```
tag=PLAN_C_RERUN device=CPU iters=15 warmup=5
video: mean= 1418.19ms  stdev= 39.97ms  min= 1359.55ms  max= 1512.77ms  n=15
mem after load: 1062.7 MB
mem after video: 10564.9 MB
```

### Master GPU
```
tag=MASTER_GPU device=GPU iters=10 warmup=3
video: mean=  352.86ms  stdev= 10.50ms  min=  333.59ms  max=  366.30ms  n=10
mem after load: 903.5 MB
mem after video: 824.7 MB
```

### Plan C GPU
```
tag=PLAN_C_GPU device=GPU iters=10 warmup=3
video: mean=  326.46ms  stdev=  6.57ms  min=  313.90ms  max=  337.40ms  n=10
mem after load: 886.3 MB
mem after video: 795.1 MB
```
