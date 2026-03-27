#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test: verify xAttention can be enabled on Qwen3-MOE without errors.
Compares short-prompt output with and without xAttention — should be identical
since xAttention sparse path only activates when prompt exceeds
num_last_dense_tokens_in_prefill.

Usage:
    python test_xattention_qwen3_moe_smoke.py <model_dir> [device]
    python test_xattention_qwen3_moe_smoke.py <model_dir> GPU.0 --xattention-only
"""

import argparse
import sys
import time

import openvino_genai as ov_genai


def timed_print(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="xAttention smoke test for Qwen3-MOE")
    parser.add_argument("model_dir", help="Path to the converted Qwen3-MOE model directory")
    parser.add_argument("device", nargs="?", default="CPU", help="Device (default: CPU)")
    parser.add_argument("--xattention-only", action="store_true",
                        help="Skip baseline, only test xAttention (useful for large models on GPU)")
    args = parser.parse_args()

    prompt = "What is the capital of France?"
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 50

    # On GPU, xAttention does not support per-channel quantized key cache.
    # Disable KV cache quantization to avoid the conflict.
    device_config = {}
    if "GPU" in args.device.upper():
        device_config["KV_CACHE_PRECISION"] = "f16"
        timed_print("GPU detected: setting KV_CACHE_PRECISION=f16 (xAttention requires non-quantized key cache)")

    text_base = None

    # --- Baseline (no xAttention) ---
    if not args.xattention_only:
        print("=" * 60)
        timed_print("[1/2] Baseline — no xAttention")
        print("=" * 60)
        timed_print("Creating LLMPipeline (baseline) ...")
        pipe_base = ov_genai.LLMPipeline(args.model_dir, args.device, **device_config)
        timed_print("Pipeline created. Generating ...")
        result_base = pipe_base.generate(prompt, config)
        text_base = result_base.texts[0]
        timed_print(f"Output: {text_base}")
        del pipe_base
        timed_print("Baseline pipeline released.")
    else:
        timed_print("Skipping baseline (--xattention-only)")

    # --- With xAttention ---
    print()
    print("=" * 60)
    step = "[1/1]" if args.xattention_only else "[2/2]"
    timed_print(f"{step} With xAttention (XATTENTION mode)")
    print("=" * 60)
    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.use_sparse_attention = True
    scheduler_config.sparse_attention_config = ov_genai.SparseAttentionConfig(
        mode=ov_genai.SparseAttentionMode.XATTENTION,
        xattention_threshold=0.8,
        xattention_block_size=64,
        xattention_stride=8,
        num_last_dense_tokens_in_prefill=100,
    )

    timed_print("Creating LLMPipeline (xAttention) ...")
    pipe_xa = ov_genai.LLMPipeline(args.model_dir, args.device, scheduler_config=scheduler_config, **device_config)
    timed_print("Pipeline created. Generating ...")
    result_xa = pipe_xa.generate(prompt, config)
    text_xa = result_xa.texts[0]
    timed_print(f"Output: {text_xa}")

    # --- Compare ---
    print()
    print("=" * 60)
    timed_print("Result")
    print("=" * 60)
    if text_base is not None:
        match = text_base == text_xa
        print(f"Baseline : {text_base[:120]}")
        print(f"XAttn    : {text_xa[:120]}")
        print(f"Match    : {match}")
        if not match:
            print("\nWARNING: outputs differ on short prompt — investigate further.")
            sys.exit(1)
        else:
            print("\nPASS: short-prompt outputs match.")
    else:
        print(f"XAttn output: {text_xa[:200]}")
        print("\nPASS: xAttention pipeline created and generated successfully.")


if __name__ == "__main__":
    main()
