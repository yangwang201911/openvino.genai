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
"""

import argparse
import sys

import openvino_genai as ov_genai


def main():
    parser = argparse.ArgumentParser(description="xAttention smoke test for Qwen3-MOE")
    parser.add_argument("model_dir", help="Path to the converted Qwen3-MOE model directory")
    parser.add_argument("device", nargs="?", default="CPU", help="Device (default: CPU)")
    args = parser.parse_args()

    prompt = "What is the capital of France?"
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 50

    # --- Baseline (no xAttention) ---
    print("=" * 60)
    print("[1/2] Baseline — no xAttention")
    print("=" * 60)
    pipe_base = ov_genai.LLMPipeline(args.model_dir, args.device)
    result_base = pipe_base.generate(prompt, config)
    text_base = result_base.texts[0]
    print(f"Output: {text_base}")
    del pipe_base

    # --- With xAttention ---
    print()
    print("=" * 60)
    print("[2/2] With xAttention (XATTENTION mode)")
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

    pipe_xa = ov_genai.LLMPipeline(args.model_dir, args.device, scheduler_config=scheduler_config)
    result_xa = pipe_xa.generate(prompt, config)
    text_xa = result_xa.texts[0]
    print(f"Output: {text_xa}")

    # --- Compare ---
    print()
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    match = text_base == text_xa
    print(f"Baseline : {text_base[:120]}")
    print(f"XAttn    : {text_xa[:120]}")
    print(f"Match    : {match}")

    if not match:
        print("\nWARNING: outputs differ on short prompt — investigate further.")
        sys.exit(1)
    else:
        print("\nPASS: short-prompt outputs match.")


if __name__ == "__main__":
    main()
