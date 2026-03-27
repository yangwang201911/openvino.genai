#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
ContinuousBatching test: verify xAttention works with multiple concurrent
requests through the CB pipeline on Qwen3-MOE.

Usage:
    python test_xattention_qwen3_moe_cb.py <model_dir> [device]
"""

import argparse
import sys

import openvino_genai as ov_genai


def main():
    parser = argparse.ArgumentParser(description="xAttention CB test for Qwen3-MOE")
    parser.add_argument("model_dir", help="Path to the converted Qwen3-MOE model directory")
    parser.add_argument("device", nargs="?", default="CPU", help="Device (default: CPU)")
    args = parser.parse_args()

    scheduler_config = ov_genai.SchedulerConfig()
    scheduler_config.use_sparse_attention = True
    scheduler_config.dynamic_split_fuse = True
    scheduler_config.sparse_attention_config = ov_genai.SparseAttentionConfig(
        mode=ov_genai.SparseAttentionMode.XATTENTION,
        xattention_threshold=0.8,
        xattention_block_size=64,
        xattention_stride=8,
    )

    # On GPU, xAttention does not support per-channel quantized key cache.
    device_config = {}
    if "GPU" in args.device.upper():
        device_config["KV_CACHE_PRECISION"] = "f16"
        print("GPU detected: setting KV_CACHE_PRECISION=f16")

    print("Creating ContinuousBatchingPipeline with xAttention ...")
    pipe = ov_genai.ContinuousBatchingPipeline(args.model_dir, scheduler_config, args.device, {}, device_config)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 30

    prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about winter.",
    ]

    print(f"Generating {len(prompts)} requests in a batch ...\n")
    results = pipe.generate(prompts, [config] * len(prompts))

    all_ok = True
    for i, result in enumerate(results):
        status = result.m_status
        text = result.m_generation_ids[0] if result.m_generation_ids else "<empty>"
        ok = status == ov_genai.GenerationStatus.FINISHED
        print(f"[{i}] Status: {status}  OK: {ok}")
        print(f"    Output: {text[:120]}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("PASS: all requests finished successfully with xAttention + CB.")
    else:
        print("FAIL: some requests did not finish.")
        sys.exit(1)


if __name__ == "__main__":
    main()
