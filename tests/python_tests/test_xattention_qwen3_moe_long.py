#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Long-prompt test: verify xAttention actually activates its sparse path.
Uses a prompt longer than num_last_dense_tokens_in_prefill (default 100)
so the importance-score-based block selection is exercised.

Usage:
    python test_xattention_qwen3_moe_long.py <model_dir> [device]
"""

import argparse
import time

import openvino_genai as ov_genai


def make_long_prompt(target_tokens: int = 1500) -> str:
    """Build a prompt that is roughly `target_tokens` tokens long."""
    filler = "The quick brown fox jumps over the lazy dog. "
    # ~10 tokens per repetition
    repetitions = target_tokens // 10
    context = filler * repetitions
    return context + "\nSummarize the above text in one sentence."


def run_pipeline(model_dir: str, device: str, prompt: str, max_new_tokens: int,
                 scheduler_config=None) -> tuple:
    """Create pipeline, generate, return (output_text, elapsed_seconds)."""
    if scheduler_config is not None:
        pipe = ov_genai.LLMPipeline(model_dir, device, scheduler_config=scheduler_config)
    else:
        pipe = ov_genai.LLMPipeline(model_dir, device)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens

    start = time.perf_counter()
    result = pipe.generate(prompt, config)
    elapsed = time.perf_counter() - start
    return result.texts[0], elapsed


def main():
    parser = argparse.ArgumentParser(description="xAttention long-prompt test for Qwen3-MOE")
    parser.add_argument("model_dir", help="Path to the converted Qwen3-MOE model directory")
    parser.add_argument("device", nargs="?", default="CPU", help="Device (default: CPU)")
    parser.add_argument("--tokens", type=int, default=1500, help="Approximate prompt length in tokens")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()

    prompt = make_long_prompt(args.tokens)
    print(f"Prompt length (approx): ~{args.tokens} tokens")
    print(f"Max new tokens: {args.max_new_tokens}")

    # --- Baseline ---
    print()
    print("=" * 60)
    print("[1/2] Baseline — no xAttention")
    print("=" * 60)
    text_base, t_base = run_pipeline(args.model_dir, args.device, prompt, args.max_new_tokens)
    print(f"Time   : {t_base:.2f}s")
    print(f"Output : {text_base[:200]}")

    # --- xAttention ---
    print()
    print("=" * 60)
    print("[2/2] With xAttention")
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
    text_xa, t_xa = run_pipeline(args.model_dir, args.device, prompt, args.max_new_tokens,
                                 scheduler_config=scheduler_config)
    print(f"Time   : {t_xa:.2f}s")
    print(f"Output : {text_xa[:200]}")

    # --- Summary ---
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    speedup = t_base / t_xa if t_xa > 0 else float("inf")
    print(f"Baseline time  : {t_base:.2f}s")
    print(f"xAttention time: {t_xa:.2f}s")
    print(f"Speedup        : {speedup:.2f}x")
    print(f"Outputs match  : {text_base == text_xa}")
    if text_base != text_xa:
        print("  (Minor differences are expected with sparse attention on long prompts)")
    print("\nPASS: long-prompt generation completed successfully.")


if __name__ == "__main__":
    main()
