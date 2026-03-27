#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Interactive chat with xAttention enabled on Qwen3-MOE.
Based on chat_sample.py but with SchedulerConfig for xAttention.

Usage:
    python chat_xattention_qwen3_moe.py <model_dir> [device]
"""

import argparse
import openvino_genai as ov_genai


def streamer(subword):
    print(subword, end="", flush=True)
    return ov_genai.StreamingStatus.RUNNING


def main():
    parser = argparse.ArgumentParser(description="Chat with Qwen3-MOE + xAttention")
    parser.add_argument("model_dir", help="Path to the converted Qwen3-MOE model directory")
    parser.add_argument("device", nargs="?", default="CPU", help="Device (default: CPU)")
    parser.add_argument("--no-xattention", action="store_true", help="Disable xAttention (baseline)")
    args = parser.parse_args()

    if args.no_xattention:
        print("Mode: baseline (no xAttention)")
        pipe = ov_genai.LLMPipeline(args.model_dir, args.device)
    else:
        print("Mode: xAttention enabled")
        scheduler_config = ov_genai.SchedulerConfig()
        scheduler_config.use_sparse_attention = True
        scheduler_config.sparse_attention_config = ov_genai.SparseAttentionConfig(
            mode=ov_genai.SparseAttentionMode.XATTENTION,
            xattention_threshold=0.8,
            xattention_block_size=64,
            xattention_stride=8,
            num_last_dense_tokens_in_prefill=100,
        )
        pipe = ov_genai.LLMPipeline(args.model_dir, args.device, scheduler_config=scheduler_config)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 200

    chat_history = ov_genai.ChatHistory()
    print("Type your message (Ctrl+D or Ctrl+Z to quit):\n")
    while True:
        try:
            prompt = input("question:\n")
        except EOFError:
            break
        chat_history.append({"role": "user", "content": prompt})
        decoded_results = pipe.generate(chat_history, config, streamer)
        chat_history.append({"role": "assistant", "content": decoded_results.texts[0]})
        print("\n----------")


if __name__ == "__main__":
    main()
