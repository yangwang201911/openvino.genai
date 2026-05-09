#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Prepare a DFlash draft model from an existing OpenVINO model.

This script takes a standard OpenVINO model (exported via optimum-cli) and adds
dflash_mode runtime info to make it recognizable as a DFlash draft model by the
GenAI pipeline factory.

Usage:
  python prepare_dflash_draft_model.py \
    --source-model-dir <path_to_ov_model> \
    --output-dir <path_to_dflash_draft_model> \
    [--block-size 16] \
    [--mask-token-id -1] \
    [--target-layer-ids 2 12 22]

Example:
  # First export a model with optimum-cli:
  optimum-cli export openvino --model HuggingFaceTB/SmolLM2-135M \
      --task text-generation-with-past --trust-remote-code ./smollm2-135m-ov

  # Then prepare both target and draft from the same model (for testing):
  python prepare_dflash_draft_model.py \
      --source-model-dir ./smollm2-135m-ov \
      --output-dir ./smollm2-135m-dflash-draft
"""

import argparse
import json
import shutil
from pathlib import Path

import openvino as ov


def prepare_dflash_draft(source_dir: Path, output_dir: Path,
                         block_size: int = 16,
                         mask_token_id: int = -1,
                         target_layer_ids: list = None):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    model_xml = source_dir / "openvino_model.xml"
    if not model_xml.exists():
        raise FileNotFoundError(f"No openvino_model.xml found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy all files from source to output
    for item in source_dir.iterdir():
        dst = output_dir / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    # Load the model and add DFlash rt_info
    core = ov.Core()
    model = core.read_model(str(model_xml))

    model.set_rt_info(True, "dflash_mode")
    model.set_rt_info(block_size, "dflash_block_size")
    model.set_rt_info(mask_token_id, "dflash_mask_token_id")

    if target_layer_ids is None:
        # Read num_hidden_layers from config.json for auto selection
        config_json = source_dir / "config.json"
        if config_json.exists():
            with open(config_json) as f:
                config = json.load(f)
            n_layers = config.get("num_hidden_layers", 4)
            if n_layers >= 10:
                target_layer_ids = [2, n_layers // 2, n_layers - 3]
            else:
                target_layer_ids = [0, n_layers // 2, n_layers - 1]
        else:
            target_layer_ids = [0, 1, 2]

    model.set_rt_info(target_layer_ids, "hidden_layers_list")

    # Save the modified model
    ov.save_model(model, str(output_dir / "openvino_model.xml"))

    # Update config.json with dflash_config section
    config_json = output_dir / "config.json"
    if config_json.exists():
        with open(config_json) as f:
            config = json.load(f)
    else:
        config = {}

    config["dflash_config"] = {
        "target_layer_ids": target_layer_ids,
        "block_size": block_size,
        "mask_token_id": mask_token_id,
    }

    with open(config_json, "w") as f:
        json.dump(config, f, indent=2)

    print(f"DFlash draft model prepared at: {output_dir}")
    print(f"  block_size: {block_size}")
    print(f"  mask_token_id: {mask_token_id}")
    print(f"  target_layer_ids: {target_layer_ids}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a DFlash draft model")
    parser.add_argument("--source-model-dir", type=str, required=True,
                        help="Path to an existing OpenVINO model directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for the DFlash draft model")
    parser.add_argument("--block-size", type=int, default=16,
                        help="DFlash block size (default: 16)")
    parser.add_argument("--mask-token-id", type=int, default=-1,
                        help="Mask token ID (default: -1)")
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=None,
                        help="Target layer IDs for hidden state extraction")
    args = parser.parse_args()

    prepare_dflash_draft(
        source_dir=args.source_model_dir,
        output_dir=args.output_dir,
        block_size=args.block_size,
        mask_token_id=args.mask_token_id,
        target_layer_ids=args.target_layer_ids,
    )
