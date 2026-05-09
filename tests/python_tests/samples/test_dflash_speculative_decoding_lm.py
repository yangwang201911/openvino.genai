# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import json
import shutil
import sys
import pytest

from pathlib import Path
from conftest import SAMPLES_CPP_DIR, convert_model
from test_utils import run_sample

convert_draft_model = convert_model


def _prepare_dflash_draft(source_model_dir: str, block_size: int = 4):
    """
    Prepare a DFlash draft model by copying source model and adding dflash_mode rt_info.
    Returns the path to the prepared draft model directory.
    """
    import openvino as ov

    draft_dir = Path(source_model_dir).parent / (Path(source_model_dir).name + "_dflash_draft")
    if draft_dir.exists():
        return str(draft_dir)

    draft_dir.mkdir(parents=True, exist_ok=True)
    for item in Path(source_model_dir).iterdir():
        dst = draft_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    core = ov.Core()
    model = core.read_model(str(Path(source_model_dir) / "openvino_model.xml"))

    config_json = Path(source_model_dir) / "config.json"
    target_layer_ids = [0, 1, 2]
    if config_json.exists():
        with open(config_json) as f:
            config = json.load(f)
        n_layers = config.get("num_hidden_layers", 4)
        if n_layers >= 10:
            target_layer_ids = [2, n_layers // 2, n_layers - 3]
        elif n_layers >= 4:
            target_layer_ids = [0, n_layers // 2, n_layers - 1]

    model.set_rt_info(True, "dflash_mode")
    model.set_rt_info(block_size, "dflash_block_size")
    model.set_rt_info(-1, "dflash_mask_token_id")
    model.set_rt_info(target_layer_ids, "hidden_layers_list")

    ov.save_model(model, str(draft_dir / "openvino_model.xml"))

    out_config = draft_dir / "config.json"
    if out_config.exists():
        with open(out_config) as f:
            cfg = json.load(f)
    else:
        cfg = {}
    cfg["dflash_config"] = {
        "target_layer_ids": target_layer_ids,
        "block_size": block_size,
        "mask_token_id": -1,
    }
    with open(out_config, "w") as f:
        json.dump(cfg, f, indent=2)

    return str(draft_dir)


class TestDFlashSpeculativeDecodingLM:
    @pytest.mark.dflash_decoding
    @pytest.mark.parametrize(
        "convert_model, sample_args",
        [
            pytest.param("SmolLM2-135M", "Alan Turing was a"),
        ],
        indirect=["convert_model"],
    )
    def test_dflash_speculative_decoding_lm(self, convert_model, sample_args):
        """
        Test DFlash speculative decoding pipeline:
        1. Converts SmolLM2-135M as target model
        2. Creates a DFlash draft model copy with dflash_mode rt_info
        3. Runs C++ dflash_speculative_decoding sample
        4. Verifies the pipeline produces non-empty output matching greedy baseline
        """
        if sys.platform == "darwin":
            pytest.xfail("Ticket 173586")

        env = os.environ.copy()
        env["OPENVINO_LOG_LEVEL"] = "0"

        draft_model_path = _prepare_dflash_draft(convert_model, block_size=4)

        # Run C++ DFlash sample
        cpp_sample = SAMPLES_CPP_DIR / "dflash_speculative_decoding"
        cpp_command = [cpp_sample, convert_model, draft_model_path, sample_args]
        cpp_result = run_sample(cpp_command, env=env)

        assert cpp_result.returncode == 0, (
            f"DFlash sample failed with return code {cpp_result.returncode}: {cpp_result.stderr}"
        )
        assert len(cpp_result.stdout.strip()) > 0, "DFlash sample produced no output"

        # Compare with greedy causal LM baseline
        cpp_sample_ref = SAMPLES_CPP_DIR / "greedy_causal_lm"
        cpp_command_ref = [cpp_sample_ref, convert_model, sample_args]
        cpp_result_ref = run_sample(cpp_command_ref, env=env)

        assert cpp_result_ref.stdout.strip() in cpp_result.stdout.strip(), (
            "DFlash speculative decoding output should match greedy decoding output"
        )
