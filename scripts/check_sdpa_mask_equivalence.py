#!/usr/bin/env python
"""Compare SetSwitch eager attention and SDPA on the same custom 4D mask.

Use this before trusting a SetSwitch+SDPA run. Passing this check means the
current model/transformers/torch path appears to honor the dense SetSwitch mask
for this batch. It is still a validation step, not a mathematical guarantee for
all future library versions.
"""

from __future__ import annotations

import argparse
import copy

import torch

from set_switch.data.collator import SetSwitchCollator
from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchDocument, SetSwitchExample
from set_switch.modeling.load_model import load_tokenizer_and_model
from set_switch.training.train import (
    apply_interface_overrides,
    attention_mask_dtype_from_config,
)
from set_switch.utils.io import read_yaml
from set_switch.utils.seed import set_seed


def fixture_example() -> SetSwitchExample:
    return SetSwitchExample(
        example_id="sdpa-check",
        instruction=("Use the provided passages or options to answer the question."),
        question="What is the launch year of project NARU-17?",
        documents=[
            SetSwitchDocument("d0", "Project LOMA-42 has launch year 1986.", False),
            SetSwitchDocument("d1", "Project NARU-17 has launch year 2004.", True),
            SetSwitchDocument("d2", "Project VELA-09 has launch year 1972.", False),
        ],
        answer="2004",
        source="fixture",
        metadata={"set_type": "documents"},
    )


def _model_cfg(cfg: dict, attn_implementation: str) -> dict:
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["attn_implementation"] = attn_implementation
    return model_cfg


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "position_ids", "labels"}
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flashrag.yaml")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    cfg["model_interface"] = "setswitch"
    cfg = apply_interface_overrides(cfg)
    set_seed(int(cfg.get("seed", 42)))

    eager_cfg = _model_cfg(cfg, "eager")
    sdpa_cfg = _model_cfg(cfg, "sdpa")
    tokenizer, eager_model = load_tokenizer_and_model(eager_cfg, add_setswitch_tokens=True)
    _, sdpa_model = load_tokenizer_and_model(sdpa_cfg, add_setswitch_tokens=True)
    sdpa_model.load_state_dict(eager_model.state_dict(), strict=True)

    mask_dtype = attention_mask_dtype_from_config(eager_cfg, cfg.get("train", {}), eager_model)
    rendered = render_example(fixture_example(), tokenizer, {"data": cfg.get("data", {})})
    batch = SetSwitchCollator(
        tokenizer=tokenizer,
        attention_mode=cfg.get("mask", {}).get("doc_attention", "doc_causal"),
        mask_dtype=mask_dtype,
    )([rendered])

    device = torch.device(args.device)
    eager_model.to(device).eval()
    sdpa_model.to(device).eval()
    model_batch = _move_batch(batch, device)
    model_batch["use_cache"] = False

    eager_logits = eager_model(**model_batch).logits.float().cpu()
    sdpa_logits = sdpa_model(**model_batch).logits.float().cpu()
    diff = (eager_logits - sdpa_logits).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    print(f"attention_mode: {cfg.get('mask', {}).get('doc_attention', 'doc_causal')}")
    print(f"mask_dtype: {mask_dtype}")
    print(f"seq_len: {model_batch['input_ids'].shape[1]}")
    print(f"max_abs_diff: {max_abs:.6f}")
    print(f"mean_abs_diff: {mean_abs:.6f}")
    if max_abs > args.threshold:
        raise SystemExit(
            "SDPA differs from eager above threshold; do not trust SetSwitch+SDPA for this run."
        )
    print("SDPA and eager are close on this SetSwitch masked batch.")


if __name__ == "__main__":
    main()
