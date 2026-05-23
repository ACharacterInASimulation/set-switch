#!/usr/bin/env python
"""Compare eager attention and SDPA on custom 4D masks.

Use this before trusting a custom-mask SDPA run for a new model/library stack.
The check loads the same weights into eager and SDPA models, runs one masked
batch, and reports logit differences.
"""

from __future__ import annotations

import argparse
import copy

import torch

from set_switch.data.collator import SetSwitchCollator
from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchDocument, SetSwitchExample
from set_switch.data.setfuse_collator import SetFuseCollator
from set_switch.data.setfuse_render import render_setfuse_example
from set_switch.data.setllm_collator import SetLLMCollator
from set_switch.data.setllm_render import render_setllm_example
from set_switch.modeling.layer_masking import (
    install_setfuse_layerwise_attention,
    resolve_fuse_start_layer,
    set_setfuse_masks,
)
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
        instruction="Use the provided passages or options to answer the question.",
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


def _move_tensors(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def _render_and_collate(interface: str, tokenizer, cfg: dict, mask_dtype: torch.dtype) -> dict:
    render_cfg = {"data": cfg.get("data", {})}
    example = fixture_example()
    if interface == "setswitch":
        rendered = render_example(example, tokenizer, render_cfg)
        return SetSwitchCollator(
            tokenizer=tokenizer,
            attention_mode=cfg.get("mask", {}).get("doc_attention", "doc_causal"),
            answer_attends_raw_docs=bool(cfg.get("mask", {}).get("answer_attends_raw_docs", False)),
            answer_attends_reads=bool(cfg.get("mask", {}).get("answer_attends_reads", False)),
            mask_dtype=mask_dtype,
        )([rendered])
    if interface == "setllm":
        rendered = render_setllm_example(example, tokenizer, render_cfg)
        return SetLLMCollator(tokenizer=tokenizer, mask_dtype=mask_dtype)([rendered])
    if interface == "setfuse":
        rendered = render_setfuse_example(example, tokenizer, render_cfg)
        return SetFuseCollator(
            tokenizer=tokenizer,
            mask_dtype=mask_dtype,
            setfuse_answer_attends_docs_in_early_layers=bool(
                cfg.get("mask", {}).get("setfuse_answer_attends_docs_in_early_layers", False)
            ),
            setfuse_late_prefix_doc_bidir=bool(
                cfg.get("mask", {}).get("setfuse_late_prefix_doc_bidir", True)
            ),
        )([rendered])
    raise ValueError("SDPA mask parity is only relevant for setswitch, setllm, and setfuse")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flashrag.yaml")
    parser.add_argument(
        "--interface",
        choices=["setswitch", "setllm", "setfuse"],
        default="setswitch",
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    cfg["model_interface"] = args.interface
    cfg = apply_interface_overrides(cfg)
    set_seed(int(cfg.get("seed", 42)))

    eager_cfg = _model_cfg(cfg, "eager")
    sdpa_cfg = _model_cfg(cfg, "sdpa")
    add_setswitch_tokens = args.interface == "setswitch"
    tokenizer, eager_model = load_tokenizer_and_model(
        eager_cfg,
        add_setswitch_tokens=add_setswitch_tokens,
    )
    _, sdpa_model = load_tokenizer_and_model(
        sdpa_cfg,
        add_setswitch_tokens=add_setswitch_tokens,
    )
    sdpa_model.load_state_dict(eager_model.state_dict(), strict=True)

    if args.interface == "setfuse":
        fuse_start_layer = resolve_fuse_start_layer(
            eager_model,
            cfg.get("mask", {}).get("fuse_start_layer", "auto_half"),
        )
        install_setfuse_layerwise_attention(eager_model, fuse_start_layer)
        install_setfuse_layerwise_attention(sdpa_model, fuse_start_layer)

    mask_dtype = attention_mask_dtype_from_config(eager_cfg, cfg.get("train", {}), eager_model)
    batch = _render_and_collate(args.interface, tokenizer, cfg, mask_dtype)

    device = torch.device(args.device)
    eager_model.to(device).eval()
    sdpa_model.to(device).eval()
    batch = _move_tensors(batch, device)

    if args.interface == "setfuse":
        model_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask_early"],
            "position_ids": batch["position_ids"],
            "labels": batch["labels"],
            "use_cache": False,
        }
        with set_setfuse_masks(
            eager_model, batch["attention_mask_early"], batch["attention_mask_late"]
        ):
            eager_logits = eager_model(**model_batch).logits.float().cpu()
        with set_setfuse_masks(
            sdpa_model, batch["attention_mask_early"], batch["attention_mask_late"]
        ):
            sdpa_logits = sdpa_model(**model_batch).logits.float().cpu()
    else:
        model_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "position_ids": batch["position_ids"],
            "labels": batch["labels"],
            "use_cache": False,
        }
        eager_logits = eager_model(**model_batch).logits.float().cpu()
        sdpa_logits = sdpa_model(**model_batch).logits.float().cpu()

    diff = (eager_logits - sdpa_logits).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())

    print(f"interface: {args.interface}")
    print(f"mask_dtype: {mask_dtype}")
    print(f"seq_len: {batch['input_ids'].shape[1]}")
    print(f"max_abs_diff: {max_abs:.6f}")
    print(f"mean_abs_diff: {mean_abs:.6f}")
    if max_abs > args.threshold:
        raise SystemExit("SDPA differs from eager above threshold for this model/batch.")
    print("SDPA and eager are close on this custom masked batch.")


if __name__ == "__main__":
    main()
