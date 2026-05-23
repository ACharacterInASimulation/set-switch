#!/usr/bin/env python
"""Print the SetSwitch allowed-attention matrix for a tiny example."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchDocument, SetSwitchExample
from set_switch.data.setfuse_render import render_setfuse_example
from set_switch.data.setllm_render import render_setllm_example
from set_switch.modeling.attention_mask import build_setswitch_attention_mask
from set_switch.modeling.setfuse_attention_mask import build_setfuse_attention_mask
from set_switch.modeling.setllm import build_setllm_attention_mask
from set_switch.modeling.special_tokens import add_setswitch_special_tokens
from set_switch.utils.io import read_yaml


def fixture_example() -> SetSwitchExample:
    return SetSwitchExample(
        example_id="debug-mask",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flashrag.yaml")
    parser.add_argument("--model")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    model_name = args.model or cfg.get("model", {}).get(
        "name_or_path", "HuggingFaceTB/SmolLM2-360M"
    )
    interface = cfg.get(
        "model_interface",
        cfg.get("interface", cfg.get("model", {}).get("interface", "setswitch")),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if interface == "setswitch":
        add_setswitch_special_tokens(tokenizer, None)
    example = fixture_example()
    if interface == "setllm":
        rendered = render_setllm_example(example, tokenizer, cfg)
        mask = build_setllm_attention_mask(rendered["role_ids"], rendered["item_ids"])
    elif interface == "setfuse":
        rendered = render_setfuse_example(example, tokenizer, cfg)
        mask = build_setfuse_attention_mask(
            rendered["role_ids"],
            rendered["item_ids"],
            stage=str(cfg.get("mask", {}).get("setfuse_debug_stage", "early")),
            setfuse_answer_attends_docs_in_early_layers=bool(
                cfg.get("mask", {}).get("setfuse_answer_attends_docs_in_early_layers", False)
            ),
            setfuse_late_prefix_doc_bidir=bool(
                cfg.get("mask", {}).get("setfuse_late_prefix_doc_bidir", True)
            ),
        )
    elif interface == "chat_baseline":
        raise ValueError("chat_baseline uses the model's ordinary causal mask; no custom matrix")
    else:
        rendered = render_example(example, tokenizer, cfg)
        mask = build_setswitch_attention_mask(
            rendered["role_ids"],
            rendered["item_ids"],
            rendered["read_slot_ids"],
            rendered["gather_slot_ids"],
            attention_mode=cfg.get("mask", {}).get("doc_attention", "doc_causal"),
            answer_attends_raw_docs=bool(cfg.get("mask", {}).get("answer_attends_raw_docs", False)),
            answer_attends_reads=bool(cfg.get("mask", {}).get("answer_attends_reads", False)),
        )
    allowed = (mask[0, 0] == 0).int()
    print(allowed.numpy())


if __name__ == "__main__":
    main()
