#!/usr/bin/env python
"""Inspect one rendered SetSwitch batch."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from set_switch.constants import ROLE_NAMES
from set_switch.data.baseline_render import render_chat_baseline_example
from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchDocument, SetSwitchExample
from set_switch.data.setfuse_render import render_setfuse_example
from set_switch.data.setllm_render import render_setllm_example
from set_switch.modeling.special_tokens import add_setswitch_special_tokens
from set_switch.utils.io import read_examples_jsonl, read_yaml


def fixture_example(index: int = 0) -> SetSwitchExample:
    del index
    return SetSwitchExample(
        example_id="inspect-fixture",
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--input")
    parser.add_argument("--model")
    parser.add_argument("--index", type=int, default=0)
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
    if args.input:
        example = read_examples_jsonl(args.input)[args.index]
    else:
        example = fixture_example(args.index)

    if interface == "setswitch":
        rendered = render_example(example, tokenizer, cfg)
    elif interface == "setllm":
        rendered = render_setllm_example(example, tokenizer, cfg)
    elif interface == "setfuse":
        rendered = render_setfuse_example(example, tokenizer, cfg)
    elif interface == "chat_baseline":
        rendered = render_chat_baseline_example(example, tokenizer, cfg)
    else:
        raise ValueError(f"Unknown model_interface {interface!r}")

    tokens = tokenizer.convert_ids_to_tokens(rendered["input_ids"])
    for idx, token in enumerate(tokens):
        label = rendered["labels"][idx]
        label_text = "" if label == -100 else tokenizer.decode([label])
        if "role_ids" not in rendered:
            print(
                f"{idx:04d} id={rendered['input_ids'][idx]:6d} "
                f"label={label:6d} {label_text!r} token={token!r}"
            )
            continue
        print(
            f"{idx:04d} id={rendered['input_ids'][idx]:6d} "
            f"role={ROLE_NAMES[rendered['role_ids'][idx]]:12s} "
            f"item={rendered['item_ids'][idx]:2d} "
            f"read={rendered.get('read_slot_ids', [-1] * len(tokens))[idx]:2d} "
            f"gather={rendered.get('gather_slot_ids', [-1] * len(tokens))[idx]:2d} "
            f"pos={rendered['position_ids'][idx]:4d} "
            f"label={label:6d} {label_text!r} token={token!r}"
        )
    print(f"answer_start={rendered['answer_start']}")


if __name__ == "__main__":
    main()
