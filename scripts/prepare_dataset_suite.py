#!/usr/bin/env python
"""Convert the selected FlashRAG sources to SetSwitch JSONL."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from set_switch.data.dataset_suite import (
    load_flashrag_selected_examples,
    normalize_flashrag_sources,
)
from set_switch.data.length_filter import (
    max_rendered_length,
    normalize_length_filter_interfaces,
)
from set_switch.modeling.special_tokens import add_setswitch_special_tokens, ensure_tokenizer_has_pad_token
from set_switch.utils.io import read_yaml, write_examples_jsonl


def _load_length_filter_tokenizer(cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"], use_fast=True)
    ensure_tokenizer_has_pad_token(tokenizer)
    add_setswitch_special_tokens(tokenizer, None)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-render-tokens", type=int)
    parser.add_argument("--length-filter-interfaces", default=None)
    parser.add_argument("--no-length-filter", action="store_true")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    data_cfg = cfg.get("data", {})
    if data_cfg.get("source", "flashrag") != "flashrag":
        raise ValueError("Only data.source='flashrag' is supported")
    instruction = data_cfg.get(
        "instruction",
        "Use the provided passages or options to answer the question. Treat the items as an unordered set.",
    )
    total_key = "total_train_examples" if args.split == "train" else "total_val_examples"
    max_render_tokens = (
        args.max_render_tokens
        if args.max_render_tokens is not None
        else data_cfg.get("train_max_render_tokens")
    )
    apply_length_filter = (
        args.split == "train" and max_render_tokens is not None and not args.no_length_filter
    )
    dropped = 0
    example_filter = None
    if apply_length_filter:
        max_render_tokens = int(max_render_tokens)
        tokenizer = _load_length_filter_tokenizer(cfg)
        interfaces = normalize_length_filter_interfaces(
            args.length_filter_interfaces or data_cfg.get("length_filter_interfaces", "all")
        )

        def example_filter(example):
            nonlocal dropped
            keep = (
                max_rendered_length(
                    example=example,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    interfaces=interfaces,
                )
                <= max_render_tokens
            )
            dropped += int(not keep)
            return keep

    examples = load_flashrag_selected_examples(
        dataset_name=data_cfg.get("dataset_name", "RUC-NLPIR/FlashRAG_datasets"),
        selections=normalize_flashrag_sources(data_cfg, args.split),
        max_docs=int(data_cfg.get("max_docs", 8)),
        instruction=instruction,
        total_examples=data_cfg.get(total_key),
        sample_allocation=data_cfg.get("sample_allocation", "task_balanced_equal"),
        sample_allocation_alpha=float(data_cfg.get("sample_allocation_alpha", 0.5)),
        example_filter=example_filter,
    )

    write_examples_jsonl(args.output, examples)
    print(f"Wrote {len(examples)} examples to {args.output}")
    if apply_length_filter:
        print(
            "Length filter: "
            f"kept={len(examples)} dropped={dropped} "
            f"max_render_tokens={max_render_tokens}"
        )


if __name__ == "__main__":
    main()
