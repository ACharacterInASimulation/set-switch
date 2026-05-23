#!/usr/bin/env python
"""Convert the selected FlashRAG sources to SetSwitch JSONL."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer
from tqdm import tqdm

from set_switch.config_validation import validate_config
from set_switch.data.dataset_suite import (
    iter_flashrag_selected_examples,
    normalize_flashrag_sources,
)
from set_switch.data.length_filter import (
    max_rendered_length,
    normalize_length_filter_interfaces,
)
from set_switch.modeling.special_tokens import (
    add_setswitch_special_tokens,
    ensure_tokenizer_has_pad_token,
)
from set_switch.utils.io import read_yaml, write_examples_jsonl


def _load_length_filter_tokenizer(cfg: dict, add_setswitch_tokens_for_filter: bool):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name_or_path"], use_fast=True)
    ensure_tokenizer_has_pad_token(tokenizer)
    if add_setswitch_tokens_for_filter:
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
    parser.add_argument(
        "--max-examples",
        default=None,
        help="Integer cap for this prepared split, or 'all'. Defaults to train cap for train and all for val/test.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    validate_config(cfg)
    data_cfg = cfg.get("data", {})
    if data_cfg.get("source", "flashrag") != "flashrag":
        raise ValueError("Only data.source='flashrag' is supported")
    instruction = data_cfg.get(
        "instruction",
        "Use the provided passages or options to answer the question.",
    )
    if args.max_examples is not None:
        total_examples = (
            None
            if str(args.max_examples).strip().lower() in {"all", "none", "null"}
            else int(args.max_examples)
        )
    elif args.split == "train":
        total_examples = data_cfg.get("total_train_examples")
    else:
        total_examples = None
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
    interfaces = None
    if apply_length_filter:
        max_render_tokens = int(max_render_tokens)
        interfaces = normalize_length_filter_interfaces(
            args.length_filter_interfaces or data_cfg.get("length_filter_interfaces", "all")
        )
        if "setfuse" not in interfaces:
            interfaces = (*interfaces, "setfuse")
        tokenizer = _load_length_filter_tokenizer(
            cfg,
            add_setswitch_tokens_for_filter="setswitch" in interfaces,
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

    selections = normalize_flashrag_sources(data_cfg, args.split)
    if args.verbose:
        print(
            "Preparing dataset suite: "
            f"split={args.split} output={args.output} "
            f"target={total_examples if total_examples is not None else 'all'} "
            f"max_docs={data_cfg.get('max_docs', 8)}"
        )
        print(
            "Sources: "
            + ", ".join(
                f"{selection.name}[{selection.split}]"
                + (f":{selection.max_examples}" if selection.max_examples is not None else "")
                for selection in selections
            )
        )
        if apply_length_filter:
            print(
                "Length filter: "
                f"max_render_tokens={max_render_tokens} interfaces={','.join(interfaces or ())}"
            )

    iterator = iter_flashrag_selected_examples(
        dataset_name=data_cfg.get("dataset_name", "RUC-NLPIR/FlashRAG_datasets"),
        selections=selections,
        max_docs=int(data_cfg.get("max_docs", 8)),
        instruction=instruction,
        total_examples=total_examples,
        sample_allocation=data_cfg.get("sample_allocation", "task_balanced_equal"),
        sample_allocation_alpha=float(data_cfg.get("sample_allocation_alpha", 0.5)),
        example_filter=example_filter,
        verbose=bool(args.verbose),
    )
    examples = list(
        tqdm(
            iterator,
            total=total_examples,
            desc=f"Building {args.split}",
            disable=not args.verbose,
        )
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
