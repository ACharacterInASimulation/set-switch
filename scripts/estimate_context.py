#!/usr/bin/env python
"""Estimate SetSwitch sequence length and attention-mask memory."""

from __future__ import annotations

import argparse

from set_switch.utils.io import read_yaml

DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
}


def estimate_tokens(
    max_docs: int,
    max_doc_tokens: int,
    num_reads_per_doc: int,
    num_gather_tokens: int,
    prefix_tokens: int,
    answer_tokens: int,
    item_overhead_tokens: int,
    set_overhead_tokens: int,
) -> int:
    return (
        prefix_tokens
        + set_overhead_tokens
        + max_docs * (max_doc_tokens + num_reads_per_doc + item_overhead_tokens)
        + num_gather_tokens
        + answer_tokens
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flashrag.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dtype")
    parser.add_argument(
        "--assumed-doc-tokens",
        type=int,
        default=512,
        help="Used only when data.max_doc_tokens is null.",
    )
    parser.add_argument("--prefix-tokens", type=int, default=128)
    parser.add_argument("--answer-tokens", type=int, default=64)
    parser.add_argument("--item-overhead-tokens", type=int)
    parser.add_argument("--set-overhead-tokens", type=int)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 1))
    dtype_name = (args.dtype or model_cfg.get("dtype", "bf16")).lower()
    dtype_bytes = DTYPE_BYTES.get(dtype_name, 2)
    max_doc_tokens = data_cfg.get("max_doc_tokens")
    if max_doc_tokens is None:
        max_doc_tokens = args.assumed_doc_tokens
    compact_format = bool(data_cfg.get("compact_special_token_format", False))
    item_overhead_tokens = args.item_overhead_tokens
    if item_overhead_tokens is None:
        item_overhead_tokens = 2 if compact_format else 6
    set_overhead_tokens = args.set_overhead_tokens
    if set_overhead_tokens is None:
        set_overhead_tokens = 2 if compact_format else 4
    seq_len = estimate_tokens(
        max_docs=int(data_cfg.get("max_docs", 8)),
        max_doc_tokens=int(max_doc_tokens),
        num_reads_per_doc=int(data_cfg.get("num_reads_per_doc", 2)),
        num_gather_tokens=int(data_cfg.get("num_gather_tokens", 4)),
        prefix_tokens=args.prefix_tokens,
        answer_tokens=args.answer_tokens,
        item_overhead_tokens=item_overhead_tokens,
        set_overhead_tokens=set_overhead_tokens,
    )
    mask_gib = batch_size * seq_len * seq_len * dtype_bytes / 1024**3

    print(f"estimated_seq_len: {seq_len}")
    print(f"assumed_max_doc_tokens: {max_doc_tokens}")
    print(f"batch_size: {batch_size}")
    print(f"compact_special_token_format: {compact_format}")
    print(f"item_overhead_tokens: {item_overhead_tokens}")
    print(f"set_overhead_tokens: {set_overhead_tokens}")
    print(f"mask_dtype_bytes: {dtype_bytes}")
    print(f"one_4d_attention_mask_gib: {mask_gib:.3f}")
    print()
    print("Note: this is only the additive attention mask tensor. Model activations")
    print("and eager attention score tensors add much more memory during training.")


if __name__ == "__main__":
    main()
