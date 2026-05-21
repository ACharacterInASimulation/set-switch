#!/usr/bin/env python
"""Convert the selected FlashRAG sources to SetSwitch JSONL."""

from __future__ import annotations

import argparse

from set_switch.data.dataset_suite import (
    load_flashrag_selected_examples,
    normalize_flashrag_sources,
)
from set_switch.utils.io import read_yaml, write_examples_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output", required=True)
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
    examples = load_flashrag_selected_examples(
        dataset_name=data_cfg.get("dataset_name", "RUC-NLPIR/FlashRAG_datasets"),
        selections=normalize_flashrag_sources(data_cfg, args.split),
        max_docs=int(data_cfg.get("max_docs", 8)),
        instruction=instruction,
        total_examples=data_cfg.get(total_key),
        sample_allocation=data_cfg.get("sample_allocation", "task_balanced_equal"),
        sample_allocation_alpha=float(data_cfg.get("sample_allocation_alpha", 0.5)),
    )

    write_examples_jsonl(args.output, examples)
    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
