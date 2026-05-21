#!/usr/bin/env python
"""Run gold-document length diagnostics."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from set_switch.data.length_diagnostic import compute_length_diagnostic, save_length_histogram
from set_switch.modeling.special_tokens import add_setswitch_special_tokens
from set_switch.utils.io import read_examples_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--histogram-png")
    parser.add_argument("--whitespace", action="store_true")
    args = parser.parse_args()

    examples = read_examples_jsonl(args.input)
    tokenizer = None
    if args.model and not args.whitespace:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        add_setswitch_special_tokens(tokenizer, None)

    result = compute_length_diagnostic(examples, tokenizer=tokenizer)
    write_json(args.output, result.report)
    if args.histogram_png:
        save_length_histogram(result.gold_lengths, result.non_gold_lengths, args.histogram_png)
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
