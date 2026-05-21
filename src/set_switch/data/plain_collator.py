"""Plain causal-LM collator for the chat-template baseline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from set_switch.constants import IGNORE_INDEX


@dataclass
class PlainCausalCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id")

        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            multiple = int(self.pad_to_multiple_of)
            if max_len % multiple:
                max_len = ((max_len // multiple) + 1) * multiple

        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        answer_start: list[int] = []
        example_id: list[str] = []

        for feature in features:
            length = len(feature["input_ids"])
            pad_len = max_len - length
            input_ids.append(feature["input_ids"] + [int(self.tokenizer.pad_token_id)] * pad_len)
            labels.append(feature["labels"] + [IGNORE_INDEX] * pad_len)
            attention_mask.append([1] * length + [0] * pad_len)
            answer_start.append(int(feature["answer_start"]))
            example_id.append(feature.get("example_id", ""))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "answer_start": torch.tensor(answer_start, dtype=torch.long),
            "example_id": example_id,
        }
