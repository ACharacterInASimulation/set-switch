"""Batch collation for SetFuse-LM rendered examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from set_switch.constants import IGNORE_INDEX, ROLE_PAD
from set_switch.modeling.setfuse_attention_mask import build_setfuse_layer_masks


@dataclass
class SetFuseCollator:
    tokenizer: Any
    mask_dtype: torch.dtype = torch.float32
    pad_to_multiple_of: int | None = None
    setfuse_answer_attends_docs_in_early_layers: bool = False
    setfuse_late_prefix_doc_bidir: bool = True
    build_attention_mask: bool = True

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

        batch: dict[str, list[Any]] = {
            "input_ids": [],
            "labels": [],
            "role_ids": [],
            "item_ids": [],
            "position_ids": [],
            "pad_mask": [],
            "answer_start": [],
            "prefix_length": [],
            "max_doc_length": [],
            "example_id": [],
        }
        for feature in features:
            length = len(feature["input_ids"])
            pad_len = max_len - length
            batch["input_ids"].append(
                feature["input_ids"] + [int(self.tokenizer.pad_token_id)] * pad_len
            )
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad_len)
            batch["role_ids"].append(feature["role_ids"] + [ROLE_PAD] * pad_len)
            batch["item_ids"].append(feature["item_ids"] + [-1] * pad_len)
            batch["position_ids"].append(feature["position_ids"] + [0] * pad_len)
            batch["pad_mask"].append([True] * length + [False] * pad_len)
            batch["answer_start"].append(int(feature["answer_start"]))
            batch["prefix_length"].append(int(feature["prefix_length"]))
            batch["max_doc_length"].append(int(feature["max_doc_length"]))
            batch["example_id"].append(feature.get("example_id", ""))

        role_ids = torch.tensor(batch["role_ids"], dtype=torch.long)
        item_ids = torch.tensor(batch["item_ids"], dtype=torch.long)
        pad_mask = torch.tensor(batch["pad_mask"], dtype=torch.bool)
        output = {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "position_ids": torch.tensor(batch["position_ids"], dtype=torch.long),
            "role_ids": role_ids,
            "item_ids": item_ids,
            "pad_mask": pad_mask,
            "answer_start": torch.tensor(batch["answer_start"], dtype=torch.long),
            "prefix_length": torch.tensor(batch["prefix_length"], dtype=torch.long),
            "max_doc_length": torch.tensor(batch["max_doc_length"], dtype=torch.long),
            "example_id": batch["example_id"],
        }
        if self.build_attention_mask:
            layer_masks = build_setfuse_layer_masks(
                role_ids=role_ids,
                item_ids=item_ids,
                pad_mask=pad_mask,
                dtype=self.mask_dtype,
                setfuse_answer_attends_docs_in_early_layers=(
                    self.setfuse_answer_attends_docs_in_early_layers
                ),
                setfuse_late_prefix_doc_bidir=self.setfuse_late_prefix_doc_bidir,
            )
            output["attention_mask_early"] = layer_masks["early"]
            output["attention_mask_late"] = layer_masks["late"]
        return output
