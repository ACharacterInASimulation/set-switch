"""Simple validation and invariance diagnostics."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import DataLoader

from set_switch.data.collator import SetSwitchCollator
from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchExample


@torch.no_grad()
def evaluate_answer_ce(model: Any, dataloader: DataLoader, max_batches: int | None = None) -> float:
    model.eval()
    device = next(model.parameters()).device
    losses: list[float] = []
    for batch_idx, batch in enumerate(dataloader):
        model_batch = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
            "use_cache": False,
        }
        if "position_ids" in batch:
            model_batch["position_ids"] = batch["position_ids"].to(device)
        outputs = model(**model_batch)
        losses.append(float(outputs.loss.detach().cpu()))
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def permutation_invariance_delta(
    model: Any,
    tokenizer: Any,
    example: SetSwitchExample,
    cfg: dict[str, Any],
    attention_mode: str = "doc_causal",
) -> float:
    """Compare answer-start logits after reversing document order."""

    if len(example.documents) < 2:
        return 0.0

    model.eval()
    reversed_example = deepcopy(example)
    reversed_example.documents = list(reversed(example.documents))

    rendered_a = render_example(example, tokenizer, cfg)
    rendered_b = render_example(reversed_example, tokenizer, cfg)
    collator = SetSwitchCollator(tokenizer=tokenizer, attention_mode=attention_mode)
    batch = collator([rendered_a, rendered_b])
    device = next(model.parameters()).device
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        position_ids=batch["position_ids"].to(device),
        labels=None,
        use_cache=False,
    )
    start_a = int(batch["answer_start"][0])
    start_b = int(batch["answer_start"][1])
    logits_a = outputs.logits[0, start_a - 1].float().cpu()
    logits_b = outputs.logits[1, start_b - 1].float().cpu()
    model.train()
    return float(torch.max(torch.abs(logits_a - logits_b)).item())
