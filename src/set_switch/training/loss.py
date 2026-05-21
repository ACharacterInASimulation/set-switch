"""Loss helpers."""

from __future__ import annotations

import torch

from set_switch.constants import IGNORE_INDEX


def answer_only_labels(input_ids: torch.Tensor, answer_starts: torch.Tensor) -> torch.Tensor:
    """Create labels with IGNORE_INDEX before each example's answer start."""

    labels = input_ids.clone()
    for batch_idx, answer_start in enumerate(answer_starts.tolist()):
        labels[batch_idx, : int(answer_start)] = IGNORE_INDEX
    return labels
