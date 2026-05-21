"""Set-LLM position ids and attention masks.

This implements the SetPE + SetMask construction from "Set-LLM: A
Permutation-Invariant LLM" for a single known set span. The rendered text stays
ordinary text; set membership is carried only by metadata arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from set_switch.constants import (
    BLOCKED_ATTENTION_VALUE,
    ROLE_ANSWER,
    ROLE_PAD,
)


def build_setllm_position_ids_from_chunks(
    prefix_ids: Sequence[int],
    set_item_ids: Sequence[Sequence[int]],
    suffix_ids: Sequence[int],
    answer_ids: Sequence[int],
) -> list[int]:
    """Build exact SetPE positions for ``prefix + set(items) + suffix + answer``.

    Regular prompt tokens are numbered consecutively. Every set element starts
    at the same current position. After the set, positions resume after the
    total number of set-element tokens, matching Algorithm 1 in the Set-LLM
    paper.
    """

    positions: list[int] = []
    cursor = 0

    positions.extend(range(cursor, cursor + len(prefix_ids)))
    cursor += len(prefix_ids)

    set_start = cursor
    total_set_tokens = 0
    for item_ids in set_item_ids:
        positions.extend(range(set_start, set_start + len(item_ids)))
        total_set_tokens += len(item_ids)
    cursor += total_set_tokens

    positions.extend(range(cursor, cursor + len(suffix_ids)))
    cursor += len(suffix_ids)

    positions.extend(range(cursor, cursor + len(answer_ids)))
    return positions


def _as_2d_long(values: Any) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 1D or 2D values, got shape {tuple(tensor.shape)}")
    return tensor


def _as_2d_bool(values: Any, shape: tuple[int, int]) -> torch.Tensor:
    if values is None:
        return torch.ones(shape, dtype=torch.bool)
    tensor = torch.as_tensor(values)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape != shape:
        raise ValueError(f"pad_mask shape {tuple(tensor.shape)} does not match {shape}")
    return tensor.bool()


def _vectorized_allowed(
    roles: torch.Tensor,
    items: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len = roles.shape
    q_role = roles.unsqueeze(2)
    k_role = roles.unsqueeze(1)
    q_item = items.unsqueeze(2)
    k_item = items.unsqueeze(1)

    idx = torch.arange(seq_len, device=roles.device)
    causal = idx.view(1, 1, seq_len) <= idx.view(1, seq_len, 1)
    cross_item = (q_item >= 0) & (k_item >= 0) & (q_item != k_item)
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)

    q_answer = q_role == ROLE_ANSWER
    k_answer = k_role == ROLE_ANSWER
    allowed = (~q_answer & ~k_answer) | (q_answer & (~k_answer | (k_answer & causal)))
    allowed &= ~cross_item
    allowed &= valid_pair
    allowed &= (q_role != ROLE_PAD) & (k_role != ROLE_PAD)
    return allowed.reshape(batch_size, seq_len, seq_len)


def build_setllm_attention_mask(
    role_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
    item_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
    pad_mask: Sequence[bool] | Sequence[Sequence[bool]] | torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    blocked_value: float = BLOCKED_ATTENTION_VALUE,
) -> torch.Tensor:
    """Build an additive [B, 1, T, T] SetMask attention mask.

    Prompt tokens use a prefix mask with cross-set-element edges removed.
    Response tokens attend to the full prompt and previous response tokens.
    """

    roles = _as_2d_long(role_ids)
    items = _as_2d_long(item_ids)
    items = items.to(device=roles.device)
    if roles.shape != items.shape:
        raise ValueError("role_ids and item_ids must have the same shape")

    valid = _as_2d_bool(pad_mask, tuple(roles.shape))
    valid = valid.to(device=roles.device)
    batch_size, seq_len = roles.shape
    mask_device = torch.device(device) if device is not None else roles.device
    allowed = _vectorized_allowed(roles, items, valid)

    mask = torch.full(
        (batch_size, 1, seq_len, seq_len),
        fill_value=blocked_value,
        dtype=dtype,
        device=mask_device,
    )
    mask.masked_fill_(allowed.unsqueeze(1).to(device=mask.device), 0.0)
    return mask
