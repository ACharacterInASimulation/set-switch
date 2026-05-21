"""SetSwitch 4D additive attention masks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from set_switch.constants import (
    BLOCKED_ATTENTION_VALUE,
    DOC_ATTENTION_MODES,
    DOC_BIDIR,
    DOC_CAUSAL,
    ROLE_ANSWER,
    ROLE_DOC,
    ROLE_GATHER,
    ROLE_ITEM_SPECIAL,
    ROLE_PAD,
    ROLE_PREFIX,
    ROLE_READ,
    ROLE_SET_SPECIAL,
)


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
    attention_mode: str,
) -> torch.Tensor:
    batch_size, seq_len = roles.shape
    del batch_size

    q_role = roles.unsqueeze(2)
    k_role = roles.unsqueeze(1)
    q_item = items.unsqueeze(2)
    k_item = items.unsqueeze(1)

    idx = torch.arange(seq_len, device=roles.device)
    causal = idx.view(1, 1, seq_len) <= idx.view(1, seq_len, 1)
    same_item = (q_item >= 0) & (q_item == k_item)
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)

    q_prefix = q_role == ROLE_PREFIX
    q_set_special = q_role == ROLE_SET_SPECIAL
    q_item_special = q_role == ROLE_ITEM_SPECIAL
    q_doc = q_role == ROLE_DOC
    q_read = q_role == ROLE_READ
    q_gather = q_role == ROLE_GATHER
    q_answer = q_role == ROLE_ANSWER

    k_prefix = k_role == ROLE_PREFIX
    k_set_special = k_role == ROLE_SET_SPECIAL
    k_item_special = k_role == ROLE_ITEM_SPECIAL
    k_doc = k_role == ROLE_DOC
    k_read = k_role == ROLE_READ
    k_gather = k_role == ROLE_GATHER
    k_answer = k_role == ROLE_ANSWER
    k_set_or_item = k_set_special | k_item_special

    allowed = torch.zeros(
        (roles.shape[0], seq_len, seq_len),
        dtype=torch.bool,
        device=roles.device,
    )
    allowed |= q_prefix & k_prefix
    allowed |= q_set_special & (k_prefix | (k_set_or_item & causal))
    allowed |= q_item_special & (k_prefix | (same_item & k_set_or_item & causal))

    if attention_mode == DOC_BIDIR:
        doc_to_doc = k_doc
    else:
        doc_to_doc = k_doc & causal
    allowed |= q_doc & (k_prefix | (same_item & (doc_to_doc | (k_item_special & causal))))
    allowed |= q_read & (k_prefix | (same_item & (k_doc | k_read | k_item_special)))
    allowed |= q_gather & (k_prefix | k_read | k_gather)
    allowed |= q_answer & (k_prefix | k_gather | (k_answer & causal))

    allowed &= valid_pair
    allowed &= (q_role != ROLE_PAD) & (k_role != ROLE_PAD)
    return allowed


def build_setswitch_attention_mask(
    role_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
    item_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
    read_slot_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor | None = None,
    gather_slot_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor | None = None,
    attention_mode: str = DOC_CAUSAL,
    pad_mask: Sequence[bool] | Sequence[Sequence[bool]] | torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    blocked_value: float = BLOCKED_ATTENTION_VALUE,
) -> torch.Tensor:
    """Build an additive [B, 1, T, T] SetSwitch attention mask.

    Allowed positions are 0. Blocked positions are a large negative value.
    """

    del read_slot_ids, gather_slot_ids
    if attention_mode not in DOC_ATTENTION_MODES:
        raise ValueError(
            f"Unknown attention mode {attention_mode!r}; expected one of {DOC_ATTENTION_MODES}"
        )

    roles = _as_2d_long(role_ids)
    items = _as_2d_long(item_ids)
    items = items.to(device=roles.device)
    if roles.shape != items.shape:
        raise ValueError("role_ids and item_ids must have the same shape")

    valid = _as_2d_bool(pad_mask, tuple(roles.shape))
    valid = valid.to(device=roles.device)
    batch_size, seq_len = roles.shape
    mask_device = torch.device(device) if device is not None else roles.device
    allowed = _vectorized_allowed(
        roles=roles,
        items=items,
        valid=valid,
        attention_mode=attention_mode,
    )

    mask = torch.full(
        (batch_size, 1, seq_len, seq_len),
        fill_value=blocked_value,
        dtype=dtype,
        device=mask_device,
    )
    mask.masked_fill_(allowed.unsqueeze(1).to(device=mask.device), 0.0)
    return mask


def allowed_attention_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return a boolean allowed matrix for tests/debugging."""

    return mask >= 0
