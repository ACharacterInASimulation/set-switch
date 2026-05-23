"""Dense early/late attention masks for SetFuse-LM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from set_switch.constants import (
    ROLE_ANSWER,
    ROLE_DOC,
    ROLE_ITEM_SPECIAL,
    ROLE_PAD,
    ROLE_PREFIX,
    ROLE_SET_SPECIAL,
)

SETFUSE_BLOCKED_ATTENTION_VALUE = -3.4028235e38


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
    if tuple(tensor.shape) != shape:
        raise ValueError(f"pad_mask shape {tuple(tensor.shape)} does not match {shape}")
    return tensor.bool()


def _blocked_value_for_dtype(value: float, dtype: torch.dtype) -> float:
    if not dtype.is_floating_point:
        return value
    finfo = torch.finfo(dtype)
    return max(float(value), float(finfo.min))


def build_setfuse_allowed(
    role_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
    item_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
    pad_mask: Sequence[bool] | Sequence[Sequence[bool]] | torch.Tensor | None = None,
    stage: str = "early",
    setfuse_answer_attends_docs_in_early_layers: bool = False,
    setfuse_late_prefix_doc_bidir: bool = True,
) -> torch.Tensor:
    """Return a boolean SetFuse-LM allowed-attention matrix ``[B, T, T]``."""

    if stage not in {"early", "late"}:
        raise ValueError("stage must be 'early' or 'late'")

    roles = _as_2d_long(role_ids)
    items = _as_2d_long(item_ids).to(device=roles.device)
    if roles.shape != items.shape:
        raise ValueError("role_ids and item_ids must have the same shape")
    valid = _as_2d_bool(pad_mask, tuple(roles.shape)).to(device=roles.device)

    batch_size, seq_len = roles.shape
    q_role = roles.unsqueeze(2)
    k_role = roles.unsqueeze(1)
    q_item = items.unsqueeze(2)
    k_item = items.unsqueeze(1)

    idx = torch.arange(seq_len, device=roles.device)
    causal = idx.view(1, 1, seq_len) <= idx.view(1, seq_len, 1)
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)

    q_prefix = q_role == ROLE_PREFIX
    q_answer = q_role == ROLE_ANSWER
    q_doc = q_role == ROLE_DOC
    q_item_special = q_role == ROLE_ITEM_SPECIAL
    q_set_special = q_role == ROLE_SET_SPECIAL
    q_evidence = q_doc | q_item_special | q_set_special

    k_prefix = k_role == ROLE_PREFIX
    k_answer = k_role == ROLE_ANSWER
    k_doc = k_role == ROLE_DOC
    k_item_special = k_role == ROLE_ITEM_SPECIAL
    k_set_special = k_role == ROLE_SET_SPECIAL
    k_evidence = k_doc | k_item_special | k_set_special

    same_item = (q_item >= 0) & (q_item == k_item)
    same_document_evidence = same_item & (q_doc | q_item_special) & (k_doc | k_item_special)
    same_set_structure = q_set_special & k_set_special

    allowed = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.bool, device=roles.device)
    if stage == "early":
        allowed |= q_prefix & k_prefix
        allowed |= q_evidence & k_prefix
        allowed |= q_evidence & (same_document_evidence | same_set_structure)
        answer_keys = k_prefix | (k_answer & causal)
        if setfuse_answer_attends_docs_in_early_layers:
            answer_keys = answer_keys | k_evidence
        allowed |= q_answer & answer_keys
    else:
        global_evidence_keys = k_prefix | k_evidence
        prefix_keys = global_evidence_keys if setfuse_late_prefix_doc_bidir else k_prefix
        allowed |= q_prefix & prefix_keys
        allowed |= q_evidence & global_evidence_keys
        allowed |= q_answer & (global_evidence_keys | (k_answer & causal))

    allowed &= valid_pair
    allowed &= (q_role != ROLE_PAD) & (k_role != ROLE_PAD)
    pad_batch_idx, pad_query_idx = torch.nonzero(~valid, as_tuple=True)
    if pad_batch_idx.numel():
        allowed[pad_batch_idx, pad_query_idx, pad_query_idx] = True
    return allowed


def build_setfuse_attention_mask(
    role_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
    item_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
    pad_mask: Sequence[bool] | Sequence[Sequence[bool]] | torch.Tensor | None = None,
    stage: str = "early",
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    blocked_value: float = SETFUSE_BLOCKED_ATTENTION_VALUE,
    setfuse_answer_attends_docs_in_early_layers: bool = False,
    setfuse_late_prefix_doc_bidir: bool = True,
) -> torch.Tensor:
    """Return an additive SetFuse-LM mask ``[B, 1, T, T]``.

    Allowed entries are 0; blocked entries receive ``blocked_value`` clipped to
    the requested floating dtype's finite range.
    """

    roles = _as_2d_long(role_ids)
    items = _as_2d_long(item_ids).to(device=roles.device)
    if roles.shape != items.shape:
        raise ValueError("role_ids and item_ids must have the same shape")
    valid = _as_2d_bool(pad_mask, tuple(roles.shape)).to(device=roles.device)

    batch_size, seq_len = roles.shape
    mask_device = torch.device(device) if device is not None else roles.device
    allowed = build_setfuse_allowed(
        roles,
        items,
        pad_mask=valid,
        stage=stage,
        setfuse_answer_attends_docs_in_early_layers=(setfuse_answer_attends_docs_in_early_layers),
        setfuse_late_prefix_doc_bidir=setfuse_late_prefix_doc_bidir,
    )
    mask = torch.full(
        (batch_size, 1, seq_len, seq_len),
        fill_value=_blocked_value_for_dtype(blocked_value, dtype),
        dtype=dtype,
        device=mask_device,
    )
    mask.masked_fill_(allowed.unsqueeze(1).to(device=mask.device), 0.0)
    return mask


def build_setfuse_layer_masks(
    role_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
    item_ids: Sequence[int] | Sequence[Sequence[int]] | torch.Tensor,
    pad_mask: Sequence[bool] | Sequence[Sequence[bool]] | torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    blocked_value: float = SETFUSE_BLOCKED_ATTENTION_VALUE,
    setfuse_answer_attends_docs_in_early_layers: bool = False,
    setfuse_late_prefix_doc_bidir: bool = True,
) -> dict[str, torch.Tensor]:
    """Return the two dense masks used by the layer-wise SetFuse schedule."""

    return {
        "early": build_setfuse_attention_mask(
            role_ids=role_ids,
            item_ids=item_ids,
            pad_mask=pad_mask,
            stage="early",
            dtype=dtype,
            device=device,
            blocked_value=blocked_value,
            setfuse_answer_attends_docs_in_early_layers=(
                setfuse_answer_attends_docs_in_early_layers
            ),
            setfuse_late_prefix_doc_bidir=setfuse_late_prefix_doc_bidir,
        ),
        "late": build_setfuse_attention_mask(
            role_ids=role_ids,
            item_ids=item_ids,
            pad_mask=pad_mask,
            stage="late",
            dtype=dtype,
            device=device,
            blocked_value=blocked_value,
            setfuse_answer_attends_docs_in_early_layers=(
                setfuse_answer_attends_docs_in_early_layers
            ),
            setfuse_late_prefix_doc_bidir=setfuse_late_prefix_doc_bidir,
        ),
    }
