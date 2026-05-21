from __future__ import annotations

import torch

from set_switch.constants import (
    DOC_BIDIR,
    DOC_CAUSAL,
    ROLE_ANSWER,
    ROLE_DOC,
    ROLE_GATHER,
    ROLE_ITEM_SPECIAL,
    ROLE_PREFIX,
    ROLE_READ,
    ROLE_SET_SPECIAL,
)
from set_switch.modeling.attention_mask import build_setswitch_attention_mask


def _toy_roles():
    role_ids = [
        ROLE_PREFIX,
        ROLE_PREFIX,
        ROLE_SET_SPECIAL,
        ROLE_ITEM_SPECIAL,
        ROLE_DOC,
        ROLE_DOC,
        ROLE_READ,
        ROLE_ITEM_SPECIAL,
        ROLE_ITEM_SPECIAL,
        ROLE_DOC,
        ROLE_DOC,
        ROLE_READ,
        ROLE_ITEM_SPECIAL,
        ROLE_SET_SPECIAL,
        ROLE_GATHER,
        ROLE_GATHER,
        ROLE_ANSWER,
        ROLE_ANSWER,
    ]
    item_ids = [-1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
    read_slots = [-1] * len(role_ids)
    read_slots[6] = 0
    read_slots[11] = 0
    gather_slots = [-1] * len(role_ids)
    gather_slots[14] = 0
    gather_slots[15] = 1
    return role_ids, item_ids, read_slots, gather_slots


def _allowed(mode: str):
    role_ids, item_ids, read_slots, gather_slots = _toy_roles()
    mask = build_setswitch_attention_mask(
        role_ids,
        item_ids,
        read_slots,
        gather_slots,
        attention_mode=mode,
    )
    return mask[0, 0] == 0


def test_doc_attention_is_document_local_and_causal_or_bidir():
    causal = _allowed(DOC_CAUSAL)
    bidir = _allowed(DOC_BIDIR)

    assert causal[1, 0]
    assert causal[0, 1]
    assert not causal[0, 4]
    assert not causal[0, 6]
    assert not causal[0, 14]

    assert causal[4, 0]
    assert not causal[4, 9]
    assert not causal[9, 4]
    assert not causal[4, 5]
    assert bidir[4, 5]
    assert not causal[4, 6]
    assert not causal[4, 14]
    assert not causal[4, 16]


def test_read_gather_and_answer_rules():
    allowed = _allowed(DOC_CAUSAL)

    assert allowed[6, 4]
    assert not allowed[6, 9]
    assert not allowed[6, 14]
    assert not allowed[6, 16]

    assert allowed[14, 6]
    assert allowed[14, 11]
    assert allowed[14, 15]
    assert allowed[15, 14]
    assert not allowed[14, 4]

    assert allowed[16, 14]
    assert allowed[16, 15]
    assert not allowed[16, 4]
    assert not allowed[16, 6]
    assert allowed[17, 16]
    assert not allowed[16, 17]


def test_padding_is_never_attended():
    role_ids, item_ids, read_slots, gather_slots = _toy_roles()
    role_ids = role_ids + [ROLE_PREFIX]
    item_ids = item_ids + [-1]
    read_slots = read_slots + [-1]
    gather_slots = gather_slots + [-1]
    pad_mask = [True] * (len(role_ids) - 1) + [False]

    mask = build_setswitch_attention_mask(
        role_ids,
        item_ids,
        read_slots,
        gather_slots,
        attention_mode=DOC_CAUSAL,
        pad_mask=pad_mask,
    )
    allowed = mask[0, 0] == 0

    assert not allowed[:, -1].any()
    assert not allowed[-1, :].any()


def test_custom_mask_dtype_is_configurable():
    role_ids, item_ids, read_slots, gather_slots = _toy_roles()
    mask = build_setswitch_attention_mask(
        role_ids,
        item_ids,
        read_slots,
        gather_slots,
        attention_mode=DOC_CAUSAL,
        dtype=torch.bfloat16,
    )

    assert mask.dtype == torch.bfloat16
