"""Custom SetSwitch position ids."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from set_switch.constants import ROLE_ANSWER, ROLE_DOC, ROLE_GATHER, ROLE_ITEM_SPECIAL, ROLE_READ


def build_position_ids(
    role_ids: Sequence[int],
    item_ids: Sequence[int],
    prefix_length: int,
    max_doc_length: int | None = None,
    read_slot_ids: Sequence[int] | None = None,
    gather_slot_ids: Sequence[int] | None = None,
    read_gather_position_mode: str = "zero",
) -> list[int]:
    """Build SetPE-style SetSwitch position ids.

    Prefix positions are assumed to have already occupied token indexes 0..prefix_length-1.
    Raw document positions reset for every document, starting at prefix_length.
    Set/item tokens use 0. In ``zero`` mode, read/gather tokens also use 0.
    In ``continuous`` mode, read tokens continue after their local document and
    gather/answer tokens continue after the longest document plus aggregation span.
    """

    if len(role_ids) != len(item_ids):
        raise ValueError("role_ids and item_ids must have the same length")
    if read_slot_ids is None:
        read_slot_ids = [-1] * len(role_ids)
    if gather_slot_ids is None:
        gather_slot_ids = [-1] * len(role_ids)
    if len(read_slot_ids) != len(role_ids) or len(gather_slot_ids) != len(role_ids):
        raise ValueError("read_slot_ids and gather_slot_ids must match role_ids length")
    if read_gather_position_mode not in {"zero", "continuous"}:
        raise ValueError("read_gather_position_mode must be 'zero' or 'continuous'")

    doc_offsets: defaultdict[int, int] = defaultdict(int)
    doc_lengths: defaultdict[int, int] = defaultdict(int)
    for role, item_id in zip(role_ids, item_ids, strict=True):
        if role == ROLE_DOC:
            doc_lengths[int(item_id)] += 1
    if max_doc_length is None:
        max_doc_length = max(doc_lengths.values(), default=0)
    num_read_positions = max([int(slot) for slot in read_slot_ids if int(slot) >= 0], default=-1) + 1
    num_gather_positions = (
        max([int(slot) for slot in gather_slot_ids if int(slot) >= 0], default=-1) + 1
    )
    aggregation_offset = (
        num_read_positions + num_gather_positions
        if read_gather_position_mode == "continuous"
        else 0
    )
    answer_offset = 0
    position_ids: list[int] = []

    for idx, role in enumerate(role_ids):
        item_id = int(item_ids[idx])
        if idx < prefix_length and role not in {
            ROLE_DOC,
            ROLE_READ,
            ROLE_GATHER,
            ROLE_ITEM_SPECIAL,
            ROLE_ANSWER,
        }:
            position_ids.append(idx)
        elif role == ROLE_DOC:
            position_ids.append(prefix_length + doc_offsets[item_id])
            doc_offsets[item_id] += 1
        elif role == ROLE_READ and read_gather_position_mode == "continuous":
            read_slot = max(0, int(read_slot_ids[idx]))
            position_ids.append(prefix_length + doc_lengths[item_id] + read_slot)
        elif role == ROLE_GATHER and read_gather_position_mode == "continuous":
            gather_slot = max(0, int(gather_slot_ids[idx]))
            position_ids.append(prefix_length + int(max_doc_length) + num_read_positions + gather_slot)
        elif role == ROLE_ANSWER:
            position_ids.append(
                prefix_length + int(max_doc_length) + aggregation_offset + answer_offset
            )
            answer_offset += 1
        else:
            position_ids.append(0)

    return position_ids
