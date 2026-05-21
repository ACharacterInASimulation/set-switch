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
) -> list[int]:
    """Build SetPE-style SetSwitch position ids.

    Prefix positions are assumed to have already occupied token indexes 0..prefix_length-1.
    Raw document positions reset for every document, starting at prefix_length.
    Set/item/read/gather tokens use 0. Answer tokens resume after the longest
    document in the current set, i.e. prefix_length + max_doc_length.
    """

    if len(role_ids) != len(item_ids):
        raise ValueError("role_ids and item_ids must have the same length")

    doc_offsets: defaultdict[int, int] = defaultdict(int)
    if max_doc_length is None:
        doc_lengths: defaultdict[int, int] = defaultdict(int)
        for role, item_id in zip(role_ids, item_ids, strict=True):
            if role == ROLE_DOC:
                doc_lengths[int(item_id)] += 1
        max_doc_length = max(doc_lengths.values(), default=0)
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
        elif role == ROLE_ANSWER:
            position_ids.append(prefix_length + int(max_doc_length) + answer_offset)
            answer_offset += 1
        else:
            position_ids.append(0)

    return position_ids
