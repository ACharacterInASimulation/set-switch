from __future__ import annotations

from copy import deepcopy

from set_switch.constants import ROLE_ANSWER, ROLE_DOC, ROLE_GATHER, ROLE_READ
from set_switch.data.render import render_example
from set_switch.modeling.special_tokens import add_setswitch_special_tokens


def test_doc_positions_reset_and_read_gather_are_zero(tokenizer, example):
    add_setswitch_special_tokens(tokenizer, None)
    rendered = render_example(example, tokenizer, {"num_reads_per_doc": 2, "num_gather_tokens": 4})

    first_doc_positions = {}
    for idx, role in enumerate(rendered["role_ids"]):
        if role == ROLE_DOC:
            first_doc_positions.setdefault(rendered["item_ids"][idx], rendered["position_ids"][idx])
    assert set(first_doc_positions.values()) == {rendered["prefix_length"]}

    for idx, role in enumerate(rendered["role_ids"]):
        if role in {ROLE_READ, ROLE_GATHER}:
            assert rendered["position_ids"][idx] == 0


def test_answer_positions_use_current_set_max_doc_length(tokenizer, example):
    add_setswitch_special_tokens(tokenizer, None)
    short = deepcopy(example)
    long = deepcopy(example)
    long.documents[0].text += " Extra context appears here. More context appears here."

    rendered_short = render_example(
        short, tokenizer, {"num_reads_per_doc": 2, "num_gather_tokens": 4}
    )
    rendered_long = render_example(
        long, tokenizer, {"num_reads_per_doc": 2, "num_gather_tokens": 4}
    )

    short_answer_positions = [
        pos
        for pos, role in zip(rendered_short["position_ids"], rendered_short["role_ids"])
        if role == ROLE_ANSWER
    ]
    long_answer_positions = [
        pos
        for pos, role in zip(rendered_long["position_ids"], rendered_long["role_ids"])
        if role == ROLE_ANSWER
    ]
    short_doc_lengths = {}
    for role, item_id in zip(rendered_short["role_ids"], rendered_short["item_ids"], strict=True):
        if role == ROLE_DOC:
            short_doc_lengths[item_id] = short_doc_lengths.get(item_id, 0) + 1
    long_doc_lengths = {}
    for role, item_id in zip(rendered_long["role_ids"], rendered_long["item_ids"], strict=True):
        if role == ROLE_DOC:
            long_doc_lengths[item_id] = long_doc_lengths.get(item_id, 0) + 1

    assert short_answer_positions[0] == rendered_short["prefix_length"] + max(
        short_doc_lengths.values()
    )
    assert long_answer_positions[0] == rendered_long["prefix_length"] + max(
        long_doc_lengths.values()
    )
    assert long_answer_positions[0] > short_answer_positions[0]
