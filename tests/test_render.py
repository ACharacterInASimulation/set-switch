from __future__ import annotations

from set_switch.constants import (
    GATHER_TOKENS,
    IGNORE_INDEX,
    READ_TOKENS,
    ROLE_GATHER,
    ROLE_READ,
    SETSWITCH_SPECIAL_TOKENS,
)
from set_switch.data.render import render_example
from set_switch.modeling.special_tokens import add_setswitch_special_tokens


def test_render_contains_tokens_and_answer_labels(tokenizer, example):
    add_setswitch_special_tokens(tokenizer, None)
    rendered = render_example(example, tokenizer, {"num_reads_per_doc": 2, "num_gather_tokens": 4})
    decoded = tokenizer.decode(rendered["input_ids"])

    for token in ["<set>", "</set>", "<item>", "</item>", READ_TOKENS[0], GATHER_TOKENS[0]]:
        assert token in decoded
    for token in SETSWITCH_SPECIAL_TOKENS:
        assert len(tokenizer.encode(token, add_special_tokens=False)) == 1

    answer_start = rendered["answer_start"]
    assert tokenizer.decode(rendered["input_ids"][answer_start:]) == example.answer
    assert all(label == IGNORE_INDEX for label in rendered["labels"][:answer_start])
    assert any(label != IGNORE_INDEX for label in rendered["labels"][answer_start:])
    assert ROLE_READ in rendered["role_ids"]
    assert ROLE_GATHER in rendered["role_ids"]
    assert rendered["role_ids"].count(ROLE_READ) == len(example.documents) * 2
    assert rendered["role_ids"].count(ROLE_GATHER) == 4


def test_setswitch_can_append_eos_to_answer_labels(tokenizer, example):
    add_setswitch_special_tokens(tokenizer, None)
    rendered = render_example(
        example,
        tokenizer,
        {"num_reads_per_doc": 2, "num_gather_tokens": 4, "append_eos_token": True},
    )

    assert rendered["input_ids"][-1] == tokenizer.eos_token_id
    assert rendered["labels"][-1] == tokenizer.eos_token_id
    assert tokenizer.decode(rendered["input_ids"][rendered["answer_start"] : -1]) == example.answer
