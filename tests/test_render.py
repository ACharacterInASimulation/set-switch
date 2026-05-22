from __future__ import annotations

from set_switch.constants import (
    END_ITEM_TOKEN,
    END_SET_TOKEN,
    GATHER_TOKENS,
    IGNORE_INDEX,
    ITEM_TOKEN,
    READ_TOKENS,
    ROLE_GATHER,
    ROLE_ANSWER,
    ROLE_READ,
    SET_TOKEN,
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


def test_compact_special_token_format_removes_structural_spacing(tokenizer, example):
    add_setswitch_special_tokens(tokenizer, None)
    legacy = render_example(
        example,
        tokenizer,
        {
            "num_reads_per_doc": 2,
            "num_gather_tokens": 4,
            "compact_special_token_format": False,
        },
    )
    compact = render_example(
        example,
        tokenizer,
        {
            "num_reads_per_doc": 2,
            "num_gather_tokens": 4,
            "compact_special_token_format": True,
        },
    )

    decoded_prompt = tokenizer.decode(compact["input_ids"][: compact["answer_start"]])

    assert len(compact["input_ids"]) < len(legacy["input_ids"])
    assert f"{SET_TOKEN}{ITEM_TOKEN}" in decoded_prompt
    assert f"{READ_TOKENS[0]}{READ_TOKENS[1]}{END_ITEM_TOKEN}" in decoded_prompt
    assert f"{END_SET_TOKEN}{GATHER_TOKENS[0]}{GATHER_TOKENS[1]}" in decoded_prompt
    assert compact["role_ids"].count(ROLE_READ) == len(example.documents) * 2
    assert compact["role_ids"].count(ROLE_GATHER) == 4


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


def test_setswitch_answer_prefix_is_masked_but_answer_visible(tokenizer, example):
    add_setswitch_special_tokens(tokenizer, None)
    rendered = render_example(
        example,
        tokenizer,
        {
            "num_reads_per_doc": 2,
            "num_gather_tokens": 4,
            "setswitch_answer_prefix": "\nAnswer:\n",
        },
    )

    answer_start = rendered["answer_start"]
    decoded_prompt = tokenizer.decode(rendered["input_ids"][:answer_start])

    assert decoded_prompt.endswith("\nAnswer:\n")
    assert tokenizer.decode(rendered["input_ids"][answer_start:]) == example.answer
    assert all(label == IGNORE_INDEX for label in rendered["labels"][:answer_start])
    prefix_len = len(tokenizer.encode("\nAnswer:\n", add_special_tokens=False))
    assert rendered["role_ids"][answer_start - prefix_len : answer_start] == [
        ROLE_ANSWER
    ] * prefix_len


def test_setswitch_can_render_without_boundary_tokens(tokenizer, example):
    add_setswitch_special_tokens(tokenizer, None)
    rendered = render_example(
        example,
        tokenizer,
        {
            "num_reads_per_doc": 2,
            "num_gather_tokens": 4,
            "setswitch_boundary_tokens": False,
            "compact_special_token_format": True,
        },
    )
    decoded = tokenizer.decode(rendered["input_ids"][: rendered["answer_start"]])

    assert SET_TOKEN not in decoded
    assert ITEM_TOKEN not in decoded
    assert END_SET_TOKEN not in decoded
    assert END_ITEM_TOKEN not in decoded
    assert READ_TOKENS[0] in decoded
    assert GATHER_TOKENS[0] in decoded
    assert rendered["role_ids"].count(ROLE_READ) == len(example.documents) * 2
    assert rendered["role_ids"].count(ROLE_GATHER) == 4
