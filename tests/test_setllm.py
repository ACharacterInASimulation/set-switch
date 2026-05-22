from __future__ import annotations

import torch

from set_switch.constants import IGNORE_INDEX, ROLE_ANSWER, ROLE_DOC, ROLE_PREFIX
from set_switch.data.schema import SetSwitchDocument, SetSwitchExample
from set_switch.data.setllm_collator import SetLLMCollator
from set_switch.data.setllm_render import render_setllm_example


def _option_example() -> SetSwitchExample:
    return SetSwitchExample(
        example_id="setllm-options",
        instruction="Use the provided options.",
        question="Which option is correct?",
        documents=[
            SetSwitchDocument("a", "Candidate answer: red", False, {"choice_text": "red"}),
            SetSwitchDocument("b", "Candidate answer: blue", True, {"choice_text": "blue"}),
        ],
        answer="blue",
        source="fixture",
        metadata={"set_type": "options"},
    )


def test_setllm_uses_modified_unnumbered_choice_prompt(tokenizer):
    rendered = render_setllm_example(_option_example(), tokenizer)
    decoded = tokenizer.decode(rendered["input_ids"])

    assert decoded.startswith(
        "Instruction: Use the provided options.\n\n"
        "Question: Which option is correct?\n\nChoices:\n"
    )
    assert "\nred\nblue\n\nAnswer:\nblue" in decoded
    assert "<set>" not in decoded
    assert "Option A" not in decoded
    assert "Candidate answer:" not in decoded
    assert all(label == IGNORE_INDEX for label in rendered["labels"][: rendered["answer_start"]])
    assert any(label != IGNORE_INDEX for label in rendered["labels"][rendered["answer_start"] :])


def test_setllm_setpe_resets_each_item_start(tokenizer):
    rendered = render_setllm_example(_option_example(), tokenizer)
    item0 = [idx for idx, item in enumerate(rendered["item_ids"]) if item == 0]
    item1 = [idx for idx, item in enumerate(rendered["item_ids"]) if item == 1]

    assert item0
    assert item1
    assert rendered["position_ids"][item0[0]] == rendered["position_ids"][item1[0]]
    assert rendered["position_ids"][item0[1]] == rendered["position_ids"][item0[0]] + 1
    assert rendered["role_ids"][item0[0]] == ROLE_DOC
    assert rendered["role_ids"][0] == ROLE_PREFIX
    assert rendered["role_ids"][rendered["answer_start"]] == ROLE_ANSWER


def test_setllm_setmask_blocks_cross_item_edges(tokenizer):
    rendered = render_setllm_example(_option_example(), tokenizer)
    batch = SetLLMCollator(tokenizer)([rendered])
    allowed = batch["attention_mask"][0, 0] == 0

    item0 = [idx for idx, item in enumerate(rendered["item_ids"]) if item == 0]
    item1 = [idx for idx, item in enumerate(rendered["item_ids"]) if item == 1]
    answer = rendered["answer_start"]

    assert not bool(allowed[item0[0], item1[0]])
    assert not bool(allowed[item1[0], item0[0]])
    assert bool(allowed[answer, item0[0]])
    assert bool(allowed[answer, item1[0]])
    assert not bool(allowed[0, answer])
    assert bool(torch.all(batch["labels"][batch["pad_mask"] == 0] == IGNORE_INDEX))


def test_setllm_padding_rows_have_defined_attention(tokenizer):
    short = _option_example()
    long = _option_example()
    long.documents[1] = SetSwitchDocument(
        "b",
        "Candidate answer: blue",
        True,
        {"choice_text": "blue with a much longer explanatory option"},
    )
    batch = SetLLMCollator(tokenizer)(
        [render_setllm_example(short, tokenizer), render_setllm_example(long, tokenizer)]
    )
    allowed = batch["attention_mask"][:, 0] == 0

    assert allowed.any(dim=-1).all()
    for batch_idx in range(allowed.shape[0]):
        valid = batch["pad_mask"][batch_idx]
        assert not allowed[batch_idx][valid][:, ~valid].any()
