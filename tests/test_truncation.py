from __future__ import annotations

from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchDocument, SetSwitchExample
from set_switch.data.truncation import truncate_text_by_tokens
from set_switch.modeling.special_tokens import add_setswitch_special_tokens


def test_answer_window_truncation_keeps_answer(tokenizer):
    text = " ".join([f"left{i}" for i in range(30)] + ["needle"] + [f"right{i}" for i in range(30)])

    ids = truncate_text_by_tokens(
        tokenizer,
        text,
        max_tokens=12,
        answer_texts=["needle"],
        prefer_answer_window=True,
    )

    decoded = tokenizer.decode(ids)
    assert "needle" in decoded
    assert "left0" not in decoded


def test_setswitch_gold_document_truncates_around_answer(tokenizer):
    add_setswitch_special_tokens(tokenizer, None)
    long_gold = " ".join(
        [f"before{i}" for i in range(40)]
        + ["project-answer-777"]
        + [f"after{i}" for i in range(40)]
    )
    example = SetSwitchExample(
        example_id="truncation",
        instruction="Use documents.",
        question="What is the answer?",
        documents=[SetSwitchDocument("gold", long_gold, is_gold=True)],
        answer="project-answer-777",
        source="test",
        metadata={"set_type": "documents", "golden_answers": ["project-answer-777"]},
    )

    rendered = render_example(
        example,
        tokenizer,
        {"max_doc_tokens": 16, "num_reads_per_doc": 1, "num_gather_tokens": 1},
    )
    decoded = tokenizer.decode(rendered["input_ids"])

    assert "project-answer-777" in decoded
    assert "before0" not in decoded
