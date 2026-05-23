from __future__ import annotations

from set_switch.data.length_filter import (
    filter_examples_by_rendered_length,
    max_rendered_length,
    normalize_length_filter_interfaces,
)
from set_switch.data.schema import SetSwitchDocument, SetSwitchExample
from set_switch.modeling.special_tokens import add_setswitch_special_tokens


def test_length_filter_uses_rendered_lengths(tokenizer):
    add_setswitch_special_tokens(tokenizer, None)
    short = SetSwitchExample(
        example_id="short",
        instruction="Use docs.",
        question="Q?",
        documents=[SetSwitchDocument("d0", "short doc", True)],
        answer="A",
        source="fixture",
        metadata={"set_type": "documents"},
    )
    long = SetSwitchExample(
        example_id="long",
        instruction="Use docs.",
        question="Q?",
        documents=[SetSwitchDocument("d0", " ".join(["word"] * 200), True)],
        answer="A",
        source="fixture",
        metadata={"set_type": "documents"},
    )
    cfg = {"data": {"num_reads_per_doc": 2, "num_gather_tokens": 4}}

    short_length = max_rendered_length(short, tokenizer, cfg, ("setswitch",))
    kept, dropped = filter_examples_by_rendered_length(
        [short, long],
        tokenizer=tokenizer,
        cfg=cfg,
        max_tokens=short_length + 5,
        interfaces=("setswitch",),
    )

    assert [example.example_id for example in kept] == ["short"]
    assert dropped[0]["example_id"] == "long"
    assert dropped[0]["rendered_length"] > short_length


def test_normalize_length_filter_interfaces():
    assert normalize_length_filter_interfaces("all") == (
        "setswitch",
        "setllm",
        "chat_baseline",
    )
    assert normalize_length_filter_interfaces("setswitch,setllm") == ("setswitch", "setllm")
