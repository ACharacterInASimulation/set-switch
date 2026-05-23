from __future__ import annotations

from set_switch.data.schema import SetSwitchDocument, SetSwitchExample
from set_switch.eval.official_metrics import (
    hotpot_f1,
    score_musique_answer_only,
)


def test_hotpot_yes_no_guard_zeroes_non_exact_yes_no():
    f1, precision, recall = hotpot_f1("yes it is", "yes")
    assert (f1, precision, recall) == (0.0, 0.0, 0.0)


def test_alias_max_counts_matching_alias_as_exact():
    example = SetSwitchExample(
        example_id="alias",
        instruction="Use docs.",
        question="Q?",
        documents=[SetSwitchDocument("d0", "NYC is also New York City.", True)],
        answer="New York City",
        source="flashrag_musique",
        metadata={"answer_aliases": ["NYC"]},
    )

    score = score_musique_answer_only(example, "NYC")

    assert score["answer_em"] == 1.0
    assert score["answer_f1"] == 1.0


def test_musique_aliases_from_converted_golden_answers_count():
    example = SetSwitchExample(
        example_id="alias",
        instruction="Use docs.",
        question="Q?",
        documents=[SetSwitchDocument("d0", "NYC is also New York City.", True)],
        answer="New York City",
        source="musique",
        metadata={"golden_answers": ["New York City", "NYC"]},
    )

    score = score_musique_answer_only(example, "NYC")

    assert score["answer_em"] == 1.0
