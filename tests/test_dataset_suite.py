from __future__ import annotations

from set_switch.data.dataset_suite import (
    FlashRAGSourceSelection,
    allocate_flashrag_source_limits,
    convert_flashrag_row,
    convert_native_musique_row,
    normalize_flashrag_sources,
    task_group_for_source,
)
from set_switch.data.dataset_suite import _limit_documents_prefer_gold, _source_selection_limit
from set_switch.data.schema import SetSwitchDocument


def test_flashrag_context_row_converts_to_documents():
    row = {
        "id": "train_0",
        "question": "Which document has the answer?",
        "golden_answers": ["Document B"],
        "metadata": {
            "type": "bridge",
            "supporting_facts": {"title": ["B"], "sent_id": [0]},
            "context": {
                "title": ["A", "B"],
                "sentences": [["Distractor sentence."], ["Answer sentence."]],
            },
        },
    }

    example = convert_flashrag_row(
        row=row,
        example_idx=0,
        max_docs=8,
        instruction="Use documents.",
        config_name="hotpotqa",
    )

    assert example is not None
    assert example.source == "flashrag_hotpotqa"
    assert example.answer == "Document B"
    assert len(example.documents) == 2
    assert [doc.is_gold for doc in example.documents] == [False, True]
    assert example.metadata["eval_task_group"] == "hotpotqa_bridge"


def test_flashrag_multihop_subtype_metadata_is_reported():
    row = {
        "id": "2wiki_0",
        "question": "Which entity is older?",
        "golden_answers": ["Entity A"],
        "metadata": {
            "type": "bridge_comparison",
            "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
            "context": {
                "title": ["A", "B"],
                "sentences": [["A was born in 1900."], ["B was born in 1910."]],
            },
        },
    }

    example = convert_flashrag_row(
        row=row,
        example_idx=0,
        max_docs=8,
        instruction="Use documents.",
        config_name="2wikimultihopqa",
    )

    assert example is not None
    assert example.metadata["question_type"] == "bridge_comparison"
    assert example.metadata["eval_task_group"] == "2wikimultihopqa_bridge_comparison"


def test_flashrag_msmarco_row_converts_passages():
    row = {
        "id": "train_1",
        "question": "What is the answer?",
        "golden_answers": ["42"],
        "metadata": {
            "passages": {
                "is_selected": [0, 1],
                "passage_text": ["No answer here.", "The answer is 42."],
                "url": ["a", "b"],
            }
        },
    }

    example = convert_flashrag_row(
        row=row,
        example_idx=1,
        max_docs=8,
        instruction="Use documents.",
        config_name="msmarco-qa",
    )

    assert example is not None
    assert example.source == "flashrag_msmarco-qa"
    assert len(example.documents) == 2
    assert [doc.is_gold for doc in example.documents] == [False, True]


def test_flashrag_option_row_uses_choice_text_not_position_label():
    row = {
        "id": "train_2",
        "question": "A person wants to cool down. What should they do?",
        "choices": [
            "stand near a heater",
            "drink water",
            "wear a heavy coat",
            "close the window",
        ],
        "golden_answers": [1],
        "metadata": {},
    }

    example = convert_flashrag_row(
        row=row,
        example_idx=2,
        max_docs=8,
        instruction="Choose the correct answer.",
        config_name="arc",
    )

    assert example is not None
    assert example.metadata["set_type"] == "options"
    assert example.answer == "drink water"
    assert [doc.is_gold for doc in example.documents] == [False, True, False, False]
    assert all("Option A" not in doc.text and "A)" not in doc.text for doc in example.documents)
    assert example.metadata["task_group"] == "normal_mcq"


def test_flashrag_option_row_keeps_zero_index_answer():
    row = {
        "id": "train_zero",
        "question": "Which choice is first?",
        "choices": ["alpha", "beta", "gamma"],
        "golden_answers": [0],
        "metadata": {},
    }

    example = convert_flashrag_row(
        row=row,
        example_idx=0,
        max_docs=8,
        instruction="Choose the correct answer.",
        config_name="openbookqa",
    )

    assert example is not None
    assert example.answer == "alpha"
    assert [doc.is_gold for doc in example.documents] == [True, False, False]


def test_flashrag_quartz_uses_metadata_choice_texts_not_labels():
    row = {
        "id": "quartz-0",
        "question": "As population increases, water availability is",
        "choices": ["A", "B"],
        "golden_answers": [0],
        "metadata": {
            "answerKey": "A",
            "choices": [
                {"label": "A", "text": "scarce"},
                {"label": "B", "text": "plentiful"},
            ],
        },
    }

    example = convert_flashrag_row(
        row=row,
        example_idx=0,
        max_docs=8,
        instruction="Choose the correct answer.",
        config_name="quartz",
    )

    assert example is not None
    assert example.answer == "scarce"
    assert [doc.metadata["choice_text"] for doc in example.documents] == ["scarce", "plentiful"]
    assert [doc.is_gold for doc in example.documents] == [True, False]


def test_flashrag_squad_and_boolq_convert_single_passages():
    squad = convert_flashrag_row(
        row={
            "id": "squad-0",
            "question": "Who appeared?",
            "golden_answers": ["Saint Bernadette"],
            "metadata": {"title": "Lourdes", "text": "Mary appeared to Saint Bernadette."},
        },
        example_idx=3,
        max_docs=8,
        instruction="Use documents.",
        config_name="squad",
    )
    boolq = convert_flashrag_row(
        row={
            "id": "boolq-0",
            "question": "is this true",
            "golden_answers": [True],
            "metadata": {"passage": "The passage supports yes."},
        },
        example_idx=4,
        max_docs=8,
        instruction="Use documents.",
        config_name="boolq",
    )

    assert squad is not None
    assert squad.documents[0].is_gold
    assert squad.metadata["task_group"] == "rag_single_hop"
    assert boolq is not None
    assert boolq.answer == "yes"
    assert boolq.documents[0].is_gold
    assert task_group_for_source("flashrag_boolq") == "rag_single_hop"


def test_native_musique_uses_all_paragraphs_and_marks_support():
    example = convert_native_musique_row(
        row={
            "id": "2hop__fixture",
            "question": "When was the owner founded?",
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Distractor",
                    "paragraph_text": "This paragraph is unrelated.",
                    "is_supporting": False,
                },
                {
                    "idx": 5,
                    "title": "The Collegian",
                    "paragraph_text": "The Collegian is owned by Houston Baptist University.",
                    "is_supporting": True,
                },
                {
                    "idx": 9,
                    "title": "Houston",
                    "paragraph_text": "Houston Baptist University was founded in 1960.",
                    "is_supporting": True,
                },
            ],
            "question_decomposition": [
                {"question": "The Collegian >> owned by", "paragraph_support_idx": 5},
                {"question": "Houston Baptist University >> founded", "paragraph_support_idx": 9},
            ],
            "answer": "1960",
            "answer_aliases": ["nineteen sixty"],
            "answerable": True,
        },
        example_idx=0,
        max_docs=8,
        instruction="Use documents.",
    )

    assert example is not None
    assert example.source == "musique"
    assert [doc.is_gold for doc in example.documents] == [False, True, True]
    assert example.metadata["golden_answers"] == ["1960", "nineteen sixty"]
    assert example.metadata["context_policy"] == "native_all_paragraphs_with_distractors"
    assert example.metadata["eval_task_group"] == "musique_2hop"


def test_document_limit_keeps_gold_without_moving_gold_to_front():
    docs = [
        SetSwitchDocument("n0", "non-gold 0", False),
        SetSwitchDocument("n1", "non-gold 1", False),
        SetSwitchDocument("g2", "gold 2", True),
        SetSwitchDocument("n3", "non-gold 3", False),
        SetSwitchDocument("g4", "gold 4", True),
    ]

    limited = _limit_documents_prefer_gold(docs, max_docs=3)

    assert [doc.doc_id for doc in limited] == ["n0", "g2", "g4"]


def test_percent_selection_uses_known_flashrag_count():
    limit = _source_selection_limit(
        FlashRAGSourceSelection(name="hotpotqa", split="train", percent=0.1),
        split="train",
    )

    assert limit == 9044


def test_compact_percentage_source_syntax():
    selections = normalize_flashrag_sources(
        {"datasets": ["HotpotQA[:0.5]", "mmlu[dev:10%]", "msmarco[:0.01]"]},
        split="train",
    )

    assert selections[0] == FlashRAGSourceSelection(name="hotpotqa", split="train", percent=0.5)
    assert selections[1] == FlashRAGSourceSelection(name="mmlu", split="dev", percent=0.1)
    assert selections[2] == FlashRAGSourceSelection(name="msmarco-qa", split="train", percent=0.01)


def test_split_specific_flashrag_limits_only_affect_requested_split():
    train_selection = normalize_flashrag_sources(
        {"datasets": [{"name": "hotpotqa", "train_max_examples": 6000}]},
        split="train",
    )
    dev_selection = normalize_flashrag_sources(
        {"datasets": [{"name": "hotpotqa", "train_max_examples": 6000}]},
        split="dev",
    )

    assert train_selection[0].max_examples == 6000
    assert dev_selection[0].max_examples is None


def test_test_split_selection_skips_sources_without_labeled_test_split():
    selections = normalize_flashrag_sources(
        {
            "datasets": [
                "commonsenseqa",
                "openbookqa",
                "arc",
                "hellaswag",
                "mmlu",
                "quartz",
                "hotpotqa",
            ]
        },
        split="test",
    )

    assert [selection.name for selection in selections] == [
        "openbookqa",
        "arc",
        "mmlu",
        "quartz",
    ]
    assert all(selection.split == "test" for selection in selections)


def test_task_balanced_equal_allocation_prevents_msmarco_domination():
    selections = normalize_flashrag_sources(
        {
            "datasets": [
                "commonsenseqa",
                "openbookqa",
                "arc",
                "hellaswag",
                "mmlu",
                "quartz",
                "msmarco-qa",
                "squad",
                "boolq",
                "hotpotqa",
                "2wikimultihopqa",
                "musique",
                "ambig_qa",
            ]
        },
        split="train",
    )

    limits = allocate_flashrag_source_limits(
        selections,
        total_examples=100_000,
        strategy="task_balanced_equal",
    )
    by_name = {selection.name: limit for selection, limit in zip(selections, limits, strict=True)}

    assert sum(limit or 0 for limit in limits) == 100_000
    assert by_name["ambig_qa"] == 10_036
    assert by_name["boolq"] == 9_427
    assert abs(by_name["msmarco-qa"] - by_name["squad"]) <= 1
    assert by_name["msmarco-qa"] < 11_000
