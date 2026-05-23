from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import torch

from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchDocument, SetSwitchExample
from set_switch.data.setfuse_render import render_setfuse_example
from set_switch.data.baseline_render import render_chat_baseline_example
from set_switch.modeling.special_tokens import add_setswitch_special_tokens

_EVALUATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
_SPEC = importlib.util.spec_from_file_location("set_switch_evaluate_script", _EVALUATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_EVALUATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVALUATE)
_greedy_setswitch = _EVALUATE._greedy_setswitch
_greedy_setfuse = _EVALUATE._greedy_setfuse
gold_sweep_status = _EVALUATE.gold_sweep_status
gold_position_sweep_enabled = _EVALUATE.gold_position_sweep_enabled
option_permutation_sweep_enabled = _EVALUATE.option_permutation_sweep_enabled
permute_option_documents = _EVALUATE.permute_option_documents
_include_in_reported_summary = _EVALUATE._include_in_reported_summary
_majority_prediction = _EVALUATE._majority_prediction
_summarize_option_counts = _EVALUATE._summarize_option_counts
_update_option_summary_count = _EVALUATE._update_option_summary_count
load_eval_tokenizer_and_model = _EVALUATE.load_eval_tokenizer_and_model
score_prediction = _EVALUATE.score_prediction
score_mcq_options = _EVALUATE.score_mcq_options
_load_eval_examples = _EVALUATE._load_eval_examples
_parse_max_examples = _EVALUATE._parse_max_examples


class RecordingGreedyModel:
    def __init__(self, first_token_id: int, eos_token_id: int, vocab_size: int = 64) -> None:
        self.first_token_id = first_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.position_calls: list[torch.Tensor] = []
        self.attention_mask_calls: list[torch.Tensor] = []

    def parameters(self):
        yield self.param

    def __call__(self, **kwargs):
        position_ids = kwargs["position_ids"].detach().cpu()
        self.position_calls.append(position_ids)
        self.attention_mask_calls.append(kwargs["attention_mask"].detach().cpu())
        seq_len = int(kwargs["input_ids"].shape[1])
        logits = torch.zeros((1, seq_len, self.vocab_size), device=kwargs["input_ids"].device)
        next_id = self.first_token_id if len(self.position_calls) == 1 else self.eos_token_id
        logits[0, -1, next_id] = 1.0
        return SimpleNamespace(logits=logits)


def test_setswitch_greedy_generation_uses_training_answer_positions(tokenizer, example):
    add_setswitch_special_tokens(tokenizer, None)
    rendered = render_example(example, tokenizer, {"num_reads_per_doc": 2, "num_gather_tokens": 4})
    first_answer_pos = rendered["position_ids"][rendered["answer_start"]]
    assert first_answer_pos != rendered["prefix_length"]

    model = RecordingGreedyModel(first_token_id=2, eos_token_id=int(tokenizer.eos_token_id))

    _greedy_setswitch(
        model=model,
        tokenizer=tokenizer,
        rendered=rendered,
        cfg={"mask": {"doc_attention": "doc_causal"}},
        max_new_tokens=2,
    )

    assert len(model.position_calls) == 2
    assert int(model.position_calls[1][0, -1]) == first_answer_pos


def test_setfuse_greedy_generation_rebuilds_masks_and_decodes_answer_only(tokenizer, example):
    rendered = render_setfuse_example(example, tokenizer)
    first_answer_pos = rendered["position_ids"][rendered["answer_start"]]
    model = RecordingGreedyModel(first_token_id=2, eos_token_id=int(tokenizer.eos_token_id))

    prediction = _greedy_setfuse(
        model=model,
        tokenizer=tokenizer,
        rendered=rendered,
        cfg={"mask": {"fuse_start_layer": "auto_half"}},
        max_new_tokens=2,
    )

    assert prediction == tokenizer.decode([2])
    assert len(model.position_calls) == 2
    assert model.attention_mask_calls[0].shape[-1] == rendered["answer_start"]
    assert model.attention_mask_calls[1].shape[-1] == rendered["answer_start"] + 1
    assert int(model.position_calls[1][0, -1]) == first_answer_pos


def test_gold_sweep_status_reports_unmovable_examples():
    example = SetSwitchExample(
        example_id="no-gold",
        instruction="Use docs.",
        question="Q?",
        documents=[
            SetSwitchDocument("d0", "doc 0", False),
            SetSwitchDocument("d1", "doc 1", False),
        ],
        answer="A",
        source="fixture",
    )

    assert gold_sweep_status(example) == {
        "num_gold_documents": 0,
        "num_non_gold_documents": 2,
        "gold_sweep_movable": False,
    }


def test_only_causal_baseline_uses_explicit_gold_position_sweep():
    assert gold_position_sweep_enabled("chat_baseline") is True
    assert gold_position_sweep_enabled("setllm") is False
    assert gold_position_sweep_enabled("setswitch") is False
    assert gold_position_sweep_enabled("setfuse") is False


def test_only_causal_baseline_uses_option_permutation_sweep():
    assert option_permutation_sweep_enabled("chat_baseline") is True
    assert option_permutation_sweep_enabled("setllm") is False
    assert option_permutation_sweep_enabled("setswitch") is False
    assert option_permutation_sweep_enabled("setfuse") is False


def test_invariant_reported_summary_counts_each_example_once():
    included = [
        _include_in_reported_summary(
            row_index=idx,
            sweep_gold_positions=False,
            sweep_option_order=False,
        )
        for idx in range(5)
    ]
    assert included == [True, False, False, False, False]


def test_option_permutation_is_seeded_and_preserves_choices():
    example = SetSwitchExample(
        example_id="mcq",
        instruction="Use docs.",
        question="Q?",
        documents=[
            SetSwitchDocument("d0", "Candidate answer: a", False, {"choice_index": 0}),
            SetSwitchDocument("d1", "Candidate answer: b", True, {"choice_index": 1}),
            SetSwitchDocument("d2", "Candidate answer: c", False, {"choice_index": 2}),
            SetSwitchDocument("d3", "Candidate answer: d", False, {"choice_index": 3}),
        ],
        answer="b",
        source="flashrag_arc",
        metadata={"set_type": "options", "golden_answers": ["b"]},
    )

    first = permute_option_documents(example, permutation_index=0, seed=7)
    second = permute_option_documents(example, permutation_index=0, seed=7)

    assert [doc.doc_id for doc in first.documents] == [doc.doc_id for doc in second.documents]
    assert sorted(doc.doc_id for doc in first.documents) == ["d0", "d1", "d2", "d3"]


def test_option_permutation_summary_reports_majority_vote_accuracy():
    counts = {"overall": _EVALUATE._empty_option_summary_count()}
    scores = [
        {"primary_score": 1.0, "correct": True},
        {"primary_score": 0.0, "correct": False},
        {"primary_score": 1.0, "correct": True},
        {"primary_score": 0.0, "correct": False},
    ]
    majority_score = {"primary_score": 1.0, "correct": True}

    _update_option_summary_count(counts, "overall", scores, majority_score)
    summary = _summarize_option_counts(counts)

    assert _majority_prediction(["blue", "red", "blue"]) == "blue"
    assert summary["overall"]["permutation_accuracy_mean"] == 0.5
    assert summary["overall"]["majority_vote_accuracy"] == 1.0
    assert summary["overall"]["any_permutation_accuracy"] == 1.0
    assert summary["overall"]["all_permutations_accuracy"] == 0.0


def test_mcq_logprob_scoring_selects_best_length_normalized_option(tokenizer):
    example = SetSwitchExample(
        example_id="mcq",
        instruction="Use docs.",
        question="Q?",
        documents=[
            SetSwitchDocument("d0", "Candidate answer: red", False, {"choice_text": "red"}),
            SetSwitchDocument("d1", "Candidate answer: blue", True, {"choice_text": "blue"}),
        ],
        answer="blue",
        source="flashrag_arc",
        metadata={"set_type": "options", "golden_answers": ["blue"]},
    )
    rendered = render_chat_baseline_example(example, tokenizer)
    blue_id = tokenizer.encode("blue", add_special_tokens=False)[0]

    class BlueModel:
        def __init__(self):
            self.param = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            yield self.param

        def __call__(self, **kwargs):
            seq_len = int(kwargs["input_ids"].shape[1])
            logits = torch.zeros((1, seq_len, 128), device=kwargs["input_ids"].device)
            logits[:, :, blue_id] = 5.0
            return SimpleNamespace(logits=logits)

    result = score_mcq_options(
        interface="chat_baseline",
        model=BlueModel(),
        tokenizer=tokenizer,
        rendered=rendered,
        option_texts=["red", "blue"],
        cfg={"model": {}, "train": {}},
    )

    assert result["prediction"] == "blue"
    assert result["score_key"] == "length_normalized_logprob"


def test_eval_loads_peft_adapter_checkpoint_from_base_model(monkeypatch, tmp_path):
    adapter_config = {"base_model_name_or_path": "base-model-from-adapter"}
    (tmp_path / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    calls: dict[str, object] = {}

    def fake_load_tokenizer_and_model(model_cfg, add_setswitch_tokens):
        calls["model_cfg"] = dict(model_cfg)
        calls["add_setswitch_tokens"] = add_setswitch_tokens
        return "tokenizer", "base-model"

    fake_peft = ModuleType("peft")

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, checkpoint):
            calls["adapter_model"] = model
            calls["adapter_checkpoint"] = checkpoint
            return "adapter-wrapped-model"

    fake_peft.PeftModel = FakePeftModel
    monkeypatch.setattr(_EVALUATE, "load_tokenizer_and_model", fake_load_tokenizer_and_model)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    tokenizer, model = load_eval_tokenizer_and_model(
        {"model": {"name_or_path": "config-base"}},
        interface="setllm",
        checkpoint=str(tmp_path),
    )

    assert tokenizer == "tokenizer"
    assert model == "adapter-wrapped-model"
    assert calls["model_cfg"]["name_or_path"] == "base-model-from-adapter"
    assert calls["add_setswitch_tokens"] is False
    assert calls["adapter_model"] == "base-model"
    assert calls["adapter_checkpoint"] == str(tmp_path)


def test_eval_loads_fixed_dev_jsonl(tmp_path, example):
    path = tmp_path / "dev.jsonl"
    path.write_text(
        json.dumps(
            {
                "example_id": example.example_id,
                "instruction": example.instruction,
                "question": example.question,
                "documents": [
                    {
                        "doc_id": doc.doc_id,
                        "text": doc.text,
                        "is_gold": doc.is_gold,
                        "metadata": doc.metadata,
                    }
                    for doc in example.documents
                ],
                "answer": example.answer,
                "source": example.source,
                "metadata": example.metadata,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = _load_eval_examples({"data": {"dev_jsonl": str(path)}}, split="dev", max_examples=1)

    assert [item.example_id for item in loaded] == [example.example_id]


def test_eval_max_examples_all_is_uncapped():
    assert _parse_max_examples("all") is None
    assert _parse_max_examples(None) is None
    assert _parse_max_examples("7") == 7


def test_eval_all_ignores_total_val_examples_cap(monkeypatch):
    calls = {}

    def fake_normalize(data_cfg, split):
        calls["data_cfg"] = data_cfg
        calls["split"] = split
        return [SimpleNamespace(name="hotpotqa", split="dev", max_examples=None)]

    def fake_load(**kwargs):
        calls["total_examples"] = kwargs["total_examples"]
        calls["verbose"] = kwargs["verbose"]
        return []

    monkeypatch.setattr(_EVALUATE, "normalize_flashrag_sources", fake_normalize)
    monkeypatch.setattr(_EVALUATE, "load_flashrag_selected_examples", fake_load)

    loaded = _load_eval_examples(
        {"data": {"total_val_examples": 1000, "datasets": ["hotpotqa"]}},
        split="dev",
        max_examples=None,
        verbose=True,
    )

    assert loaded == []
    assert calls["total_examples"] is None
    assert calls["verbose"] is True


def test_eval_uses_accuracy_for_options_and_f1_for_qa():
    option_example = SetSwitchExample(
        example_id="mcq",
        instruction="Use docs.",
        question="Q?",
        documents=[SetSwitchDocument("d0", "Candidate answer: blue", True)],
        answer="blue",
        source="flashrag_commonsenseqa",
        metadata={"set_type": "options", "golden_answers": ["blue"]},
    )
    qa_example = SetSwitchExample(
        example_id="qa",
        instruction="Use docs.",
        question="Q?",
        documents=[SetSwitchDocument("d0", "Paris is in France.", True)],
        answer="Paris France",
        source="flashrag_squad",
        metadata={"golden_answers": ["Paris France"]},
    )

    option_score = score_prediction(option_example, "blue")
    qa_score = score_prediction(qa_example, "Paris")

    assert option_score["primary_metric"] == "accuracy"
    assert option_score["primary_score"] == 1.0
    assert qa_score["primary_metric"] == "token_f1"
    assert 0.0 < qa_score["primary_score"] < 1.0


def test_eval_uses_rouge_l_primary_for_msmarco_qa():
    example = SetSwitchExample(
        example_id="msmarco",
        instruction="Use docs.",
        question="Q?",
        documents=[SetSwitchDocument("d0", "The answer is a blue car.", True)],
        answer="a blue car",
        source="flashrag_msmarco-qa",
        metadata={"golden_answers": ["a blue car"]},
    )

    score = score_prediction(example, "blue")

    assert score["primary_metric"] == "rouge_l"
    assert score["primary_score"] == score["rouge_l"]
    assert 0.0 < score["primary_score"] < 1.0


def test_exact_match_is_strict_after_normalization():
    example = SetSwitchExample(
        example_id="qa",
        instruction="Use docs.",
        question="Q?",
        documents=[SetSwitchDocument("d0", "Paris is in France.", True)],
        answer="blue",
        source="flashrag_hotpotqa",
        metadata={"golden_answers": ["blue"]},
    )

    exact_score = score_prediction(example, "blue")
    extra_text_score = score_prediction(example, "blue car")

    assert exact_score["exact_match"] == 1.0
    assert extra_text_score["exact_match"] == 0.0
    assert extra_text_score["token_f1"] > 0.0
