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
from set_switch.modeling.special_tokens import add_setswitch_special_tokens

_EVALUATE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
_SPEC = importlib.util.spec_from_file_location("set_switch_evaluate_script", _EVALUATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_EVALUATE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVALUATE)
_greedy_setswitch = _EVALUATE._greedy_setswitch
gold_sweep_status = _EVALUATE.gold_sweep_status
load_eval_tokenizer_and_model = _EVALUATE.load_eval_tokenizer_and_model


class RecordingGreedyModel:
    def __init__(self, first_token_id: int, eos_token_id: int, vocab_size: int = 64) -> None:
        self.first_token_id = first_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.position_calls: list[torch.Tensor] = []

    def parameters(self):
        yield self.param

    def __call__(self, **kwargs):
        position_ids = kwargs["position_ids"].detach().cpu()
        self.position_calls.append(position_ids)
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
