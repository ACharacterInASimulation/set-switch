from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from set_switch.data.schema import SetSwitchDocument, SetSwitchExample


class SimpleTokenizer:
    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
        self.inv_vocab = {0: "<pad>", 1: "<eos>", 2: "<bos>"}
        self.additional_special_tokens: list[str] = []
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.bos_token = "<bos>"
        self.bos_token_id = 2

    def __len__(self) -> int:
        return len(self.vocab)

    def _add_token(self, token: str) -> int:
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
        return self.vocab[token]

    def add_special_tokens(self, special_tokens_dict: dict[str, Any]) -> int:
        added = 0
        for token in special_tokens_dict.get("additional_special_tokens", []):
            before = len(self.vocab)
            self._add_token(token)
            if len(self.vocab) > before:
                added += 1
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
        if "pad_token" in special_tokens_dict:
            token = special_tokens_dict["pad_token"]
            before = len(self.vocab)
            token_id = self._add_token(token)
            if len(self.vocab) > before:
                added += 1
            self.pad_token = token
            self.pad_token_id = token_id
        return added

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        pos = 0
        specials = sorted(self.additional_special_tokens, key=len, reverse=True)
        while pos < len(text):
            matched = None
            for special in specials:
                if text.startswith(special, pos):
                    matched = special
                    break
            if matched is not None:
                ids.append(self._add_token(matched))
                pos += len(matched)
                continue

            if text[pos].isspace():
                ids.append(self._add_token(text[pos]))
                pos += 1
                continue

            end = pos + 1
            while end < len(text) and not text[end].isspace():
                if any(text.startswith(special, end) for special in specials):
                    break
                end += 1
            ids.append(self._add_token(text[pos:end]))
            pos = end
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv_vocab[int(idx)] for idx in ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.inv_vocab[int(idx)] for idx in ids]

    def save_pretrained(self, path: str) -> None:
        del path


@dataclass
class TinyModelFactory:
    vocab_size: int

    def make(self) -> LlamaForCausalLM:
        config = LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
        )
        model = LlamaForCausalLM(config)
        model.train()
        return model


@pytest.fixture()
def tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer()


@pytest.fixture()
def example() -> SetSwitchExample:
    return SetSwitchExample(
        example_id="example-0",
        instruction="Use the provided documents to answer the question. Treat the documents as an unordered set.",
        question="What is the launch year of project NARU-17?",
        documents=[
            SetSwitchDocument("d0", "Project LOMA-42 has launch year 1986.", False),
            SetSwitchDocument("d1", "Project NARU-17 has launch year 2004.", True),
            SetSwitchDocument("d2", "Project VELA-09 has launch year 1972.", False),
        ],
        answer="2004",
        source="test",
    )


def make_fixture_example(idx: int = 0) -> SetSwitchExample:
    year = "2004" if idx % 2 == 0 else "1997"
    target = "NARU-17" if idx % 2 == 0 else "KITO-11"
    documents = [
        SetSwitchDocument(f"d{idx}-0", "Project LOMA-42 has launch year 1986.", False),
        SetSwitchDocument(f"d{idx}-1", f"Project {target} has launch year {year}.", True),
        SetSwitchDocument(f"d{idx}-2", "Project VELA-09 has launch year 1972.", False),
    ]
    return SetSwitchExample(
        example_id=f"fixture-{idx}",
        instruction="Use the provided passages or options to answer the question. Treat the items as an unordered set.",
        question=f"What is the launch year of project {target}?",
        documents=documents,
        answer=year,
        source="fixture",
        metadata={"set_type": "documents"},
    )


@pytest.fixture(autouse=True)
def deterministic_torch() -> None:
    torch.manual_seed(0)


@pytest.fixture()
def make_tiny_model():
    def _make(vocab_size: int) -> LlamaForCausalLM:
        return TinyModelFactory(vocab_size=vocab_size).make()

    return _make


@pytest.fixture()
def fixture_example_factory():
    return make_fixture_example
