"""Internal data schema and JSONL helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SetSwitchDocument:
    doc_id: str
    text: str
    is_gold: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SetSwitchExample:
    example_id: str
    instruction: str
    question: str
    documents: list[SetSwitchDocument]
    answer: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def document_from_dict(data: dict[str, Any]) -> SetSwitchDocument:
    return SetSwitchDocument(
        doc_id=str(data["doc_id"]),
        text=str(data["text"]),
        is_gold=bool(data.get("is_gold", False)),
        metadata=dict(data.get("metadata", {})),
    )


def example_from_dict(data: dict[str, Any]) -> SetSwitchExample:
    return SetSwitchExample(
        example_id=str(data["example_id"]),
        instruction=str(data["instruction"]),
        question=str(data["question"]),
        documents=[document_from_dict(doc) for doc in data["documents"]],
        answer=str(data["answer"]),
        source=str(data["source"]),
        metadata=dict(data.get("metadata", {})),
    )


def example_to_dict(example: SetSwitchExample) -> dict[str, Any]:
    return asdict(example)
