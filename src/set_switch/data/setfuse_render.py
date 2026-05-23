"""SetFuse-LM prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from set_switch.constants import (
    IGNORE_INDEX,
    ROLE_ANSWER,
    ROLE_DOC,
    ROLE_ITEM_SPECIAL,
    ROLE_PREFIX,
)
from set_switch.data.schema import SetSwitchExample
from set_switch.data.truncation import answer_candidates, truncate_text_by_tokens
from set_switch.modeling.position_ids import build_position_ids


@dataclass(frozen=True)
class SetFuseRenderConfig:
    max_doc_tokens: int | None = None
    doc_truncation: str = "answer_window"
    add_bos_token: bool = False
    append_eos_token: bool = False
    answer_prefix: str = ""
    document_prefix: str = "\nPassage:\n"
    document_suffix: str = "\n"


def setfuse_render_config_from_obj(cfg: Any | None) -> SetFuseRenderConfig:
    if cfg is None:
        return SetFuseRenderConfig()
    if isinstance(cfg, SetFuseRenderConfig):
        return cfg
    data = cfg.get("data", cfg) if isinstance(cfg, dict) else cfg

    def get(name: str, default: Any) -> Any:
        if isinstance(data, dict):
            return data.get(name, default)
        return getattr(data, name, default)

    return SetFuseRenderConfig(
        max_doc_tokens=get("max_doc_tokens", None),
        doc_truncation=str(get("doc_truncation", "answer_window")),
        add_bos_token=bool(get("add_bos_token", False)),
        append_eos_token=bool(get("append_eos_token", False)),
        answer_prefix=str(get("setfuse_answer_prefix", get("answer_prefix", ""))),
        document_prefix=str(get("setfuse_document_prefix", "\nPassage:\n")),
        document_suffix=str(get("setfuse_document_suffix", "\n")),
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _append(
    input_ids: list[int],
    role_ids: list[int],
    item_ids: list[int],
    ids: list[int],
    role: int,
    item_id: int = -1,
) -> None:
    input_ids.extend(ids)
    role_ids.extend([role] * len(ids))
    item_ids.extend([item_id] * len(ids))


def _render_prefix(example: SetSwitchExample) -> str:
    return f"Instruction: {example.instruction}\n\nQuestion: {example.question}\n\n"


def render_setfuse_example(
    example: SetSwitchExample,
    tokenizer: Any,
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Render one SetFuse-LM example without read/gather aggregation tokens."""

    rcfg = setfuse_render_config_from_obj(cfg)
    input_ids: list[int] = []
    role_ids: list[int] = []
    item_ids: list[int] = []
    doc_lengths: dict[int, int] = {}

    if rcfg.add_bos_token and tokenizer.bos_token_id is not None:
        _append(input_ids, role_ids, item_ids, [int(tokenizer.bos_token_id)], ROLE_PREFIX)

    prefix_ids = _encode(tokenizer, _render_prefix(example))
    _append(input_ids, role_ids, item_ids, prefix_ids, ROLE_PREFIX)
    prefix_length = len(input_ids)

    answers = answer_candidates(example.answer, example.metadata)
    document_prefix_ids = _encode(tokenizer, rcfg.document_prefix)
    document_suffix_ids = _encode(tokenizer, rcfg.document_suffix)
    for doc_idx, doc in enumerate(example.documents):
        _append(
            input_ids,
            role_ids,
            item_ids,
            document_prefix_ids,
            ROLE_ITEM_SPECIAL,
            item_id=doc_idx,
        )

        doc_ids = truncate_text_by_tokens(
            tokenizer=tokenizer,
            text=doc.text.rstrip(),
            max_tokens=rcfg.max_doc_tokens,
            answer_texts=answers,
            prefer_answer_window=rcfg.doc_truncation == "answer_window" and doc.is_gold,
        )
        _append(input_ids, role_ids, item_ids, doc_ids, ROLE_DOC, item_id=doc_idx)
        doc_lengths[doc_idx] = len(doc_ids)

        _append(
            input_ids,
            role_ids,
            item_ids,
            document_suffix_ids,
            ROLE_ITEM_SPECIAL,
            item_id=doc_idx,
        )

    if rcfg.answer_prefix:
        _append(input_ids, role_ids, item_ids, _encode(tokenizer, rcfg.answer_prefix), ROLE_PREFIX)

    answer_start = len(input_ids)
    answer_ids = _encode(tokenizer, example.answer)
    if rcfg.append_eos_token and getattr(tokenizer, "eos_token_id", None) is not None:
        answer_ids = answer_ids + [int(tokenizer.eos_token_id)]
    if not answer_ids:
        raise ValueError(f"Example {example.example_id} has an empty tokenized answer")
    _append(input_ids, role_ids, item_ids, answer_ids, ROLE_ANSWER)

    labels = [IGNORE_INDEX] * answer_start + list(answer_ids)
    max_doc_length = max(doc_lengths.values(), default=0)
    position_ids = build_position_ids(
        role_ids=role_ids,
        item_ids=item_ids,
        prefix_length=prefix_length,
        max_doc_length=max_doc_length,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "role_ids": role_ids,
        "item_ids": item_ids,
        "position_ids": position_ids,
        "answer_start": answer_start,
        "prefix_length": prefix_length,
        "max_doc_length": max_doc_length,
        "attention_metadata": {"doc_lengths": doc_lengths},
        "example_id": example.example_id,
        "interface": "setfuse",
    }
