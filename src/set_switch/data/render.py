"""Canonical SetSwitch example rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from set_switch.constants import (
    DEFAULT_NUM_GATHER_TOKENS,
    DEFAULT_NUM_READS_PER_DOC,
    END_ITEM_TOKEN,
    END_SET_TOKEN,
    GATHER_TOKENS,
    IGNORE_INDEX,
    ITEM_TOKEN,
    READ_TOKENS,
    ROLE_ANSWER,
    ROLE_DOC,
    ROLE_GATHER,
    ROLE_ITEM_SPECIAL,
    ROLE_PREFIX,
    ROLE_READ,
    ROLE_SET_SPECIAL,
    SET_TOKEN,
)
from set_switch.data.schema import SetSwitchExample
from set_switch.data.truncation import answer_candidates, truncate_text_by_tokens
from set_switch.modeling.position_ids import build_position_ids


@dataclass(frozen=True)
class RenderConfig:
    num_reads_per_doc: int = DEFAULT_NUM_READS_PER_DOC
    num_gather_tokens: int = DEFAULT_NUM_GATHER_TOKENS
    max_doc_tokens: int | None = None
    doc_truncation: str = "answer_window"
    add_bos_token: bool = False
    append_eos_token: bool = False


def render_config_from_obj(cfg: Any | None) -> RenderConfig:
    if cfg is None:
        return RenderConfig()
    if isinstance(cfg, RenderConfig):
        return cfg
    data = cfg.get("data", cfg) if isinstance(cfg, dict) else cfg

    def get(name: str, default: Any) -> Any:
        if isinstance(data, dict):
            return data.get(name, default)
        return getattr(data, name, default)

    return RenderConfig(
        num_reads_per_doc=int(get("num_reads_per_doc", DEFAULT_NUM_READS_PER_DOC)),
        num_gather_tokens=int(get("num_gather_tokens", DEFAULT_NUM_GATHER_TOKENS)),
        max_doc_tokens=get("max_doc_tokens", None),
        doc_truncation=str(get("doc_truncation", "answer_window")),
        add_bos_token=bool(get("add_bos_token", False)),
        append_eos_token=bool(get("append_eos_token", False)),
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _append(
    input_ids: list[int],
    role_ids: list[int],
    item_ids: list[int],
    read_slot_ids: list[int],
    gather_slot_ids: list[int],
    ids: list[int],
    role: int,
    item_id: int = -1,
    read_slot_id: int = -1,
    gather_slot_id: int = -1,
) -> None:
    input_ids.extend(ids)
    role_ids.extend([role] * len(ids))
    item_ids.extend([item_id] * len(ids))
    read_slot_ids.extend([read_slot_id] * len(ids))
    gather_slot_ids.extend([gather_slot_id] * len(ids))


def _special_id(tokenizer: Any, token: str) -> int:
    ids = _encode(tokenizer, token)
    if len(ids) != 1:
        raise ValueError(f"Special token {token!r} must encode to one id, got {ids}")
    return int(ids[0])


def _render_prefix(example: SetSwitchExample) -> str:
    return f"Instruction: {example.instruction}\n\nQuestion: {example.question}\n\n"


def render_prompt_text(
    example: SetSwitchExample,
    num_reads_per_doc: int = DEFAULT_NUM_READS_PER_DOC,
    num_gather_tokens: int = DEFAULT_NUM_GATHER_TOKENS,
) -> str:
    """Render human-readable canonical prompt text, without the answer."""

    lines = [_render_prefix(example), f"{SET_TOKEN}\n"]
    for doc in example.documents:
        lines.append(f"{ITEM_TOKEN}\n")
        lines.append(doc.text.rstrip() + "\n")
        lines.append(" ".join(READ_TOKENS[:num_reads_per_doc]) + "\n")
        lines.append(f"{END_ITEM_TOKEN}\n")
    lines.append(f"{END_SET_TOKEN}\n")
    lines.append(" ".join(GATHER_TOKENS[:num_gather_tokens]) + "\n\n")
    return "".join(lines)


def render_example(
    example: SetSwitchExample, tokenizer: Any, cfg: Any | None = None
) -> dict[str, Any]:
    """Render an example into ids, roles, labels, metadata, and custom positions.

    Pieces are tokenized separately so role spans are explicit and robust.
    """

    rcfg = render_config_from_obj(cfg)
    if rcfg.num_reads_per_doc > len(READ_TOKENS):
        raise ValueError("num_reads_per_doc exceeds the available read tokens")
    if rcfg.num_gather_tokens > len(GATHER_TOKENS):
        raise ValueError("num_gather_tokens exceeds the available gather tokens")

    input_ids: list[int] = []
    role_ids: list[int] = []
    item_ids: list[int] = []
    read_slot_ids: list[int] = []
    gather_slot_ids: list[int] = []

    if rcfg.add_bos_token and tokenizer.bos_token_id is not None:
        _append(
            input_ids,
            role_ids,
            item_ids,
            read_slot_ids,
            gather_slot_ids,
            [int(tokenizer.bos_token_id)],
            ROLE_PREFIX,
        )

    prefix_ids = _encode(tokenizer, _render_prefix(example))
    _append(input_ids, role_ids, item_ids, read_slot_ids, gather_slot_ids, prefix_ids, ROLE_PREFIX)
    prefix_length = len(input_ids)

    _append(
        input_ids,
        role_ids,
        item_ids,
        read_slot_ids,
        gather_slot_ids,
        [_special_id(tokenizer, SET_TOKEN)],
        ROLE_SET_SPECIAL,
    )
    newline_ids = _encode(tokenizer, "\n")
    _append(
        input_ids, role_ids, item_ids, read_slot_ids, gather_slot_ids, newline_ids, ROLE_SET_SPECIAL
    )

    for doc_idx, doc in enumerate(example.documents):
        _append(
            input_ids,
            role_ids,
            item_ids,
            read_slot_ids,
            gather_slot_ids,
            [_special_id(tokenizer, ITEM_TOKEN)],
            ROLE_ITEM_SPECIAL,
            item_id=doc_idx,
        )
        _append(
            input_ids,
            role_ids,
            item_ids,
            read_slot_ids,
            gather_slot_ids,
            newline_ids,
            ROLE_ITEM_SPECIAL,
            item_id=doc_idx,
        )

        doc_ids = truncate_text_by_tokens(
            tokenizer=tokenizer,
            text=doc.text.rstrip(),
            max_tokens=rcfg.max_doc_tokens,
            answer_texts=answer_candidates(example.answer, example.metadata),
            prefer_answer_window=rcfg.doc_truncation == "answer_window" and doc.is_gold,
        )
        _append(
            input_ids,
            role_ids,
            item_ids,
            read_slot_ids,
            gather_slot_ids,
            doc_ids,
            ROLE_DOC,
            item_id=doc_idx,
        )
        _append(
            input_ids,
            role_ids,
            item_ids,
            read_slot_ids,
            gather_slot_ids,
            newline_ids,
            ROLE_ITEM_SPECIAL,
            item_id=doc_idx,
        )

        for read_idx, token in enumerate(READ_TOKENS[: rcfg.num_reads_per_doc]):
            if read_idx > 0:
                _append(
                    input_ids,
                    role_ids,
                    item_ids,
                    read_slot_ids,
                    gather_slot_ids,
                    _encode(tokenizer, " "),
                    ROLE_ITEM_SPECIAL,
                    item_id=doc_idx,
                )
            _append(
                input_ids,
                role_ids,
                item_ids,
                read_slot_ids,
                gather_slot_ids,
                [_special_id(tokenizer, token)],
                ROLE_READ,
                item_id=doc_idx,
                read_slot_id=read_idx,
            )
        _append(
            input_ids,
            role_ids,
            item_ids,
            read_slot_ids,
            gather_slot_ids,
            newline_ids,
            ROLE_ITEM_SPECIAL,
            item_id=doc_idx,
        )
        _append(
            input_ids,
            role_ids,
            item_ids,
            read_slot_ids,
            gather_slot_ids,
            [_special_id(tokenizer, END_ITEM_TOKEN)],
            ROLE_ITEM_SPECIAL,
            item_id=doc_idx,
        )
        _append(
            input_ids,
            role_ids,
            item_ids,
            read_slot_ids,
            gather_slot_ids,
            newline_ids,
            ROLE_ITEM_SPECIAL,
            item_id=doc_idx,
        )

    _append(
        input_ids,
        role_ids,
        item_ids,
        read_slot_ids,
        gather_slot_ids,
        [_special_id(tokenizer, END_SET_TOKEN)],
        ROLE_SET_SPECIAL,
    )
    _append(
        input_ids, role_ids, item_ids, read_slot_ids, gather_slot_ids, newline_ids, ROLE_SET_SPECIAL
    )

    for gather_idx, token in enumerate(GATHER_TOKENS[: rcfg.num_gather_tokens]):
        if gather_idx > 0:
            _append(
                input_ids,
                role_ids,
                item_ids,
                read_slot_ids,
                gather_slot_ids,
                _encode(tokenizer, " "),
                ROLE_SET_SPECIAL,
            )
        _append(
            input_ids,
            role_ids,
            item_ids,
            read_slot_ids,
            gather_slot_ids,
            [_special_id(tokenizer, token)],
            ROLE_GATHER,
            gather_slot_id=gather_idx,
        )

    _append(
        input_ids,
        role_ids,
        item_ids,
        read_slot_ids,
        gather_slot_ids,
        _encode(tokenizer, "\n\n"),
        ROLE_SET_SPECIAL,
    )

    answer_start = len(input_ids)
    answer_ids = _encode(tokenizer, example.answer)
    if rcfg.append_eos_token and getattr(tokenizer, "eos_token_id", None) is not None:
        answer_ids = answer_ids + [int(tokenizer.eos_token_id)]
    if not answer_ids:
        raise ValueError(f"Example {example.example_id} has an empty tokenized answer")
    _append(input_ids, role_ids, item_ids, read_slot_ids, gather_slot_ids, answer_ids, ROLE_ANSWER)

    labels = list(input_ids)
    for idx in range(answer_start):
        labels[idx] = IGNORE_INDEX

    position_ids = build_position_ids(role_ids, item_ids, prefix_length=prefix_length)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "role_ids": role_ids,
        "item_ids": item_ids,
        "read_slot_ids": read_slot_ids,
        "gather_slot_ids": gather_slot_ids,
        "position_ids": position_ids,
        "answer_start": answer_start,
        "prefix_length": prefix_length,
        "example_id": example.example_id,
    }
