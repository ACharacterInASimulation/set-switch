"""Set-LLM prompt rendering.

For option sets this follows the modified Set-LLM prompt format:

Question: {question}

Choices:
{choice0}
{choice1}

Answer:
{answer}<EOS>

The choices/passages are ordinary text, not numbered, and set membership is carried by
metadata used for SetPE and SetMask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from set_switch.constants import IGNORE_INDEX, ROLE_ANSWER, ROLE_DOC, ROLE_PREFIX
from set_switch.data.schema import SetSwitchExample
from set_switch.data.truncation import answer_candidates, truncate_text_by_tokens
from set_switch.modeling.setllm import build_setllm_position_ids_from_chunks


@dataclass(frozen=True)
class SetLLMRenderConfig:
    max_doc_tokens: int | None = None
    doc_truncation: str = "answer_window"
    append_eos_token: bool = True


def setllm_render_config_from_obj(cfg: Any | None) -> SetLLMRenderConfig:
    if cfg is None:
        return SetLLMRenderConfig()
    data = cfg.get("data", cfg) if isinstance(cfg, dict) else cfg

    def get(name: str, default: Any) -> Any:
        if isinstance(data, dict):
            return data.get(name, default)
        return getattr(data, name, default)

    return SetLLMRenderConfig(
        max_doc_tokens=get("max_doc_tokens", None),
        doc_truncation=str(get("doc_truncation", "answer_window")),
        append_eos_token=bool(get("setllm_append_eos_token", True)),
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _truncate_ids(ids: list[int], max_tokens: int | None) -> list[int]:
    if max_tokens is None:
        return ids
    return ids[: int(max_tokens)]


def _set_item_texts(example: SetSwitchExample) -> tuple[str, list[str]]:
    set_type = example.metadata.get("set_type", "documents")
    if set_type == "options":
        choices = [
            str(
                doc.metadata.get("choice_text") or doc.text.replace("Candidate answer:", "", 1)
            ).strip()
            for doc in example.documents
        ]
        return "Choices", choices
    return "Passages", [doc.text.strip() for doc in example.documents]


def render_setllm_prompt_text(example: SetSwitchExample) -> str:
    set_name, item_texts = _set_item_texts(example)
    return (
        f"Instruction: {example.instruction}\n\n"
        f"Question: {example.question}\n\n"
        f"{set_name}:\n" + "\n".join(item_texts) + "\n\nAnswer:\n"
    )


def render_setllm_example(
    example: SetSwitchExample,
    tokenizer: Any,
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Render a Set-LLM instruction-tuning example with explicit SetPE metadata."""

    rcfg = setllm_render_config_from_obj(cfg)
    set_name, item_texts = _set_item_texts(example)

    prefix_ids = _encode(
        tokenizer,
        f"Instruction: {example.instruction}\n\nQuestion: {example.question}\n\n{set_name}:\n",
    )
    if example.metadata.get("set_type", "documents") == "options":
        set_item_ids = [
            _truncate_ids(_encode(tokenizer, text.strip() + "\n"), rcfg.max_doc_tokens)
            for text in item_texts
            if text.strip()
        ]
    else:
        answers = answer_candidates(example.answer, example.metadata)
        set_item_ids = [
            truncate_text_by_tokens(
                tokenizer=tokenizer,
                text=doc.text.strip() + "\n",
                max_tokens=rcfg.max_doc_tokens,
                answer_texts=answers,
                prefer_answer_window=rcfg.doc_truncation == "answer_window" and doc.is_gold,
            )
            for doc in example.documents
            if doc.text.strip()
        ]
    if not set_item_ids:
        raise ValueError(f"Example {example.example_id} has no non-empty set items")
    suffix_ids = _encode(tokenizer, "\nAnswer:\n")
    answer_ids = _encode(tokenizer, example.answer)
    if rcfg.append_eos_token and getattr(tokenizer, "eos_token_id", None) is not None:
        answer_ids = answer_ids + [int(tokenizer.eos_token_id)]
    if not answer_ids:
        raise ValueError(f"Example {example.example_id} has an empty tokenized answer")

    input_ids: list[int] = []
    role_ids: list[int] = []
    item_ids: list[int] = []

    input_ids.extend(prefix_ids)
    role_ids.extend([ROLE_PREFIX] * len(prefix_ids))
    item_ids.extend([-1] * len(prefix_ids))

    for item_idx, ids in enumerate(set_item_ids):
        input_ids.extend(ids)
        role_ids.extend([ROLE_DOC] * len(ids))
        item_ids.extend([item_idx] * len(ids))

    input_ids.extend(suffix_ids)
    role_ids.extend([ROLE_PREFIX] * len(suffix_ids))
    item_ids.extend([-1] * len(suffix_ids))

    answer_start = len(input_ids)
    input_ids.extend(answer_ids)
    role_ids.extend([ROLE_ANSWER] * len(answer_ids))
    item_ids.extend([-1] * len(answer_ids))

    labels = [IGNORE_INDEX] * answer_start + list(answer_ids)
    position_ids = build_setllm_position_ids_from_chunks(
        prefix_ids=prefix_ids,
        set_item_ids=set_item_ids,
        suffix_ids=suffix_ids,
        answer_ids=answer_ids,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "role_ids": role_ids,
        "item_ids": item_ids,
        "position_ids": position_ids,
        "answer_start": answer_start,
        "example_id": example.example_id,
        "interface": "setllm",
    }
