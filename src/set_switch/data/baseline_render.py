"""Normal decoder-only chat-template baseline rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from set_switch.constants import IGNORE_INDEX
from set_switch.data.schema import SetSwitchExample
from set_switch.data.truncation import answer_candidates, truncate_text_by_tokens


@dataclass(frozen=True)
class BaselineRenderConfig:
    max_doc_tokens: int | None = None
    doc_truncation: str = "answer_window"
    append_eos_token: bool = False


def baseline_render_config_from_obj(cfg: Any | None) -> BaselineRenderConfig:
    if cfg is None:
        return BaselineRenderConfig()
    data = cfg.get("data", cfg) if isinstance(cfg, dict) else cfg

    def get(name: str, default: Any) -> Any:
        if isinstance(data, dict):
            return data.get(name, default)
        return getattr(data, name, default)

    return BaselineRenderConfig(
        max_doc_tokens=get("max_doc_tokens", None),
        doc_truncation=str(get("doc_truncation", "answer_window")),
        append_eos_token=bool(get("append_eos_token", False)),
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def render_baseline_prompt_text(
    example: SetSwitchExample,
    tokenizer: Any | None = None,
    max_doc_tokens: int | None = None,
    doc_truncation: str = "answer_window",
) -> str:
    set_type = example.metadata.get("set_type", "documents")
    item_name = "Option" if set_type == "options" else "Passage"
    set_name = "Options" if set_type == "options" else "Passages"
    final_instruction = (
        "Answer with the correct option text."
        if set_type == "options"
        else "Answer using only the provided passages."
    )

    docs: list[str] = []
    for doc in example.documents:
        text = doc.text.strip()
        if tokenizer is not None:
            ids = truncate_text_by_tokens(
                tokenizer=tokenizer,
                text=text,
                max_tokens=max_doc_tokens,
                answer_texts=answer_candidates(example.answer, example.metadata),
                prefer_answer_window=doc_truncation == "answer_window" and doc.is_gold,
            )
            text = tokenizer.decode(ids).strip() if hasattr(tokenizer, "decode") else text
        docs.append(f"{item_name}:\n{text}")

    return (
        f"{example.instruction}\n\n"
        f"Question: {example.question}\n\n"
        f"{set_name}:\n" + "\n\n".join(docs) + f"\n\n{final_instruction}"
    )


def _chat_prefix_ids(tokenizer: Any, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]
    return _encode(tokenizer, prompt + "\n\nAnswer:")


def render_chat_baseline_example(
    example: SetSwitchExample,
    tokenizer: Any,
    cfg: Any | None = None,
) -> dict[str, Any]:
    """Render a normal decoder-only SFT example with the model's own chat template."""

    rcfg = baseline_render_config_from_obj(cfg)
    prompt = render_baseline_prompt_text(
        example,
        tokenizer,
        rcfg.max_doc_tokens,
        doc_truncation=rcfg.doc_truncation,
    )
    prefix_ids = _chat_prefix_ids(tokenizer, prompt)
    answer_ids = _encode(tokenizer, example.answer)
    if rcfg.append_eos_token and getattr(tokenizer, "eos_token_id", None) is not None:
        answer_ids = answer_ids + [int(tokenizer.eos_token_id)]
    if not answer_ids:
        raise ValueError(f"Example {example.example_id} has an empty tokenized answer")

    input_ids = prefix_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prefix_ids) + list(answer_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "answer_start": len(prefix_ids),
        "example_id": example.example_id,
        "interface": "chat_baseline",
    }
