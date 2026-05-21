"""Document truncation helpers."""

from __future__ import annotations

from typing import Any


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def answer_candidates(example_answer: str, metadata: dict[str, Any] | None = None) -> list[str]:
    candidates: list[str] = []
    metadata = metadata or {}
    raw_answers = metadata.get("golden_answers")
    if isinstance(raw_answers, list):
        candidates.extend(str(answer).strip() for answer in raw_answers if str(answer).strip())
    if example_answer.strip():
        candidates.append(example_answer.strip())
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        key = candidate.lower()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def truncate_text_by_tokens(
    tokenizer: Any,
    text: str,
    max_tokens: int | None,
    answer_texts: list[str] | None = None,
    prefer_answer_window: bool = False,
) -> list[int]:
    """Tokenize text and optionally keep a window around an answer mention."""

    ids = _encode(tokenizer, text)
    if max_tokens is None or len(ids) <= int(max_tokens):
        return ids

    max_tokens = int(max_tokens)
    if not prefer_answer_window:
        return ids[:max_tokens]

    lower_text = text.lower()
    for answer in answer_texts or []:
        answer = answer.strip()
        if not answer:
            continue
        char_start = lower_text.find(answer.lower())
        if char_start < 0:
            continue

        answer_token_start = len(_encode(tokenizer, text[:char_start]))
        answer_token_len = max(
            1, len(_encode(tokenizer, text[char_start : char_start + len(answer)]))
        )
        answer_token_end = min(len(ids), answer_token_start + answer_token_len)
        answer_span = max(1, answer_token_end - answer_token_start)
        left_budget = max(0, (max_tokens - answer_span) // 2)
        start = max(0, answer_token_start - left_budget)
        end = start + max_tokens
        if end > len(ids):
            end = len(ids)
            start = max(0, end - max_tokens)
        return ids[start:end]

    return ids[:max_tokens]
