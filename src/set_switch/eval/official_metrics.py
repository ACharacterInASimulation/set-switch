"""Source-standard answer metrics used by evaluation scripts."""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from set_switch.data.schema import SetSwitchExample

YES_NO_NOANSWER = {"yes", "no", "noanswer"}


def normalize_answer_hotpot(text: Any) -> str:
    """Hotpot/SQuAD-style answer normalization."""

    value = str(text or "").lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _tokens(text: Any, normalizer: Callable[[Any], str] = normalize_answer_hotpot) -> list[str]:
    return normalizer(text).split()


def hotpot_f1(prediction: Any, ground_truth: Any) -> tuple[float, float, float]:
    """Return official-style Hotpot answer F1, precision, and recall."""

    normalized_prediction = normalize_answer_hotpot(prediction)
    normalized_gold = normalize_answer_hotpot(ground_truth)
    if normalized_prediction in YES_NO_NOANSWER or normalized_gold in YES_NO_NOANSWER:
        if normalized_prediction == normalized_gold:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    if not pred_tokens or not gold_tokens:
        return (1.0, 1.0, 1.0) if pred_tokens == gold_tokens else (0.0, 0.0, 0.0)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall), precision, recall


def exact_match(
    prediction: Any,
    ground_truth: Any,
    normalizer: Callable[[Any], str] = normalize_answer_hotpot,
) -> bool:
    return normalizer(prediction) == normalizer(ground_truth)


def max_over_ground_truths(
    metric_fn: Callable[[Any, Any], Any],
    prediction: Any,
    answers: Sequence[Any],
) -> Any:
    if not answers:
        return metric_fn(prediction, "")
    scores = [metric_fn(prediction, answer) for answer in answers]
    first = scores[0]
    if isinstance(first, tuple):
        return max(scores, key=lambda item: item[0])
    return max(scores)


def _metadata_answers(example: SetSwitchExample) -> list[str]:
    answers: list[str] = []
    for key in ("golden_answers", "answers", "answer_aliases", "aliases"):
        raw = example.metadata.get(key)
        if isinstance(raw, list):
            answers.extend(
                "yes" if item is True else "no" if item is False else str(item) for item in raw
            )
    answers.append(example.answer)
    deduped: list[str] = []
    seen: set[str] = set()
    for answer in answers:
        normalized = normalize_answer_hotpot(answer)
        if answer.strip() and normalized not in seen:
            deduped.append(answer)
            seen.add(normalized)
    return deduped


def _answer_only_score(example: SetSwitchExample, prediction: str) -> dict[str, Any]:
    prediction = prediction.splitlines()[0] if prediction else ""
    answers = _metadata_answers(example)
    em = float(max_over_ground_truths(exact_match, prediction, answers))
    f1, precision, recall = max_over_ground_truths(hotpot_f1, prediction, answers)
    return {
        "exact_match": em,
        "token_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "answers": answers,
    }


def score_hotpot_answer_only(example: SetSwitchExample, prediction: str) -> dict[str, Any]:
    return _answer_only_score(example, prediction)


def score_2wiki_answer_only(example: SetSwitchExample, prediction: str) -> dict[str, Any]:
    return _answer_only_score(example, prediction)


def score_musique_answer_only(example: SetSwitchExample, prediction: str) -> dict[str, Any]:
    answers = [example.answer]
    for key in ("answer_aliases", "golden_answers"):
        raw_aliases = example.metadata.get(key)
        if isinstance(raw_aliases, list):
            answers.extend(str(alias) for alias in raw_aliases)
    answers = list(dict.fromkeys(answer for answer in answers if str(answer).strip()))
    prediction = prediction.splitlines()[0] if prediction else ""
    em = float(max_over_ground_truths(exact_match, prediction, answers))
    f1, precision, recall = max_over_ground_truths(hotpot_f1, prediction, answers)
    return {
        "answer_em": em,
        "answer_f1": float(f1),
        "exact_match": em,
        "token_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "answers": answers,
    }


def score_squad_answer_only(example: SetSwitchExample, prediction: str) -> dict[str, Any]:
    return _answer_only_score(example, prediction)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def _rouge_l(prediction: str, answer: str) -> float:
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(answer)
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _bleu_n(prediction: str, answer: str, n: int) -> float:
    pred_tokens = _tokens(prediction)
    gold_tokens = _tokens(answer)
    if len(pred_tokens) < n or len(gold_tokens) < n:
        return 0.0
    pred_ngrams = Counter(
        tuple(pred_tokens[idx : idx + n]) for idx in range(len(pred_tokens) - n + 1)
    )
    gold_ngrams = Counter(
        tuple(gold_tokens[idx : idx + n]) for idx in range(len(gold_tokens) - n + 1)
    )
    overlap = sum((pred_ngrams & gold_ngrams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(pred_ngrams.values()))
    brevity_penalty = 1.0
    if len(pred_tokens) < len(gold_tokens):
        brevity_penalty = math.exp(1.0 - len(gold_tokens) / max(1, len(pred_tokens)))
    return brevity_penalty * precision


def score_msmarco_official_adapter(
    candidate_jsonl: str | Path | None = None,
    reference_jsonl: str | Path | None = None,
    predictions: dict[str, str] | None = None,
    examples: Sequence[SetSwitchExample] | None = None,
) -> dict[str, float]:
    """Evaluate MS MARCO QA-style candidate answers.

    The accepted candidate format is one JSON object per line:
    ``{"query_id": "...", "answers": ["predicted answer"]}``.
    """

    if candidate_jsonl is not None:
        predictions = {}
        for line in Path(candidate_jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            answers = item.get("answers") or [""]
            predictions[str(item["query_id"])] = str(answers[0] if answers else "")
    if examples is None and reference_jsonl is not None:
        references: dict[str, list[str]] = {}
        for line in Path(reference_jsonl).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            query_id = str(item.get("query_id", item.get("example_id", "")))
            answers = item.get("answers") or item.get("golden_answers") or [item.get("answer", "")]
            references[query_id] = [str(answer) for answer in answers]
    elif examples is not None:
        references = {example.example_id: _metadata_answers(example) for example in examples}
    else:
        references = {}

    predictions = predictions or {}
    totals = {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0, "rouge_l": 0.0}
    total = 0
    for query_id, answers in references.items():
        prediction = predictions.get(query_id, "")
        totals["bleu1"] += max((_bleu_n(prediction, answer, 1) for answer in answers), default=0.0)
        totals["bleu2"] += max((_bleu_n(prediction, answer, 2) for answer in answers), default=0.0)
        totals["bleu3"] += max((_bleu_n(prediction, answer, 3) for answer in answers), default=0.0)
        totals["bleu4"] += max((_bleu_n(prediction, answer, 4) for answer in answers), default=0.0)
        totals["rouge_l"] += max((_rouge_l(prediction, answer) for answer in answers), default=0.0)
        total += 1
    return {key: value / max(1, total) for key, value in totals.items()}


@torch.no_grad()
def score_mcq_logprobs(
    model: Any,
    tokenizer: Any,
    prompt_input_ids: Sequence[int],
    option_texts: Sequence[str],
    length_normalize: bool = True,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Score multiple-choice options by teacher-forced option log probability."""

    if device is None:
        device = next(model.parameters()).device
    prompt = list(int(token_id) for token_id in prompt_input_ids)
    scores: list[dict[str, float]] = []
    for option in option_texts:
        option_ids = list(tokenizer.encode(str(option), add_special_tokens=False))
        if not option_ids:
            scores.append(
                {"sum_logprob": float("-inf"), "length_normalized_logprob": float("-inf")}
            )
            continue
        ids = torch.tensor([prompt + option_ids], dtype=torch.long, device=device)
        logits = model(input_ids=ids, use_cache=False).logits.float()
        log_probs = torch.log_softmax(logits[0, len(prompt) - 1 : -1], dim=-1)
        target = torch.tensor(option_ids, dtype=torch.long, device=device)
        token_logprobs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        total = float(token_logprobs.sum().detach().cpu())
        scores.append(
            {
                "sum_logprob": total,
                "length_normalized_logprob": total / len(option_ids),
            }
        )
    score_key = "length_normalized_logprob" if length_normalize else "sum_logprob"
    best_index = max(range(len(scores)), key=lambda idx: scores[idx][score_key]) if scores else -1
    return {"best_index": best_index, "scores": scores, "score_key": score_key}
