"""Gold-vs-non-gold document length diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Any

import numpy as np

from set_switch.data.schema import SetSwitchExample


@dataclass(frozen=True)
class LengthDiagnosticResult:
    report: dict[str, Any]
    gold_lengths: list[int]
    non_gold_lengths: list[int]


def _token_len(text: str, tokenizer: Any | None) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def _summary(lengths: list[int]) -> dict[str, float | int | None]:
    if not lengths:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(lengths),
        "mean": float(mean(lengths)),
        "median": float(median(lengths)),
        "std": float(pstdev(lengths)),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
    }


def compute_length_diagnostic(
    examples: list[SetSwitchExample],
    tokenizer: Any | None = None,
    compute_auc: bool = True,
) -> LengthDiagnosticResult:
    gold_lengths: list[int] = []
    non_gold_lengths: list[int] = []
    labels: list[int] = []
    lengths: list[int] = []

    for example in examples:
        for doc in example.documents:
            doc_len = _token_len(doc.text, tokenizer)
            lengths.append(doc_len)
            labels.append(1 if doc.is_gold else 0)
            if doc.is_gold:
                gold_lengths.append(doc_len)
            else:
                non_gold_lengths.append(doc_len)

    gold_summary = _summary(gold_lengths)
    non_gold_summary = _summary(non_gold_lengths)
    gold_mean = gold_summary["mean"]
    non_gold_mean = non_gold_summary["mean"]
    mean_diff = (
        None if gold_mean is None or non_gold_mean is None else float(gold_mean - non_gold_mean)
    )

    report: dict[str, Any] = {
        "num_examples": len(examples),
        "num_documents": len(lengths),
        "gold": gold_summary,
        "non_gold": non_gold_summary,
        "mean_difference_gold_minus_non_gold": mean_diff,
    }

    if compute_auc and len(set(labels)) == 2:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score

            x = np.asarray(lengths, dtype=np.float32).reshape(-1, 1)
            y = np.asarray(labels, dtype=np.int64)
            clf = LogisticRegression().fit(x, y)
            prob = clf.predict_proba(x)[:, 1]
            report["length_only_gold_auc"] = float(roc_auc_score(y, prob))
            report["length_only_logreg_coef"] = float(clf.coef_[0][0])
            report["length_only_logreg_intercept"] = float(clf.intercept_[0])
        except Exception as exc:  # pragma: no cover - optional dependency/runtime detail
            report["length_only_auc_error"] = str(exc)

    return LengthDiagnosticResult(
        report=report,
        gold_lengths=gold_lengths,
        non_gold_lengths=non_gold_lengths,
    )


def save_length_histogram(
    gold_lengths: list[int],
    non_gold_lengths: list[int],
    output_png: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    if non_gold_lengths:
        plt.hist(non_gold_lengths, bins=30, alpha=0.6, label="non-gold")
    if gold_lengths:
        plt.hist(gold_lengths, bins=30, alpha=0.6, label="gold")
    plt.xlabel("Document length")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
