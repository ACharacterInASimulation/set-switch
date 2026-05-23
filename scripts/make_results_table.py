#!/usr/bin/env python
"""Build paper-style result tables from evaluation JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DATASET_ORDER = [
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "msmarco-qa",
    "squad",
    "commonsenseqa",
    "openbookqa",
    "arc",
    "hellaswag",
    "mmlu",
    "quartz",
]

INTERFACE_LABELS = {
    "chat_baseline": "Chat Baseline",
    "setllm": "SetLLM",
    "setswitch": "SetRelay",
    "setfuse": "SetFuse-LM",
}

DATASET_LABELS = {
    "2wikimultihopqa": "2WikiMQA",
    "msmarco-qa": "MS MARCO QA",
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
    "squad": "SQuAD",
    "commonsenseqa": "CSQA",
    "openbookqa": "OBQA",
    "arc": "ARC",
    "hellaswag": "HellaSwag",
    "mmlu": "MMLU",
    "quartz": "Quartz",
}

METRIC_LABELS = {
    "primary_score": "Primary",
    "accuracy": "Acc.",
    "exact_match": "EM",
    "token_f1": "F1",
    "rouge_l": "ROUGE-L",
    "bleu1": "BLEU-1",
    "bleu2": "BLEU-2",
    "bleu3": "BLEU-3",
    "bleu4": "BLEU-4",
}


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    report["_path"] = str(path)
    return report


def _sort_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        name: idx for idx, name in enumerate(["chat_baseline", "setllm", "setswitch", "setfuse"])
    }
    return sorted(reports, key=lambda item: order.get(str(item.get("interface")), 999))


def _datasets(reports: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    for report in reports:
        seen.update(report.get("dataset_summary", {}).keys())
    ordered = [dataset for dataset in DATASET_ORDER if dataset in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _metric_value(summary: dict[str, Any] | None, metric: str) -> float | None:
    if not summary:
        return None
    value = summary.get(metric)
    if value is None:
        return None
    return float(value)


def _format_value(value: float | None, scale: str, digits: int) -> str:
    if value is None:
        return "-"
    if scale == "percent":
        value *= 100.0
    return f"{value:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _latex_table(headers: list[str], rows: list[list[str]]) -> str:
    column_spec = "l" + "r" * (len(headers) - 1)
    escaped_headers = [header.replace("_", "\\_") for header in headers]
    lines = [
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(escaped_headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        escaped = [cell.replace("_", "\\_") for cell in row]
        lines.append(" & ".join(escaped) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def build_wide_table(
    reports: list[dict[str, Any]],
    metric: str,
    scale: str,
    digits: int,
) -> tuple[list[str], list[list[str]]]:
    datasets = _datasets(reports)
    headers = ["Method", f"Overall {METRIC_LABELS.get(metric, metric)}"]
    headers.extend(DATASET_LABELS.get(dataset, dataset) for dataset in datasets)
    rows: list[list[str]] = []
    for report in _sort_reports(reports):
        interface = str(report.get("interface", "unknown"))
        overall = report.get("reported_overall_summary", {}).get("overall")
        row = [
            INTERFACE_LABELS.get(interface, interface),
            _format_value(_metric_value(overall, metric), scale, digits),
        ]
        dataset_summary = report.get("dataset_summary", {})
        for dataset in datasets:
            row.append(
                _format_value(_metric_value(dataset_summary.get(dataset), metric), scale, digits)
            )
        rows.append(row)
    return headers, rows


def build_long_rows(reports: list[dict[str, Any]], scale: str) -> list[dict[str, Any]]:
    factor = 100.0 if scale == "percent" else 1.0
    rows: list[dict[str, Any]] = []
    for report in _sort_reports(reports):
        interface = str(report.get("interface", "unknown"))
        summaries = {"overall": report.get("reported_overall_summary", {}).get("overall", {})}
        summaries.update(report.get("dataset_summary", {}))
        for dataset, summary in summaries.items():
            row = {
                "interface": interface,
                "method": INTERFACE_LABELS.get(interface, interface),
                "dataset": dataset,
                "total": int(summary.get("total", 0)),
            }
            for metric in METRIC_LABELS:
                value = summary.get(metric)
                row[metric] = "" if value is None else float(value) * factor
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", help="Evaluation JSON reports")
    parser.add_argument("--metric", default="primary_score", choices=sorted(METRIC_LABELS))
    parser.add_argument("--scale", choices=["unit", "percent"], default="percent")
    parser.add_argument("--digits", type=int, default=2)
    parser.add_argument("--output-md", default="outputs/results_table.md")
    parser.add_argument("--output-csv", default="outputs/results_table.csv")
    parser.add_argument("--output-tex", default="outputs/results_table.tex")
    args = parser.parse_args()

    reports = [_load_report(Path(path)) for path in args.reports]
    headers, rows = build_wide_table(reports, args.metric, args.scale, args.digits)
    markdown = _markdown_table(headers, rows)
    latex = _latex_table(headers, rows)
    long_rows = build_long_rows(reports, args.scale)

    md_path = Path(args.output_md)
    csv_path = Path(args.output_csv)
    tex_path = Path(args.output_tex)
    for path in (md_path, csv_path, tex_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(markdown, encoding="utf-8")
    tex_path.write_text(latex, encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "interface",
            "method",
            "dataset",
            "total",
            *METRIC_LABELS.keys(),
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)

    print(markdown)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
