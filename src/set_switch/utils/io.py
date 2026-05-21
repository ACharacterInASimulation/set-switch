"""Small JSON/YAML/JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from set_switch.data.schema import SetSwitchExample, example_from_dict, example_to_dict


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def write_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_examples_jsonl(path: str | Path) -> list[SetSwitchExample]:
    return [example_from_dict(row) for row in read_jsonl(path)]


def write_examples_jsonl(path: str | Path, examples: list[SetSwitchExample]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example_to_dict(example), ensure_ascii=False) + "\n")
