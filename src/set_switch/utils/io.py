"""Small JSON/YAML/JSONL helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from set_switch.data.schema import SetSwitchExample, example_from_dict, example_to_dict


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        data = dict(yaml.safe_load(handle) or {})
    extends = data.pop("extends", None)
    if extends is None:
        return data

    parent_paths = [extends] if isinstance(extends, str) else list(extends)
    merged: dict[str, Any] = {}
    for parent_path in parent_paths:
        resolved = Path(parent_path)
        if not resolved.is_absolute():
            resolved = input_path.parent / resolved
        merged = _deep_merge_dicts(merged, read_yaml(resolved))
    return _deep_merge_dicts(merged, data)


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
