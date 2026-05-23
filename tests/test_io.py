from __future__ import annotations

from pathlib import Path

from set_switch.utils.io import read_yaml


def test_read_yaml_extends_and_deep_merges_relative_paths(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "model_interface: setswitch\n"
        "model:\n"
        "  dtype: bfloat16\n"
        "  attn_implementation: eager\n"
        "train:\n"
        "  learning_rate: 0.0001\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\n"
        "model_interface: setllm\n"
        "model:\n"
        "  attn_implementation: sdpa\n",
        encoding="utf-8",
    )

    cfg = read_yaml(tmp_path / "child.yaml")

    assert cfg["model_interface"] == "setllm"
    assert cfg["model"]["dtype"] == "bfloat16"
    assert cfg["model"]["attn_implementation"] == "sdpa"
    assert cfg["train"]["learning_rate"] == 0.0001


def test_default_flashrag_suite_is_set_focused_and_totals_100k():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = read_yaml(repo_root / "configs" / "flashrag.yaml")
    datasets = cfg["data"]["datasets"]
    names = [item["name"] for item in datasets]

    assert names == [
        "commonsenseqa",
        "openbookqa",
        "arc",
        "hellaswag",
        "mmlu",
        "quartz",
        "msmarco-qa",
        "squad",
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
    ]
    assert {"boolq", "qasc", "ambig_qa"}.isdisjoint(names)
    assert sum(int(item.get("train_max_examples", 0)) for item in datasets) == 100_000
    assert cfg["data"]["train_max_render_tokens"] == 4096
