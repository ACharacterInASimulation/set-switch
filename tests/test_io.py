from __future__ import annotations

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
