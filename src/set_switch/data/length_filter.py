"""Rendered-length filtering utilities for train-set construction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from set_switch.data.baseline_render import render_chat_baseline_example
from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchExample
from set_switch.data.setfuse_render import render_setfuse_example
from set_switch.data.setllm_render import render_setllm_example

LENGTH_FILTER_INTERFACES = ("chat_baseline", "setllm", "setswitch", "setfuse")


def normalize_length_filter_interfaces(value: Any, default: str = "all") -> tuple[str, ...]:
    """Normalize config/CLI interface names for rendered-length filtering."""

    if value is None:
        value = default
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in value if str(item).strip()]
    if not values or values == ["all"]:
        return LENGTH_FILTER_INTERFACES
    interfaces = tuple(values)
    unknown = sorted(set(interfaces) - set(LENGTH_FILTER_INTERFACES))
    if unknown:
        raise ValueError(
            f"Unknown length-filter interface(s): {unknown}; "
            f"expected one of {LENGTH_FILTER_INTERFACES} or 'all'"
        )
    return interfaces


def rendered_length(
    example: SetSwitchExample,
    tokenizer: Any,
    cfg: dict[str, Any],
    interface: str,
) -> int:
    """Return rendered token length for one interface."""

    render_cfg = {"data": cfg.get("data", cfg)}
    if interface == "setswitch":
        return len(render_example(example, tokenizer, render_cfg)["input_ids"])
    if interface == "setllm":
        return len(render_setllm_example(example, tokenizer, render_cfg)["input_ids"])
    if interface == "chat_baseline":
        return len(render_chat_baseline_example(example, tokenizer, render_cfg)["input_ids"])
    if interface == "setfuse":
        return len(render_setfuse_example(example, tokenizer, render_cfg)["input_ids"])
    raise ValueError(f"Unknown interface {interface!r}")


def max_rendered_length(
    example: SetSwitchExample,
    tokenizer: Any,
    cfg: dict[str, Any],
    interfaces: Sequence[str],
) -> int:
    """Return the maximum rendered length across selected interfaces."""

    return max(rendered_length(example, tokenizer, cfg, interface) for interface in interfaces)


def filter_examples_by_rendered_length(
    examples: Iterable[SetSwitchExample],
    tokenizer: Any,
    cfg: dict[str, Any],
    max_tokens: int,
    interfaces: Sequence[str],
) -> tuple[list[SetSwitchExample], list[dict[str, Any]]]:
    """Keep examples whose rendered length is at most ``max_tokens``."""

    kept: list[SetSwitchExample] = []
    dropped: list[dict[str, Any]] = []
    for example in examples:
        length = max_rendered_length(example, tokenizer, cfg, interfaces)
        if length <= max_tokens:
            kept.append(example)
        else:
            dropped.append(
                {
                    "example_id": example.example_id,
                    "source": example.source,
                    "rendered_length": length,
                    "num_documents": len(example.documents),
                }
            )
    return kept, dropped
