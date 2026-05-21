#!/usr/bin/env python
"""Evaluate accuracy with task buckets and gold-position sweeps."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from set_switch.constants import ROLE_ANSWER
from set_switch.data.baseline_render import render_chat_baseline_example
from set_switch.data.dataset_suite import (
    load_flashrag_selected_examples,
    normalize_flashrag_sources,
    task_group_for_source,
)
from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchExample
from set_switch.data.setllm_render import render_setllm_example
from set_switch.modeling.attention_mask import build_setswitch_attention_mask
from set_switch.modeling.load_model import load_tokenizer_and_model
from set_switch.modeling.peft_setup import load_special_token_embeddings
from set_switch.modeling.setllm import build_setllm_attention_mask
from set_switch.training.train import apply_interface_overrides, attention_mask_dtype_from_config
from set_switch.utils.io import read_yaml

DEFAULT_GOLD_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _normalize_answer(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _answers(example: SetSwitchExample) -> list[str]:
    raw = example.metadata.get("golden_answers")
    if isinstance(raw, list):
        answers = [
            ("yes" if item is True else "no" if item is False else str(item)) for item in raw
        ]
        return [answer for answer in answers if answer.strip()]
    return [example.answer]


def _matches(prediction: str, answers: list[str]) -> bool:
    pred = _normalize_answer(prediction.splitlines()[0] if prediction else prediction)
    if not pred:
        return False
    for answer in answers:
        gold = _normalize_answer(answer)
        if pred == gold:
            return True
        if gold and pred.startswith(gold + " "):
            return True
    return False


def place_gold_documents(example: SetSwitchExample, fraction: float) -> SetSwitchExample:
    """Move gold documents/options as a block to an approximate set position."""

    placed = copy.deepcopy(example)
    gold = [doc for doc in placed.documents if doc.is_gold]
    non_gold = [doc for doc in placed.documents if not doc.is_gold]
    if not gold or not non_gold:
        return placed
    insert_at = round(float(fraction) * len(non_gold))
    placed.documents = non_gold[:insert_at] + gold + non_gold[insert_at:]
    return placed


def gold_sweep_status(example: SetSwitchExample) -> dict[str, int | bool]:
    gold_count = sum(1 for doc in example.documents if doc.is_gold)
    non_gold_count = len(example.documents) - gold_count
    return {
        "num_gold_documents": gold_count,
        "num_non_gold_documents": non_gold_count,
        "gold_sweep_movable": gold_count > 0 and non_gold_count > 0,
    }


def _render(interface: str, example: SetSwitchExample, tokenizer: Any, cfg: dict[str, Any]):
    render_cfg = {"data": cfg.get("data", {})}
    if interface == "setswitch":
        return render_example(example, tokenizer, render_cfg)
    if interface == "setllm":
        return render_setllm_example(example, tokenizer, render_cfg)
    if interface == "chat_baseline":
        return render_chat_baseline_example(example, tokenizer, render_cfg)
    raise ValueError(f"Unknown model_interface {interface!r}")


def _decode(tokenizer: Any, ids: list[int]) -> str:
    try:
        return tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(ids)


@torch.no_grad()
def _greedy_chat_baseline(
    model: Any,
    tokenizer: Any,
    rendered: dict[str, Any],
    max_new_tokens: int,
) -> str:
    device = next(model.parameters()).device
    input_ids = torch.tensor([rendered["input_ids"][: rendered["answer_start"]]], device=device)
    attention_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    return _decode(tokenizer, new_ids)


@torch.no_grad()
def _greedy_setllm(
    model: Any,
    tokenizer: Any,
    rendered: dict[str, Any],
    max_new_tokens: int,
    mask_dtype: torch.dtype = torch.float32,
) -> str:
    device = next(model.parameters()).device
    prompt_len = int(rendered["answer_start"])
    input_ids = list(rendered["input_ids"][:prompt_len])
    role_ids = list(rendered["role_ids"][:prompt_len])
    item_ids = list(rendered["item_ids"][:prompt_len])
    position_ids = list(rendered["position_ids"][:prompt_len])
    next_answer_pos = int(rendered["position_ids"][prompt_len])
    generated: list[int] = []

    for step in range(max_new_tokens):
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        roles = torch.tensor([role_ids], dtype=torch.long, device=device)
        items = torch.tensor([item_ids], dtype=torch.long, device=device)
        pos = torch.tensor([position_ids], dtype=torch.long, device=device)
        mask = build_setllm_attention_mask(roles, items, dtype=mask_dtype, device=device)
        logits = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            use_cache=False,
        ).logits
        next_id = int(torch.argmax(logits[0, -1]).detach().cpu())
        if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
            break
        generated.append(next_id)
        input_ids.append(next_id)
        role_ids.append(ROLE_ANSWER)
        item_ids.append(-1)
        position_ids.append(next_answer_pos + step)
    return _decode(tokenizer, generated)


@torch.no_grad()
def _greedy_setswitch(
    model: Any,
    tokenizer: Any,
    rendered: dict[str, Any],
    cfg: dict[str, Any],
    max_new_tokens: int,
    mask_dtype: torch.dtype = torch.float32,
) -> str:
    device = next(model.parameters()).device
    prompt_len = int(rendered["answer_start"])
    input_ids = list(rendered["input_ids"][:prompt_len])
    role_ids = list(rendered["role_ids"][:prompt_len])
    item_ids = list(rendered["item_ids"][:prompt_len])
    read_slot_ids = list(rendered["read_slot_ids"][:prompt_len])
    gather_slot_ids = list(rendered["gather_slot_ids"][:prompt_len])
    position_ids = list(rendered["position_ids"][:prompt_len])
    next_answer_pos = int(rendered["position_ids"][prompt_len])
    generated: list[int] = []

    for step in range(max_new_tokens):
        ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        roles = torch.tensor([role_ids], dtype=torch.long, device=device)
        items = torch.tensor([item_ids], dtype=torch.long, device=device)
        reads = torch.tensor([read_slot_ids], dtype=torch.long, device=device)
        gathers = torch.tensor([gather_slot_ids], dtype=torch.long, device=device)
        pos = torch.tensor([position_ids], dtype=torch.long, device=device)
        mask = build_setswitch_attention_mask(
            role_ids=roles,
            item_ids=items,
            read_slot_ids=reads,
            gather_slot_ids=gathers,
            attention_mode=cfg.get("mask", {}).get("doc_attention", "doc_causal"),
            dtype=mask_dtype,
            device=device,
        )
        logits = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            use_cache=False,
        ).logits
        next_id = int(torch.argmax(logits[0, -1]).detach().cpu())
        if tokenizer.eos_token_id is not None and next_id == int(tokenizer.eos_token_id):
            break
        generated.append(next_id)
        input_ids.append(next_id)
        role_ids.append(ROLE_ANSWER)
        item_ids.append(-1)
        read_slot_ids.append(-1)
        gather_slot_ids.append(-1)
        position_ids.append(next_answer_pos + step)
    return _decode(tokenizer, generated)


def generate_prediction(
    interface: str,
    model: Any,
    tokenizer: Any,
    rendered: dict[str, Any],
    cfg: dict[str, Any],
    max_new_tokens: int,
) -> str:
    if interface == "chat_baseline":
        return _greedy_chat_baseline(model, tokenizer, rendered, max_new_tokens)
    mask_dtype = attention_mask_dtype_from_config(
        cfg.get("model", {}),
        cfg.get("train", {}),
        model,
    )
    if interface == "setllm":
        return _greedy_setllm(model, tokenizer, rendered, max_new_tokens, mask_dtype)
    return _greedy_setswitch(model, tokenizer, rendered, cfg, max_new_tokens, mask_dtype)


def _as_float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    return [float(item) for item in value]


def _format_output_path(template: str, cfg: dict[str, Any], interface: str, split: str) -> str:
    return template.format(
        run_name=cfg.get("run_name", "set_switch_run"),
        interface=interface,
        split=split,
    )


def _load_eval_examples(cfg: dict[str, Any], split: str, max_examples: int | None):
    data_cfg = dict(cfg.get("data", {}))
    if max_examples is not None:
        data_cfg["total_val_examples"] = int(max_examples)
    selections = normalize_flashrag_sources(data_cfg, split)
    return load_flashrag_selected_examples(
        dataset_name=data_cfg.get("dataset_name", "RUC-NLPIR/FlashRAG_datasets"),
        selections=selections,
        max_docs=int(data_cfg.get("max_docs", 8)),
        instruction=data_cfg.get(
            "instruction",
            "Use the provided passages or options to answer the question. Treat the items as an unordered set.",
        ),
        total_examples=data_cfg.get("total_val_examples"),
        sample_allocation=data_cfg.get("sample_allocation", "task_balanced_equal"),
        sample_allocation_alpha=float(data_cfg.get("sample_allocation_alpha", 0.5)),
    )


def _is_peft_adapter_checkpoint(checkpoint: str | Path | None) -> bool:
    if checkpoint is None:
        return False
    checkpoint_path = Path(checkpoint)
    return checkpoint_path.is_dir() and (checkpoint_path / "adapter_config.json").is_file()


def _peft_adapter_base_model(checkpoint: str | Path) -> str | None:
    config_path = Path(checkpoint) / "adapter_config.json"
    try:
        adapter_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError:
        return None
    base_model = adapter_cfg.get("base_model_name_or_path")
    return str(base_model) if base_model else None


def load_eval_tokenizer_and_model(
    cfg: dict[str, Any],
    interface: str,
    checkpoint: str | None,
):
    model_cfg = dict(cfg["model"])
    if checkpoint and _is_peft_adapter_checkpoint(checkpoint):
        model_cfg["name_or_path"] = (
            model_cfg.get("base_model_name_or_path")
            or _peft_adapter_base_model(checkpoint)
            or model_cfg["name_or_path"]
        )
        tokenizer, model = load_tokenizer_and_model(
            model_cfg,
            add_setswitch_tokens=interface == "setswitch",
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, checkpoint)
        if interface == "setswitch":
            model = load_special_token_embeddings(model, checkpoint)
        return tokenizer, model

    if checkpoint:
        model_cfg["name_or_path"] = checkpoint
    tokenizer, model = load_tokenizer_and_model(
        model_cfg,
        add_setswitch_tokens=interface == "setswitch",
    )
    if checkpoint and interface == "setswitch":
        model = load_special_token_embeddings(model, checkpoint)
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--interface", choices=["setswitch", "chat_baseline", "setllm"])
    parser.add_argument("--split")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--gold-positions", nargs="+", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    if args.interface:
        cfg["model_interface"] = args.interface
    cfg = apply_interface_overrides(cfg)
    eval_cfg = cfg.get("eval", {})
    interface = cfg.get("model_interface", "setswitch")
    split = args.split or eval_cfg.get("split", "dev")
    max_examples = (
        args.max_examples
        if args.max_examples is not None
        else int(eval_cfg.get("max_examples", 200))
    )
    max_new_tokens = (
        args.max_new_tokens
        if args.max_new_tokens is not None
        else int(eval_cfg.get("max_new_tokens", 32))
    )
    gold_positions = (
        list(args.gold_positions)
        if args.gold_positions is not None
        else _as_float_list(eval_cfg.get("gold_positions"), DEFAULT_GOLD_POSITIONS)
    )
    checkpoint = args.checkpoint or eval_cfg.get("checkpoint")

    tokenizer, model = load_eval_tokenizer_and_model(cfg, interface, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    examples = _load_eval_examples(cfg, split, max_examples)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    rows = []

    for example in examples:
        task = example.metadata.get("task_group") or task_group_for_source(example.source)
        sweep_status = gold_sweep_status(example)
        for fraction in gold_positions:
            placed = place_gold_documents(example, fraction)
            rendered = _render(interface, placed, tokenizer, cfg)
            prediction = generate_prediction(
                interface=interface,
                model=model,
                tokenizer=tokenizer,
                rendered=rendered,
                cfg=cfg,
                max_new_tokens=max_new_tokens,
            )
            correct = _matches(prediction, _answers(example))
            key = f"{task}@{fraction:g}"
            counts[key]["correct"] += int(correct)
            counts[key]["total"] += 1
            counts[key]["movable_total"] += int(bool(sweep_status["gold_sweep_movable"]))
            rows.append(
                {
                    "example_id": example.example_id,
                    "task_group": task,
                    "gold_position": fraction,
                    "prediction": prediction,
                    "answers": _answers(example),
                    "correct": correct,
                    **sweep_status,
                }
            )

    summary = {
        key: {
            "accuracy": value["correct"] / max(1, value["total"]),
            "correct": value["correct"],
            "total": value["total"],
            "movable_total": value["movable_total"],
            "movable_fraction": value["movable_total"] / max(1, value["total"]),
        }
        for key, value in sorted(counts.items())
    }
    report = {
        "interface": interface,
        "split": split,
        "max_examples": max_examples,
        "gold_positions": gold_positions,
        "summary": summary,
        "rows": rows,
    }

    output_template = args.output or eval_cfg.get(
        "output", "outputs/{run_name}_{interface}_{split}_eval.json"
    )
    output_path = Path(_format_output_path(output_template, cfg, interface, split))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
