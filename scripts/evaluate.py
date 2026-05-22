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


def _answer_tokens(text: Any) -> list[str]:
    return _normalize_answer(text).split()


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


def _token_f1(prediction: str, answers: list[str]) -> float:
    pred_tokens = _answer_tokens(prediction.splitlines()[0] if prediction else prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        gold_tokens = _answer_tokens(answer)
        if not gold_tokens:
            continue
        common = set(pred_tokens) & set(gold_tokens)
        overlap = sum(min(pred_tokens.count(token), gold_tokens.count(token)) for token in common)
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _lcs_length(left: list[str], right: list[str]) -> int:
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


def _rouge_l(prediction: str, answers: list[str]) -> float:
    pred_tokens = _answer_tokens(prediction.splitlines()[0] if prediction else prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        gold_tokens = _answer_tokens(answer)
        if not gold_tokens:
            continue
        lcs = _lcs_length(pred_tokens, gold_tokens)
        if lcs == 0:
            continue
        precision = lcs / len(pred_tokens)
        recall = lcs / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _primary_metric_for_example(example: SetSwitchExample) -> str:
    source = example.source.removeprefix("flashrag_")
    if example.metadata.get("set_type") == "options" or task_group_for_source(source) == "normal_mcq":
        return "accuracy"
    if source == "qasc":
        return "accuracy"
    if source == "boolq":
        return "accuracy"
    if source == "msmarco-qa":
        return "rouge_l"
    return "token_f1"


def score_prediction(example: SetSwitchExample, prediction: str) -> dict[str, Any]:
    answers = _answers(example)
    exact = _matches(prediction, answers)
    f1 = _token_f1(prediction, answers)
    rouge_l = _rouge_l(prediction, answers)
    primary_metric = _primary_metric_for_example(example)
    metric_values = {
        "accuracy": float(exact),
        "token_f1": f1,
        "rouge_l": rouge_l,
    }
    primary_score = metric_values[primary_metric]
    return {
        "primary_metric": primary_metric,
        "primary_score": primary_score,
        "exact_match": float(exact),
        "token_f1": f1,
        "rouge_l": rouge_l,
        "correct": bool(exact),
    }


METRIC_POLICY = {
    "normal_mcq": "normalized exact-match accuracy over option text",
    "options": "normalized exact-match accuracy over option text",
    "boolq": "normalized yes/no accuracy",
    "msmarco-qa": "max normalized ROUGE-L F1 over golden answers, with EM/F1 also logged",
    "qa": "max normalized token F1 over golden answers, with exact_match also logged",
}


def _empty_count() -> dict[str, float]:
    return {
        "correct": 0.0,
        "total": 0.0,
        "score_sum": 0.0,
        "exact_sum": 0.0,
        "f1_sum": 0.0,
        "rouge_l_sum": 0.0,
        "movable_total": 0.0,
    }


def _update_count(
    counts: dict[str, dict[str, float]],
    key: str,
    score: dict[str, Any],
    sweep_status: dict[str, int | bool],
) -> None:
    counts[key]["correct"] += int(score["correct"])
    counts[key]["total"] += 1
    counts[key]["score_sum"] += float(score["primary_score"])
    counts[key]["exact_sum"] += float(score["exact_match"])
    counts[key]["f1_sum"] += float(score["token_f1"])
    counts[key]["rouge_l_sum"] += float(score["rouge_l"])
    counts[key]["movable_total"] += int(bool(sweep_status["gold_sweep_movable"]))


def _summarize_counts(counts: dict[str, dict[str, float]]) -> dict[str, dict[str, float | int]]:
    return {
        key: {
            "primary_score": value["score_sum"] / max(1, value["total"]),
            "accuracy": value["correct"] / max(1, value["total"]),
            "exact_match": value["exact_sum"] / max(1, value["total"]),
            "token_f1": value["f1_sum"] / max(1, value["total"]),
            "rouge_l": value["rouge_l_sum"] / max(1, value["total"]),
            "correct": int(value["correct"]),
            "total": int(value["total"]),
            "movable_total": int(value["movable_total"]),
            "movable_fraction": value["movable_total"] / max(1, value["total"]),
        }
        for key, value in sorted(counts.items())
    }


def gold_position_sweep_enabled(interface: str) -> bool:
    """Only the ordinary causal baseline needs an explicit gold-position sweep."""

    return interface == "chat_baseline"


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
            answer_attends_raw_docs=bool(
                cfg.get("mask", {}).get("answer_attends_raw_docs", False)
            ),
            answer_attends_reads=bool(cfg.get("mask", {}).get("answer_attends_reads", False)),
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
        try:
            from peft import PeftModel
        except Exception as exc:
            raise ImportError(
                "This checkpoint is a PEFT adapter, but PEFT could not be imported. "
                "Install a compatible transformers/peft pair before evaluation."
            ) from exc

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
    sweep_gold_positions = gold_position_sweep_enabled(interface)
    task_counts: dict[str, dict[str, float]] = defaultdict(_empty_count)
    source_counts: dict[str, dict[str, float]] = defaultdict(_empty_count)
    overall_counts: dict[str, dict[str, float]] = defaultdict(_empty_count)
    rows = []

    for example in examples:
        task = (
            example.metadata.get("eval_task_group")
            or example.metadata.get("task_group")
            or task_group_for_source(example.source)
        )
        source = example.source.removeprefix("flashrag_")
        sweep_status = gold_sweep_status(example)
        scored_predictions: list[tuple[float, float | None, str, dict[str, Any]]] = []
        if sweep_gold_positions:
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
                scored_predictions.append(
                    (fraction, fraction, prediction, score_prediction(example, prediction))
                )
        else:
            rendered = _render(interface, example, tokenizer, cfg)
            prediction = generate_prediction(
                interface=interface,
                model=model,
                tokenizer=tokenizer,
                rendered=rendered,
                cfg=cfg,
                max_new_tokens=max_new_tokens,
            )
            score = score_prediction(example, prediction)
            scored_predictions = [
                (fraction, None, prediction, score) for fraction in gold_positions
            ]

        for fraction, effective_fraction, prediction, score in scored_predictions:
            _update_count(task_counts, f"{task}@{fraction:g}", score, sweep_status)
            _update_count(source_counts, f"{source}@{fraction:g}", score, sweep_status)
            _update_count(overall_counts, f"overall@{fraction:g}", score, sweep_status)
            rows.append(
                {
                    "example_id": example.example_id,
                    "source": example.source,
                    "source_config": source,
                    "task_group": task,
                    "gold_position": fraction,
                    "effective_gold_position": effective_fraction,
                    "gold_position_swept": sweep_gold_positions,
                    "prediction": prediction,
                    "answers": _answers(example),
                    **score,
                    **sweep_status,
                }
            )

    summary = _summarize_counts(task_counts)
    source_summary = _summarize_counts(source_counts)
    overall_summary = _summarize_counts(overall_counts)
    report = {
        "interface": interface,
        "split": split,
        "max_examples": max_examples,
        "gold_positions": gold_positions,
        "gold_position_sweep": sweep_gold_positions,
        "metric_policy": METRIC_POLICY,
        "summary": summary,
        "task_summary": summary,
        "source_summary": source_summary,
        "overall_summary": overall_summary,
        "rows": rows,
    }

    output_template = args.output or eval_cfg.get(
        "output", "outputs/{run_name}_{interface}_{split}_eval.json"
    )
    output_path = Path(_format_output_path(output_template, cfg, interface, split))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "task_summary": summary,
                "source_summary": source_summary,
                "overall_summary": overall_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
