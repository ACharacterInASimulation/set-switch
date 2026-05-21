"""Minimal Accelerate training loop."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from set_switch.data.baseline_render import render_chat_baseline_example
from set_switch.data.collator import SetSwitchCollator
from set_switch.data.dataset_suite import (
    load_flashrag_selected_examples,
    normalize_flashrag_sources,
)
from set_switch.data.plain_collator import PlainCausalCollator
from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchExample
from set_switch.data.setllm_collator import SetLLMCollator
from set_switch.data.setllm_render import render_setllm_example
from set_switch.modeling.load_model import dtype_from_name, load_tokenizer_and_model
from set_switch.modeling.peft_setup import apply_trainable_parameter_policy, maybe_apply_lora
from set_switch.modeling.special_tokens import token_id_map
from set_switch.training.eval_simple import evaluate_answer_ce
from set_switch.utils.io import read_examples_jsonl, read_yaml
from set_switch.utils.logging import JsonlMetricLogger
from set_switch.utils.seed import set_seed


class RenderedSetSwitchDataset(Dataset):
    def __init__(
        self,
        examples: list[SetSwitchExample],
        tokenizer: Any,
        render_cfg: dict[str, Any],
        interface: str = "setswitch",
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.render_cfg = render_cfg
        self.interface = interface

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.interface == "setswitch":
            return render_example(self.examples[idx], self.tokenizer, self.render_cfg)
        if self.interface == "chat_baseline":
            return render_chat_baseline_example(self.examples[idx], self.tokenizer, self.render_cfg)
        if self.interface == "setllm":
            return render_setllm_example(self.examples[idx], self.tokenizer, self.render_cfg)
        raise ValueError(f"Unknown model interface {self.interface!r}")


def _load_examples(cfg: dict[str, Any], split: str) -> list[SetSwitchExample]:
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "flashrag")

    jsonl_key = f"{split}_jsonl"
    if data_cfg.get(jsonl_key):
        return read_examples_jsonl(data_cfg[jsonl_key])

    if source == "flashrag":
        total_key = "total_train_examples" if split == "train" else "total_val_examples"
        selections = normalize_flashrag_sources(data_cfg, split)
        return load_flashrag_selected_examples(
            dataset_name=data_cfg.get("dataset_name", "RUC-NLPIR/FlashRAG_datasets"),
            selections=selections,
            max_docs=int(data_cfg.get("max_docs", 8)),
            instruction=data_cfg.get(
                "instruction",
                "Use the provided passages or options to answer the question. Treat the items as an unordered set.",
            ),
            total_examples=data_cfg.get(total_key),
            sample_allocation=data_cfg.get("sample_allocation", "task_balanced_equal"),
            sample_allocation_alpha=float(data_cfg.get("sample_allocation_alpha", 0.5)),
        )

    raise ValueError("Only data.source='flashrag' is supported in the simplified training path")


def _lr_lambda(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(warmup_steps))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def apply_interface_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    interface = cfg.get("model_interface", cfg.get("interface", "setswitch"))
    overrides = cfg.get("method_overrides", {}).get(interface, {})
    return _deep_merge(cfg, overrides)


def attention_mask_dtype_from_config(
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any] | None = None,
    model: torch.nn.Module | None = None,
) -> torch.dtype:
    """Choose the additive custom-mask dtype for SetSwitch/SetLLM."""

    explicit_dtype = model_cfg.get("mask_dtype") or model_cfg.get("attention_mask_dtype")
    if explicit_dtype:
        dtype = dtype_from_name(str(explicit_dtype))
        if isinstance(dtype, torch.dtype):
            return dtype
        raise ValueError("mask_dtype cannot be 'auto'; choose bf16, fp16, or fp32")

    model_dtype = dtype_from_name(model_cfg.get("dtype", "auto"))
    if isinstance(model_dtype, torch.dtype):
        return model_dtype

    mixed_precision = str((train_cfg or {}).get("mixed_precision", "no")).lower()
    if mixed_precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if mixed_precision in {"fp16", "float16"}:
        return torch.float16

    if model is not None:
        try:
            param_dtype = next(model.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32
        if param_dtype.is_floating_point:
            return param_dtype

    return torch.float32


def infinite_dataloader(dataloader: DataLoader) -> Iterator[Any]:
    """Repeat a dataloader without caching batches between epochs."""

    while True:
        yielded = False
        for batch in dataloader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("Cannot train with an empty dataloader")


def train_from_config(cfg: dict[str, Any]) -> None:
    cfg = apply_interface_overrides(cfg)
    if cfg.get("_cli_run_name"):
        cfg["run_name"] = cfg.pop("_cli_run_name")
    set_seed(int(cfg.get("seed", 42)))
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    mask_cfg = cfg.get("mask", {})
    interface = cfg.get("model_interface", cfg.get("interface", "setswitch"))
    if interface not in {"setswitch", "chat_baseline", "setllm"}:
        raise ValueError("model_interface must be one of: setswitch, chat_baseline, setllm")

    accelerator = Accelerator(mixed_precision=train_cfg.get("mixed_precision", "no"))
    tokenizer, model = load_tokenizer_and_model(
        model_cfg,
        add_setswitch_tokens=interface == "setswitch",
    )
    model = maybe_apply_lora(model, model_cfg)
    trainable_token_ids = (
        list(token_id_map(tokenizer).values()) if interface == "setswitch" else None
    )
    model = apply_trainable_parameter_policy(model, model_cfg, trainable_token_ids)

    train_examples = _load_examples(cfg, "train")
    val_examples = _load_examples(cfg, "val")
    render_cfg = {"data": cfg.get("data", {})}
    custom_mask_dtype = attention_mask_dtype_from_config(model_cfg, train_cfg, model)

    train_dataset = RenderedSetSwitchDataset(train_examples, tokenizer, render_cfg, interface)
    val_dataset = RenderedSetSwitchDataset(val_examples, tokenizer, render_cfg, interface)
    if interface == "setswitch":
        collator = SetSwitchCollator(
            tokenizer=tokenizer,
            attention_mode=mask_cfg.get("doc_attention", "doc_causal"),
            mask_dtype=custom_mask_dtype,
        )
    elif interface == "setllm":
        collator = SetLLMCollator(tokenizer=tokenizer, mask_dtype=custom_mask_dtype)
    else:
        collator = PlainCausalCollator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(step, int(train_cfg.get("warmup_steps", 0))),
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    max_steps = int(train_cfg.get("max_steps", 100))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    log_every = int(train_cfg.get("log_every", 10))
    eval_every = int(train_cfg.get("eval_every", 0))
    save_every = int(train_cfg.get("save_every", 0))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    output_dir = Path(cfg.get("output_dir", "outputs")) / cfg.get("run_name", "set_switch_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / str(train_cfg.get("metrics_file", "metrics.jsonl"))
    metric_logger = JsonlMetricLogger(
        metrics_path,
        enabled=accelerator.is_local_main_process and bool(train_cfg.get("local_logging", True)),
    )
    metric_logger.log(
        event="run_start",
        run_name=cfg.get("run_name", "set_switch_run"),
        interface=interface,
        train_examples=len(train_examples),
        val_examples=len(val_examples),
        max_steps=max_steps,
        grad_accum_steps=grad_accum_steps,
        batch_size=int(train_cfg.get("batch_size", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
        custom_mask_dtype=str(custom_mask_dtype).replace("torch.", ""),
    )

    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    train_iter = infinite_dataloader(train_loader)

    for step in progress:
        for _ in range(grad_accum_steps):
            batch = next(train_iter)
            model_batch = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
                "use_cache": False,
            }
            if "position_ids" in batch:
                model_batch["position_ids"] = batch["position_ids"]
            outputs = model(**model_batch)
            loss = outputs.loss / grad_accum_steps
            accelerator.backward(loss)
            running_loss += float(loss.detach().cpu())

        if max_grad_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if accelerator.is_local_main_process and (step + 1) % log_every == 0:
            avg_loss = running_loss / log_every
            progress.set_postfix(loss=f"{avg_loss:.4f}")
            metric_logger.log(
                event="train",
                step=step + 1,
                loss=avg_loss,
                learning_rate=float(scheduler.get_last_lr()[0]),
            )
            running_loss = 0.0

        if eval_every and (step + 1) % eval_every == 0:
            val_loss = evaluate_answer_ce(model, val_loader, max_batches=10)
            accelerator.print(f"step={step + 1} val_loss={val_loss:.4f}")
            metric_logger.log(event="eval", step=step + 1, val_loss=val_loss)

        if save_every and (step + 1) % save_every == 0:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)
            save_path = output_dir / f"step-{step + 1}"
            unwrapped.save_pretrained(save_path, save_function=accelerator.save)
            if accelerator.is_local_main_process:
                tokenizer.save_pretrained(save_path)
                metric_logger.log(event="save", step=step + 1, path=str(save_path))

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    final_path = output_dir / "final"
    unwrapped.save_pretrained(final_path, save_function=accelerator.save)
    if accelerator.is_local_main_process:
        tokenizer.save_pretrained(final_path)
        metric_logger.log(event="save_final", step=max_steps, path=str(final_path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--interface", choices=["setswitch", "chat_baseline", "setllm"])
    parser.add_argument("--run-name")
    args = parser.parse_args(argv)
    cfg = read_yaml(args.config)
    if args.interface:
        cfg["model_interface"] = args.interface
    if args.run_name:
        cfg["_cli_run_name"] = args.run_name
    train_from_config(cfg)


if __name__ == "__main__":
    main()
