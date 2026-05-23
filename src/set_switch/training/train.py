"""Minimal Accelerate training loop."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from set_switch.config_validation import validate_config
from set_switch.data.baseline_render import render_chat_baseline_example
from set_switch.data.collator import SetSwitchCollator
from set_switch.data.dataset_suite import (
    load_flashrag_selected_examples,
    normalize_flashrag_sources,
)
from set_switch.data.length_filter import max_rendered_length, normalize_length_filter_interfaces
from set_switch.data.plain_collator import PlainCausalCollator
from set_switch.data.render import render_example
from set_switch.data.schema import SetSwitchExample
from set_switch.data.setfuse_collator import SetFuseCollator
from set_switch.data.setfuse_render import render_setfuse_example
from set_switch.data.setllm_collator import SetLLMCollator
from set_switch.data.setllm_render import render_setllm_example
from set_switch.modeling.attention_mask import build_setswitch_attention_mask
from set_switch.modeling.layer_masking import (
    install_setfuse_layerwise_attention,
    resolve_fuse_start_layer,
    set_setfuse_masks,
)
from set_switch.modeling.load_model import dtype_from_name, load_tokenizer_and_model
from set_switch.modeling.peft_setup import (
    apply_trainable_parameter_policy,
    configure_special_token_lr_multipliers,
    merged_special_token_state_dict,
    maybe_apply_lora,
    save_special_token_embeddings,
)
from set_switch.modeling.setfuse_attention_mask import build_setfuse_layer_masks
from set_switch.modeling.setllm import build_setllm_attention_mask
from set_switch.modeling.special_tokens import active_token_id_map
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
        if self.interface == "setfuse":
            return render_setfuse_example(self.examples[idx], self.tokenizer, self.render_cfg)
        raise ValueError(f"Unknown model interface {self.interface!r}")


def _load_examples(
    cfg: dict[str, Any],
    split: str,
    example_filter: Any | None = None,
) -> list[SetSwitchExample]:
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
                "Use the provided passages or options to answer the question.",
            ),
            total_examples=data_cfg.get(total_key),
            sample_allocation=data_cfg.get("sample_allocation", "task_balanced_equal"),
            sample_allocation_alpha=float(data_cfg.get("sample_allocation_alpha", 0.5)),
            example_filter=example_filter,
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
    interface = cfg.get(
        "model_interface",
        cfg.get("interface", cfg.get("model", {}).get("interface", "setswitch")),
    )
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
    if bool((train_cfg or {}).get("bf16", False)):
        return torch.bfloat16
    if bool((train_cfg or {}).get("fp16", False)):
        return torch.float16
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


def mixed_precision_from_train_config(train_cfg: dict[str, Any]) -> str:
    if train_cfg.get("mixed_precision") is not None:
        return str(train_cfg.get("mixed_precision"))
    if bool(train_cfg.get("bf16", False)):
        return "bf16"
    if bool(train_cfg.get("fp16", False)):
        return "fp16"
    return "no"


def custom_mask_build_on_device(interface: str, mask_cfg: dict[str, Any]) -> bool:
    """Whether training should build large custom masks after device placement."""

    return interface in {"setswitch", "setllm", "setfuse"} and bool(
        mask_cfg.get("build_on_device", True)
    )


def make_train_collator(
    interface: str,
    tokenizer: Any,
    mask_cfg: dict[str, Any],
    mask_dtype: torch.dtype,
    build_attention_mask: bool = True,
):
    """Create the collator for an interface.

    For custom-mask methods, ``build_attention_mask=False`` pads only compact
    metadata. The dense mask can then be built directly on the accelerator
    device, avoiding a large host-to-device transfer each batch.
    """

    if interface == "setswitch":
        return SetSwitchCollator(
            tokenizer=tokenizer,
            attention_mode=mask_cfg.get("doc_attention", "doc_causal"),
            answer_attends_raw_docs=bool(mask_cfg.get("answer_attends_raw_docs", False)),
            answer_attends_reads=bool(mask_cfg.get("answer_attends_reads", False)),
            mask_dtype=mask_dtype,
            build_attention_mask=build_attention_mask,
        )
    if interface == "setllm":
        return SetLLMCollator(
            tokenizer=tokenizer,
            mask_dtype=mask_dtype,
            build_attention_mask=build_attention_mask,
        )
    if interface == "setfuse":
        return SetFuseCollator(
            tokenizer=tokenizer,
            mask_dtype=mask_dtype,
            setfuse_answer_attends_docs_in_early_layers=bool(
                mask_cfg.get("setfuse_answer_attends_docs_in_early_layers", False)
            ),
            setfuse_late_prefix_doc_bidir=bool(mask_cfg.get("setfuse_late_prefix_doc_bidir", True)),
            build_attention_mask=build_attention_mask,
        )
    return PlainCausalCollator(tokenizer=tokenizer)


def build_custom_attention_masks_from_batch(
    batch: dict[str, Any],
    interface: str,
    mask_cfg: dict[str, Any],
    mask_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Return custom masks, building them on the batch device when absent."""

    device = batch["input_ids"].device
    if interface == "setswitch":
        if "attention_mask" not in batch:
            batch["attention_mask"] = build_setswitch_attention_mask(
                role_ids=batch["role_ids"],
                item_ids=batch["item_ids"],
                read_slot_ids=batch.get("read_slot_ids"),
                gather_slot_ids=batch.get("gather_slot_ids"),
                attention_mode=mask_cfg.get("doc_attention", "doc_causal"),
                answer_attends_raw_docs=bool(mask_cfg.get("answer_attends_raw_docs", False)),
                answer_attends_reads=bool(mask_cfg.get("answer_attends_reads", False)),
                pad_mask=batch.get("pad_mask"),
                dtype=mask_dtype,
                device=device,
            )
        return {"attention_mask": batch["attention_mask"]}

    if interface == "setllm":
        if "attention_mask" not in batch:
            batch["attention_mask"] = build_setllm_attention_mask(
                role_ids=batch["role_ids"],
                item_ids=batch["item_ids"],
                pad_mask=batch.get("pad_mask"),
                dtype=mask_dtype,
                device=device,
            )
        return {"attention_mask": batch["attention_mask"]}

    if interface == "setfuse":
        if "attention_mask_early" not in batch or "attention_mask_late" not in batch:
            layer_masks = build_setfuse_layer_masks(
                role_ids=batch["role_ids"],
                item_ids=batch["item_ids"],
                pad_mask=batch.get("pad_mask"),
                dtype=mask_dtype,
                device=device,
                setfuse_answer_attends_docs_in_early_layers=bool(
                    mask_cfg.get("setfuse_answer_attends_docs_in_early_layers", False)
                ),
                setfuse_late_prefix_doc_bidir=bool(
                    mask_cfg.get("setfuse_late_prefix_doc_bidir", True)
                ),
            )
            batch["attention_mask_early"] = layer_masks["early"]
            batch["attention_mask_late"] = layer_masks["late"]
        return {
            "attention_mask_early": batch["attention_mask_early"],
            "attention_mask_late": batch["attention_mask_late"],
        }

    raise ValueError(f"Interface {interface!r} does not use a custom attention mask")


def infinite_dataloader(dataloader: DataLoader) -> Iterator[Any]:
    """Repeat a dataloader without caching batches between epochs."""

    while True:
        yielded = False
        for batch in dataloader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("Cannot train with an empty dataloader")


def _nonfinite_batch_summary(batch: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "example_id": batch.get("example_id", []),
        "input_shape": tuple(batch["input_ids"].shape),
    }
    if "pad_mask" in batch:
        summary["input_lengths"] = batch["pad_mask"].sum(dim=1).detach().cpu().tolist()
    if "answer_start" in batch:
        summary["answer_start"] = batch["answer_start"].detach().cpu().tolist()
    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"]
        summary["attention_mask_finite"] = bool(torch.isfinite(attention_mask).all().item())
        summary["attention_rows_with_allowed_key"] = bool(
            (attention_mask[:, 0] == 0).any(dim=-1).all().item()
        )
    if "attention_mask_early" in batch and "attention_mask_late" in batch:
        for stage in ("early", "late"):
            attention_mask = batch[f"attention_mask_{stage}"]
            summary[f"attention_mask_{stage}_finite"] = bool(
                torch.isfinite(attention_mask).all().item()
            )
            summary[f"attention_rows_with_allowed_key_{stage}"] = bool(
                (attention_mask[:, 0] == 0).any(dim=-1).all().item()
            )
    return summary


def optimizer_param_groups(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    special_token_learning_rate: float | None = None,
) -> list[dict[str, Any]]:
    base_params: list[torch.nn.Parameter] = []
    special_token_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".special_embeddings." in name or name.endswith("special_embeddings.weight"):
            special_token_params.append(param)
        else:
            base_params.append(param)

    groups: list[dict[str, Any]] = []
    if base_params:
        groups.append(
            {
                "params": base_params,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "name": "base",
            }
        )
    if special_token_params:
        groups.append(
            {
                "params": special_token_params,
                "lr": special_token_learning_rate or learning_rate,
                "weight_decay": weight_decay,
                "name": "setswitch_special_tokens",
            }
        )
    if not groups:
        raise ValueError("No trainable parameters found")
    return groups


def _save_training_checkpoint(
    model: Any,
    tokenizer: Any,
    save_path: Path,
    accelerator: Accelerator,
) -> None:
    unwrapped = accelerator.unwrap_model(model)
    state_dict = merged_special_token_state_dict(unwrapped)
    unwrapped.save_pretrained(
        save_path,
        save_function=accelerator.save,
        state_dict=state_dict,
    )
    if accelerator.is_local_main_process:
        save_special_token_embeddings(unwrapped, save_path)
        tokenizer.save_pretrained(save_path)


def train_from_config(cfg: dict[str, Any]) -> None:
    cfg = apply_interface_overrides(cfg)
    if cfg.get("_cli_run_name"):
        cfg["run_name"] = cfg.pop("_cli_run_name")
    validate_config(cfg)
    set_seed(int(cfg.get("seed", 42)))
    train_cfg = cfg.get("train", cfg.get("training", {}))
    model_cfg = cfg.get("model", {})
    mask_cfg = cfg.get("mask", {})
    interface = cfg.get(
        "model_interface",
        cfg.get("interface", model_cfg.get("interface", "setswitch")),
    )
    if interface not in {"setswitch", "chat_baseline", "setllm", "setfuse"}:
        raise ValueError(
            "model_interface must be one of: setswitch, chat_baseline, setllm, setfuse"
        )

    accelerator = Accelerator(mixed_precision=mixed_precision_from_train_config(train_cfg))
    tokenizer, model = load_tokenizer_and_model(
        model_cfg,
        add_setswitch_tokens=interface == "setswitch",
    )
    model = maybe_apply_lora(model, model_cfg)
    setswitch_token_ids = None
    if interface == "setswitch":
        setswitch_token_ids = active_token_id_map(tokenizer, dict(cfg.get("data", {})))
    trainable_token_ids = list(setswitch_token_ids.values()) if setswitch_token_ids else None
    model = apply_trainable_parameter_policy(model, model_cfg, trainable_token_ids)
    configure_special_token_lr_multipliers(model, setswitch_token_ids, train_cfg)
    fuse_start_layer: int | None = None
    if interface == "setfuse":
        fuse_start_layer = resolve_fuse_start_layer(
            model,
            mask_cfg.get("fuse_start_layer", "auto_half"),
        )
        install_setfuse_layerwise_attention(model, fuse_start_layer)
    if bool(train_cfg.get("gradient_checkpointing", False)):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    data_cfg = cfg.get("data", {})
    train_length_dropped = 0
    train_max_render_tokens = data_cfg.get("train_max_render_tokens")
    train_example_filter = None
    if train_max_render_tokens is not None:
        train_max_render_tokens = int(train_max_render_tokens)
        length_filter_interfaces = list(
            normalize_length_filter_interfaces(data_cfg.get("length_filter_interfaces", "all"))
        )
        if "setfuse" not in length_filter_interfaces:
            length_filter_interfaces.append("setfuse")

        def train_example_filter(example: SetSwitchExample) -> bool:
            nonlocal train_length_dropped
            keep = (
                max_rendered_length(
                    example=example,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    interfaces=tuple(length_filter_interfaces),
                )
                <= train_max_render_tokens
            )
            train_length_dropped += int(not keep)
            return keep

    train_examples = _load_examples(cfg, "train", train_example_filter)
    val_examples = _load_examples(cfg, "val")
    render_cfg = {"data": cfg.get("data", {})}
    custom_mask_dtype = attention_mask_dtype_from_config(model_cfg, train_cfg, model)
    build_masks_on_device = custom_mask_build_on_device(interface, mask_cfg)

    train_dataset = RenderedSetSwitchDataset(train_examples, tokenizer, render_cfg, interface)
    val_dataset = RenderedSetSwitchDataset(val_examples, tokenizer, render_cfg, interface)
    train_collator = make_train_collator(
        interface,
        tokenizer,
        mask_cfg,
        custom_mask_dtype,
        build_attention_mask=not build_masks_on_device,
    )
    val_collator = make_train_collator(
        interface,
        tokenizer,
        mask_cfg,
        custom_mask_dtype,
        build_attention_mask=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        collate_fn=train_collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=False,
        collate_fn=val_collator,
    )

    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    special_token_learning_rate = train_cfg.get("special_token_learning_rate")
    special_token_learning_rate = (
        float(special_token_learning_rate) if special_token_learning_rate is not None else None
    )
    optimizer = AdamW(
        optimizer_param_groups(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            special_token_learning_rate=special_token_learning_rate,
        )
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(step, int(train_cfg.get("warmup_steps", 0))),
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    max_steps = int(train_cfg.get("max_steps", 100))
    grad_accum_steps = int(
        train_cfg.get("grad_accum_steps", train_cfg.get("gradient_accumulation_steps", 1))
    )
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
        learning_rate=learning_rate,
        special_token_learning_rate=special_token_learning_rate,
        custom_mask_dtype=str(custom_mask_dtype).replace("torch.", ""),
        custom_mask_build_on_device=build_masks_on_device,
        train_max_render_tokens=train_max_render_tokens,
        train_length_dropped=train_length_dropped,
        fuse_start_layer=fuse_start_layer,
    )

    model.train()
    running_loss = 0.0
    running_active_tokens = 0
    running_padded_tokens = 0
    running_examples = 0
    running_log_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process)
    train_iter = infinite_dataloader(train_loader)

    for step in progress:
        for _ in range(grad_accum_steps):
            batch = next(train_iter)
            running_padded_tokens += int(batch["input_ids"].numel())
            running_examples += int(batch["input_ids"].shape[0])
            if "pad_mask" in batch:
                running_active_tokens += int(batch["pad_mask"].sum().item())
            else:
                running_active_tokens += int(batch["input_ids"].numel())
            if interface == "setfuse":
                masks = build_custom_attention_masks_from_batch(
                    batch,
                    interface,
                    mask_cfg,
                    custom_mask_dtype,
                )
                model_batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": masks["attention_mask_early"],
                    "labels": batch["labels"],
                    "use_cache": False,
                }
                if "position_ids" in batch:
                    model_batch["position_ids"] = batch["position_ids"]
                with set_setfuse_masks(
                    model,
                    masks["attention_mask_early"],
                    masks["attention_mask_late"],
                ):
                    outputs = model(**model_batch)
                    if not torch.isfinite(outputs.loss):
                        summary = _nonfinite_batch_summary(batch)
                        raise FloatingPointError(
                            f"Non-finite loss at optimizer_step={step + 1}: {summary}"
                        )
                    loss = outputs.loss / grad_accum_steps
                    accelerator.backward(loss)
            else:
                if interface in {"setswitch", "setllm"}:
                    attention_mask = build_custom_attention_masks_from_batch(
                        batch,
                        interface,
                        mask_cfg,
                        custom_mask_dtype,
                    )["attention_mask"]
                else:
                    attention_mask = batch["attention_mask"]
                model_batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": attention_mask,
                    "labels": batch["labels"],
                    "use_cache": False,
                }
                if "position_ids" in batch:
                    model_batch["position_ids"] = batch["position_ids"]
                outputs = model(**model_batch)
                if not torch.isfinite(outputs.loss):
                    summary = _nonfinite_batch_summary(batch)
                    raise FloatingPointError(
                        f"Non-finite loss at optimizer_step={step + 1}: {summary}"
                    )
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
            elapsed = max(time.perf_counter() - running_log_start, 1.0e-6)
            active_tokens_per_second = running_active_tokens / elapsed
            padded_tokens_per_second = running_padded_tokens / elapsed
            avg_active_tokens = running_active_tokens / max(running_examples, 1)
            avg_padded_tokens = running_padded_tokens / max(running_examples, 1)
            progress.set_postfix(loss=f"{avg_loss:.4f}", tok_s=f"{active_tokens_per_second:.0f}")
            metric_logger.log(
                event="train",
                step=step + 1,
                loss=avg_loss,
                learning_rate=float(scheduler.get_last_lr()[0]),
                active_tokens_per_second=active_tokens_per_second,
                padded_tokens_per_second=padded_tokens_per_second,
                avg_active_tokens_per_example=avg_active_tokens,
                avg_padded_tokens_per_example=avg_padded_tokens,
            )
            running_loss = 0.0
            running_active_tokens = 0
            running_padded_tokens = 0
            running_examples = 0
            running_log_start = time.perf_counter()

        if eval_every and (step + 1) % eval_every == 0:
            val_loss = evaluate_answer_ce(model, val_loader, max_batches=10)
            accelerator.print(f"step={step + 1} val_loss={val_loss:.4f}")
            metric_logger.log(event="eval", step=step + 1, val_loss=val_loss)

        if save_every and (step + 1) % save_every == 0:
            accelerator.wait_for_everyone()
            save_path = output_dir / f"step-{step + 1}"
            _save_training_checkpoint(model, tokenizer, save_path, accelerator)
            if accelerator.is_local_main_process:
                metric_logger.log(event="save", step=step + 1, path=str(save_path))

    accelerator.wait_for_everyone()
    final_path = output_dir / "final"
    _save_training_checkpoint(model, tokenizer, final_path, accelerator)
    if accelerator.is_local_main_process:
        metric_logger.log(event="save_final", step=max_steps, path=str(final_path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--interface",
        choices=["setswitch", "chat_baseline", "setllm", "setfuse"],
    )
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
