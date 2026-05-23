"""Configuration validation for interface combinations."""

from __future__ import annotations

from typing import Any

from set_switch.constants import DOC_ATTENTION_MODES, GATHER_TOKENS, READ_TOKENS

INTERFACES = {"setswitch", "chat_baseline", "setllm", "setfuse"}
CUSTOM_MASK_INTERFACES = {"setswitch", "setllm", "setfuse"}
MIXED_PRECISION_VALUES = {"no", "fp16", "float16", "bf16", "bfloat16"}
MCQ_SCORING_VALUES = {"logprob", "teacher_forced", "teacher-forced", "generative"}


def _interface(cfg: dict[str, Any]) -> str:
    model_cfg = cfg.get("model", {})
    return str(
        cfg.get("model_interface", cfg.get("interface", model_cfg.get("interface", "setswitch")))
    )


def _active_setswitch_token_count(interface: str, data_cfg: dict[str, Any]) -> int:
    if interface != "setswitch":
        return 0
    num_reads = int(data_cfg.get("num_reads_per_doc", 2))
    num_gathers = int(data_cfg.get("num_gather_tokens", 4))
    boundary = bool(data_cfg.get("setswitch_boundary_tokens", True))
    return (4 if boundary else 0) + num_reads + num_gathers


def _require_bool(section: dict[str, Any], section_name: str, key: str) -> None:
    if key in section and not isinstance(section[key], bool):
        raise ValueError(
            f"{section_name}.{key} must be a YAML boolean true/false, got {section[key]!r}"
        )


def validate_config(cfg: dict[str, Any], require_model_name: bool = True) -> None:
    """Raise clear errors for unsupported config combinations."""

    interface = _interface(cfg)
    if interface not in INTERFACES:
        raise ValueError(
            f"Unknown model_interface {interface!r}; expected one of {sorted(INTERFACES)}"
        )

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    mask_cfg = cfg.get("mask", {})
    train_cfg = cfg.get("train", cfg.get("training", {}))
    eval_cfg = cfg.get("eval", {})

    if require_model_name and not model_cfg.get("name_or_path"):
        raise ValueError("model.name_or_path is required")

    for key in ("use_lora", "allow_non_eager_custom_masks"):
        _require_bool(model_cfg, "model", key)
    for key in (
        "setswitch_boundary_tokens",
        "compact_special_token_format",
        "append_eos_token",
        "setllm_append_eos_token",
    ):
        _require_bool(data_cfg, "data", key)
    for key in (
        "build_on_device",
        "answer_attends_raw_docs",
        "answer_attends_reads",
        "setfuse_answer_attends_docs_in_early_layers",
        "setfuse_late_prefix_doc_bidir",
    ):
        _require_bool(mask_cfg, "mask", key)
    for key in ("bf16", "fp16", "gradient_checkpointing", "local_logging"):
        _require_bool(train_cfg, "train", key)

    attn_impl = model_cfg.get("attn_implementation")
    if interface in CUSTOM_MASK_INTERFACES and attn_impl not in {None, "eager"}:
        if not bool(model_cfg.get("allow_non_eager_custom_masks", False)):
            raise ValueError(
                f"{interface} uses custom 4D masks; set model.attn_implementation: eager "
                "or set model.allow_non_eager_custom_masks: true after validating parity."
            )

    trainable = str(model_cfg.get("trainable", "full")).lower()
    if interface == "setfuse" and trainable in {"setswitch_tokens", "special_tokens_only"}:
        raise ValueError(
            "SetFuse-LM does not use SetSwitch special tokens; use LoRA or full training instead "
            "of trainable=setswitch_tokens/special_tokens_only."
        )
    if trainable in {"setswitch_tokens", "special_tokens_only"}:
        active_tokens = _active_setswitch_token_count(interface, data_cfg)
        if active_tokens <= 0:
            raise ValueError(
                f"trainable={trainable!r} for {interface} requires at least one rendered "
                "SetSwitch special token. Enable setswitch_boundary_tokens or use LoRA/full training."
            )

    if mask_cfg.get("doc_attention", "doc_causal") not in DOC_ATTENTION_MODES:
        raise ValueError(
            f"mask.doc_attention must be one of {sorted(DOC_ATTENTION_MODES)}, "
            f"got {mask_cfg.get('doc_attention')!r}"
        )

    fuse_start_layer = mask_cfg.get("fuse_start_layer", "auto_half")
    if fuse_start_layer != "auto_half":
        try:
            if int(fuse_start_layer) < 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "mask.fuse_start_layer must be 'auto_half' or a non-negative integer"
            ) from exc

    num_reads = int(data_cfg.get("num_reads_per_doc", 2))
    num_gathers = int(data_cfg.get("num_gather_tokens", 4))
    if not 0 <= num_reads <= len(READ_TOKENS):
        raise ValueError(f"data.num_reads_per_doc must be in [0, {len(READ_TOKENS)}]")
    if not 0 <= num_gathers <= len(GATHER_TOKENS):
        raise ValueError(f"data.num_gather_tokens must be in [0, {len(GATHER_TOKENS)}]")

    max_docs = int(data_cfg.get("max_docs", 8))
    if max_docs <= 0:
        raise ValueError("data.max_docs must be positive")
    max_doc_tokens = data_cfg.get("max_doc_tokens")
    if max_doc_tokens is not None and int(max_doc_tokens) <= 0:
        raise ValueError("data.max_doc_tokens must be positive when set")
    train_max_render_tokens = data_cfg.get("train_max_render_tokens")
    if train_max_render_tokens is not None and int(train_max_render_tokens) <= 0:
        raise ValueError("data.train_max_render_tokens must be positive when set")

    batch_size = int(train_cfg.get("batch_size", 1))
    grad_accum = int(
        train_cfg.get("grad_accum_steps", train_cfg.get("gradient_accumulation_steps", 1))
    )
    if batch_size <= 0:
        raise ValueError("train.batch_size must be positive")
    if grad_accum <= 0:
        raise ValueError("train.grad_accum_steps/gradient_accumulation_steps must be positive")
    if bool(train_cfg.get("bf16", False)) and bool(train_cfg.get("fp16", False)):
        raise ValueError("train.bf16 and train.fp16 cannot both be true")
    mixed_precision = str(train_cfg.get("mixed_precision", "no")).lower()
    if mixed_precision not in MIXED_PRECISION_VALUES:
        raise ValueError(
            f"train.mixed_precision must be one of {sorted(MIXED_PRECISION_VALUES)}, "
            f"got {mixed_precision!r}"
        )

    mcq_scoring = str(eval_cfg.get("mcq_scoring", "logprob")).lower()
    if mcq_scoring not in MCQ_SCORING_VALUES:
        raise ValueError(
            f"eval.mcq_scoring must be one of {sorted(MCQ_SCORING_VALUES)}, got {mcq_scoring!r}"
        )
