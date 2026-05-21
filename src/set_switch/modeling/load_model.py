"""Model/tokenizer loading."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from set_switch.modeling.special_tokens import (
    add_setswitch_special_tokens,
    ensure_tokenizer_has_pad_token,
)


def dtype_from_name(name: str | None) -> torch.dtype | str:
    if name is None or name == "auto":
        return "auto"
    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype {name!r}")


def load_tokenizer_and_model(model_cfg: dict[str, Any], add_setswitch_tokens: bool = True):
    name_or_path = model_cfg["name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    ensure_tokenizer_has_pad_token(tokenizer)
    dtype = dtype_from_name(model_cfg.get("dtype", "auto"))
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
    }
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    if add_setswitch_tokens:
        add_setswitch_special_tokens(tokenizer, model)
    elif model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    return tokenizer, model
