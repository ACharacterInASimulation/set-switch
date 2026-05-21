"""PEFT/LoRA setup."""

from __future__ import annotations

from typing import Any

import torch


def maybe_apply_lora(model: Any, model_cfg: dict[str, Any]) -> Any:
    if not model_cfg.get("use_lora", False):
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = model_cfg.get("lora_target_modules", "all-linear")
    if isinstance(target_modules, tuple):
        target_modules = list(target_modules)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(model_cfg.get("lora_r", 8)),
        lora_alpha=int(model_cfg.get("lora_alpha", 1)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.0)),
        target_modules=target_modules,
        modules_to_save=model_cfg.get("lora_modules_to_save"),
    )
    lora_model = get_peft_model(model, config)
    if hasattr(lora_model, "print_trainable_parameters"):
        lora_model.print_trainable_parameters()
    return lora_model


def _mask_embedding_gradients(embedding: Any, token_ids: list[int]) -> None:
    mask = torch.zeros_like(embedding.weight, dtype=torch.bool)
    mask[token_ids] = True

    def hook(grad: torch.Tensor) -> torch.Tensor:
        return grad * mask.to(device=grad.device, dtype=grad.dtype)

    embedding.weight.register_hook(hook)


def apply_trainable_parameter_policy(
    model: Any,
    model_cfg: dict[str, Any],
    trainable_token_ids: list[int] | None = None,
) -> Any:
    """Apply non-LoRA trainability policies.

    ``use_lora=true`` is handled by PEFT. With ``use_lora=false`` the default is
    full fine-tuning unless ``trainable`` asks for a narrower policy.
    """

    if model_cfg.get("use_lora", False):
        return model

    policy = str(model_cfg.get("trainable", "full")).lower()
    if policy == "full":
        return model
    if policy not in {"setswitch_tokens", "special_tokens_only"}:
        raise ValueError(f"Unknown trainable policy {policy!r}; expected full or setswitch_tokens")
    if not trainable_token_ids:
        raise ValueError("trainable=setswitch_tokens requires SetSwitch token ids")

    for param in model.parameters():
        param.requires_grad = False

    input_embeddings = model.get_input_embeddings()
    input_embeddings.weight.requires_grad = True
    _mask_embedding_gradients(input_embeddings, trainable_token_ids)

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings.weight is not input_embeddings.weight:
        output_embeddings.weight.requires_grad = True
        _mask_embedding_gradients(output_embeddings, trainable_token_ids)

    return model
