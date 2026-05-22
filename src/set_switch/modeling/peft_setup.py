"""PEFT/LoRA setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from set_switch.constants import (
    END_ITEM_TOKEN,
    END_SET_TOKEN,
    GATHER_TOKENS,
    ITEM_TOKEN,
    READ_TOKENS,
    SET_TOKEN,
)

SPECIAL_TOKEN_EMBEDDINGS_FILE = "setswitch_special_embeddings.pt"


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


class SpecialTokenEmbeddingWrapper(nn.Module):
    """Freeze a base embedding and replace selected token ids with trainable rows."""

    def __init__(self, base_embedding: nn.Embedding, token_ids: list[int]) -> None:
        super().__init__()
        if not token_ids:
            raise ValueError("SpecialTokenEmbeddingWrapper requires at least one token id")
        self.base_embedding = base_embedding
        for param in self.base_embedding.parameters():
            param.requires_grad = False

        token_tensor = torch.tensor([int(token_id) for token_id in token_ids], dtype=torch.long)
        if len(torch.unique(token_tensor)) != len(token_tensor):
            raise ValueError("Special token ids must be unique")
        self.register_buffer("special_token_ids", token_tensor, persistent=True)

        id_to_slot = torch.full((int(token_tensor.max().item()) + 1,), -1, dtype=torch.long)
        id_to_slot[token_tensor] = torch.arange(len(token_tensor), dtype=torch.long)
        self.register_buffer("_id_to_slot", id_to_slot, persistent=False)

        self.special_embeddings = nn.Embedding(
            num_embeddings=len(token_tensor),
            embedding_dim=base_embedding.embedding_dim,
            device=base_embedding.weight.device,
            dtype=base_embedding.weight.dtype,
        )
        with torch.no_grad():
            self.special_embeddings.weight.copy_(base_embedding.weight[token_tensor])
        self.register_buffer(
            "_lr_multipliers",
            torch.ones(len(token_tensor), dtype=base_embedding.weight.dtype),
            persistent=False,
        )
        self.special_embeddings.weight.register_hook(self._scale_special_embedding_grad)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_embedding.weight

    @property
    def embedding_dim(self) -> int:
        return int(self.base_embedding.embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.base_embedding(input_ids)
        in_range = input_ids < self._id_to_slot.shape[0]
        if not bool(in_range.any()):
            return embeddings

        slot_ids = torch.full_like(input_ids, -1)
        slot_ids[in_range] = self._id_to_slot[input_ids[in_range]]
        special_mask = slot_ids >= 0
        if not bool(special_mask.any()):
            return embeddings

        embeddings = embeddings.clone()
        embeddings[special_mask] = self.special_embeddings(slot_ids[special_mask]).to(
            dtype=embeddings.dtype
        )
        return embeddings

    def _scale_special_embedding_grad(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self._lr_multipliers.to(device=grad.device, dtype=grad.dtype).unsqueeze(1)

    def set_lr_multipliers_by_token_id(self, multipliers: dict[int, float]) -> None:
        values = torch.ones_like(self._lr_multipliers)
        token_to_slot = {
            int(token_id): slot
            for slot, token_id in enumerate(self.special_token_ids.detach().cpu().tolist())
        }
        for token_id, multiplier in multipliers.items():
            slot = token_to_slot.get(int(token_id))
            if slot is not None:
                values[slot] = float(multiplier)
        self._lr_multipliers.copy_(values.to(device=self._lr_multipliers.device))

    def merged_weight(self) -> torch.Tensor:
        weight = self.base_embedding.weight.detach().clone()
        weight[self.special_token_ids.to(weight.device)] = self.special_embeddings.weight.detach().to(
            device=weight.device,
            dtype=weight.dtype,
        )
        return weight


def _apply_special_token_embedding_wrapper(model: Any, token_ids: list[int]) -> Any:
    input_embeddings = model.get_input_embeddings()
    if isinstance(input_embeddings, SpecialTokenEmbeddingWrapper):
        return model
    wrapper = SpecialTokenEmbeddingWrapper(input_embeddings, token_ids)
    model.set_input_embeddings(wrapper)
    return model


def special_token_embedding_wrapper(model: Any) -> SpecialTokenEmbeddingWrapper | None:
    embeddings = model.get_input_embeddings()
    return embeddings if isinstance(embeddings, SpecialTokenEmbeddingWrapper) else None


def configure_special_token_lr_multipliers(
    model: Any,
    token_ids: dict[str, int] | None,
    train_cfg: dict[str, Any],
) -> None:
    wrapper = special_token_embedding_wrapper(model)
    if wrapper is None or not token_ids:
        return
    config = train_cfg.get("special_token_lr_multipliers", {})
    if not isinstance(config, dict):
        raise ValueError("special_token_lr_multipliers must be a mapping")
    boundary_tokens = {SET_TOKEN, END_SET_TOKEN, ITEM_TOKEN, END_ITEM_TOKEN}
    multipliers: dict[int, float] = {}
    for token, token_id in token_ids.items():
        if token in READ_TOKENS:
            key = "read"
        elif token in GATHER_TOKENS:
            key = "gather"
        elif token in boundary_tokens:
            key = "boundary"
        else:
            key = "other"
        multipliers[int(token_id)] = float(
            config.get(token, config.get(key, config.get("default", 1.0)))
        )
    wrapper.set_lr_multipliers_by_token_id(multipliers)


def save_special_token_embeddings(model: Any, output_dir: str | Path) -> None:
    wrapper = special_token_embedding_wrapper(model)
    if wrapper is None:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "token_ids": wrapper.special_token_ids.detach().cpu(),
            "weight": wrapper.special_embeddings.weight.detach().cpu(),
        },
        output_path / SPECIAL_TOKEN_EMBEDDINGS_FILE,
    )


def load_special_token_embeddings(model: Any, checkpoint_dir: str | Path) -> Any:
    checkpoint_path = Path(checkpoint_dir) / SPECIAL_TOKEN_EMBEDDINGS_FILE
    if not checkpoint_path.is_file():
        return model
    data = torch.load(checkpoint_path, map_location="cpu")
    token_ids = [int(token_id) for token_id in data["token_ids"].tolist()]
    _apply_special_token_embedding_wrapper(model, token_ids)
    wrapper = special_token_embedding_wrapper(model)
    if wrapper is None:
        raise ValueError("Failed to attach SetSwitch special-token embeddings")
    wrapper.special_embeddings.weight.data.copy_(
        data["weight"].to(
            device=wrapper.special_embeddings.weight.device,
            dtype=wrapper.special_embeddings.weight.dtype,
        )
    )
    return model


def merged_special_token_state_dict(model: Any) -> dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    wrapper = special_token_embedding_wrapper(model)
    if wrapper is None:
        return state_dict

    input_embeddings = model.get_input_embeddings()
    wrapper_name = None
    for name, module in model.named_modules():
        if module is input_embeddings:
            wrapper_name = name
            break
    if not wrapper_name:
        return state_dict

    merged = dict(state_dict)
    prefix = f"{wrapper_name}."
    for key in list(merged):
        if key.startswith(prefix):
            del merged[key]
    merged[f"{wrapper_name}.weight"] = wrapper.merged_weight()
    return merged


def apply_trainable_parameter_policy(
    model: Any,
    model_cfg: dict[str, Any],
    trainable_token_ids: list[int] | None = None,
) -> Any:
    """Apply non-LoRA trainability policies.

    ``use_lora=true`` is handled by PEFT. With ``use_lora=false`` the default is
    full fine-tuning unless ``trainable`` asks for a narrower policy.
    """

    policy = str(model_cfg.get("trainable", "full")).lower()
    if policy == "full" or (model_cfg.get("use_lora", False) and policy in {"lora", "full"}):
        return model
    if policy not in {"setswitch_tokens", "special_tokens_only"}:
        raise ValueError(f"Unknown trainable policy {policy!r}; expected full or setswitch_tokens")
    if not trainable_token_ids:
        raise ValueError("trainable=setswitch_tokens requires SetSwitch token ids")

    if not model_cfg.get("use_lora", False):
        for param in model.parameters():
            param.requires_grad = False
    return _apply_special_token_embedding_wrapper(model, trainable_token_ids)
