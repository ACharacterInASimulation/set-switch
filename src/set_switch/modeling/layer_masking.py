"""Layer-wise attention-mask routing for SetFuse-LM."""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


def _get_path(root: Any, path: str) -> Any | None:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def find_decoder_layers(model: Any) -> list[nn.Module]:
    """Return decoder layer modules across raw models and common PEFT wrappers."""

    candidate_paths = (
        "model.layers",
        "model.model.layers",
        "base_model.model.model.layers",
        "base_model.model.layers",
        "module.model.layers",
        "module.model.model.layers",
        "module.base_model.model.model.layers",
        "module.base_model.model.layers",
    )
    for path in candidate_paths:
        layers = _get_path(model, path)
        if isinstance(layers, (nn.ModuleList, list, tuple)) and layers:
            if all(isinstance(layer, nn.Module) for layer in layers):
                return list(layers)

    if hasattr(model, "get_base_model"):
        try:
            base_layers = find_decoder_layers(model.get_base_model())
        except ValueError:
            base_layers = []
        if base_layers:
            return base_layers

    for name, module in model.named_modules():
        if not name.endswith("layers") or not isinstance(module, nn.ModuleList) or not module:
            continue
        if all(isinstance(layer, nn.Module) for layer in module):
            return list(module)

    raise ValueError("Could not locate decoder layers for SetFuse layer-wise masking")


def _mask_owner_candidates(model: Any) -> list[Any]:
    candidates = [model]
    for attr in ("module", "base_model", "model"):
        child = getattr(model, attr, None)
        if child is not None and child is not model:
            candidates.append(child)
    return candidates


def _find_installed_owner(model: Any) -> Any:
    seen: set[int] = set()
    stack = list(_mask_owner_candidates(model))
    while stack:
        current = stack.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if getattr(current, "_setfuse_layerwise_installed", False):
            return current
        stack.extend(_mask_owner_candidates(current))
    return model


def _attention_mask_arg_index(forward: Any) -> int | None:
    try:
        params = list(inspect.signature(forward).parameters)
    except (TypeError, ValueError):
        return None
    try:
        return params.index("attention_mask")
    except ValueError:
        return None


def _replace_attention_mask(
    original_forward: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    mask: torch.Tensor,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if "attention_mask" in kwargs:
        kwargs = dict(kwargs)
        kwargs["attention_mask"] = mask
        return args, kwargs

    arg_index = _attention_mask_arg_index(original_forward)
    if arg_index is not None and arg_index < len(args):
        new_args = list(args)
        new_args[arg_index] = mask
        return tuple(new_args), kwargs

    kwargs = dict(kwargs)
    kwargs["attention_mask"] = mask
    return args, kwargs


def install_setfuse_layerwise_attention(model: Any, fuse_start_layer: int) -> None:
    """Patch decoder layers so each layer substitutes the scheduled SetFuse mask."""

    layers = find_decoder_layers(model)
    if not 0 <= int(fuse_start_layer) <= len(layers):
        raise ValueError(f"fuse_start_layer={fuse_start_layer} is outside [0, {len(layers)}]")

    owner = _find_installed_owner(model)
    owner._setfuse_layerwise_installed = True
    owner._setfuse_fuse_start_layer = int(fuse_start_layer)
    owner._setfuse_num_layers = len(layers)

    for layer_idx, layer in enumerate(layers):
        if getattr(layer, "_setfuse_attention_patched", False):
            layer._setfuse_layer_idx = layer_idx
            continue

        original_forward = layer.forward

        @functools.wraps(original_forward)
        def wrapped_forward(*args: Any, __orig=original_forward, __idx=layer_idx, **kwargs: Any):
            early_mask = getattr(owner, "_setfuse_attention_mask_early", None)
            late_mask = getattr(owner, "_setfuse_attention_mask_late", None)
            if early_mask is not None and late_mask is not None:
                mask = early_mask if __idx < owner._setfuse_fuse_start_layer else late_mask
                args, kwargs = _replace_attention_mask(__orig, args, kwargs, mask)
            return __orig(*args, **kwargs)

        layer._setfuse_original_forward = original_forward
        layer._setfuse_attention_patched = True
        layer._setfuse_layer_idx = layer_idx
        layer.forward = wrapped_forward


def resolve_fuse_start_layer(model: Any, fuse_start_layer: int | str | None) -> int:
    """Resolve ``auto_half`` against the located decoder depth."""

    layers = find_decoder_layers(model)
    if fuse_start_layer is None or str(fuse_start_layer).lower() == "auto_half":
        return len(layers) // 2
    value = int(fuse_start_layer)
    if not 0 <= value <= len(layers):
        raise ValueError(f"fuse_start_layer={value} is outside [0, {len(layers)}]")
    return value


@contextmanager
def set_setfuse_masks(model: Any, early_mask: torch.Tensor, late_mask: torch.Tensor):
    """Temporarily attach SetFuse masks to the model owner used by patched layers."""

    owner = _find_installed_owner(model)
    old_early = getattr(owner, "_setfuse_attention_mask_early", None)
    old_late = getattr(owner, "_setfuse_attention_mask_late", None)
    had_early = hasattr(owner, "_setfuse_attention_mask_early")
    had_late = hasattr(owner, "_setfuse_attention_mask_late")
    owner._setfuse_attention_mask_early = early_mask
    owner._setfuse_attention_mask_late = late_mask
    try:
        yield
    finally:
        if had_early:
            owner._setfuse_attention_mask_early = old_early
        elif hasattr(owner, "_setfuse_attention_mask_early"):
            delattr(owner, "_setfuse_attention_mask_early")
        if had_late:
            owner._setfuse_attention_mask_late = old_late
        elif hasattr(owner, "_setfuse_attention_mask_late"):
            delattr(owner, "_setfuse_attention_mask_late")
