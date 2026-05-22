"""SetSwitch tokenizer/model special-token setup."""

from __future__ import annotations

from typing import Any

import torch

from set_switch.constants import (
    DEFAULT_NUM_GATHER_TOKENS,
    DEFAULT_NUM_READS_PER_DOC,
    END_ITEM_TOKEN,
    END_SET_TOKEN,
    GATHER_TOKENS,
    ITEM_TOKEN,
    READ_TOKENS,
    SETSWITCH_SPECIAL_TOKENS,
    SET_TOKEN,
)


SEMANTIC_INIT_TEXTS = {
    "<set>": ["documents", "context", "set"],
    "</set>": ["end", "context"],
    "<item>": ["document", "passage"],
    "</item>": ["end", "document"],
    **{token: ["read", "evidence", "summarize"] for token in READ_TOKENS},
    **{token: ["gather", "combine", "aggregate", "answer"] for token in GATHER_TOKENS},
}


def _encode_without_specials(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def token_id_map(tokenizer: Any) -> dict[str, int]:
    """Return the single id for each SetSwitch token after validating atomicity."""

    ids: dict[str, int] = {}
    for token in SETSWITCH_SPECIAL_TOKENS:
        encoded = _encode_without_specials(tokenizer, token)
        if len(encoded) != 1:
            raise ValueError(f"SetSwitch token {token!r} encoded to {encoded}, expected one id")
        ids[token] = int(encoded[0])
    return ids


def active_token_id_map(tokenizer: Any, data_cfg: dict[str, Any] | None = None) -> dict[str, int]:
    """Return SetSwitch token ids that are actually rendered by the current config."""

    data_cfg = data_cfg or {}
    all_ids = token_id_map(tokenizer)
    num_reads = int(data_cfg.get("num_reads_per_doc", DEFAULT_NUM_READS_PER_DOC))
    num_gathers = int(data_cfg.get("num_gather_tokens", DEFAULT_NUM_GATHER_TOKENS))
    if num_reads > len(READ_TOKENS):
        raise ValueError("num_reads_per_doc exceeds the available read tokens")
    if num_gathers > len(GATHER_TOKENS):
        raise ValueError("num_gather_tokens exceeds the available gather tokens")

    active_tokens = [*READ_TOKENS[:num_reads], *GATHER_TOKENS[:num_gathers]]
    if bool(data_cfg.get("setswitch_boundary_tokens", True)):
        active_tokens = [
            SET_TOKEN,
            END_SET_TOKEN,
            ITEM_TOKEN,
            END_ITEM_TOKEN,
            *active_tokens,
        ]
    return {token: all_ids[token] for token in active_tokens}


def ensure_tokenizer_has_pad_token(tokenizer: Any) -> None:
    """Ensure padding is defined without adding new tokens unless necessary."""

    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _semantic_seed_ids(
    tokenizer: Any,
    texts: list[str],
    special_ids: set[int],
    max_vocab_size: int,
) -> list[int]:
    ids: list[int] = []
    for text in texts:
        for token_id in _encode_without_specials(tokenizer, text):
            token_id = int(token_id)
            if token_id not in special_ids and token_id < max_vocab_size:
                ids.append(token_id)
    return ids


def initialize_setswitch_special_token_embeddings(
    tokenizer: Any,
    model: Any,
    token_ids: dict[str, int],
    strategy: str = "resize_default",
    noise_std: float = 0.01,
) -> None:
    """Initialize SetSwitch token rows after model.resize_token_embeddings.

    ``resize_default`` preserves the HF resize initialization. ``vocab_mean`` and
    ``semantic`` keep the new rows near the base embedding distribution and add a
    small per-token perturbation to avoid identical special-token rows.
    """

    strategy = str(strategy or "resize_default").lower()
    if strategy in {"resize_default", "default", "hf"}:
        return
    embeddings = model.get_input_embeddings()
    weight = embeddings.weight
    special_ids = {int(token_id) for token_id in token_ids.values()}
    base_indices = [
        idx for idx in range(weight.shape[0]) if idx not in special_ids
    ]
    if not base_indices:
        raise ValueError("Cannot initialize SetSwitch tokens without base vocabulary rows")
    base_weight = weight[torch.tensor(base_indices, device=weight.device)]
    vocab_mean = base_weight.mean(dim=0)
    global_std = torch.clamp(base_weight.float().std(), min=1.0e-6).to(weight.dtype)
    noise_scale = float(noise_std) * global_std

    with torch.no_grad():
        for token, token_id in token_ids.items():
            if strategy in {"vocab_mean", "mean"}:
                vector = vocab_mean
            elif strategy in {"semantic", "semantic_mean"}:
                seed_ids = _semantic_seed_ids(
                    tokenizer,
                    SEMANTIC_INIT_TEXTS.get(token, [token.strip("<>/_0123456789")]),
                    special_ids,
                    max_vocab_size=weight.shape[0],
                )
                if seed_ids:
                    seed_tensor = torch.tensor(seed_ids, device=weight.device)
                    vector = weight[seed_tensor].mean(dim=0)
                else:
                    vector = vocab_mean
            elif strategy == "zero":
                vector = torch.zeros_like(vocab_mean)
            else:
                raise ValueError(
                    "setswitch_token_init must be one of: resize_default, vocab_mean, semantic, zero"
                )
            if noise_scale > 0 and strategy != "zero":
                vector = vector + torch.randn_like(vector) * noise_scale
            weight[int(token_id)].copy_(vector.to(dtype=weight.dtype))


def add_setswitch_special_tokens(
    tokenizer: Any,
    model: Any | None = None,
    init_strategy: str = "resize_default",
    init_noise_std: float = 0.01,
) -> dict[str, int]:
    """Add SetSwitch tokens, resize model embeddings, and validate atomic encoding."""

    tokenizer.add_special_tokens({"additional_special_tokens": SETSWITCH_SPECIAL_TOKENS})
    ensure_tokenizer_has_pad_token(tokenizer)

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))

    ids = token_id_map(tokenizer)
    if model is not None:
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if embedding_size != len(tokenizer):
            raise ValueError(
                f"Model embedding rows ({embedding_size}) do not match tokenizer length ({len(tokenizer)})"
            )
        initialize_setswitch_special_token_embeddings(
            tokenizer=tokenizer,
            model=model,
            token_ids=ids,
            strategy=init_strategy,
            noise_std=init_noise_std,
        )
    return ids
