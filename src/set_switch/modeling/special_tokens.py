"""SetSwitch tokenizer/model special-token setup."""

from __future__ import annotations

from typing import Any

from set_switch.constants import SETSWITCH_SPECIAL_TOKENS


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


def ensure_tokenizer_has_pad_token(tokenizer: Any) -> None:
    """Ensure padding is defined without adding new tokens unless necessary."""

    if tokenizer.pad_token is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})


def add_setswitch_special_tokens(tokenizer: Any, model: Any | None = None) -> dict[str, int]:
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
    return ids
