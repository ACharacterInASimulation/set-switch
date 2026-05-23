from __future__ import annotations

import pytest

from set_switch.config_validation import validate_config


def _cfg(interface: str = "setswitch"):
    return {
        "model_interface": interface,
        "model": {
            "name_or_path": "tiny/model",
            "attn_implementation": "eager",
        },
        "data": {
            "max_docs": 4,
            "num_reads_per_doc": 2,
            "num_gather_tokens": 4,
            "setswitch_boundary_tokens": True,
        },
        "mask": {"doc_attention": "doc_causal"},
        "train": {"batch_size": 1, "grad_accum_steps": 1, "mixed_precision": "no"},
        "eval": {"mcq_scoring": "logprob"},
    }


def test_validate_config_accepts_default_custom_mask_setup():
    validate_config(_cfg("setswitch"))
    validate_config(_cfg("setllm"))
    validate_config(_cfg("setfuse"))


def test_validate_config_rejects_non_eager_custom_masks_without_override():
    cfg = _cfg("setfuse")
    cfg["model"]["attn_implementation"] = "sdpa"

    with pytest.raises(ValueError, match="custom 4D masks"):
        validate_config(cfg)

    cfg["model"]["allow_non_eager_custom_masks"] = True
    validate_config(cfg)


def test_validate_config_rejects_setfuse_special_token_training():
    cfg = _cfg("setfuse")
    cfg["model"]["trainable"] = "setswitch_tokens"

    with pytest.raises(ValueError, match="does not use SetSwitch special tokens"):
        validate_config(cfg)


def test_validate_config_rejects_unsafe_training_combinations():
    cfg = _cfg("setswitch")
    cfg["train"]["bf16"] = True
    cfg["train"]["fp16"] = True
    with pytest.raises(ValueError, match="cannot both be true"):
        validate_config(cfg)

    cfg = _cfg("setswitch")
    cfg["eval"]["mcq_scoring"] = "roulette"
    with pytest.raises(ValueError, match="eval.mcq_scoring"):
        validate_config(cfg)


def test_validate_config_rejects_string_booleans():
    cfg = _cfg("setswitch")
    cfg["mask"]["answer_attends_raw_docs"] = "false"

    with pytest.raises(ValueError, match="YAML boolean"):
        validate_config(cfg)


def test_validate_config_accepts_eval_all_examples():
    cfg = _cfg("setswitch")
    cfg["eval"]["max_examples"] = "all"

    validate_config(cfg)
