from __future__ import annotations

import torch

from set_switch.training.train import attention_mask_dtype_from_config


def test_explicit_custom_mask_dtype_wins():
    dtype = attention_mask_dtype_from_config(
        {"dtype": "bfloat16", "mask_dtype": "float32"},
        {"mixed_precision": "bf16"},
    )

    assert dtype == torch.float32


def test_mixed_precision_sets_custom_mask_dtype_when_model_dtype_is_auto():
    dtype = attention_mask_dtype_from_config(
        {"dtype": "auto"},
        {"mixed_precision": "bf16"},
    )

    assert dtype == torch.bfloat16
