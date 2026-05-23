from __future__ import annotations

import torch

from set_switch.data.collator import SetSwitchCollator
from set_switch.data.render import render_example
from set_switch.data.setllm_collator import SetLLMCollator
from set_switch.data.setllm_render import render_setllm_example
from set_switch.modeling.special_tokens import add_setswitch_special_tokens
from set_switch.training.train import build_custom_attention_masks_from_batch


def test_tiny_decoder_forward_returns_finite_loss(tokenizer, example, make_tiny_model):
    add_setswitch_special_tokens(tokenizer, None)
    rendered = render_example(example, tokenizer, {"num_reads_per_doc": 2, "num_gather_tokens": 4})
    model = make_tiny_model(len(tokenizer))
    batch = SetSwitchCollator(tokenizer)([rendered])

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        labels=batch["labels"],
        use_cache=False,
    )
    assert torch.isfinite(outputs.loss)


def test_tiny_decoder_forward_setllm_returns_finite_loss(tokenizer, example, make_tiny_model):
    rendered = render_setllm_example(example, tokenizer)
    model = make_tiny_model(len(tokenizer))
    batch = SetLLMCollator(tokenizer)([rendered])

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        labels=batch["labels"],
        use_cache=False,
    )
    assert torch.isfinite(outputs.loss)


def test_custom_training_masks_can_be_built_after_collation(tokenizer, example):
    add_setswitch_special_tokens(tokenizer, None)
    rendered = render_example(example, tokenizer, {"num_reads_per_doc": 2, "num_gather_tokens": 4})
    batch = SetSwitchCollator(tokenizer, build_attention_mask=False)([rendered])
    assert "attention_mask" not in batch

    masks = build_custom_attention_masks_from_batch(
        batch,
        "setswitch",
        {"doc_attention": "doc_causal"},
        torch.float32,
    )

    assert masks["attention_mask"].shape[-2:] == batch["input_ids"].shape[-1:] * 2
    assert "attention_mask" in batch


def test_setllm_training_mask_can_be_built_after_collation(tokenizer, example):
    rendered = render_setllm_example(example, tokenizer)
    batch = SetLLMCollator(tokenizer, build_attention_mask=False)([rendered])
    assert "attention_mask" not in batch

    masks = build_custom_attention_masks_from_batch(batch, "setllm", {}, torch.float32)

    assert masks["attention_mask"].shape[-2:] == batch["input_ids"].shape[-1:] * 2
