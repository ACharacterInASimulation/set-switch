from __future__ import annotations

import torch

from set_switch.data.collator import SetSwitchCollator
from set_switch.data.render import render_example
from set_switch.data.setllm_collator import SetLLMCollator
from set_switch.data.setllm_render import render_setllm_example
from set_switch.modeling.special_tokens import add_setswitch_special_tokens


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
