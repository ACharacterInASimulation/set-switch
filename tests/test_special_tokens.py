from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from set_switch.constants import SETSWITCH_SPECIAL_TOKENS
from set_switch.modeling.special_tokens import active_token_id_map, add_setswitch_special_tokens


def test_special_tokens_are_atomic_and_resize_model(tokenizer):
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    ids = add_setswitch_special_tokens(tokenizer, model)

    assert set(ids) == set(SETSWITCH_SPECIAL_TOKENS)
    for token in SETSWITCH_SPECIAL_TOKENS:
        assert len(tokenizer.encode(token, add_special_tokens=False)) == 1
    assert model.get_input_embeddings().weight.shape[0] == len(tokenizer)


def test_special_tokens_can_initialize_from_vocab_mean(tokenizer):
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    ids = add_setswitch_special_tokens(
        tokenizer,
        model,
        init_strategy="vocab_mean",
        init_noise_std=0.0,
    )
    weight = model.get_input_embeddings().weight.detach()
    special_ids = set(ids.values())
    base_ids = [idx for idx in range(weight.shape[0]) if idx not in special_ids]
    expected = weight[base_ids].mean(dim=0)

    for token_id in ids.values():
        assert torch.allclose(weight[token_id], expected)


def test_active_token_id_map_excludes_unused_boundary_tokens(tokenizer):
    add_setswitch_special_tokens(tokenizer, None)
    ids = active_token_id_map(
        tokenizer,
        {
            "setswitch_boundary_tokens": False,
            "num_reads_per_doc": 1,
            "num_gather_tokens": 2,
        },
    )

    assert set(ids) == {"<read_0>", "<gather_0>", "<gather_1>"}
