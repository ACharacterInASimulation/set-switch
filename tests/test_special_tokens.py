from __future__ import annotations

from transformers import LlamaConfig, LlamaForCausalLM

from set_switch.constants import SETSWITCH_SPECIAL_TOKENS
from set_switch.modeling.special_tokens import add_setswitch_special_tokens


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
