from __future__ import annotations

import torch

from set_switch.modeling.peft_setup import apply_trainable_parameter_policy
from set_switch.modeling.special_tokens import add_setswitch_special_tokens, token_id_map


def test_setswitch_tokens_policy_freezes_base_and_masks_embedding_grads(tokenizer, make_tiny_model):
    add_setswitch_special_tokens(tokenizer, None)
    model = make_tiny_model(len(tokenizer))
    token_ids = list(token_id_map(tokenizer).values())

    apply_trainable_parameter_policy(
        model,
        {"use_lora": False, "trainable": "setswitch_tokens"},
        trainable_token_ids=token_ids,
    )

    input_embeddings = model.get_input_embeddings()
    assert input_embeddings.weight.requires_grad
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert "model.embed_tokens.weight" in trainable_names
    assert trainable_names <= {"model.embed_tokens.weight", "lm_head.weight"}

    non_special_id = 0
    special_id = token_ids[0]
    input_ids = torch.tensor([[non_special_id, special_id, non_special_id]])
    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    outputs.loss.backward()

    assert input_embeddings.weight.grad is not None
    assert input_embeddings.weight.grad[non_special_id].abs().sum() == 0
    assert input_embeddings.weight.grad[special_id].abs().sum() > 0
