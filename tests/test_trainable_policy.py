from __future__ import annotations

import torch

from set_switch.modeling.peft_setup import (
    SPECIAL_TOKEN_EMBEDDINGS_FILE,
    SpecialTokenEmbeddingWrapper,
    apply_trainable_parameter_policy,
    configure_special_token_lr_multipliers,
    load_special_token_embeddings,
    merged_special_token_state_dict,
    save_special_token_embeddings,
)
from set_switch.modeling.special_tokens import active_token_id_map, add_setswitch_special_tokens, token_id_map
from set_switch.training.train import optimizer_param_groups


def test_setswitch_tokens_policy_trains_only_small_special_embedding_table(
    tokenizer, make_tiny_model
):
    add_setswitch_special_tokens(tokenizer, None)
    model = make_tiny_model(len(tokenizer))
    token_ids = list(token_id_map(tokenizer).values())

    apply_trainable_parameter_policy(
        model,
        {"use_lora": False, "trainable": "setswitch_tokens"},
        trainable_token_ids=token_ids,
    )

    input_embeddings = model.get_input_embeddings()
    assert isinstance(input_embeddings, SpecialTokenEmbeddingWrapper)
    assert not input_embeddings.base_embedding.weight.requires_grad
    assert input_embeddings.special_embeddings.weight.requires_grad
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names == {"model.embed_tokens.special_embeddings.weight"}

    non_special_id = 0
    special_id = token_ids[0]
    input_ids = torch.tensor([[non_special_id, special_id, non_special_id]])
    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    outputs.loss.backward()

    assert input_embeddings.base_embedding.weight.grad is None
    assert input_embeddings.special_embeddings.weight.grad is not None
    assert input_embeddings.special_embeddings.weight.grad[0].abs().sum() > 0


def test_special_token_embeddings_can_be_saved_loaded_and_merged(
    tmp_path, tokenizer, make_tiny_model
):
    add_setswitch_special_tokens(tokenizer, None)
    model = make_tiny_model(len(tokenizer))
    token_ids = list(token_id_map(tokenizer).values())
    apply_trainable_parameter_policy(
        model,
        {"use_lora": False, "trainable": "setswitch_tokens"},
        trainable_token_ids=token_ids,
    )

    wrapper = model.get_input_embeddings()
    assert isinstance(wrapper, SpecialTokenEmbeddingWrapper)
    with torch.no_grad():
        wrapper.special_embeddings.weight.add_(1.0)

    save_special_token_embeddings(model, tmp_path)
    assert (tmp_path / SPECIAL_TOKEN_EMBEDDINGS_FILE).is_file()

    reloaded = make_tiny_model(len(tokenizer))
    load_special_token_embeddings(reloaded, tmp_path)
    reloaded_wrapper = reloaded.get_input_embeddings()
    assert isinstance(reloaded_wrapper, SpecialTokenEmbeddingWrapper)
    assert torch.allclose(
        reloaded_wrapper.special_embeddings.weight,
        wrapper.special_embeddings.weight,
    )

    state_dict = merged_special_token_state_dict(model)
    merged_weight = state_dict["model.embed_tokens.weight"]
    assert torch.allclose(merged_weight[token_ids[0]], wrapper.special_embeddings.weight[0])
    assert "model.embed_tokens.special_embeddings.weight" not in state_dict


def test_special_token_optimizer_group_can_use_separate_lr(tokenizer, make_tiny_model):
    add_setswitch_special_tokens(tokenizer, None)
    model = make_tiny_model(len(tokenizer))
    token_ids = list(token_id_map(tokenizer).values())
    apply_trainable_parameter_policy(
        model,
        {"use_lora": False, "trainable": "setswitch_tokens"},
        trainable_token_ids=token_ids,
    )

    groups = optimizer_param_groups(
        model,
        learning_rate=3e-4,
        weight_decay=0.0,
        special_token_learning_rate=1e-3,
    )

    assert len(groups) == 1
    assert groups[0]["name"] == "setswitch_special_tokens"
    assert groups[0]["lr"] == 1e-3


def test_special_token_lr_multipliers_scale_embedding_gradients(tokenizer, make_tiny_model):
    add_setswitch_special_tokens(tokenizer, None)
    model = make_tiny_model(len(tokenizer))
    token_ids = token_id_map(tokenizer)
    apply_trainable_parameter_policy(
        model,
        {"use_lora": False, "trainable": "setswitch_tokens"},
        trainable_token_ids=list(token_ids.values()),
    )
    configure_special_token_lr_multipliers(
        model,
        token_ids,
        {"special_token_lr_multipliers": {"read": 0.5, "gather": 2.0}},
    )
    wrapper = model.get_input_embeddings()
    assert isinstance(wrapper, SpecialTokenEmbeddingWrapper)

    wrapper.special_embeddings.weight.grad = None
    loss = wrapper.special_embeddings.weight.sum()
    loss.backward()
    token_to_slot = {
        int(token_id): slot
        for slot, token_id in enumerate(wrapper.special_token_ids.detach().cpu().tolist())
    }

    assert torch.allclose(
        wrapper.special_embeddings.weight.grad[token_to_slot[token_ids["<read_0>"]]],
        torch.full((wrapper.embedding_dim,), 0.5),
    )
    assert torch.allclose(
        wrapper.special_embeddings.weight.grad[token_to_slot[token_ids["<gather_0>"]]],
        torch.full((wrapper.embedding_dim,), 2.0),
    )


def test_trainable_policy_can_wrap_only_active_boundaryless_tokens(tokenizer, make_tiny_model):
    add_setswitch_special_tokens(tokenizer, None)
    model = make_tiny_model(len(tokenizer))
    active_ids = active_token_id_map(
        tokenizer,
        {
            "setswitch_boundary_tokens": False,
            "num_reads_per_doc": 1,
            "num_gather_tokens": 1,
        },
    )
    apply_trainable_parameter_policy(
        model,
        {"use_lora": False, "trainable": "setswitch_tokens"},
        trainable_token_ids=list(active_ids.values()),
    )
    wrapper = model.get_input_embeddings()
    assert isinstance(wrapper, SpecialTokenEmbeddingWrapper)

    assert wrapper.special_embeddings.num_embeddings == 2
    assert sorted(wrapper.special_token_ids.detach().cpu().tolist()) == sorted(active_ids.values())
