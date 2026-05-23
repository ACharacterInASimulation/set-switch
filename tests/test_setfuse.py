from __future__ import annotations

from copy import deepcopy

import torch

from set_switch.constants import ROLE_ANSWER, ROLE_DOC, ROLE_PAD, ROLE_PREFIX
from set_switch.data.setfuse_collator import SetFuseCollator
from set_switch.data.setfuse_render import render_setfuse_example
from set_switch.modeling.layer_masking import (
    install_setfuse_layerwise_attention,
    set_setfuse_masks,
)
from set_switch.modeling.setfuse_attention_mask import build_setfuse_allowed
from set_switch.training.train import build_custom_attention_masks_from_batch


def _tiny_layout():
    role_ids = [
        ROLE_PREFIX,
        ROLE_PREFIX,
        ROLE_DOC,
        ROLE_DOC,
        ROLE_DOC,
        ROLE_DOC,
        ROLE_DOC,
        ROLE_ANSWER,
        ROLE_ANSWER,
        ROLE_PAD,
    ]
    item_ids = [-1, -1, 0, 0, 0, 1, 1, -1, -1, -1]
    pad_mask = [True] * 9 + [False]
    return role_ids, item_ids, pad_mask


def test_setfuse_early_late_masks_and_padding_rules():
    role_ids, item_ids, pad_mask = _tiny_layout()
    early = build_setfuse_allowed(role_ids, item_ids, pad_mask, stage="early")[0]
    late = build_setfuse_allowed(role_ids, item_ids, pad_mask, stage="late")[0]

    assert early[0, 1]
    assert not early[0, 2]
    assert early[2, 0]
    assert early[2, 4]
    assert early[4, 2]
    assert not early[2, 5]
    assert not early[5, 2]
    assert early[7, 0]
    assert not early[7, 2]
    assert early[8, 7]
    assert not early[7, 8]
    assert not early[0, 7]
    assert not early[2, 7]

    assert late[0, 2]
    assert late[2, 5]
    assert late[5, 2]
    assert late[7, 2]
    assert late[8, 7]
    assert not late[7, 8]
    assert not late[0, 7]
    assert not late[2, 7]

    assert early[9, 9]
    assert not early[9, :9].any()
    assert not early[:9, 9].any()


def test_setfuse_positions_reset_docs_and_answer_starts_after_max_doc(tokenizer, example):
    rendered = render_setfuse_example(example, tokenizer)
    doc_positions: dict[int, list[int]] = {}
    for role, item_id, position in zip(
        rendered["role_ids"],
        rendered["item_ids"],
        rendered["position_ids"],
        strict=True,
    ):
        if role == ROLE_DOC:
            doc_positions.setdefault(item_id, []).append(position)

    assert {positions[0] for positions in doc_positions.values()} == {rendered["prefix_length"]}
    answer_positions = [
        position
        for role, position in zip(rendered["role_ids"], rendered["position_ids"], strict=True)
        if role == ROLE_ANSWER
    ]
    assert answer_positions[0] == rendered["prefix_length"] + rendered["max_doc_length"]

    permuted = deepcopy(example)
    permuted.documents = list(reversed(permuted.documents))
    rendered_permuted = render_setfuse_example(permuted, tokenizer)
    local_positions = sorted(
        position - rendered["prefix_length"]
        for role, position in zip(rendered["role_ids"], rendered["position_ids"], strict=True)
        if role == ROLE_DOC
    )
    permuted_local_positions = sorted(
        position - rendered_permuted["prefix_length"]
        for role, position in zip(
            rendered_permuted["role_ids"],
            rendered_permuted["position_ids"],
            strict=True,
        )
        if role == ROLE_DOC
    )
    assert local_positions == permuted_local_positions


def test_setfuse_render_does_not_require_setswitch_special_tokens(tokenizer, example):
    rendered = render_setfuse_example(example, tokenizer)
    rendered_text = tokenizer.decode(rendered["input_ids"][: rendered["answer_start"]])

    assert "<set>" not in rendered_text
    assert "<item>" not in rendered_text
    assert "Passage:" in rendered_text


def test_tiny_decoder_forward_backward_setfuse(tokenizer, example, make_tiny_model):
    rendered = render_setfuse_example(example, tokenizer)
    model = make_tiny_model(len(tokenizer))
    install_setfuse_layerwise_attention(model, fuse_start_layer=0)
    batch = SetFuseCollator(tokenizer)([rendered])

    with set_setfuse_masks(model, batch["attention_mask_early"], batch["attention_mask_late"]):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask_early"],
            position_ids=batch["position_ids"],
            labels=batch["labels"],
            use_cache=False,
        )
        assert torch.isfinite(outputs.loss)
        outputs.loss.backward()

    trainable_grads = [
        param.grad for param in model.parameters() if param.requires_grad and param.grad is not None
    ]
    assert trainable_grads
    assert all(torch.isfinite(grad).all() for grad in trainable_grads)


def test_setfuse_training_masks_can_be_built_after_collation(tokenizer, example):
    rendered = render_setfuse_example(example, tokenizer)
    batch = SetFuseCollator(tokenizer, build_attention_mask=False)([rendered])
    assert "attention_mask_early" not in batch
    assert "attention_mask_late" not in batch

    masks = build_custom_attention_masks_from_batch(batch, "setfuse", {}, torch.float32)

    assert masks["attention_mask_early"].shape[-2:] == batch["input_ids"].shape[-1:] * 2
    assert masks["attention_mask_late"].shape == masks["attention_mask_early"].shape
    assert "attention_mask_early" in batch


def test_setfuse_layerwise_attention_routes_masks_by_layer():
    class RecordingLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, hidden_states, attention_mask=None):
            self.seen.append(attention_mask)
            return hidden_states

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([RecordingLayer(), RecordingLayer()])

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeBackbone()

    model = FakeModel()
    install_setfuse_layerwise_attention(model, fuse_start_layer=1)
    hidden = torch.zeros(1, 2, 3)
    placeholder = torch.zeros(1, 1, 2, 2)
    early = torch.ones(1, 1, 2, 2)
    late = torch.full((1, 1, 2, 2), 2.0)

    with set_setfuse_masks(model, early, late):
        model.model.layers[0](hidden, attention_mask=placeholder)
        model.model.layers[1](hidden, attention_mask=placeholder)

    assert model.model.layers[0].seen[-1] is early
    assert model.model.layers[1].seen[-1] is late
    assert not hasattr(model, "_setfuse_attention_mask_early")
