from __future__ import annotations

import pytest
import torch
from torch.optim import AdamW

from set_switch.data.collator import SetSwitchCollator
from set_switch.data.render import render_example
from set_switch.modeling.special_tokens import add_setswitch_special_tokens


@pytest.mark.slow
def test_tiny_fixture_overfit_loss_decreases(tokenizer, make_tiny_model, fixture_example_factory):
    add_setswitch_special_tokens(tokenizer, None)
    examples = [fixture_example_factory(idx % 2) for idx in range(32)]
    rendered = [
        render_example(example, tokenizer, {"num_reads_per_doc": 2, "num_gather_tokens": 4})
        for example in examples
    ]
    collator = SetSwitchCollator(tokenizer)
    batches = [collator(rendered[idx : idx + 4]) for idx in range(0, len(rendered), 4)]

    model = make_tiny_model(len(tokenizer))
    optimizer = AdamW(model.parameters(), lr=5e-3)
    losses: list[float] = []

    for epoch in range(4):
        for batch in batches:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
                labels=batch["labels"],
                use_cache=False,
            )
            loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if epoch == 0 or epoch == 3:
                losses.append(float(loss.detach()))

    assert torch.isfinite(torch.tensor(losses)).all()
    early = sum(losses[: len(batches)]) / len(batches)
    late = sum(losses[-len(batches) :]) / len(batches)
    assert late < early
