from __future__ import annotations

from set_switch.constants import IGNORE_INDEX
from set_switch.data.collator import SetSwitchCollator
from set_switch.data.render import render_example
from set_switch.modeling.special_tokens import add_setswitch_special_tokens


def test_labels_before_answer_and_padding_are_ignored(tokenizer, fixture_example_factory):
    add_setswitch_special_tokens(tokenizer, None)
    examples = [fixture_example_factory(0), fixture_example_factory(1)]
    rendered = [
        render_example(example, tokenizer, {"num_reads_per_doc": 2, "num_gather_tokens": 4})
        for example in examples
    ]
    batch = SetSwitchCollator(tokenizer)(rendered)

    for batch_idx, feature in enumerate(rendered):
        start = feature["answer_start"]
        length = len(feature["input_ids"])
        assert batch["labels"][batch_idx, :start].eq(IGNORE_INDEX).all()
        assert (
            batch["labels"][batch_idx, start:length]
            .eq(batch["input_ids"][batch_idx, start:length])
            .all()
        )
        assert batch["labels"][batch_idx, length:].eq(IGNORE_INDEX).all()

        first_supervised_target = batch["labels"][batch_idx, start]
        assert first_supervised_target == batch["input_ids"][batch_idx, start]
