from __future__ import annotations

from set_switch.constants import IGNORE_INDEX
from set_switch.data.baseline_render import (
    _chat_prefix_ids,
    render_baseline_prompt_text,
    render_chat_baseline_example,
)


def test_chat_baseline_uses_prompt_prefix_labels(tokenizer, example):
    rendered = render_chat_baseline_example(example, tokenizer, {"max_doc_tokens": 32})

    assert rendered["interface"] == "chat_baseline"
    assert rendered["answer_start"] > 0
    assert all(label == IGNORE_INDEX for label in rendered["labels"][: rendered["answer_start"]])
    assert (
        rendered["labels"][rendered["answer_start"] :]
        == rendered["input_ids"][rendered["answer_start"] :]
    )


def test_baseline_prompt_has_no_document_indices(tokenizer, example):
    prompt = render_baseline_prompt_text(example, tokenizer, max_doc_tokens=32)

    assert "Document 1" not in prompt
    assert "Document 2" not in prompt
    assert "Passage:\n" in prompt


def test_chat_prefix_ids_accepts_dict_chat_template_response(tokenizer):
    class DictChatTokenizer(type(tokenizer)):
        chat_template = "template"

        def apply_chat_template(self, *args, **kwargs):
            del args, kwargs
            return {"input_ids": [7, 8, 9]}

    assert _chat_prefix_ids(DictChatTokenizer(), "hello") == [7, 8, 9]


def test_chat_prefix_ids_accepts_batched_chat_template_response(tokenizer):
    class BatchedChatTokenizer(type(tokenizer)):
        chat_template = "template"

        def apply_chat_template(self, *args, **kwargs):
            del args, kwargs
            return {"input_ids": [[7, 8, 9]]}

    assert _chat_prefix_ids(BatchedChatTokenizer(), "hello") == [7, 8, 9]
