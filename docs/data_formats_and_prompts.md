# Data Formats And Prompts

This file documents the shared example schema, dataset adapters, and rendered prompt
formats used by the four interfaces:

- `chat_baseline`: ordinary causal decoder baseline.
- `setllm`: SetLLM-style SetPE/SetMask baseline.
- `setswitch`: current read/gather model, called SetRelay in paper text.
- `setfuse`: layer-scheduled SetFuse-LM.

## Shared Internal Example

After conversion, every dataset becomes a `SetSwitchExample` JSONL row:

```json
{
  "example_id": "flashrag-hotpotqa-...",
  "instruction": "Use the provided passages or options to answer the question.",
  "question": "Which city ...?",
  "documents": [
    {
      "doc_id": "flashrag-hotpotqa-0-doc-0",
      "text": "Document title\nDocument body ...",
      "is_gold": true,
      "metadata": {
        "title": "Document title",
        "source_doc_index": 0
      }
    },
    {
      "doc_id": "flashrag-hotpotqa-0-doc-1",
      "text": "Distractor title\nDistractor body ...",
      "is_gold": false,
      "metadata": {
        "title": "Distractor title",
        "source_doc_index": 1
      }
    }
  ],
  "answer": "final answer text",
  "source": "flashrag_hotpotqa",
  "metadata": {
    "golden_answers": ["final answer text"],
    "set_type": "documents",
    "task_group": "rag_multi_hop"
  }
}
```

All interfaces use the exact same converted examples when `train_jsonl` and
`dev_jsonl` are configured.

## Dataset Conversion

| Dataset | Internal set type | What each item is | Gold marking | Distractors included |
|---|---:|---|---|---|
| HotpotQA | `documents` | Wikipedia context paragraph, rendered as `title + body` | Context title in supporting facts | Yes, non-supporting context paragraphs are included up to `max_docs` |
| 2WikiMultiHopQA | `documents` | Wikipedia context paragraph, rendered as `title + body` | Context title in supporting facts | Yes, non-supporting context paragraphs are included up to `max_docs` |
| MuSiQue | `documents` | Native paragraph from `paragraphs`, rendered as `title + paragraph_text` | `is_supporting` or decomposition support index | Yes, native all-paragraph adapter keeps distractor paragraphs up to `max_docs` |
| MS MARCO QA | `documents` | Retrieved passage text | `passages.is_selected` | Yes, unselected retrieved passages are included up to `max_docs` |
| SQuAD | `documents` | Single gold paragraph/context | Always gold | No, this is a single-passage control |
| CommonsenseQA | `options` | Candidate answer text | Candidate equals gold answer | Yes, wrong answer choices are option distractors |
| OpenBookQA | `options` | Candidate answer text | Candidate equals gold answer | Yes |
| ARC | `options` | Candidate answer text | Candidate equals gold answer | Yes |
| HellaSwag | `options` | Candidate continuation text | Candidate equals gold answer | Yes |
| MMLU | `options` | Candidate answer text | Candidate equals gold answer | Yes |
| Quartz | `options` | Candidate answer text | Candidate equals gold answer | Yes |

When more than `max_docs` documents are available, conversion keeps gold/supporting
documents first and fills the rest with non-gold documents in source order.

## Stored Rendered Fields

Renderers produce token-level fields:

```text
input_ids       token ids fed to the model
labels          -100 before answer, answer token ids on answer tokens
role_ids        prefix/doc/read/gather/answer/etc. role per token
item_ids        document/option item id per token, or -1
position_ids    custom position ids for SetLLM/SetRelay/SetFuse
answer_start    token index where answer labels begin
prefix_length   token count before the set starts, for set interfaces
max_doc_length  longest raw-document token length, for SetFuse
```

## Prompt Shapes

### `chat_baseline`

Document QA:

```text
Instruction: {instruction}

Question: {question}

Passages:
Passage:
{document_0_text}

Passage:
{document_1_text}

Answer using only the provided passages.

Answer:{answer}
```

Multiple choice:

```text
Instruction: {instruction}

Question: {question}

Options:
Option:
{choice_0}

Option:
{choice_1}

Answer with the correct option text.

Answer:{answer}
```

If the tokenizer has a chat template, the prompt prefix is wrapped with that template.

### `setllm`

Document QA:

```text
Instruction: {instruction}

Question: {question}

Passages:
{document_0_text}
{document_1_text}

Answer:
{answer}
```

Multiple choice:

```text
Instruction: {instruction}

Question: {question}

Choices:
{choice_0}
{choice_1}

Answer:
{answer}
```

No choices are numbered. Set membership is carried by `item_ids`; SetPE resets item
positions and SetMask blocks cross-item communication where required.

### `setswitch` / SetRelay

With compact special-token format:

```text
Instruction: {instruction}

Question: {question}

<set><item>{document_0_text}<read_0> <read_1></item><item>{document_1_text}<read_0> <read_1></item></set><gather_0> <gather_1> <gather_2> <gather_3>
Answer:
{answer}
```

The information path is raw document tokens to same-document read tokens to global
gather tokens to answer tokens. By default, answer tokens do not attend directly to raw
documents or read tokens.

### `setfuse` / SetFuse-LM

SetFuse-LM does not add or train SetSwitch special tokens. Documents are rendered
with ordinary text separators and separated by role/item metadata used only for
position ids and masks:

```text
Instruction: {instruction}

Question: {question}

Passage:
{document_0_text}

Passage:
{document_1_text}
Answer:
{answer}
```

SetFuse has no read/gather tokens and no SetSwitch boundary tokens. Early layers
isolate documents. Late layers allow all prefix/document/structure evidence tokens
to fuse globally. Answer tokens always have causal self-attention and can attend
evidence in late layers.

## Evaluation Output

Each eval JSON contains:

```text
rows                         one row per prediction/condition
dataset_summary              reported per-source summaries, one example counted once for invariant interfaces
reported_overall_summary     reported overall summary, one example counted once for invariant interfaces
source_summary               condition-expanded summaries for visualization
overall_summary              condition-expanded overall summaries for visualization
metric_policy                metric names and caveats by source
```

For paper tables, use `dataset_summary` and `reported_overall_summary`, not
condition-expanded `source_summary` or `overall_summary`.
