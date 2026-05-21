# SetSwitch

SetSwitch is a small research codebase for token-triggered set reading in decoder-only
language models. It renders retrieved documents as an unordered evidence set using
special set, item, read, and gather tokens, then supplies custom position ids and a
custom 4D attention mask.

The v0 path is deliberately narrow: answer-only cross entropy, no auxiliary losses,
and no permutation-expanded training. The main SetSwitch run keeps LoRA disabled and
trains only the SetSwitch special-token embedding rows, so we can test whether the
special tokens, mask, and position policy are sufficient for learning. The normal
decoder baseline and SetLLM baseline use LoRA.

## Install

```bash
cd /home/badrinath.chandana/git/ACharacterInASimulation/set-switch
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` if you want local Hugging Face cache/output paths.
Never commit real tokens.

## Run Tests

```bash
pytest -q
```

The suite covers special-token atomicity, canonical rendering, custom positions,
attention-mask rules, answer-only labels, a tiny decoder forward pass, FlashRAG
conversion, and a tiny fixture overfit run.

## Canonical Format

SetSwitch v0 uses exactly this non-chat layout:

```text
Instruction: {instruction}

Question: {question}

<set>
<item>
{document_0_text}
<read_0> <read_1>
</item>
...
</set>
<gather_0> <gather_1> <gather_2> <gather_3>

{answer_text}
```

There are no custom query, instruction, or answer tokens. Labels are `-100` before the
answer start, and answer tokens use ordinary causal LM labels.

## Read And Gather Tokens

Read tokens live inside each document item and summarize only that document. They can
attend to the prefix, their own raw document tokens, and same-document read tokens.

Gather tokens are placed after `</set>` and before the answer. They attend to the
prefix, all read tokens, and all gather tokens bidirectionally. In v0, answer tokens
attend to prefix tokens, gather tokens, and previous answer tokens. They do not attend
directly to raw documents or read tokens.

Prefix tokens attend bidirectionally within the prefix only. They do not attend to raw
documents, reads, gathers, or answer tokens. This keeps the instruction/question as
shared conditioning without turning it into a cross-document communication path.

The intended information path is:

```text
raw document tokens -> same-document read tokens -> global gather tokens -> answer
```

## Position Policy

The renderer passes explicit `position_ids`:

- Prefix tokens use normal local positions.
- Raw document token positions reset inside every document, starting at `prefix_length`.
- `<set>`, `</set>`, `<item>`, and `</item>` use position id `0`.
- Read and gather tokens use position id `0`, so RoPE-based models do not receive a
  meaningful rotation for those aggregation tokens.
- Answer positions start at `prefix_length + max_doc_len_in_that_set`, where
  `max_doc_len_in_that_set` is the longest rendered document in the current example.
  This is order-invariant because the maximum does not depend on document slot order.

## Attention Modes

The custom additive mask has shape `[B, 1, T, T]` and supports:

- `doc_causal`: raw document tokens attend causally within their own document.
- `doc_bidir`: raw document tokens attend bidirectionally within their own document.

Both modes keep documents isolated until read tokens feed into gather tokens.
Prefix tokens are bidirectional within the prefix in both modes, but the prefix does not
attend into the document set.

## Dataset Suite

The data path is FlashRAG-only. `RUC-NLPIR/FlashRAG_datasets` aggregates common RAG and
multiple-choice QA sources behind one Hugging Face dataset family and gives stable
train/dev splits for the sources we use.

Final v0 task buckets:

- `normal_mcq`: `commonsenseqa`, `openbookqa`, `arc`, `hellaswag`, `mmlu`, `quartz`
- `rag_single_hop`: `msmarco-qa`, `squad`, `boolq`
- `rag_multi_hop`: `hotpotqa`, `2wikimultihopqa`, `musique`
- `aggregation`: `ambig_qa`

Rationale: SetLLM is strongest and most directly motivated for unordered multiple
choice. SetSwitch additionally targets document sets, so the suite adds single-hop RAG,
multi-hop RAG, and an ambiguous/multi-answer bucket. FlashRAG `piqa` and `siqa` are not
default training sources because the normalized FlashRAG rows do not expose clean
choice/gold-label structure for the exact SetLLM-style setup; they can be added later
through native dataset adapters.

Document/passage QA sources include:

- `hotpotqa`: multi-document Wikipedia QA with supporting facts.
- `2wikimultihopqa`: multi-hop Wikipedia QA with supporting facts.
- `musique`: compositional multi-hop QA. In the FlashRAG form, MuSiQue exposes the
  supporting paragraphs directly; this is useful for learning but has fewer distractors
  than the original MuSiQue adapter.
- `msmarco-qa`: retrieved-passage QA with selected passage flags.
- `squad`: single-passage extractive QA.
- `boolq`: single-passage yes/no QA.
- `ambig_qa`: ambiguous/multi-answer QA converted from available search-result snippets.

SetSwitch also supports unordered option sets. For multiple-choice datasets, each
choice is rendered as a separate `<item>` and the target answer is the correct choice
text, not a position label such as `A` or `B`. This avoids training a positional option
bias into the interface.

Option-set sources:

- `commonsenseqa`
- `openbookqa`
- `arc`
- `hellaswag`
- `mmlu`
- `quartz`

## SetLLM Baseline

SetLLM uses the paper's architectural recipe: SetPE position ids, SetMask attention, and
LoRA finetuning. For option datasets, the prompt follows the paper's modified sample
format:

```text
Question: {question}

Choices:
{choice0}
{choice1}
...

Answer:
{answer}
```

Choices are not numbered. For document QA, the analogous shared-suite prompt uses
`Passages:` with unnumbered passage text. SetLLM does not use SetSwitch read/gather
special tokens.

## Single Config

Use [configs/flashrag.yaml](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/configs/flashrag.yaml)
for both SetSwitch and the chat baseline. Important fields:

```yaml
model_interface: setswitch  # or chat_baseline or setllm

data:
  source: flashrag
  datasets: [commonsenseqa, openbookqa, arc, hellaswag, mmlu, quartz, msmarco-qa, squad, boolq, hotpotqa, 2wikimultihopqa, musique, ambig_qa]
  total_train_examples: 100000
  total_val_examples: 10000
  sample_allocation: task_balanced_equal
```

If `total_train_examples` is set, the default allocation is `task_balanced_equal`:
first split the budget across task buckets, then split each task bucket across its
datasets. Small datasets cap at their available split size and unused budget is
redistributed. This is deliberately not proportional to source size, so MS MARCO cannot
dominate just because it has many more rows.

For the default 100k train budget, this gives roughly:

- `aggregation`: all available AmbigQA train examples.
- `rag_single_hop`: about 10k each from MS MARCO-QA, SQuAD, and BoolQ.
- `rag_multi_hop`: about 10k each from HotpotQA, 2WikiMultiHopQA, and MuSiQue.
- `normal_mcq`: spread across the MCQ datasets, with small sources capped and the
  remainder redistributed inside MCQ.

The default set size is `max_docs: 20` and keeps full converted documents
(`max_doc_tokens: null`). This avoids silently cutting away evidence. If you need a
memory-controlled run, set `max_doc_tokens` to an integer such as `512`; then gold docs
use `doc_truncation: answer_window` to keep an answer-centered window when the answer
string is present. The custom SetSwitch and SetLLM masks scale quadratically in sequence
length, so check the rough mask footprint before pushing context length higher:

```bash
python scripts/estimate_context.py --config configs/flashrag.yaml
```

You can also control datasets individually:

```yaml
data:
  datasets: ["hotpotqa[:0.5]", "2wiki[:50%]", "msmarco[:0.01]"]
```

The compact form is:

```text
dataset_name[split:percent]
dataset_name[:percent]
```

Examples: `hotpotqa[:0.5]` means 50% of the train split selected by the current run;
`mmlu[dev:10%]` means 10% of MMLU dev.

For more explicit control:

```yaml
data:
  datasets:
    - name: hotpotqa
      split: train
      max_examples: 5000
    - name: msmarco-qa
      split: train
      percent: 0.01
```

`percent` means a percentage of the known source split size when that count is known.
For unknown counts, use `max_examples`.

Prepare JSONL from the selected FlashRAG sources:

```bash
python scripts/prepare_dataset_suite.py \
  --config configs/flashrag.yaml \
  --split train \
  --output data/flashrag_train.jsonl
```

Train:

```bash
accelerate launch scripts/train.py --config configs/flashrag.yaml --interface setswitch
accelerate launch scripts/train.py --config configs/flashrag.yaml --interface setllm
accelerate launch scripts/train.py --config configs/flashrag.yaml --interface chat_baseline
```

The training loop uses Accelerate with a custom PyTorch loop, custom masks, custom
position ids, and answer-only CE. The config uses Qwen/Qwen3-4B by default. Method
overrides train only SetSwitch special-token embeddings for SetSwitch and enable LoRA
for the normal decoder baseline and SetLLM.

Learning rates are method-specific:

- `setswitch`: `3e-3`, because only the SetSwitch special-token embedding rows are
  trainable. This is closer to soft-prompt/special-token tuning, where tiny trainable
  parameter sets usually need higher LR.
- `setllm`: `3e-4`, matching the lower end of the SetLLM paper's LoRA sweep.
- `chat_baseline`: `3e-4`, so the normal decoder baseline has comparable LoRA
  adaptation strength.

Runs write local JSONL metrics even without W&B:

```text
outputs/{run_name}/metrics.jsonl
```

Each row records events such as `run_start`, `train`, `eval`, `save`, and `save_final`.

Attention implementation differs by method. The normal decoder baseline uses
`attn_implementation: sdpa`, so PyTorch/HF can use optimized causal attention kernels
when available. SetSwitch and SetLLM use `attn_implementation: eager` because they pass
custom 4D non-causal masks; switching those to FlashAttention/SDPA must be validated
separately so the custom mask is not ignored or simplified.
The dense custom masks use the configured model/mixed-precision dtype by default, so
bf16 runs do not keep the `[B, 1, T, T]` masks in fp32 unless `model.mask_dtype`
explicitly requests it.

The baseline uses `tokenizer.apply_chat_template(..., add_generation_prompt=True)` when
the tokenizer provides a chat template. If the tokenizer is not chat-tuned, it falls
back to a plain instruction/document/question prompt.

Visible item text should not contain document or option indices such as `Document 1` or
`Option A`. The SetSwitch renderer uses only repeated `<item>` delimiters; the baseline
renderer uses repeated unnumbered `Passage:` or `Option:` blocks.

Evaluate held-out accuracy with gold item placement sweeps:

```bash
python scripts/evaluate.py --config configs/flashrag.yaml --interface setswitch
```

The `eval:` block in the config controls split, sample count, generation length, output
path, and gold-position sweep. The report separates task buckets and evaluates gold
items placed around 0%, 25%, 50%, 75%, and 100% of the set. Many FlashRAG sources
provide `dev` rather than `test`; use `--split test` only for sources that actually
expose a test split.

## Inspect And Debug

Inspect one rendered example:

```bash
python scripts/inspect_batch.py --config configs/flashrag.yaml
```

Print an allowed-attention matrix:

```bash
python scripts/debug_mask.py --config configs/flashrag.yaml
```

## Gold-Length Diagnostic

Gold-document length is only a diagnostic in v0, not a sampling constraint.

```bash
python scripts/report_gold_length_bias.py \
  --input data/flashrag_train.jsonl \
  --output outputs/flashrag_length_report.json \
  --histogram-png outputs/flashrag_length_hist.png
```

The report includes gold and non-gold length distributions, summary statistics, mean
difference, and an optional length-only logistic-regression/AUC check.
