# SetSwitch

SetSwitch is a small research codebase for token-triggered set reading in decoder-only
language models. It renders retrieved documents as an unordered evidence set using
special set, item, read, and gather tokens, then supplies custom position ids and a
custom 4D attention mask.

The v0 path is deliberately narrow: answer-only cross entropy, no auxiliary losses,
and no permutation-expanded training. The main comparison uses the same LoRA budget
for SetSwitch, SetLLM, and the normal decoder baseline. SetSwitch additionally trains
only the special-token rows that are actually rendered by the current configuration.

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

The main data path uses `RUC-NLPIR/FlashRAG_datasets`, which aggregates common RAG
and multiple-choice QA sources behind one Hugging Face dataset family and gives stable
train/dev splits for most sources. MuSiQue is the exception: it is loaded from
`dgslibisey/MuSiQue` so the rendered set contains all paragraphs, including
distractors, rather than FlashRAG's support-only MuSiQue conversion.

Final v0 task buckets:

- `normal_mcq`: `commonsenseqa`, `openbookqa`, `arc`, `hellaswag`, `mmlu`, `quartz`
- `retrieved_qa`: `msmarco-qa`
- `single_passage_control`: `squad`
- `rag_multi_hop`: `hotpotqa`, `2wikimultihopqa`, `musique`

Rationale: SetLLM is strongest and most directly motivated for unordered multiple
choice. SetSwitch additionally targets document sets, so the suite adds retrieved QA
and multi-hop RAG. Default examples are mostly set-shaped: either multiple answer
choices or multiple passages/documents. We keep one small SQuAD bucket as a
single-passage extractive QA control. FlashRAG `piqa`, `siqa`, `boolq`, and `ambig_qa`
are not default training sources because they are ambiguous, too weak for the set
interface, or missing a clean paper-defensible target for the exact SetLLM-style setup.
They can be added later through native dataset adapters.

The main supervised target is always the dataset final answer, not an intermediate
reasoning trace, so the setup stays comparable to standard QA and SetLLM-style training.

Document/passage QA sources include:

- `hotpotqa`: multi-document Wikipedia QA with supporting facts.
- `2wikimultihopqa`: multi-hop Wikipedia QA with supporting facts.
- `musique`: compositional multi-hop QA loaded from the native all-paragraph adapter.
  Supporting paragraphs are marked, but distractor paragraphs are still rendered.
- `msmarco-qa`: retrieved-passage QA with selected passage flags.
- `squad`: single-passage extractive QA, used only as a small control bucket.

`boolq` remains an opt-in adapter path, but it is not in the default paper suite because
yes/no single-passage examples are not useful for the unordered-set claim.

`ambig_qa` remains an opt-in adapter for diagnostics, but it is not in the default
paper suite because the proper answer-set metric is not implemented here.

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

Detailed evaluation buckets are also emitted when metadata is available:

- HotpotQA: `hotpotqa_bridge`, `hotpotqa_comparison`
- 2WikiMultiHopQA: for example `2wikimultihopqa_compositional`,
  `2wikimultihopqa_comparison`, and `2wikimultihopqa_bridge_comparison`
- MuSiQue: `musique_2hop`, `musique_3hop`, `musique_4hop`

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

## Configs

The shared base config is
[configs/flashrag.yaml](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/configs/flashrag.yaml).
Use the three thin wrapper configs for actual runs:

- [configs/flashrag_setswitch.yaml](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/configs/flashrag_setswitch.yaml)
- [configs/flashrag_setllm.yaml](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/configs/flashrag_setllm.yaml)
- [configs/flashrag_decoder.yaml](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/configs/flashrag_decoder.yaml)

Each wrapper uses `extends: flashrag.yaml` and only sets `model_interface`, so data,
training, and evaluation settings stay synchronized across methods.

Important base fields:

```yaml
data:
  source: flashrag
  datasets: [...]
  total_train_examples: 100000
  total_val_examples: 1000
  sample_allocation: task_balanced_equal
```

The default 100k train mix is explicit:

- normal MCQ: 9k total, 1.5k from each normal MCQ source
- retrieved QA: 17k total from MS MARCO QA
- single-passage control: 10k total from SQuAD
- multi-hop RAG: 64k total

The train-only limits do not affect validation or evaluation. The default validation
budget remains 1k and is allocated across the selected dev sources.

Training examples are additionally guarded by `train_max_render_tokens: 4096`. During
train JSONL preparation, and during direct FlashRAG training, examples whose rendered
prompt plus answer exceed this cap are skipped and replaced by later examples from the
same source allocation. Validation and test examples are not length-filtered.

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

Override the train-only length filter:

```bash
python scripts/prepare_dataset_suite.py \
  --config configs/flashrag.yaml \
  --split train \
  --output data/flashrag_train.jsonl \
  --max-render-tokens 4096
```

Use `--no-length-filter` to export the uncapped train split.

## Train

```bash
accelerate launch scripts/train.py --config configs/flashrag_setswitch.yaml
accelerate launch scripts/train.py --config configs/flashrag_setllm.yaml
accelerate launch scripts/train.py --config configs/flashrag_decoder.yaml
```

The training loop uses Accelerate with a custom PyTorch loop, custom masks, custom
position ids, and answer-only CE. The config uses Qwen/Qwen3-4B by default. Method
overrides enable LoRA for all three methods. SetSwitch additionally wraps and trains
the rendered SetSwitch special-token rows with a separate optimizer group.

Learning rates are method-specific:

- SetSwitch LoRA/base group: `3e-4`
- SetSwitch special-token table: `1e-3`
- SetLLM LoRA: `3e-4`
- decoder baseline LoRA: `3e-4`

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

## Evaluate

After training, evaluate the final checkpoints:

```bash
python scripts/evaluate.py \
  --config configs/flashrag_setswitch.yaml \
  --checkpoint outputs/qwen3_4b_setswitch/final

python scripts/evaluate.py \
  --config configs/flashrag_setllm.yaml \
  --checkpoint outputs/qwen3_4b_setllm_lora/final

python scripts/evaluate.py \
  --config configs/flashrag_decoder.yaml \
  --checkpoint outputs/qwen3_4b_chat_baseline/final
```

The `eval:` block in the config controls split, sample count, generation length, output
path, and gold-position sweep. The decoder baseline performs the explicit
gold-position sweep for document QA. For MCQ examples, the decoder baseline also runs
`eval.option_permutations: 4` deterministic random option orders and writes
permutation-mean, any/all-permutation, and majority-vote accuracies under
`option_order_summary`. SetSwitch and SetLLM generate once per example and report the
same score across the sweep buckets because their set interfaces are structurally
permutation-invariant.

Evaluate the labeled test split:

```bash
python scripts/evaluate.py \
  --config configs/flashrag_setswitch.yaml \
  --checkpoint outputs/qwen3_4b_setswitch/final \
  --split test
```

For the current suite, `--split test` includes only selected sources with labeled test
splits: OpenBookQA, ARC, MMLU, and QuaRTz. Their known total is 18,874 examples;
the default `eval.max_examples: 1000` samples from them.

The report writes condition-specific summaries, a plain `dataset_summary` table,
`reported_overall_summary`, and per-row predictions. Primary metrics follow the
original dataset families: normalized accuracy for MCQ and BoolQ, ROUGE-L for MS MARCO
QA with BLEU-1 also logged, and SQuAD/Hotpot/2Wiki/MuSiQue-style normalized exact
match plus token F1 for extractive and multi-hop QA. HotpotQA supporting-fact and joint
metrics are not reported because these models emit only final answers.

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
