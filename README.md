# SetSwitch

Research code for comparing four decoder-only interfaces on set-shaped QA:

- `chat_baseline`: ordinary causal decoder baseline.
- `setllm`: SetLLM-style SetPE/SetMask baseline.
- `setswitch`: read/gather model, called **SetRelay** in paper text.
- `setfuse`: layer-scheduled fusion model, called **SetFuse-LM** in paper text.

The main paper workflow is: prepare fixed train/eval JSONL files once, train all four
interfaces on the same train IDs, evaluate all four on the same paper-eval IDs, and build
a paper table from the JSON reports.

## Install

```bash
cd /home/badrinath.chandana/git/ACharacterInASimulation/set-switch
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` only for local cache/output settings. Do not commit real
tokens.

## Quick Checks

```bash
ruff check .
pytest -q
```

Current expected status:

```text
93 passed
```

## Interfaces

`chat_baseline` uses normal causal attention and ordinary prompt text.

`setllm` uses SetLLM-style custom position ids and a custom 4D SetMask. It does not use
SetSwitch read/gather special tokens.

`setswitch` / **SetRelay** renders SetSwitch boundary/read/gather tokens and uses the
read/gather information path:

```text
raw document tokens -> same-document read tokens -> global gather tokens -> answer
```

`setfuse` / **SetFuse-LM** does not add or train SetSwitch special tokens. It renders
ordinary document separators, then uses layer-specific masks:

```text
early layers: documents are isolated; answer sees prefix + previous answer
late layers: documents/evidence fuse globally; answer sees prefix + all evidence + previous answer
```

See [docs/data_formats_and_prompts.md](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/docs/data_formats_and_prompts.md)
for exact rendered sample shapes.

## Dataset Suite

The default suite uses `RUC-NLPIR/FlashRAG_datasets`, with native MuSiQue loaded from
`dgslibisey/MuSiQue` so all paragraphs, including distractors, are available.

Default sources:

```text
commonsenseqa, openbookqa, arc, hellaswag, mmlu, quartz,
msmarco-qa, squad, hotpotqa, 2wikimultihopqa, musique
```

Distractors are included where the source provides them:

- multi-hop/document QA keeps non-gold passages up to `max_docs`
- MS MARCO keeps unselected retrieved passages up to `max_docs`
- MCQ datasets render wrong choices as option distractors
- SQuAD is a single-passage control

## Paper Eval Split

The default eval split is:

```yaml
eval:
  split: paper
  max_examples: all
```

`paper` means:

```text
use public labeled test split where available;
otherwise use labeled dev/validation split.
```

Current expected paper-eval split usage:

| Dataset | Split | Examples |
|---|---:|---:|
| commonsenseqa | dev | 1,221 |
| openbookqa | test | 500 |
| arc | test | 3,548 |
| hellaswag | dev | 10,042 |
| mmlu | test | 14,042 |
| quartz | test | 784 |
| msmarco-qa | dev | 10,000 random sample |
| squad | dev | 10,570 |
| hotpotqa | dev | 7,405 |
| 2wikimultihopqa | dev | 12,576 |
| musique | validation | 2,417 |
| **Total** |  | **73,105** |

The MS MARCO cap is eval-only and deterministic (`paper_sample_seed: 42`).
Training still uses the configured `train_max_examples`.

Pure `--split dev` is also available and gives 148,608 examples across all 11 datasets.
Pure `--split test` gives 18,874 examples across the four public labeled test datasets:
OpenBookQA, ARC, MMLU, and QuaRTz.

## Prepare Fixed JSONL

Preparing JSONL first is recommended for fair paper runs. It freezes example IDs and
prevents each method from seeing a different accepted subset.

```bash
cd /home/badrinath.chandana/git/ACharacterInASimulation/set-switch
mkdir -p data/fixed
```

Prepare shared 100k train:

```bash
python scripts/prepare_dataset_suite.py \
  --config configs/flashrag.yaml \
  --split train \
  --output data/fixed/train.jsonl \
  --verbose
```

Prepare paper eval with the 10k random MS MARCO cap:

```bash
python scripts/prepare_dataset_suite.py \
  --config configs/flashrag.yaml \
  --split paper \
  --output data/fixed/paper_eval.jsonl \
  --max-examples all \
  --no-length-filter \
  --verbose
```

Check counts:

```bash
wc -l data/fixed/train.jsonl data/fixed/paper_eval.jsonl
```

Expected:

```text
100000 data/fixed/train.jsonl
73105 data/fixed/paper_eval.jsonl
```

Then add these under `data:` in `configs/flashrag.yaml`:

```yaml
train_jsonl: data/fixed/train.jsonl
paper_jsonl: data/fixed/paper_eval.jsonl
```

## Configs

The shared base config is [configs/flashrag.yaml](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/configs/flashrag.yaml).
The thin wrapper configs set only the active interface:

- [configs/flashrag_decoder.yaml](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/configs/flashrag_decoder.yaml)
- [configs/flashrag_setllm.yaml](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/configs/flashrag_setllm.yaml)
- [configs/flashrag_setswitch.yaml](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/configs/flashrag_setswitch.yaml)
- [configs/flashrag_setfuse.yaml](/home/badrinath.chandana/git/ACharacterInASimulation/set-switch/configs/flashrag_setfuse.yaml)

Important defaults:

```yaml
model:
  name_or_path: Qwen/Qwen3-4B
  dtype: bfloat16

data:
  total_train_examples: 100000
  max_docs: 20
  train_max_render_tokens: 4096

train:
  batch_size: 1
  grad_accum_steps: 8
  gradient_checkpointing: true
  mixed_precision: bf16

eval:
  split: paper
  max_examples: all
  max_new_tokens: 32
```

`max_steps: 5000` is 5,000 optimizer steps, not 5,000 examples. With `batch_size: 1`
and `grad_accum_steps: 8`, that sees about 40,000 examples. For roughly one pass over
100k examples, set:

```yaml
train:
  max_steps: 12500
```

## Attention Backend

All four method configs currently use SDPA where appropriate. The custom-mask methods
(`setllm`, `setswitch`, `setfuse`) pass dense 4D masks. In this repository's current
Torch/Transformers stack, SDPA matches eager on parity checks for those masks.

Run exact-model parity checks before large runs on a new machine/library stack:

```bash
python scripts/check_sdpa_mask_equivalence.py \
  --config configs/flashrag_setswitch.yaml \
  --interface setswitch \
  --device cuda

python scripts/check_sdpa_mask_equivalence.py \
  --config configs/flashrag_setllm.yaml \
  --interface setllm \
  --device cuda

python scripts/check_sdpa_mask_equivalence.py \
  --config configs/flashrag_setfuse.yaml \
  --interface setfuse \
  --device cuda
```

Dense attention remains `O(T^2)`. SetFuse is dense SDPA, not FlexAttention/block-sparse.

## Train

Train each method:

```bash
accelerate launch scripts/train.py --config configs/flashrag_decoder.yaml
accelerate launch scripts/train.py --config configs/flashrag_setllm.yaml
accelerate launch scripts/train.py --config configs/flashrag_setswitch.yaml
accelerate launch scripts/train.py --config configs/flashrag_setfuse.yaml
```

Outputs:

```text
outputs/{run_name}/metrics.jsonl
outputs/{run_name}/final
```

Training uses answer-only CE. Prefix, document, structure, read/gather, and pad tokens
receive `-100` labels.

## Evaluate

Evaluate a checkpoint on the paper split:

```bash
python scripts/evaluate.py \
  --config configs/flashrag_setfuse.yaml \
  --checkpoint outputs/qwen3_4b_setfuse_lora/final \
  --verbose
```

Run all four:

```bash
python scripts/evaluate.py --config configs/flashrag_decoder.yaml --checkpoint outputs/qwen3_4b_chat_baseline/final --verbose
python scripts/evaluate.py --config configs/flashrag_setllm.yaml --checkpoint outputs/qwen3_4b_setllm_lora/final --verbose
python scripts/evaluate.py --config configs/flashrag_setswitch.yaml --checkpoint outputs/qwen3_4b_setswitch/final --verbose
python scripts/evaluate.py --config configs/flashrag_setfuse.yaml --checkpoint outputs/qwen3_4b_setfuse_lora/final --verbose
```

The eval report includes:

```text
dataset_summary
reported_overall_summary
reported_task_summary
source_split_summary
metric_policy
rows
```

`source_split_summary` records which split each dataset used, so the paper table is
auditable.

## Metrics

Reported metrics follow source conventions:

- HotpotQA: answer-only normalized EM/F1
- 2WikiMultiHopQA: answer-only normalized EM/F1
- MuSiQue: answer EM/F1 with aliases
- SQuAD-style datasets: normalized EM/F1
- MS MARCO QA: BLEU-1/2/3/4 and ROUGE-L adapter
- MCQ datasets: accuracy, using deterministic length-normalized option log-prob scoring

Supporting-fact, sufficiency, and joint metrics are not reported unless the model emits
those fields. These models emit final answers only.

For invariant interfaces (`setllm`, `setswitch`, `setfuse`), reported summaries count
each original example once. Visualization rows may include condition copies, but totals
are not inflated.

## Results Table

Build paper-ready tables from the four eval JSON reports:

```bash
python scripts/make_results_table.py \
  outputs/qwen3_4b_chat_baseline_chat_baseline_paper_eval.json \
  outputs/qwen3_4b_setllm_lora_setllm_paper_eval.json \
  outputs/qwen3_4b_setswitch_setswitch_paper_eval.json \
  outputs/qwen3_4b_setfuse_lora_setfuse_paper_eval.json \
  --metric primary_score \
  --scale percent \
  --output-md outputs/results_table.md \
  --output-csv outputs/results_table.csv \
  --output-tex outputs/results_table.tex
```

The CSV includes per-dataset split information.

## Inspect And Debug

Inspect one rendered example:

```bash
python scripts/inspect_batch.py --config configs/flashrag_setfuse.yaml
```

Print an allowed-attention matrix:

```bash
python scripts/debug_mask.py --config configs/flashrag_setswitch.yaml
```

Estimate context/mask footprint:

```bash
python scripts/estimate_context.py --config configs/flashrag.yaml
```
