"""Dataset-suite adapters for document- and option-grounded QA training."""

from __future__ import annotations

import re
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

from set_switch.constants import DEFAULT_INSTRUCTION
from set_switch.data.schema import SetSwitchDocument, SetSwitchExample

FLASHRAG_KNOWN_COUNTS: dict[str, dict[str, int]] = {
    "squad": {"train": 87_599, "dev": 10_570},
    "msmarco-qa": {"train": 808_731, "dev": 101_093},
    "ambig_qa": {"train": 10_036, "dev": 2_002},
    "commonsenseqa": {"train": 9_741, "dev": 1_221},
    "boolq": {"train": 9_427, "dev": 3_270},
    "hotpotqa": {"train": 90_447, "dev": 7_405},
    "2wikimultihopqa": {"train": 15_000, "dev": 12_576},
    "musique": {"train": 19_938, "dev": 2_417},
    "mmlu": {"train": 99_842, "dev": 1_531, "test": 14_042},
    "hellaswag": {"train": 39_905, "dev": 10_042},
    "arc": {"train": 3_370, "dev": 869, "test": 3_548},
    "openbookqa": {"train": 4_957, "dev": 500, "test": 500},
    "quartz": {"train": 2_696, "dev": 384, "test": 784},
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    train_split: str
    validation_split: str
    notes: str


@dataclass(frozen=True)
class FlashRAGSourceSelection:
    name: str
    split: str
    max_examples: int | None = None
    percent: float | None = None


TASK_MCQ = "normal_mcq"
TASK_RAG_SINGLE_HOP = "rag_single_hop"
TASK_RAG_MULTI_HOP = "rag_multi_hop"
TASK_AGGREGATION = "aggregation"

FLASHRAG_TASK_GROUPS: dict[str, str] = {
    "commonsenseqa": TASK_MCQ,
    "openbookqa": TASK_MCQ,
    "arc": TASK_MCQ,
    "hellaswag": TASK_MCQ,
    "mmlu": TASK_MCQ,
    "quartz": TASK_MCQ,
    "msmarco-qa": TASK_RAG_SINGLE_HOP,
    "squad": TASK_RAG_SINGLE_HOP,
    "boolq": TASK_RAG_SINGLE_HOP,
    "hotpotqa": TASK_RAG_MULTI_HOP,
    "2wikimultihopqa": TASK_RAG_MULTI_HOP,
    "musique": TASK_RAG_MULTI_HOP,
    "ambig_qa": TASK_AGGREGATION,
}


def _clean(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _clean_htmlish(text: Any) -> str:
    cleaned = re.sub(r"<[^>]+>", "", _clean(text))
    return re.sub(r"\s+", " ", cleaned).strip()


def _answer_text(answers: Any, config_name: str | None = None) -> str:
    if isinstance(answers, list):
        if config_name == "boolq" and answers and isinstance(answers[0], bool):
            return "yes" if answers[0] else "no"
        cleaned = [_clean(answer) for answer in answers if _clean(answer)]
        if not cleaned:
            return ""
        if config_name == "ambig_qa":
            return "; ".join(cleaned)
        return cleaned[0]
    if isinstance(answers, bool):
        return "yes" if answers else "no"
    return _clean(answers)


def _sequence_to_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], dict):
            return [dict(item) for item in value]
        return [{"value": item} for item in value]
    if isinstance(value, dict):
        keys = list(value.keys())
        lengths = [len(value[key]) for key in keys if isinstance(value.get(key), (list, tuple))]
        if not lengths:
            return [dict(value)]
        count = min(lengths)
        records: list[dict[str, Any]] = []
        for idx in range(count):
            record = {}
            for key in keys:
                item = value[key]
                record[key] = item[idx] if isinstance(item, (list, tuple)) else item
            records.append(record)
        return records
    return []


def _context_documents(
    context: Any,
    source: str,
    example_idx: int,
    support_titles: set[str] | None = None,
    max_docs: int = 8,
) -> list[SetSwitchDocument]:
    support_titles = support_titles or set()
    records = _sequence_to_records(context)
    documents: list[SetSwitchDocument] = []
    for doc_idx, record in enumerate(records):
        title = _clean(record.get("title", ""))
        sentences = record.get("sentences", record.get("content", record.get("paragraphs", "")))
        if isinstance(sentences, (list, tuple)):
            body = " ".join(_clean(sentence) for sentence in sentences if _clean(sentence))
        else:
            body = _clean(sentences)
        if not body:
            continue
        text = f"{title}\n{body}" if title else body
        documents.append(
            SetSwitchDocument(
                doc_id=f"{source}-{example_idx}-doc-{doc_idx}",
                text=text,
                is_gold=title in support_titles,
                metadata={"title": title, "source_doc_index": doc_idx},
            )
        )
    return _limit_documents_prefer_gold(documents, max_docs)


def _support_titles(row: dict[str, Any]) -> set[str]:
    supporting_facts = row.get("supporting_facts", {})
    titles = supporting_facts.get("title", []) if isinstance(supporting_facts, dict) else []
    return {_clean(title) for title in titles if _clean(title)}


def _limit_documents_prefer_gold(
    documents: list[SetSwitchDocument],
    max_docs: int,
) -> list[SetSwitchDocument]:
    if len(documents) <= max_docs:
        return documents
    gold_indices = [idx for idx, doc in enumerate(documents) if doc.is_gold]
    if not gold_indices:
        return documents[:max_docs]
    gold_to_keep = set(gold_indices[:max_docs])
    non_gold_quota = max_docs - len(gold_to_keep)
    selected: list[SetSwitchDocument] = []
    for idx, doc in enumerate(documents):
        if idx in gold_to_keep:
            selected.append(doc)
        elif non_gold_quota > 0:
            selected.append(doc)
            non_gold_quota -= 1
        if len(selected) >= max_docs:
            break
    return selected


def _choice_texts_from_row(row: dict[str, Any]) -> list[str]:
    choices = row.get("choices")
    metadata = row.get("metadata", {})
    metadata_choices = metadata.get("choices") if isinstance(metadata, dict) else None
    if isinstance(choices, list) and isinstance(metadata_choices, list):
        metadata_records = [choice for choice in metadata_choices if isinstance(choice, dict)]
        label_to_text = {
            _clean(choice.get("label")): _clean(choice.get("text"))
            for choice in metadata_records
            if _clean(choice.get("label")) and _clean(choice.get("text"))
        }
        if label_to_text and all(_clean(choice) in label_to_text for choice in choices):
            return [label_to_text[_clean(choice)] for choice in choices if _clean(choice)]
    if isinstance(choices, list):
        return [_clean(choice) for choice in choices if _clean(choice)]
    if isinstance(choices, dict):
        records = _sequence_to_records(choices)
        if records and "text" in records[0]:
            return [_clean(record.get("text")) for record in records if _clean(record.get("text"))]
        if "text" in choices and isinstance(choices["text"], list):
            return [_clean(choice) for choice in choices["text"] if _clean(choice)]

    if isinstance(metadata, dict) and isinstance(metadata.get("choices"), list):
        texts = []
        for choice in metadata["choices"]:
            text = choice.get("text", "") if isinstance(choice, dict) else choice
            if _clean(text):
                texts.append(_clean(text))
        return texts
    return []


def _answer_text_from_choices(row: dict[str, Any], choices: list[str]) -> str:
    answers = row.get("golden_answers", [])
    answer = answers[0] if isinstance(answers, list) and answers else answers
    if isinstance(answer, int) and 0 <= answer < len(choices):
        return choices[answer]
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.isdigit():
            idx = int(stripped)
            if 0 <= idx < len(choices):
                return choices[idx]
        for choice in choices:
            if choice == stripped:
                return choice
        metadata = row.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("answerKey") == stripped:
            for choice in metadata.get("choices", []):
                if isinstance(choice, dict) and choice.get("label") == stripped:
                    return _clean(choice.get("text"))
        return stripped
    return _clean(answer)


def convert_option_row(
    row: dict[str, Any],
    example_idx: int,
    max_docs: int,
    instruction: str,
    source: str,
) -> SetSwitchExample | None:
    question = _clean(row.get("question"))
    choices = _choice_texts_from_row(row)
    answer = _answer_text_from_choices(row, choices)
    if not question or not choices or not answer:
        return None

    documents: list[SetSwitchDocument] = []
    for choice_idx, choice in enumerate(choices[:max_docs]):
        documents.append(
            SetSwitchDocument(
                doc_id=f"{source}-{example_idx}-option-{choice_idx}",
                text=f"Candidate answer: {choice}",
                is_gold=choice == answer,
                metadata={
                    "set_item_type": "option",
                    "choice_index": choice_idx,
                    "choice_text": choice,
                },
            )
        )

    if not any(doc.is_gold for doc in documents):
        return None

    return SetSwitchExample(
        example_id=f"{source}-{row.get('id', example_idx)}",
        instruction=instruction,
        question=question,
        documents=documents,
        answer=answer,
        source=source,
        metadata={
            "set_type": "options",
            "raw_id": row.get("id"),
            "num_choices": len(choices),
            "task_group": task_group_for_source(source),
        },
    )


DATASET_SPECS: dict[str, DatasetSpec] = {
    "hotpotqa": DatasetSpec(
        name="hotpotqa",
        train_split="train",
        validation_split="dev",
        notes="Wikipedia multi-hop QA with paragraph contexts and supporting facts.",
    ),
    "2wikimultihopqa": DatasetSpec(
        name="2wikimultihopqa",
        train_split="train",
        validation_split="dev",
        notes="Wikipedia multi-hop QA with HotpotQA-like schema.",
    ),
    "musique": DatasetSpec(
        name="musique",
        train_split="train",
        validation_split="dev",
        notes="Compositional multi-hop QA with supporting paragraph flags.",
    ),
    "msmarco-qa": DatasetSpec(
        name="msmarco-qa",
        train_split="train",
        validation_split="dev",
        notes="Retrieved-passage QA with selected passage flags.",
    ),
    "squad": DatasetSpec(
        name="squad",
        train_split="train",
        validation_split="dev",
        notes="Single-passage extractive QA with paragraph context.",
    ),
    "boolq": DatasetSpec(
        name="boolq",
        train_split="train",
        validation_split="dev",
        notes="Single-passage yes/no QA with Wikipedia passages.",
    ),
    "ambig_qa": DatasetSpec(
        name="ambig_qa",
        train_split="train",
        validation_split="dev",
        notes="Ambiguous/multi-answer QA; converted from available search-result snippets.",
    ),
    "commonsenseqa": DatasetSpec(
        name="commonsenseqa",
        train_split="train",
        validation_split="dev",
        notes="Commonsense multiple choice; choices are unordered SetSwitch items.",
    ),
    "openbookqa": DatasetSpec(
        name="openbookqa",
        train_split="train",
        validation_split="dev",
        notes="Science multiple choice; choices are unordered SetSwitch items.",
    ),
    "arc": DatasetSpec(
        name="arc",
        train_split="train",
        validation_split="dev",
        notes="Science exam multiple choice; choices are unordered SetSwitch items.",
    ),
    "hellaswag": DatasetSpec(
        name="hellaswag",
        train_split="train",
        validation_split="dev",
        notes="Commonsense continuation multiple choice; choices are unordered SetSwitch items.",
    ),
    "mmlu": DatasetSpec(
        name="mmlu",
        train_split="train",
        validation_split="dev",
        notes="Knowledge multiple choice; choices are unordered SetSwitch items.",
    ),
    "quartz": DatasetSpec(
        name="quartz",
        train_split="train",
        validation_split="dev",
        notes="Qualitative reasoning multiple choice; choices are unordered SetSwitch items.",
    ),
}

DEFAULT_FLASHRAG_DOCUMENT_CONFIGS = [
    "msmarco-qa",
    "squad",
    "boolq",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "ambig_qa",
]

DEFAULT_FLASHRAG_OPTION_CONFIGS = [
    "commonsenseqa",
    "openbookqa",
    "arc",
    "hellaswag",
    "mmlu",
    "quartz",
]

DEFAULT_FLASHRAG_CONFIGS = DEFAULT_FLASHRAG_DOCUMENT_CONFIGS[:4]
DEFAULT_FLASHRAG_MIXED_CONFIGS = DEFAULT_FLASHRAG_OPTION_CONFIGS + DEFAULT_FLASHRAG_DOCUMENT_CONFIGS


FLASHRAG_ALIASES = {
    "hotpotqa": "hotpotqa",
    "hotpot_qa": "hotpotqa",
    "2wiki": "2wikimultihopqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
    "msmarco": "msmarco-qa",
    "ms_marco": "msmarco-qa",
    "msmarcoqa": "msmarco-qa",
    "msmarco-qa": "msmarco-qa",
    "squad": "squad",
    "boolq": "boolq",
    "ambigqa": "ambig_qa",
    "ambig_qa": "ambig_qa",
    "commonsenseqa": "commonsenseqa",
    "commonsense_qa": "commonsenseqa",
    "openbookqa": "openbookqa",
    "openbook_qa": "openbookqa",
    "arc": "arc",
    "hellaswag": "hellaswag",
    "mmlu": "mmlu",
    "quartz": "quartz",
}


def _canonical_source_name(name: str) -> str:
    normalized = name.strip().lower()
    return FLASHRAG_ALIASES.get(normalized, normalized)


def _parse_percent(value: str) -> float:
    stripped = value.strip()
    if stripped.endswith("%"):
        return float(stripped[:-1]) / 100.0
    return float(stripped)


def _parse_source_string(text: str, default_split: str) -> FlashRAGSourceSelection:
    """Parse compact source specs like hotpotqa[:0.5] or mmlu[dev:10%]."""

    raw = text.strip()
    if not raw:
        raise ValueError("Empty FlashRAG dataset source")
    if "[" not in raw:
        return FlashRAGSourceSelection(name=_canonical_source_name(raw), split=default_split)

    if not raw.endswith("]"):
        raise ValueError(
            f"Malformed FlashRAG dataset source {text!r}; expected name[split:percent]"
        )

    name, bracket = raw.split("[", 1)
    spec = bracket[:-1].strip()
    split = default_split
    percent: float | None = None

    if ":" in spec:
        split_part, percent_part = spec.split(":", 1)
        if split_part.strip():
            split = split_part.strip()
        if percent_part.strip():
            percent = _parse_percent(percent_part)
    elif spec:
        split = spec

    return FlashRAGSourceSelection(
        name=_canonical_source_name(name),
        split=split,
        percent=percent,
    )


def _flashrag_msmarco_documents(
    passages: Any,
    example_idx: int,
    max_docs: int,
) -> list[SetSwitchDocument]:
    if not isinstance(passages, dict):
        return []
    texts = passages.get("passage_text", [])
    selected = passages.get("is_selected", [])
    urls = passages.get("url", [])
    documents: list[SetSwitchDocument] = []
    for doc_idx, text in enumerate(list(texts)):
        body = _clean(text)
        if not body:
            continue
        documents.append(
            SetSwitchDocument(
                doc_id=f"flashrag-msmarco-{example_idx}-doc-{doc_idx}",
                text=body,
                is_gold=bool(selected[doc_idx]) if doc_idx < len(selected) else False,
                metadata={
                    "source_doc_index": doc_idx,
                    "url": urls[doc_idx] if doc_idx < len(urls) else None,
                },
            )
        )
    return _limit_documents_prefer_gold(documents, max_docs)


def _flashrag_musique_documents(
    metadata: dict[str, Any],
    example_idx: int,
    max_docs: int,
) -> list[SetSwitchDocument]:
    documents: list[SetSwitchDocument] = []
    seen: set[tuple[str, str]] = set()
    decompositions = metadata.get("question_decomposition", [])
    for doc_idx, step in enumerate(_sequence_to_records(decompositions)):
        support = step.get("support_paragraph", {})
        if not isinstance(support, dict):
            continue
        title = _clean(support.get("title", ""))
        body = _clean(support.get("paragraph_text"))
        key = (title, body)
        if not body or key in seen:
            continue
        seen.add(key)
        documents.append(
            SetSwitchDocument(
                doc_id=f"flashrag-musique-{example_idx}-doc-{len(documents)}",
                text=f"{title}\n{body}" if title else body,
                is_gold=True,
                metadata={
                    "title": title,
                    "source_doc_index": support.get("idx"),
                    "decomposition_question": step.get("question"),
                },
            )
        )
        if len(documents) >= max_docs:
            break
    return documents


def _flashrag_single_passage_documents(
    metadata: dict[str, Any],
    example_idx: int,
    config_name: str,
) -> list[SetSwitchDocument]:
    body = _clean(metadata.get("text", metadata.get("passage", "")))
    if not body:
        return []
    title = _clean(metadata.get("title", ""))
    text = f"{title}\n{body}" if title else body
    return [
        SetSwitchDocument(
            doc_id=f"flashrag-{config_name}-{example_idx}-doc-0",
            text=text,
            is_gold=True,
            metadata={"title": title, "source_doc_index": 0},
        )
    ]


def _flashrag_ambigqa_documents(
    metadata: dict[str, Any],
    example_idx: int,
    max_docs: int,
) -> list[SetSwitchDocument]:
    gold_titles = {
        _clean(title) for title in metadata.get("viewed_doc_titles", []) if _clean(title)
    }
    if _clean(metadata.get("nq_doc_title")):
        gold_titles.add(_clean(metadata.get("nq_doc_title")))

    documents: list[SetSwitchDocument] = []
    seen: set[tuple[str, str]] = set()
    for query in _sequence_to_records(metadata.get("used_queries", [])):
        for result in _sequence_to_records(query.get("results", [])):
            title = _clean_htmlish(result.get("title", ""))
            snippet = _clean_htmlish(result.get("snippet", ""))
            if not snippet:
                continue
            key = (title, snippet)
            if key in seen:
                continue
            seen.add(key)
            documents.append(
                SetSwitchDocument(
                    doc_id=f"flashrag-ambigqa-{example_idx}-doc-{len(documents)}",
                    text=f"{title}\n{snippet}" if title else snippet,
                    is_gold=title in gold_titles if title else False,
                    metadata={"title": title, "source_doc_index": len(documents)},
                )
            )
            if len(documents) >= max_docs:
                return documents
    return documents


def convert_flashrag_row(
    row: dict[str, Any],
    example_idx: int,
    max_docs: int,
    instruction: str,
    config_name: str,
) -> SetSwitchExample | None:
    answers = row.get("golden_answers", [])
    question = _clean(row.get("question"))
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {}
    if not question:
        return None

    choices = _choice_texts_from_row(row)
    if choices:
        return convert_option_row(
            row=row,
            example_idx=example_idx,
            max_docs=max_docs,
            instruction=instruction,
            source=f"flashrag_{config_name}",
        )

    answer = _answer_text(answers, config_name=config_name)
    if not answer:
        return None

    documents: list[SetSwitchDocument] = []
    if isinstance(metadata.get("context"), dict):
        documents = _context_documents(
            metadata.get("context"),
            f"flashrag-{config_name}",
            example_idx,
            support_titles=_support_titles(metadata),
            max_docs=max_docs,
        )
    elif isinstance(metadata.get("passages"), dict):
        documents = _flashrag_msmarco_documents(metadata.get("passages"), example_idx, max_docs)
    elif metadata.get("question_decomposition"):
        documents = _flashrag_musique_documents(metadata, example_idx, max_docs)
    elif config_name in {"squad", "boolq"}:
        documents = _flashrag_single_passage_documents(metadata, example_idx, config_name)
    elif config_name == "ambig_qa":
        documents = _flashrag_ambigqa_documents(metadata, example_idx, max_docs)

    if not documents:
        return None

    example_metadata = {
        "flashrag_config": config_name,
        "raw_id": row.get("id"),
        "golden_answers": answers,
        "set_type": "documents",
        "task_group": task_group_for_source(config_name),
    }
    if config_name == "musique":
        example_metadata["context_policy"] = "support_only_question_decomposition"

    return SetSwitchExample(
        example_id=f"flashrag-{config_name}-{row.get('id', example_idx)}",
        instruction=instruction,
        question=question,
        documents=documents,
        answer=answer,
        source=f"flashrag_{config_name}",
        metadata=example_metadata,
    )


def task_group_for_source(source: str) -> str:
    """Return the coarse evaluation bucket for a FlashRAG source/config name."""

    source = source.removeprefix("flashrag_")
    return FLASHRAG_TASK_GROUPS.get(_canonical_source_name(source), "unknown")


def _split_for_source(source: str, requested_split: str) -> str:
    if requested_split not in {"train", "dev", "validation", "val", "test"}:
        return requested_split
    spec = DATASET_SPECS.get(source)
    if requested_split == "train":
        return spec.train_split if spec else "train"
    if requested_split in {"dev", "validation", "val"}:
        return spec.validation_split if spec else "dev"
    return "test"


def _source_selection_limit(
    selection: FlashRAGSourceSelection,
    split: str,
    equal_limit: int | None = None,
) -> int | None:
    if selection.max_examples is not None:
        return selection.max_examples
    if equal_limit is not None:
        return equal_limit
    if selection.percent is not None:
        if not (0 < selection.percent <= 1):
            raise ValueError(f"percent for {selection.name!r} must be in (0, 1]")
        count = estimate_flashrag_source_count(selection.name, split)
        if count is None:
            raise ValueError(
                f"Cannot use percent for {selection.name!r} because no row-count estimate is known"
            )
        return max(1, int(count * selection.percent))
    return None


def estimate_flashrag_source_count(source: str, split: str) -> int | None:
    normalized_split = _split_for_source(source, split)
    return FLASHRAG_KNOWN_COUNTS.get(source, {}).get(normalized_split)


def _round_weighted_counts(
    indices: list[int],
    weights: dict[int, float],
    total: int,
) -> dict[int, int]:
    if total <= 0 or not indices:
        return {idx: 0 for idx in indices}

    weight_sum = sum(max(weights.get(idx, 0.0), 0.0) for idx in indices)
    if weight_sum <= 0:
        weights = {idx: 1.0 for idx in indices}
        weight_sum = float(len(indices))

    raw = {idx: total * max(weights.get(idx, 0.0), 0.0) / weight_sum for idx in indices}
    counts = {idx: int(math.floor(value)) for idx, value in raw.items()}
    remainder = total - sum(counts.values())
    for idx in sorted(indices, key=lambda item: (raw[item] - counts[item], -item), reverse=True)[
        :remainder
    ]:
        counts[idx] += 1
    return counts


def _weighted_capped_allocation(
    indices: list[int],
    weights: dict[int, float],
    total: int,
    caps: dict[int, int | None],
) -> dict[int, int]:
    allocation = {idx: 0 for idx in indices}
    active = list(indices)
    remaining = int(total)

    while active and remaining > 0:
        proposal = _round_weighted_counts(active, weights, remaining)
        capped: list[int] = []
        for idx in active:
            cap = caps.get(idx)
            if cap is not None and proposal[idx] >= cap:
                allocation[idx] += cap
                remaining -= cap
                capped.append(idx)

        if not capped:
            for idx, value in proposal.items():
                allocation[idx] += value
            remaining = 0
            break

        active = [idx for idx in active if idx not in capped]

    return allocation


def _size_weight(selection: FlashRAGSourceSelection, split: str, alpha: float) -> float:
    count = estimate_flashrag_source_count(selection.name, split)
    if count is None:
        return 1.0
    return float(count) ** alpha


def allocate_flashrag_source_limits(
    selections: list[FlashRAGSourceSelection],
    total_examples: int | None,
    strategy: str = "task_balanced_equal",
    alpha: float = 0.5,
) -> list[int | None]:
    """Allocate per-source example limits for a mixed FlashRAG run.

    The default is task-balanced first, then source-balanced within each task.
    This keeps RAG/MCQ/aggregation buckets visible and prevents a large source
    such as MS MARCO from dominating the first comparison.
    """

    if total_examples is None:
        return [None] * len(selections)
    if any(selection.percent is not None for selection in selections):
        raise ValueError("Use either total_examples or per-dataset percent, not both")

    fixed = [
        int(selection.max_examples) if selection.max_examples is not None else None
        for selection in selections
    ]
    fixed_total = sum(limit or 0 for limit in fixed)
    if fixed_total > int(total_examples):
        raise ValueError("Per-dataset max_examples exceeds total_examples")

    variable = [idx for idx, limit in enumerate(fixed) if limit is None]
    if not variable:
        return fixed

    remaining = int(total_examples) - fixed_total
    hf_splits = [_split_for_source(selection.name, selection.split) for selection in selections]
    caps = {
        idx: estimate_flashrag_source_count(selections[idx].name, hf_splits[idx])
        for idx in variable
    }
    strategy = strategy.lower()
    alpha = float(alpha)

    if strategy == "equal_per_source":
        weights = {idx: 1.0 for idx in variable}
        variable_limits = _weighted_capped_allocation(variable, weights, remaining, caps)
    elif strategy in {"proportional_size", "sqrt_size", "temperature"}:
        source_alpha = 1.0 if strategy == "proportional_size" else alpha
        weights = {
            idx: _size_weight(selections[idx], hf_splits[idx], source_alpha) for idx in variable
        }
        variable_limits = _weighted_capped_allocation(variable, weights, remaining, caps)
    elif strategy in {"task_balanced_equal", "task_balanced_sqrt"}:
        groups: dict[str, list[int]] = {}
        for idx in variable:
            groups.setdefault(task_group_for_source(selections[idx].name), []).append(idx)

        group_keys = sorted(groups)
        group_index = {group: idx for idx, group in enumerate(group_keys)}
        group_caps: dict[int, int | None] = {}
        for group, group_idx in group_index.items():
            group_source_caps = [caps[idx] for idx in groups[group]]
            group_caps[group_idx] = (
                None if any(cap is None for cap in group_source_caps) else sum(group_source_caps)
            )
        group_alloc = _weighted_capped_allocation(
            list(group_index.values()),
            {idx: 1.0 for idx in group_index.values()},
            remaining,
            group_caps,
        )

        variable_limits = {idx: 0 for idx in variable}
        for group, source_indices in groups.items():
            group_total = group_alloc[group_index[group]]
            if strategy == "task_balanced_equal":
                weights = {idx: 1.0 for idx in source_indices}
            else:
                weights = {
                    idx: _size_weight(selections[idx], hf_splits[idx], alpha)
                    for idx in source_indices
                }
            variable_limits.update(
                _weighted_capped_allocation(source_indices, weights, group_total, caps)
            )
    else:
        raise ValueError(
            "Unknown sample_allocation strategy "
            f"{strategy!r}; expected task_balanced_sqrt, task_balanced_equal, "
            "sqrt_size, proportional_size, or equal_per_source"
        )

    return [
        fixed[idx] if fixed[idx] is not None else variable_limits.get(idx, 0)
        for idx in range(len(selections))
    ]


def normalize_flashrag_sources(
    data_cfg: dict[str, Any], split: str
) -> list[FlashRAGSourceSelection]:
    raw_sources = data_cfg.get("datasets", data_cfg.get("configs", DEFAULT_FLASHRAG_CONFIGS))
    selections: list[FlashRAGSourceSelection] = []
    if isinstance(raw_sources, dict):
        raw_sources = [{"name": name, **(value or {})} for name, value in raw_sources.items()]

    def split_specific_value(item: dict[str, Any], field: str) -> Any:
        aliases = [split]
        if split in {"dev", "validation", "val"}:
            aliases.extend(["validation", "val", "dev", "eval"])
        for alias in dict.fromkeys(aliases):
            key = f"{alias}_{field}"
            if item.get(key) is not None:
                return item[key]
        return item.get(field)

    for item in raw_sources:
        if isinstance(item, str):
            selections.append(
                _parse_source_string(item, str(data_cfg.get(f"{split}_split", split)))
            )
            continue
        if isinstance(item, dict):
            name = item.get("name", item.get("config"))
            if not name:
                raise ValueError(f"FlashRAG dataset entry is missing name/config: {item}")
            selections.append(
                FlashRAGSourceSelection(
                    name=_canonical_source_name(str(name)),
                    split=str(
                        split_specific_value(item, "split")
                        or data_cfg.get(f"{split}_split", split)
                    ),
                    max_examples=(
                        int(split_specific_value(item, "max_examples"))
                        if split_specific_value(item, "max_examples") is not None
                        else None
                    ),
                    percent=(
                        _parse_percent(str(split_specific_value(item, "percent")))
                        if split_specific_value(item, "percent") is not None
                        else None
                    ),
                )
            )
            continue
        raise TypeError(f"Unsupported FlashRAG dataset entry: {item!r}")
    return selections


def iter_flashrag_selected_examples(
    dataset_name: str,
    selections: list[FlashRAGSourceSelection],
    max_docs: int = 8,
    instruction: str = DEFAULT_INSTRUCTION,
    total_examples: int | None = None,
    sample_allocation: str = "task_balanced_equal",
    sample_allocation_alpha: float = 0.5,
) -> Iterator[SetSwitchExample]:
    allocated_limits = allocate_flashrag_source_limits(
        selections=selections,
        total_examples=total_examples,
        strategy=sample_allocation,
        alpha=sample_allocation_alpha,
    )

    for selection_idx, selection in enumerate(selections):
        hf_split = _split_for_source(selection.name, selection.split)
        if total_examples is None:
            limit = _source_selection_limit(selection, hf_split)
        else:
            limit = allocated_limits[selection_idx]
        if limit is not None and limit <= 0:
            continue
        dataset = load_dataset(dataset_name, selection.name, split=hf_split, streaming=True)
        kept = 0
        for row_idx, row in enumerate(dataset):
            example = convert_flashrag_row(
                row=dict(row),
                example_idx=row_idx,
                max_docs=max_docs,
                instruction=instruction,
                config_name=selection.name,
            )
            if example is None:
                continue
            yield example
            kept += 1
            if limit is not None and kept >= limit:
                break


def load_flashrag_selected_examples(
    dataset_name: str,
    selections: list[FlashRAGSourceSelection],
    max_docs: int = 8,
    instruction: str = DEFAULT_INSTRUCTION,
    total_examples: int | None = None,
    sample_allocation: str = "task_balanced_equal",
    sample_allocation_alpha: float = 0.5,
) -> list[SetSwitchExample]:
    return list(
        iter_flashrag_selected_examples(
            dataset_name=dataset_name,
            selections=selections,
            max_docs=max_docs,
            instruction=instruction,
            total_examples=total_examples,
            sample_allocation=sample_allocation,
            sample_allocation_alpha=sample_allocation_alpha,
        )
    )
