from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RagSample:
    id: str
    domain: str
    language: str
    question: str
    context: str
    gold_answer: str
    source_dataset: str
    noise_level: str
    metadata: dict[str, Any]


def create_sample(
    *,
    id: str,
    domain: str,
    language: str,
    question: str,
    context: str,
    gold_answer: str,
    source_dataset: str,
    noise_level: str = "clean",
    metadata: dict[str, Any] | None = None,
) -> RagSample:

    return RagSample(
        id=normalize_whitespace(id),
        domain=normalize_whitespace(domain),
        language=normalize_whitespace(language),
        question=normalize_whitespace(question),
        context=normalize_whitespace(context),
        gold_answer=normalize_whitespace(gold_answer),
        source_dataset=normalize_whitespace(source_dataset),
        noise_level=normalize_whitespace(noise_level),
        metadata=metadata or {},
    )


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def remove_duplicate_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_control_characters(text: str) -> str:
    return re.sub(r"[\x00-\x1F\x7F]", "", text)


def clean_text(text: str) -> str:
    text = remove_control_characters(text)
    text = remove_duplicate_spaces(text)
    return text.strip()

def validate_sample(
    sample: RagSample,
    *,
    min_context_length: int = 200,
    max_context_length: int = 15000,
) -> None:

    if not sample.id:
        raise ValueError("Missing id")

    if not sample.domain:
        raise ValueError("Missing domain")

    if not sample.language:
        raise ValueError("Missing language")

    if not sample.question:
        raise ValueError("Missing question")

    if not sample.context:
        raise ValueError("Missing context")

    if not sample.gold_answer:
        raise ValueError("Missing gold_answer")

    if not sample.source_dataset:
        raise ValueError("Missing source_dataset")

    if not sample.noise_level:
        raise ValueError("Missing noise_level")

    if len(sample.context) < min_context_length:
        raise ValueError(f"Context too short: {sample.id}")

    if len(sample.context) > max_context_length:
        raise ValueError(f"Context too long: {sample.id}")

    if len(sample.question) < 5:
        raise ValueError(f"Question too short: {sample.id}")

    if len(sample.gold_answer) < 1:
        raise ValueError(f"Empty answer: {sample.id}")


def contains_answer(
    context: str,
    answer: str,
    *,
    case_sensitive: bool = False,
) -> bool:

    if not case_sensitive:
        context = context.lower()
        answer = answer.lower()

    return answer in context


def is_valid_qa_sample(
    *,
    question: str,
    answer: str,
    context: str,
) -> bool:
    
    if len(question) < 10:
        return False

    if len(answer) < 1:
        return False

    if len(context) < 100:
        return False

    if len(context) > 1000:
        return False

    return True

def save_jsonl(
    samples: list[RagSample],
    output_path: str | Path,
) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            line = json.dumps(asdict(sample), ensure_ascii=False)
            f.write(line + "\n")


def load_jsonl(input_path: str | Path) -> list[dict]:

    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def print_dataset_stats(samples: list[RagSample]) -> None:

    total = len(samples)

    avg_context = sum(len(s.context) for s in samples) // total
    avg_question = sum(len(s.question) for s in samples) // total

    print("=" * 50)
    print("Total samples:", total)
    print("Avg context length:", avg_context)
    print("Avg question length:", avg_question)
    print("=" * 50)


def remove_duplicate_questions(
    samples: list[RagSample],
) -> list[RagSample]:

    seen = set()
    unique = []

    for sample in samples:

        q = sample.question.lower().strip()

        if q in seen:
            continue

        seen.add(q)
        unique.append(sample)

    return unique
