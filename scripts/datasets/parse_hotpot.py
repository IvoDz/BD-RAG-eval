from collections import defaultdict
from typing import Any
from datasets import load_dataset

from scripts.utils import (
    create_sample,
    validate_sample,
    save_jsonl,
    print_dataset_stats,
)

def build_hotpot_context(context_data: dict[str, Any] | list[list[Any]]) -> str:

    blocks = []
    
    if isinstance(context_data, dict):
        titles = context_data.get("title", [])
        sentences_list = context_data.get("sentences", [])
        
        for title, sentences in zip(titles, sentences_list):
            if not sentences:
                continue
            
            text = " ".join(sentences)
            blocks.append(f"[{title}]\n{text}")
    
    else:
        for item in context_data:
            if len(item) != 2:
                continue

            title = item[0]
            sentences = item[1]

            if not sentences:
                continue

            text = " ".join(sentences)
            blocks.append(f"[{title}]\n{text}")

    return "\n\n".join(blocks)


def main():

    dataset = load_dataset("hotpot_qa", "fullwiki", split="train")
    dataset = dataset.shuffle(seed=42)
    data = list(dataset)
    buckets = defaultdict(list)

    for item in data:
        level = item.get("level", "unknown")
        if level not in ["easy", "medium", "hard"]:
            continue

        buckets[level].append(item)

    samples = []
    target_per_level = {"easy": 20, "medium": 20, "hard": 10}
    collected = {"easy": 0, "medium": 0, "hard": 0}
    failed_count = 0

    for level in ["easy", "medium", "hard"]:
        for item in buckets[level]:
            if collected[level] >= target_per_level[level]:
                break

            question = item.get("question", "")
            answer = item.get("answer", "")
            context = build_hotpot_context(item.get("context", {}))

            sample = create_sample(
                id=f"hotpot_{len(samples)}",
                domain="multi_hop",
                language="en",
                question=question,
                context=context,
                gold_answer=answer,
                source_dataset="HotpotQA",
                noise_level="clean",
                metadata={"level": level}
            )

            try:
                validate_sample(sample, max_context_length=500)
                samples.append(sample)
                collected[level] += 1
            except Exception as e:
                failed_count += 1
                continue

    print_dataset_stats(samples)

    save_jsonl(
        samples,
        "data/processed/hotpot_balanced_50.jsonl"
    )

if __name__ == "__main__":
    main()
