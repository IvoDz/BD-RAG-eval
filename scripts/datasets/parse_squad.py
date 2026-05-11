from datasets import load_dataset

from scripts.utils import (
    create_sample,
    validate_sample,
    save_jsonl,
    print_dataset_stats,
)

MAX_CONTEXT = 500

def main():
    dataset = load_dataset("squad", split="train")
    dataset = dataset.shuffle(seed=42)
    data = list(dataset)

    samples = []

    for item in data:

        question = item.get("question", "")
        context = item.get("context", "")
        answers = item.get("answers", {}).get("text", [])

        if not question or not context or not answers:
            continue

        answer = answers[0]

        sample = create_sample(
            id=f"squad_{len(samples)}",
            domain="single_hop",
            language="en",
            question=question,
            context=context,
            gold_answer=answer,
            source_dataset="SQuAD",
            noise_level="clean",
            metadata={}
        )

        try:
            validate_sample(sample, max_context_length=500)
            samples.append(sample)
        except Exception:
            continue

        if len(samples) >= 50:
            break

    print_dataset_stats(samples)

    save_jsonl(
        samples,
        "data/processed/squad_50.jsonl"
    )

if __name__ == "__main__":
    main()
