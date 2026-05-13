import pandas as pd
import json
import re
from dataclasses import dataclass, asdict, field
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
    metadata: dict[str, Any] = field(default_factory=dict)

def clean_context(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\[\d+\]|\[\]', '', text)
    return " ".join(text.split())

def truncate_context(text, limit=500):
    if len(text) <= limit:
        return text
    remaining = text[limit:]
    punct_match = re.search(r'[.!?]', remaining)
    if punct_match:
        return text[:limit + punct_match.start() + 1]
    return text[:limit]

configs = [
    {
        "file": "laws-lv-raw.csv",
        "domain": "legal_lv",
        "lang": "lv",
        "source": "laws-lv-raw",
        "id_prefix": "law"
    },
    {
        "file": "news-lv-raw.csv",
        "domain": "news_lv",
        "lang": "lv",
        "source": "news-lv-raw",
        "id_prefix": "news"
    },
    {
        "file": "wiki-en-raw.csv",
        "domain": "misc_en",
        "lang": "en",
        "source": "wiki-en-raw",
        "id_prefix": "wiki"
    }
]

all_samples = []

for config in configs:
    df = pd.read_csv(config['file'])

    possible_cols = ['raw_context', 'raw_content']
    context_col = next((c for c in possible_cols if c in df.columns), None)

    if not context_col:
        continue

    for idx, row in df.iterrows():
        raw_text = str(row[context_col])

        cleaned = clean_context(raw_text)
        processed_context = truncate_context(cleaned)

        sample = RagSample(
            id=f"{config['id_prefix']}_{idx}",
            domain=config['domain'],
            language=config['lang'],
            question="",
            context=processed_context,
            gold_answer="",
            source_dataset=config['source'],
            noise_level="clean",
            metadata={}
        )
        all_samples.append(asdict(sample))

final_df = pd.DataFrame(all_samples)
final_df.to_csv("combined_rag_base.csv", index=False, encoding='utf-8-sig')

with open("rag-eval-base-raw.jsonl", "w", encoding="utf-8") as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

