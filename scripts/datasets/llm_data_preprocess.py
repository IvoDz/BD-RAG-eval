import pandas as pd
import json
import re
import random
import torch
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

def call_llm(prompt, temperature=0.7):
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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

def get_qa(context, language):
    lang_name = "Latvian" if language == "lv" else "English"

    prompt = f"""Context: {context}
              Task: Generate one high-quality, challenging RAG evaluation question and a concise golden answer in {lang_name}.

              Requirements:
              1. The question must be answerable ONLY using the provided context.
              2. Avoid easiest "surface-level" questions.
              3. Focus on reasoning, relationships between concepts, or the "how/why" of the situation described.
              4. The question should be sophisticated enough that it requires understanding the full snippet, not just matching a single keyword.
              5. The answer must be a "golden answer"—precise, factually correct, and directly supported by the text.

              Return ONLY a JSON object: {{"question": "...", "answer": "..."}}
              """

    response = call_llm(prompt, temperature=0.3)
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        return json.loads(match.group()) if match else None
    except:
        return None

def get_noise(context, language):
    lang_name = "Latvian" if language == "lv" else "English"
    prompt = f"""Context: {context}
                 Task: Write 1-2 sentences in {lang_name} that are semantically related to the topic above but provide NO useful information for answering specific questions.
                 Output ONLY the sentences."""
    return call_llm(prompt, temperature=0.8)

def inject_noise(context, noise):
    sentences = re.split(r'(?<=[.!?]) +', context)
    mode = random.choice(['start', 'middle', 'end'])
    if mode == 'start':
        return f"{noise} {context}"
    elif mode == 'end' or len(sentences) < 2:
        return f"{context} {noise}"
    else:
        mid = len(sentences) // 2
        return " ".join(sentences[:mid] + [noise] + sentences[mid:])

with open("rag-eval-base-raw.jsonl", "r", encoding="utf-8") as f:
    base_data = [json.loads(line) for line in f]

clean_version = []
noisy_version = []

for sample in tqdm(base_data):
    qa = get_qa(sample['context'], sample['language'])
    if not qa:
        continue

    clean_sample = sample.copy()
    clean_sample['question'] = qa['question']
    clean_sample['gold_answer'] = qa['answer']
    clean_sample['noise_level'] = "clean"
    clean_version.append(clean_sample)

    noise_text = get_noise(sample['context'], sample['language'])

    noisy_sample = clean_sample.copy()
    noisy_sample['context'] = inject_noise(sample['context'], noise_text)
    noisy_sample['noise_level'] = "noisy"

    noisy_sample['metadata'] = noisy_sample['metadata'].copy()
    noisy_sample['metadata']['distractor'] = noise_text
    noisy_version.append(noisy_sample)

with open("rag_eval_clean.jsonl", "w", encoding="utf-8") as f:
    for s in clean_version:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

with open("rag_eval_noisy.jsonl", "w", encoding="utf-8") as f:
    for s in noisy_version:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(json.dumps(noisy_version[0], indent=2, ensure_ascii=False))

