import json
import re
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

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
        max_new_tokens=200,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def get_noise(context, language):
    lang_name = "Latvian" if language == "lv" else "English"
    prompt = f"""Context: {context}
                 Task: Write 1-2 sentences in {lang_name} that are semantically related to the topic above but provide NO useful information for answering specific questions.
                 Output ONLY the sentences."""
    return call_llm(prompt, temperature=0.8)

def inject_noise(context, noise):
    sentences = re.split(r'(?<=[.!?]) +', context)
    if len(sentences) < 2:
        return f"{context} {noise}"

    mode = random.choice(['start', 'middle', 'end'])
    if mode == 'start':
        return f"{noise} {context}"
    elif mode == 'end':
        return f"{context} {noise}"
    else:
        mid = len(sentences) // 2
        return " ".join(sentences[:mid] + [noise] + sentences[mid:])

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

squad_en = load_jsonl("squad_en.jsonl")
hotpot_en = load_jsonl("hotpot_en.jsonl")
squad_lv = load_jsonl("squad_lv.jsonl")
hotpot_lv = load_jsonl("hotpot_lv.jsonl")

en_clean_pool = squad_en + hotpot_en
lv_clean_pool = squad_lv + hotpot_lv

def process_noise_task(clean_pool, lang_label):
    noisy_results = []
    print(f"Generating noise for {lang_label} samples...")

    for sample in tqdm(clean_pool):
        distractor = get_noise(sample['context'], sample['language'])

        noisy_sample = sample.copy()
        noisy_sample['context'] = inject_noise(sample['context'], distractor)
        noisy_sample['noise_level'] = "noisy"

        if 'metadata' not in noisy_sample: noisy_sample['metadata'] = {}
        noisy_sample['metadata']['distractor'] = distractor

        noisy_results.append(noisy_sample)

    return noisy_results

en_noisy_pool = process_noise_task(en_clean_pool, "English")
lv_noisy_pool = process_noise_task(lv_clean_pool, "Latvian")

def save_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

save_jsonl(en_clean_pool, "base_clean_en.jsonl")
save_jsonl(en_noisy_pool, "base_noisy_en.jsonl")
save_jsonl(lv_clean_pool, "base_clean_lv.jsonl")
save_jsonl(lv_noisy_pool, "base_noisy_lv.jsonl")

print("Files generated successfully:")
print("- base_clean_en.jsonl (100 samples)")
print("- base_noisy_en.jsonl (100 samples)")
print("- base_clean_lv.jsonl (100 samples)")
print("- base_noisy_lv.jsonl (100 samples)")

print("\n--- Noisy LV Example Sample ---")
print(json.dumps(lv_noisy_pool[0], indent=2, ensure_ascii=False))

