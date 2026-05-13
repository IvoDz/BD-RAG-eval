import json
import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator

translator_to_lv = GoogleTranslator(source='en', target='lv')
translator_to_en = GoogleTranslator(source='lv', target='en')

def translate_sample(sample, target_lang):
    engine = translator_to_lv if target_lang == 'lv' else translator_to_en

    new_sample = sample.copy()

    new_sample['context'] = engine.translate(sample['context'])
    new_sample['question'] = engine.translate(sample['question'])
    new_sample['gold_answer'] = engine.translate(sample['gold_answer'])
    new_sample['language'] = target_lang

    new_meta = sample['metadata'].copy()
    if 'distractor' in new_meta:
        new_meta['distractor'] = engine.translate(new_meta['distractor'])
    if 'original_clean_context' in new_meta:
        new_meta['original_clean_context'] = engine.translate(new_meta['original_clean_context'])

    new_sample['metadata'] = new_meta
    return new_sample

import time
import random
import json
from tqdm import tqdm

def process_full_set(file_path, output_en_name, output_lv_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    full_en = []
    full_lv = []

    print(f"Processing {file_path}...")
    for s in tqdm(data):
        try:
            time.sleep(random.uniform(1.2, 2.8))
            if s['language'] == 'lv':
                translated = translate_sample(s, 'en')
                full_lv.append(s)
                full_en.append(translated)
            else:
                translated = translate_sample(s, 'lv')
                full_en.append(s)
                full_lv.append(translated)
        except Exception as e:
            print(f"\n[ERROR] ID: {s.get('id', 'N/A')} | File: {file_path} | {str(e)}")
            continue

    with open(output_en_name, 'w', encoding='utf-8') as f:
        for s in full_en:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    with open(output_lv_name, 'w', encoding='utf-8') as f:
        for s in full_lv:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')

process_full_set("rag_eval_clean.jsonl", "final_en_clean.jsonl", "final_lv_clean.jsonl")
process_full_set("rag_eval_noisy.jsonl", "final_en_noisy.jsonl", "final_lv_noisy.jsonl")

