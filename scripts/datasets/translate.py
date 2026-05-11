import json
import time
from pathlib import Path
from googletrans import Translator

INPUT_FILE = "data/processed/base/squad_en.jsonl"
OUTPUT_FILE = "data/processed/squad_lv.jsonl"
TARGET_LANG = "lv"
SLEEP = 0.3

translator = Translator(service_urls=['translate.google.com'])

def translate_text(text: str, cache: dict) -> str:
    if not text or not isinstance(text, str):
        return text
    if text in cache:
        return cache[text]
    
    try:
        result = translator.translate(text, dest=TARGET_LANG).text
        cache[text] = result
        return result
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def main():
    input_path = Path(INPUT_FILE)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_data = []
    cache = {}
        
    for i, line in enumerate(lines):
        if not line.strip(): continue
        
        sample = json.loads(line)
        sample["question"] = translate_text(sample.get("question", ""), cache)
        sample["context"] = translate_text(sample.get("context", ""), cache)
        sample["gold_answer"] = translate_text(sample.get("gold_answer", ""), cache)
        sample["language"] = TARGET_LANG

        output_data.append(sample)
        
        if i % 5 == 0:
            print("batch done")
            time.sleep(SLEEP)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
