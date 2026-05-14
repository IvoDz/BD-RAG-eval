import json
import pandas as pd

files = {
    "LV Clean": "data\\processed\\final\\lv\\master_lv_clean.jsonl",
    "EN Clean": "data\\processed\\final\\en\\master_en_clean.jsonl",
    "LV Noisy": "data\\processed\\final\\lv\\master_lv_noisy.jsonl",
    "EN Noisy": "data\\processed\\final\\en\\master_en_noisy.jsonl",
    "KB distractors": "data\\processed\\final\\kb_distractors_lv_en.jsonl"
}

stats = []
for label, path in files.items():
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                if label == "KB distractors":
                    for lang in ['en', 'lv']:
                        text = data.get(f'{lang}_text', '')
                        stats.append({
                            "Dataset": f"{lang.upper()} Distractor",
                            "Language": lang.upper(),
                            "Type": "Distractor",
                            "Symbols": len(text),
                            "Words": len(text.split())
                        })
                else:
                    ctx = data.get('context', '')
                    stats.append({
                        "Dataset": label,
                        "Language": label.split()[0],
                        "Type": label.split()[1],
                        "Symbols": len(ctx),
                        "Words": len(ctx.split())
                    })
    except FileNotFoundError:
        print(f"Skipping {path} (not found)")

df = pd.DataFrame(stats)

summary = df.groupby(['Language', 'Type']).agg({
    'Symbols': ['count', 'sum', 'mean', 'max'],
    'Words': ['sum', 'mean']
}).round(2)

print("--- Data Statistics ---")
print(summary)

for lang in ['EN', 'LV']:
    dist_size = df[(df['Language'] == lang) & (df['Type'] == 'Distractor')]['Symbols'].sum()
    for t_type in ['Clean', 'Noisy']:
        target_size = df[(df['Language'] == lang) & (df['Type'] == t_type)]['Symbols'].sum()
        total = dist_size + target_size
        print(f"{lang} {t_type} Bucket: {total:,} symbols")

clean_avg = df[df['Type'] == 'Clean']['Symbols'].mean()
noisy_avg = df[df['Type'] == 'Noisy']['Symbols'].mean()
overhead = ((noisy_avg - clean_avg) / clean_avg) * 100

print(f"\n--- Global ---")
print(f"Total Combined Context Volume: {df['Symbols'].sum():,} symbols")
print(f"Noise Impact: {overhead:.2f}% overhead in target contexts.")