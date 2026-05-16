import json
import os
import chromadb
import torch
import gc
from transformers import pipeline, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

TARGET_SETTING = "lv_clean"
OUTPUT_FILE = f"results_{TARGET_SETTING}.jsonl"
DB_PATH = "./rag_eval_db"

embed_model = SentenceTransformer('BAAI/bge-m3', device="cpu")

class ChromaEmbeddingFunction:
    def __call__(self, input):
        return embed_model.encode(input).tolist()

    def embed_query(self, input):
        return self.__call__(input)

    def name(self):
        return "bge-m3"

db_client = chromadb.PersistentClient(path=DB_PATH)
emb_fn = ChromaEmbeddingFunction()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

generator = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    model_kwargs={"quantization_config": bnb_config},
    device_map="auto"
)

def ingest_data():
    colls = {s: db_client.get_or_create_collection(s, embedding_function=emb_fn) for s in ["en_clean", "en_noisy", "lv_clean", "lv_noisy"]}

    with open("kb_distractors_lv_en.jsonl", "r", encoding="utf-8") as f:
        distractors = [json.loads(line) for line in f]

    for d in distractors:
        for s in ["en_clean", "en_noisy"]:
            colls[s].add(ids=[d["distractor_id"]], documents=[d["en_text"]], metadatas=[{"type": "distractor"}])
        for s in ["lv_clean", "lv_noisy"]:
            colls[s].add(ids=[d["distractor_id"]], documents=[d["lv_text"]], metadatas=[{"type": "distractor"}])

    needle_files = {
        "en_clean": "master_en_clean.jsonl",
        "en_noisy": "master_en_noisy.jsonl",
        "lv_clean": "master_lv_clean.jsonl",
        "lv_noisy": "master_lv_noisy.jsonl"
    }

    for s, path in needle_files.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                needles = [json.loads(line) for line in f]
            for n in needles:
                colls[s].add(ids=[n["id"]], documents=[n["context"]], metadatas=[{"type": "needle"}])

ingest_data()

def run_split_pipeline():
    collection = db_client.get_collection(TARGET_SETTING, embedding_function=emb_fn)

    is_lv = TARGET_SETTING.startswith("lv")
    lang_name = "Latvian" if is_lv else "English"
    unk_phrase = "Es nezinu" if is_lv else "I do not know"

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            processed_ids = {json.loads(line)["sample_id"] for line in f if line.strip()}
    else:
        processed_ids = set()

    test_files = {
        "en_clean": "master_en_clean.jsonl",
        "en_noisy": "master_en_noisy.jsonl",
        "lv_clean": "master_lv_clean.jsonl",
        "lv_noisy": "master_lv_noisy.jsonl"
    }

    with open(test_files[TARGET_SETTING], "r", encoding="utf-8") as f:
        remaining_queries = [json.loads(line) for line in f if json.loads(line)["id"] not in processed_ids]

    print(f"Starting {TARGET_SETTING} in {lang_name}...")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for q in remaining_queries:
            res = collection.query(query_texts=[q["question"]], n_results=3)
            contexts, ids = res["documents"][0], res["ids"][0]

            context_block = "\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])

            messages = [
                {
                    "role": "system",
                    "content": f"You are a factual engine. Answer in {lang_name} using ONLY the provided context. Follow rules strictly."
                },
                {
                    "role": "user",
                    "content": (
                        f"Rules:\n"
                        f"1. Use ONLY context below to answer in {lang_name}.\n"
                        f"2. If answer not found, output EXACTLY: '{unk_phrase}'.\n"
                        f"3. No outside info or filler.\n\n"
                        f"Context:\n{context_block}\n\n"
                        f"Question: {q['question']}"
                    )
                }
            ]

            prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            output = generator(
                prompt,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.0,
                return_full_text=False,
                eos_token_id=generator.tokenizer.eos_token_id
            )[0]["generated_text"]

            out_f.write(json.dumps({
                "setting": TARGET_SETTING,
                "sample_id": q["id"],
                "question": q["question"],
                "ground_truth": q["gold_answer"],
                "generated_answer": output.strip(),
                "retrieved_ids": ids
            }, ensure_ascii=False) + "\n")
            out_f.flush()

run_split_pipeline()
