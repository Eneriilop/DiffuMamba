from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER_NAME = "dbmdz/bert-base-italian-cased"
BLOCK_SIZE = 512
INPUT_FILE = "data/training_blocks.json"
OUTPUT_DIR = "data/blocks_512"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
SEP_ID = tokenizer.sep_token_id  # 102

# 1. Carica JSON Lines
dataset = load_dataset("json", data_files=INPUT_FILE, split="train")
print(f"Documenti totali: {len(dataset)}")

# 2. Tokenizza senza truncation (lunghezza naturale di ogni articolo)
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=False, padding=False)

tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizzazione"
)

# 3. Concatena tutto e taglia in blocchi esatti da 512 token
# [SEP] viene inserito come separatore tra articoli
def group_texts(examples):
    all_ids = []
    for ids in examples["input_ids"]:
        all_ids.extend(ids)
        all_ids.append(SEP_ID)

    total = (len(all_ids) // BLOCK_SIZE) * BLOCK_SIZE
    all_ids = all_ids[:total]
    blocks = [all_ids[i:i+BLOCK_SIZE] for i in range(0, total, BLOCK_SIZE)]
    return {"input_ids": blocks}

grouped = tokenized.map(
    group_texts,
    batched=True,
    batch_size=1000,
    remove_columns=tokenized.column_names,
    desc="Creazione blocchi"
)

grouped.save_to_disk(OUTPUT_DIR)
print(f"Blocchi creati: {len(grouped)} x {BLOCK_SIZE} token")
print(f"Salvati in: {OUTPUT_DIR}")