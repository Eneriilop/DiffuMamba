"""
probing_multilayer.py — Probing su tutti i layer per ogni checkpoint
"""

import os, re, csv, torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from model import DiffuMamba
import config

CHECKPOINTS_DIR = "checkpoints_sent_run4"
TRAIN_TSV       = "data/train_probe.tsv"
TEST_TSV        = "data/test_probe.tsv"
OUTPUT_BASE_DIR = "probing_results_multilayer"
BATCH_SIZE      = 64
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tsv(path):
    texts, features = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        feature_names = [c for c in reader.fieldnames if c not in ("identifier", "text")]
        for row in reader:
            texts.append(row["text"])
            features.append([float(row[fn]) for fn in feature_names])
    return texts, feature_names, np.array(features, dtype=np.float32)


def encode_all_layers(model, tokenizer, texts):
    """
    Una sola forward pass per batch: estrae mean pooling dopo ogni layer.
    Ritorna lista di array (N, hidden_size), uno per layer.
    """
    n_layers = len(model.layers)
    all_reps = [[] for _ in range(n_layers)]

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        enc = tokenizer(batch_texts, truncation=True,
                        max_length=config.MAX_SEQ_LEN,
                        padding="max_length", return_tensors="pt")
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        B = input_ids.size(0)
        t = torch.zeros(B, device=DEVICE)

        with torch.no_grad():
            x = model.token_emb(input_ids)
            t_vec = model.time_emb(t.unsqueeze(1))
            x = x + t_vec.unsqueeze(1)
            mask = attention_mask.unsqueeze(-1).float()

            for layer_idx, layer in enumerate(model.layers):
                x = layer(x)
                rep = (x * mask).sum(dim=1) / mask.sum(dim=1)  # mean pooling
                all_reps[layer_idx].append(rep.cpu().numpy())

    return [np.vstack(reps) for reps in all_reps]


def run_probing_multilayer(ckpt_path, train_texts, test_texts,
                           feature_names, Y_train, Y_test, ckpt_out_dir):
    model = DiffuMamba().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

    print(f"  Encoding train...")
    X_train_layers = encode_all_layers(model, tokenizer, train_texts)
    print(f"  Encoding test...")
    X_test_layers  = encode_all_layers(model, tokenizer, test_texts)

    for layer_idx in range(len(model.layers)):
        out_dir = os.path.join(ckpt_out_dir, f"layer_{layer_idx}")
        os.makedirs(out_dir, exist_ok=True)

        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_train_layers[layer_idx])
        X_te = scaler.transform(X_test_layers[layer_idx])

        for i, feat_name in enumerate(feature_names):
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tr, Y_train[:, i])
            y_pred = ridge.predict(X_te)
            out_path = os.path.join(out_dir, f"{feat_name}.tsv")
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["y_pred", "y_true"])
                for pred, true in zip(y_pred, Y_test[:, i]):
                    writer.writerow([pred, true])

    del model
    torch.cuda.empty_cache()


# ── Main ─────────────────────────────────────────────────────────────────────
print("Caricamento dataset probing...")
train_texts, feature_names, Y_train = load_tsv(TRAIN_TSV)
test_texts,  _,             Y_test  = load_tsv(TEST_TSV)
print(f"Train: {len(train_texts)} | Test: {len(test_texts)} | Feature: {len(feature_names)}")

ckpt_files = []
for fname in os.listdir(CHECKPOINTS_DIR):
    m = re.match(r"ckpt_step(\d+)\.pt", fname)
    if m:
        ckpt_files.append((int(m.group(1)), os.path.join(CHECKPOINTS_DIR, fname)))
ckpt_files.sort(key=lambda x: x[0])
print(f"Checkpoint trovati: {len(ckpt_files)}")

for step, ckpt_path in ckpt_files:
    ckpt_out_dir = os.path.join(OUTPUT_BASE_DIR, f"checkpoint_{step:06d}")
    if os.path.exists(ckpt_out_dir) and len(os.listdir(ckpt_out_dir)) == config.NUM_LAYERS:
        print(f"[step {step:,}] già calcolato, skip.")
        continue
    print(f"\n[step {step:,}] Probing multilayer in corso...")
    run_probing_multilayer(ckpt_path, train_texts, test_texts,
                           feature_names, Y_train, Y_test, ckpt_out_dir)
    print(f"[step {step:,}] Done.")

print("\nDone. Esegui evaluate_multilayer.py per le metriche.")