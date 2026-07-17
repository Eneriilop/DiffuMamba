"""
probing_all_checkpoints.py — Probing su tutti i checkpoint
============================================================
Per ogni checkpoint in CHECKPOINTS_DIR:
  1. Carica il modello
  2. Estrae le rappresentazioni dell'ultimo layer (mean pooling)
  3. Allena un Ridge Regressor per ogni feature
  4. Salva le predizioni grezze in una cartella separata per checkpoint

Struttura output:
  probing_results/
    checkpoint_025000/
      n_tokens.tsv
      upos_dist_NOUN.tsv
      ...
    checkpoint_050000/
      ...

Poi eseguiamo evaluate_all_checkpoints.py per calcolare Spearman ρ.

"""

import os
import re
import csv
import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

from model import DiffuMamba
import config

# ── Configurazione ───────────────────────────────────────────────────────────

CHECKPOINTS_DIR = "checkpoints_mamba_sentences"  # cartella con i checkpoint da valutare
TRAIN_TSV       = "data/train_probe.tsv"
TEST_TSV        = "data/test_probe.tsv"
OUTPUT_BASE_DIR = "probing_results_mamba_sent"
BATCH_SIZE      = 64
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Funzioni ─────────────────────────────────────────────────────────────────

def load_tsv(path):
    texts, features = [], []
    feature_names   = None
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        feature_names = [c for c in reader.fieldnames if c not in ("identifier", "text")]
        for row in reader:
            texts.append(row["text"])
            features.append([float(row[fn]) for fn in feature_names])
    return texts, feature_names, np.array(features, dtype=np.float32)


def encode_sentences(model, tokenizer, texts):
    """Mean pooling dell'ultimo layer per una lista di frasi."""
    all_reps = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        B = input_ids.size(0)
        t = torch.zeros(B, device=DEVICE)

        with torch.no_grad():
            x = model.token_emb(input_ids)
            t_vec = model.time_emb(t.unsqueeze(1))
            x = x + t_vec.unsqueeze(1)
            for layer in model.layers:
                x = layer(x)
            x = model.norm_out(x)
            mask = attention_mask.unsqueeze(-1).float()
            rep  = (x * mask).sum(dim=1) / mask.sum(dim=1)

        all_reps.append(rep.cpu().numpy())
    return np.vstack(all_reps)


def run_probing(checkpoint_path, train_texts, test_texts,
                feature_names, Y_train, Y_test, output_dir):
    """Carica il checkpoint, estrae rappresentazioni e salva predizioni."""
    os.makedirs(output_dir, exist_ok=True)

    # Carica modello
    model = DiffuMamba().to(DEVICE)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

    print(f"  Encoding train...")
    X_train = encode_sentences(model, tokenizer, train_texts)
    print(f"  Encoding test...")
    X_test  = encode_sentences(model, tokenizer, test_texts)

    # MinMaxScaler su X
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Ridge per ogni feature
    for i, feat_name in enumerate(feature_names):
        y_train = Y_train[:, i]
        y_test  = Y_test[:, i]

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)

        out_path = os.path.join(output_dir, f"{feat_name}.tsv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["y_pred", "y_true"])
            for pred, true in zip(y_pred, y_test):
                writer.writerow([pred, true])

    # Libera memoria GPU
    del model
    torch.cuda.empty_cache()


# ── Main ─────────────────────────────────────────────────────────────────────

print("Caricamento dataset probing...")
train_texts, feature_names, Y_train = load_tsv(TRAIN_TSV)
test_texts,  _,             Y_test  = load_tsv(TEST_TSV)
print(f"Train: {len(train_texts)} | Test: {len(test_texts)} | Feature: {len(feature_names)}")

# Trova tutti i checkpoint e ordinali per step
ckpt_files = []
for fname in os.listdir(CHECKPOINTS_DIR):
    m = re.match(r"ckpt_step(\d+)\.pt", fname)
    if m:
        step = int(m.group(1))
        ckpt_files.append((step, os.path.join(CHECKPOINTS_DIR, fname)))

ckpt_files.sort(key=lambda x: x[0])
print(f"\nCheckpoint trovati: {len(ckpt_files)}")
for step, path in ckpt_files:
    print(f"  step {step:>7,} → {path}")

# Probing per ogni checkpoint
for step, ckpt_path in ckpt_files:
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"checkpoint_{step:06d}")

    # Salta se già calcolato
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) == len(feature_names):
        print(f"\n[step {step:,}] già calcolato, skip.")
        continue

    print(f"\n[step {step:,}] Probing in corso...")
    run_probing(ckpt_path, train_texts, test_texts,
                feature_names, Y_train, Y_test, output_dir)
    print(f"[step {step:,}] Predizioni salvate in: {output_dir}")

print("\nDone. Esegui evaluate_all_checkpoints.py per le metriche.")