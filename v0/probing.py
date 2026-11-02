"""
probing.py — Estrazione rappresentazioni e salvataggio predizioni grezze
=========================================================================
Metodologia (da Brunato et al., ACL 2020):
  - Rappresentazione frase: mean pooling dei token non-padding dell'ultimo layer
    (sostituto del token [CLS] assente in Mamba)
  - Scaler: MinMaxScaler sulle rappresentazioni X
  - Modello di probing: Ridge Regressor
  - Output: predizioni grezze per ogni feature salvate in TSV separati


Struttura output:
  probing_results/
    last_layer/
      n_tokens.tsv
      upos_dist_NOUN.tsv
      ...  (un file per ogni feature)

Ogni TSV ha due colonne: y_pred, y_true

Poi eseguiamo evaluate_probing.py per calcolare R² e Pearson r.
"""

import os
import csv
import torch
import numpy as np
from transformers import AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

from model import DiffuMamba
import config

# ── Configurazione ───────────────────────────────────────────────────────────

CHECKPOINT   = "checkpoints/ckpt_step208000.pt"
TRAIN_TSV    = "data/train_probe.tsv"
TEST_TSV     = "data/test_probe.tsv"
OUTPUT_DIR   = "probing_results/last_layer"
BATCH_SIZE   = 64
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Carica modello ────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
print("Caricamento modello...")

tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
model     = DiffuMamba().to(DEVICE)
ckpt      = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"Modello caricato: {model.count_params():,} parametri")


# ── 2. Funzione di encoding ──────────────────────────────────────────────────

def encode_sentences(texts):
    """
    Estrae le rappresentazioni dell'ultimo layer per una lista di frasi.
    Usa mean pooling sui token non-padding (sostituto di [CLS]).
    Ritorna: np.array di shape (N, hidden_size)
    """
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

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  Encoding: {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    return np.vstack(all_reps)


# ── 3. Carica TSV ────────────────────────────────────────────────────────────

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


print("\nCaricamento dataset...")
train_texts, feature_names, Y_train = load_tsv(TRAIN_TSV)
test_texts,  _,             Y_test  = load_tsv(TEST_TSV)
print(f"Train: {len(train_texts)} frasi | Test: {len(test_texts)} frasi")
print(f"Feature linguistiche: {len(feature_names)}")


# ── 4. Estrai rappresentazioni ───────────────────────────────────────────────

print("\nEncoding train...")
X_train = encode_sentences(train_texts)
print("Encoding test...")
X_test  = encode_sentences(test_texts)
print(f"Shape rappresentazioni: train={X_train.shape}, test={X_test.shape}")

# MinMaxScaler sulle rappresentazioni X
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ── 5. Probing: Ridge Regressor + salvataggio predizioni grezze ──────────────

print("\nTraining probe e salvataggio predizioni...")

for i, feat_name in enumerate(feature_names):
    y_train = Y_train[:, i]
    y_test  = Y_test[:, i]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)

    # Salva y_pred e y_true in TSV
    out_path = os.path.join(OUTPUT_DIR, f"{feat_name}.tsv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["y_pred", "y_true"])
        for pred, true in zip(y_pred, y_test):
            writer.writerow([pred, true])

    if i % 10 == 0:
        print(f"  [{i+1}/{len(feature_names)}] {feat_name} → salvato")

print(f"\nPredizioni salvate in: {OUTPUT_DIR}/")