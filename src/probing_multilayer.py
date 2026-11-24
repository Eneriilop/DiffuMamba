"""
probing_multilayer.py — Probing su tutti i layer per ogni checkpoint

Uso (dalla cartella src):
    python probing_multilayer.py --model mamba --data sent --seed 0             → probing_results/sent_seed0
    python probing_multilayer.py --model mamba --data sent --seed 0 --no-ln     → probing_results/sent_seed0_noln
    python probing_multilayer.py --model transformer --data blocks --seed 1     → probing_results/blocks_transformer_seed1

Di default applica una LayerNorm all'output di ogni blocco prima del mean
pooling, come gli hidden_states post-LN di BERT (il residual stream che passa
al blocco successivo resta invariato). --no-ln estrae gli hidden state grezzi.

Per DiffuTransformer l'input include il positional embedding e i layer
ricevono l'attention mask, esattamente come in training.
Per Mamba AR (--model ar) non c'è timestep: il modello è causale e il
padding a destra non influenza i token reali.

Il probe (MinMaxScaler + Ridge) è deterministico: non usa seed. Il seed
indica solo quale modello addestrato leggere.

Un checkpoint già completato (file DONE nella sua cartella) viene saltato,
quindi lo script si può rilanciare dopo un'interruzione.
"""

import os, re, csv, argparse, torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from runs import get_config, checkpoints_dir, probing_dir

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["mamba", "transformer", "ar"], required=True)
parser.add_argument("--data", choices=["sent", "blocks"], required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--no-ln", action="store_true", help="niente LayerNorm sugli hidden state")
args = parser.parse_args()

MODEL    = args.model
APPLY_LN = not args.no_ln

config = get_config(MODEL, args.data)
if MODEL == "mamba":
    from model import DiffuMamba
elif MODEL == "ar":
    from model import MambaAR
else:
    from model_transformer import DiffuTransformer

# Troncatura delle frasi di probing, uguale per tutti i modelli (anche per
# quelli addestrati su blocchi da 512): stesse frasi, stesso taglio.
PROBE_MAX_LEN   = 128

CHECKPOINTS_DIR = checkpoints_dir(MODEL, args.data, args.seed)
OUTPUT_BASE_DIR = probing_dir(MODEL, args.data, args.seed, ln=APPLY_LN)
TRAIN_TSV       = "../data/train_probe.tsv"
TEST_TSV        = "../data/test_probe.tsv"
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
                        max_length=PROBE_MAX_LEN,
                        padding="max_length", return_tensors="pt")
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)
        B = input_ids.size(0)
        t = torch.zeros(B, device=DEVICE)

        with torch.no_grad():
            if MODEL == "transformer":
                x = model.embed(input_ids, t)   # token + posizione + timestep
            elif MODEL == "ar":
                x = model.token_emb(input_ids)  # Mamba AR: nessun timestep
            else:
                x = model.token_emb(input_ids)
                t_vec = model.time_emb(t.unsqueeze(1))
                x = x + t_vec.unsqueeze(1)
            mask = attention_mask.unsqueeze(-1).float()

            for layer_idx, layer in enumerate(model.layers):
                if MODEL == "transformer":
                    x = layer(x, attention_mask)   # esclude il padding dall'attenzione
                else:
                    x = layer(x)
                h = F.layer_norm(x, x.shape[-1:]) if APPLY_LN else x  # solo per il probing
                rep = (h * mask).sum(dim=1) / mask.sum(dim=1)  # mean pooling
                all_reps[layer_idx].append(rep.cpu().numpy())

    return [np.vstack(reps) for reps in all_reps]


def run_probing_multilayer(ckpt_path, train_texts, test_texts,
                           feature_names, Y_train, Y_test, ckpt_out_dir):
    if MODEL == "transformer":
        model = DiffuTransformer(config)
    elif MODEL == "ar":
        model = MambaAR(config)
    else:
        model = DiffuMamba()
    model = model.to(DEVICE)
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

    # segna il checkpoint come completo (tutti i layer e tutte le feature scritti)
    open(os.path.join(ckpt_out_dir, "DONE"), "w").close()

    del model
    torch.cuda.empty_cache()


# ── Main ─────────────────────────────────────────────────────────────────────
print(f"Modello: {MODEL} | dati: {args.data} | seed: {args.seed} | LayerNorm sui layer: {APPLY_LN}")
print(f"Checkpoint: {CHECKPOINTS_DIR} | Output: {OUTPUT_BASE_DIR}")
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
    if os.path.exists(os.path.join(ckpt_out_dir, "DONE")):
        print(f"[step {step:,}] già calcolato, skip.")
        continue
    print(f"\n[step {step:,}] Probing multilayer in corso...")
    run_probing_multilayer(ckpt_path, train_texts, test_texts,
                           feature_names, Y_train, Y_test, ckpt_out_dir)
    print(f"[step {step:,}] Done.")

print("\nDone. Esegui evaluate_multilayer.py per le metriche.")
