"""
representation_similarity.py — Similarità coseno media tra rappresentazioni di frasi

Per ogni checkpoint (fino a MAX_STEP) e ogni layer calcola la similarità coseno
media tra le rappresentazioni di frasi diverse, estratte come nel probing
(t = 0, LayerNorm, mean pooling sui token reali).

Valori vicini a 1 = rappresentazioni quasi uguali per tutte le frasi
(collasso), quindi poca informazione specifica della frase per il probe.

Output: ../results/similarity_<run>.csv  (step, layer, mean_cos)
"""

import os, re, csv, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# (run, modello, config)
RUNS = [
    ("run5_transformer", "transformer", "config_transformer"),
    ("run5",             "mamba",       "config"),
    ("run4_transformer", "transformer", "config_transformer_sentences"),
]
TEST_TSV      = "../data/test_probe.tsv"
N_SENTENCES   = 1000     # frasi del test set di probing usate
PROBE_MAX_LEN = 128      # come nel probing
MAX_STEP      = 20_000
BATCH_SIZE    = 64
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_texts(path, n):
    with open(path, encoding="utf-8") as f:
        return [row["text"] for _, row in zip(range(n), csv.DictReader(f, delimiter="\t"))]


def build_model(kind, cfg_name):
    cfg = __import__(cfg_name)
    if kind == "transformer":
        from model_transformer import DiffuTransformer
        return DiffuTransformer(cfg), cfg
    from model import DiffuMamba
    return DiffuMamba(), cfg


@torch.no_grad()
def encode_all_layers(model, kind, tokenizer, texts):
    """Come probing_multilayer.encode_all_layers (con APPLY_LN = True)."""
    reps = [[] for _ in model.layers]
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tokenizer(texts[i:i + BATCH_SIZE], truncation=True, max_length=PROBE_MAX_LEN,
                        padding="max_length", return_tensors="pt")
        ids, am = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        t = torch.zeros(ids.size(0), device=DEVICE)
        if kind == "transformer":
            x = model.embed(ids, t)
        else:
            x = model.token_emb(ids) + model.time_emb(t.unsqueeze(1)).unsqueeze(1)
        mask = am.unsqueeze(-1).float()
        for li, layer in enumerate(model.layers):
            x = layer(x, am) if kind == "transformer" else layer(x)
            h = F.layer_norm(x, x.shape[-1:])
            reps[li].append(((h * mask).sum(1) / mask.sum(1)).float())
    return [torch.cat(r) for r in reps]


def mean_pairwise_cos(X):
    Xn = F.normalize(X, dim=1)
    S = Xn @ Xn.T
    n = S.size(0)
    return ((S.sum() - S.diagonal().sum()) / (n * (n - 1))).item()


def main(only=None):
    texts = load_texts(TEST_TSV, N_SENTENCES)
    for run, kind, cfg_name in RUNS:
        if only and run not in only:
            continue
        model, cfg = build_model(kind, cfg_name)
        model = model.to(DEVICE).eval()
        tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)

        ckpt_dir = f"../checkpoints/{run}"
        ckpts = sorted((int(m.group(1)), os.path.join(ckpt_dir, f))
                       for f in os.listdir(ckpt_dir)
                       if (m := re.match(r"ckpt_step(\d+)\.pt", f)))
        rows = []
        for step, path in ckpts:
            if step > MAX_STEP:
                continue
            model.load_state_dict(torch.load(path, map_location=DEVICE)["model"])
            sims = [mean_pairwise_cos(X) for X in encode_all_layers(model, kind, tokenizer, texts)]
            rows += [{"step": step, "layer": li, "mean_cos": round(s, 4)} for li, s in enumerate(sims)]
            print(f"{run} step {step:>6}: " + " ".join(f"{s:.3f}" for s in sims), flush=True)

        out = f"../results/similarity_{run}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["step", "layer", "mean_cos"])
            w.writeheader()
            w.writerows(rows)
        print(f"Salvato: {out}\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or None)
