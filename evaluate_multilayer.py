"""
evaluate_multilayer.py — Calcola Spearman ρ per layer e checkpoint
"""

import os, re, csv
import numpy as np
from scipy.stats import spearmanr

OUTPUT_BASE_DIR = "probing_results_multilayer"
OUT_CURVE       = "learning_curve_multilayer.csv"   # avg ρ per step × layer
OUT_ALL         = "all_metrics_multilayer.csv"       # ρ per feature × layer × step


def evaluate_layer(layer_dir):
    """Calcola Spearman ρ per ogni feature in una cartella layer."""
    results = {}
    for fname in os.listdir(layer_dir):
        if not fname.endswith(".tsv"):
            continue
        feat_name = fname.replace(".tsv", "")
        y_pred, y_true = [], []
        with open(os.path.join(layer_dir, fname), encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                y_pred.append(float(row["y_pred"]))
                y_true.append(float(row["y_true"]))
        rho, _ = spearmanr(y_true, y_pred)
        results[feat_name] = round(rho, 4)
    return results


# Trova tutti i checkpoint
ckpt_dirs = []
for dname in os.listdir(OUTPUT_BASE_DIR):
    m = re.match(r"checkpoint_(\d+)", dname)
    if m:
        ckpt_dirs.append((int(m.group(1)), os.path.join(OUTPUT_BASE_DIR, dname)))
ckpt_dirs.sort(key=lambda x: x[0])
print(f"Checkpoint trovati: {len(ckpt_dirs)}")

# Trova numero di layer e feature names dal primo checkpoint
first_ckpt = ckpt_dirs[0][1]
n_layers = len([d for d in os.listdir(first_ckpt) if d.startswith("layer_")])
feature_names = sorted([
    f.replace(".tsv", "")
    for f in os.listdir(os.path.join(first_ckpt, "layer_0"))
    if f.endswith(".tsv")
])
print(f"Layer: {n_layers} | Feature: {len(feature_names)}")

# Calcola metriche
curve_rows = []   # per learning_curve_multilayer.csv
all_rows   = []   # per all_metrics_multilayer.csv

for step, ckpt_dir in ckpt_dirs:
    print(f"  Step {step:,}...")
    curve_row = {"step": step}
    for layer_idx in range(n_layers):
        layer_dir = os.path.join(ckpt_dir, f"layer_{layer_idx}")
        metrics   = evaluate_layer(layer_dir)
        avg_rho   = round(np.mean(list(metrics.values())), 4)
        curve_row[f"layer_{layer_idx}"] = avg_rho

        all_rows.append({
            "step":    step,
            "layer":   layer_idx,
            "avg_rho": avg_rho,
            **{f: metrics.get(f, float("nan")) for f in feature_names}
        })

    curve_rows.append(curve_row)

# Salva learning curve per layer
layer_cols = [f"layer_{i}" for i in range(n_layers)]
with open(OUT_CURVE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["step"] + layer_cols)
    writer.writeheader()
    writer.writerows(curve_rows)

# Salva all metrics
with open(OUT_ALL, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "layer", "avg_rho"] + feature_names)
    writer.writeheader()
    writer.writerows(all_rows)

# Stampa riepilogo
print("\n" + "="*60)
print("AVG ρ per layer all'ultimo checkpoint")
print("="*60)
last = curve_rows[-1]
for i in range(n_layers):
    print(f"  Layer {i}: {last[f'layer_{i}']:.4f}")

print(f"\nSalvato: {OUT_CURVE}")
print(f"Salvato: {OUT_ALL}")