"""
evaluate_all_checkpoints.py — Calcola Spearman ρ per tutti i checkpoint
========================================================================
Legge le predizioni grezze prodotte da probing_all_checkpoints.py
e calcola Spearman ρ per ogni feature e ogni checkpoint.

Output:
  probing_results/learning_curve.csv   — ρ medio per checkpoint (per il grafico)
  probing_results/all_metrics.csv      — ρ per ogni feature × checkpoint

Uso:
  python3 evaluate_all_checkpoints.py
"""

import os
import re
import csv
import numpy as np
from scipy.stats import spearmanr

OUTPUT_BASE_DIR  = "probing_results"
LEARNING_CURVE   = os.path.join(OUTPUT_BASE_DIR, "learning_curve.csv")
ALL_METRICS      = os.path.join(OUTPUT_BASE_DIR, "all_metrics.csv")


def evaluate_checkpoint(ckpt_dir):
    """Calcola Spearman ρ per tutte le feature in una cartella checkpoint."""
    results = {}
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".tsv"):
            continue
        feat_name = fname.replace(".tsv", "")
        y_pred, y_true = [], []
        with open(os.path.join(ckpt_dir, fname), encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                y_pred.append(float(row["y_pred"]))
                y_true.append(float(row["y_true"]))
        rho, _ = spearmanr(y_true, y_pred)
        results[feat_name] = round(rho, 4)
    return results


# Trova tutte le cartelle checkpoint
ckpt_dirs = []
for dname in os.listdir(OUTPUT_BASE_DIR):
    m = re.match(r"checkpoint_(\d+)", dname)
    if m:
        step = int(m.group(1))
        ckpt_dirs.append((step, os.path.join(OUTPUT_BASE_DIR, dname)))

ckpt_dirs.sort(key=lambda x: x[0])
print(f"Checkpoint trovati: {len(ckpt_dirs)}")

if not ckpt_dirs:
    print("Nessun checkpoint trovato. Esegui prima probing_all_checkpoints.py")
    exit()

# Calcola metriche per ogni checkpoint
all_results = []
feature_names = None

for step, ckpt_dir in ckpt_dirs:
    print(f"  Valutazione step {step:,}...")
    metrics = evaluate_checkpoint(ckpt_dir)
    if feature_names is None:
        feature_names = sorted(metrics.keys())
    avg_rho = np.mean(list(metrics.values()))
    all_results.append({"step": step, "avg_rho": round(avg_rho, 4), **metrics})
    print(f"    Media ρ = {avg_rho:.4f}")

# Salva learning curve (step → ρ medio)
with open(LEARNING_CURVE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["step", "avg_rho"])
    writer.writeheader()
    for r in all_results:
        writer.writerow({"step": r["step"], "avg_rho": r["avg_rho"]})

# Salva tutte le metriche (step × feature)
with open(ALL_METRICS, "w", newline="", encoding="utf-8") as f:
    fieldnames = ["step", "avg_rho"] + feature_names
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_results)

# Stampa learning curve
print("\n" + "="*50)
print("LEARNING CURVE — Spearman ρ medio per checkpoint")
print("="*50)
print(f"{'Step':>10}  {'Media ρ':>10}")
print("-"*25)
for r in all_results:
    print(f"{r['step']:>10,}  {r['avg_rho']:>10.4f}")

print(f"\nSalvato: {LEARNING_CURVE}")
print(f"Salvato: {ALL_METRICS}")