"""
evaluate_probing.py — Calcola metriche dalle predizioni grezze
==============================================================
Legge i file TSV prodotti da probing.py e calcola Spearman ρ
per ogni feature linguistica (metrica usata in Miaschi et al. 2020).

Uso:
  python3 evaluate_probing.py

Output:
  probing_results/last_layer/metrics.csv
"""

import os
import csv
import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = "probing_results/last_layer"
OUTPUT_CSV  = os.path.join(RESULTS_DIR, "metrics.csv")

results = []

for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith(".tsv"):
        continue

    feat_name = fname.replace(".tsv", "")
    y_pred, y_true = [], []

    with open(os.path.join(RESULTS_DIR, fname), encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            y_pred.append(float(row["y_pred"]))
            y_true.append(float(row["y_true"]))

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    spearman_rho, spearman_p = spearmanr(y_true, y_pred)

    results.append({
        "feature":     feat_name,
        "spearman_rho": round(spearman_rho, 4),
        "spearman_p":   round(spearman_p, 6),
    })

# Ordina per Spearman ρ decrescente
results.sort(key=lambda x: x["spearman_rho"], reverse=True)

# Stampa
print("="*60)
print("RISULTATI PROBING — DiffuMamba (ultimo layer, mean pooling)")
print("="*60)
print(f"{'Feature':<40} {'Spearman ρ':>12}")
print("-"*60)
for r in results:
    print(f"{r['feature']:<40} {r['spearman_rho']:>12.4f}")

avg_rho = np.mean([r["spearman_rho"] for r in results])
print("-"*60)
print(f"{'Media ρ':<40} {avg_rho:>12.4f}")

# Salva CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["feature", "spearman_rho", "spearman_p"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nMetriche salvate in: {OUTPUT_CSV}")
