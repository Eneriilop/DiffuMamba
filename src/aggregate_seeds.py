"""
aggregate_seeds.py — Media e deviazione standard dei risultati di probing sui seed

Per ogni modello (sent, sent_transformer, blocks, blocks_transformer; per
DiffuMamba anche la versione senza LN) legge i CSV dei singoli seed prodotti
da evaluate_multilayer.py e scrive:

  results/learning_curves/learning_curve_<nome>_mean.csv
      step, n_seeds, layer_<i>_mean, layer_<i>_std
  results/all_metrics/all_metrics_<nome>_mean.csv
      step, layer, n_seeds, avg_rho_mean, avg_rho_std, <feature>_mean, <feature>_std
  results/best_layer_last_checkpoint.csv
      per ogni modello e seed: layer migliore e ρ media all'ultimo checkpoint

Deviazione standard campionaria (ddof=1). Si usano solo gli step presenti
in tutti i seed disponibili; i seed mancanti vengono segnalati.

Uso (dalla cartella src): python aggregate_seeds.py
"""

import os
import pandas as pd
from seeds import SEEDS
from runs import base_name, learning_curve_csv, all_metrics_csv

# (model, data, ln)
VARIANTS = [
    ("mamba", "sent", True), ("mamba", "sent", False),
    ("transformer", "sent", True),
    ("ar", "sent", True),
    ("mamba", "blocks", True), ("mamba", "blocks", False),
    ("transformer", "blocks", True),
    ("ar", "blocks", True),
]


def mean_name(model, data, ln):
    return base_name(model, data) + ("" if ln else "_noln") + "_mean"


def load_seeds(path_fn, model, data, ln):
    """{seed: DataFrame} per i seed il cui CSV esiste."""
    out = {}
    for s in SEEDS:
        p = path_fn(model, data, s, ln=ln)
        if os.path.exists(p):
            out[s] = pd.read_csv(p)
    return out


def common_steps(dfs):
    steps = None
    for df in dfs.values():
        st = set(df["step"])
        steps = st if steps is None else steps & st
    return sorted(steps)


def aggregate(dfs, keys):
    """Media e dev. std per chiave (step o step+layer) su tutti i seed."""
    steps = common_steps(dfs)
    allv = pd.concat([df[df["step"].isin(steps)] for df in dfs.values()], ignore_index=True)
    value_cols = [c for c in allv.columns if c not in keys]
    g = allv.groupby(keys)[value_cols]
    mean, std = g.mean().add_suffix("_mean"), g.std(ddof=1).add_suffix("_std")
    out = pd.concat([mean, std], axis=1)
    # colonne alternate: x_mean, x_std
    out = out[[c for v in value_cols for c in (f"{v}_mean", f"{v}_std")]]
    out.insert(0, "n_seeds", len(dfs))
    return out.reset_index()


summary = []
for model, data, ln in VARIANTS:
    name = mean_name(model, data, ln)
    curves = load_seeds(learning_curve_csv, model, data, ln)
    metrics = load_seeds(all_metrics_csv, model, data, ln)
    if not curves:
        print(f"[{name}] nessun seed disponibile, skip.")
        continue
    missing = [s for s in SEEDS if s not in curves]
    print(f"[{name}] seed disponibili: {sorted(curves)}" + (f" | MANCANTI: {missing}" if missing else ""))

    out_curve = f"../results/learning_curves/learning_curve_{name}.csv"
    aggregate(curves, ["step"]).to_csv(out_curve, index=False)
    out_all = f"../results/all_metrics/all_metrics_{name}.csv"
    aggregate(metrics, ["step", "layer"]).to_csv(out_all, index=False)
    print(f"  Salvato: {out_curve}\n  Salvato: {out_all}")

    # layer migliore all'ultimo checkpoint, per seed
    for s, df in curves.items():
        last = df.iloc[df["step"].idxmax()]
        layer_cols = [c for c in df.columns if c.startswith("layer_")]
        best = max(layer_cols, key=lambda c: last[c])
        summary.append({"modello": base_name(model, data) + ("" if ln else "_noln"), "seed": s,
                        "step": int(last["step"]), "layer_migliore": int(best.split("_")[1]),
                        "rho_migliore": round(float(last[best]), 4)})

if summary:
    out = "../results/best_layer_last_checkpoint.csv"
    pd.DataFrame(summary).to_csv(out, index=False)
    print(f"\nSalvato: {out}")
    print(pd.DataFrame(summary).to_string(index=False))
