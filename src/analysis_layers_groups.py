"""
analysis_layers_groups.py — Analisi per layer e per gruppi di feature (media sui seed)

Usa le feature della Table 1 di Miaschi et al. condivise da tutti i modelli
(stessa selezione di plot_groups.py). Per ogni modello e formato dei dati:
  1. profilo per layer all'ultimo checkpoint (media ± std sui seed)
  2. layer migliore per ogni gruppo di feature
  3. valore per gruppo al layer migliore del modello, differenze vs DiffuMamba
  4. dinamica: step in cui ogni gruppo raggiunge il 90% del valore finale

Output in ../results/analysis/ (CSV). Uso (dalla cartella src):
    python analysis_layers_groups.py
"""

import os
import numpy as np
import pandas as pd
from seeds import SEEDS
from runs import all_metrics_csv, bert_csvs
from plot_groups import load_run, build_group_map, group_avg_by_step_layer, GROUP_ORDER

OUT = "../results/analysis"
MODELS = {"DiffuMamba": "mamba", "DiffuTransformer": "transformer", "Mamba AR": "ar"}
REACH = 0.90


def load_all(data):
    """{nome: [df per seed]}; BERT con layer riportati a 0..7."""
    runs = {name: [load_run(all_metrics_csv(m, data, s))[0] for s in SEEDS]
            for name, m in MODELS.items()}
    runs["BERT"] = [load_run(p, layer_offset=-1)[0] for p in bert_csvs(data)]
    return runs


def shared_features(runs):
    feats = set.intersection(*[set(d.columns) - {"step", "layer", "avg_rho"}
                               for dfs in runs.values() for d in dfs])
    gmap = build_group_map(sorted(feats))
    return sorted(f for f in feats if f in gmap), gmap


def per_seed_groups(dfs, used, gmap):
    """DataFrame (seed, step, layer, gruppi..., ALL) per tutti i seed."""
    out = []
    for i, d in enumerate(dfs):
        g = group_avg_by_step_layer(d, used, gmap)
        g["ALL"] = d[used].mean(axis=1).values
        g.insert(0, "seed", i)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def analyse(data):
    runs = load_all(data)
    used, gmap = shared_features(runs)
    groups = [g for g in GROUP_ORDER if any(gmap[f] == g for f in used)]
    cols = groups + ["ALL"]
    G = {name: per_seed_groups(dfs, used, gmap) for name, dfs in runs.items()}

    rows_layer, rows_bestg, rows_best, rows_dyn = [], [], [], []
    for name, g in G.items():
        last = g[g["step"] == g["step"].max()]
        mean = last.groupby("layer")[cols].mean()
        std = last.groupby("layer")[cols].std(ddof=1)
        n = last["seed"].nunique()

        # 1. profilo per layer
        prof = mean["ALL"]
        best_l = int(prof.idxmax())
        rows_layer.append({"modello": name, "n_seed": n, "layer_migliore": best_l,
                           "rho_picco": prof.max(), "rho_L0": prof.iloc[0], "rho_L7": prof.iloc[-1],
                           "calo_picco_L7": prof.max() - prof.iloc[-1],
                           **{f"L{l}": prof.loc[l] for l in prof.index}})

        # 2. layer migliore per gruppo
        for c in groups:
            rows_bestg.append({"modello": name, "gruppo": c, "layer_migliore": int(mean[c].idxmax()),
                               "rho_al_layer_migliore": mean[c].max()})

        # 3. valore per gruppo al layer migliore del modello
        for c in cols:
            rows_best.append({"modello": name, "gruppo": c, "layer": best_l,
                              "media": mean.loc[best_l, c],
                              "std": std.loc[best_l, c] if n > 1 else np.nan})

        # 4. dinamica al layer migliore: primo step con valore >= 90% del finale (per seed, poi media)
        gl = g[g["layer"] == best_l]
        for c in cols:
            steps = []
            for s, gs in gl.groupby("seed"):
                gs = gs.sort_values("step")
                final = gs[c].iloc[-1]
                steps.append(gs.loc[gs[c] >= REACH * final, "step"].min())
            rows_dyn.append({"modello": name, "gruppo": c, f"step_{int(REACH*100)}pct": np.mean(steps),
                             "std": np.std(steps, ddof=1) if len(steps) > 1 else np.nan})

    best = pd.DataFrame(rows_best)
    ref = best[best["modello"] == "DiffuMamba"].set_index("gruppo")
    best["diff_vs_DiffuMamba"] = best["media"] - best["gruppo"].map(ref["media"])
    # dev. std combinata (solo modelli con seed) per giudicare se la differenza supera il rumore
    best["std_combinata"] = np.sqrt((best["std"] ** 2 + best["gruppo"].map(ref["std"]) ** 2) / 2)
    best["diff_su_std"] = best["diff_vs_DiffuMamba"] / best["std_combinata"]

    os.makedirs(OUT, exist_ok=True)
    for tab, df in [("profilo_layer", pd.DataFrame(rows_layer)), ("layer_migliore_per_gruppo", pd.DataFrame(rows_bestg)),
                    ("gruppi_al_layer_migliore", best), ("dinamica_gruppi", pd.DataFrame(rows_dyn))]:
        df.round(4).to_csv(f"{OUT}/{tab}_{data}.csv", index=False)
    return pd.DataFrame(rows_layer), pd.DataFrame(rows_bestg), best, pd.DataFrame(rows_dyn), len(used)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    for data in ("sent", "blocks"):
        layer, bestg, best, dyn, nf = analyse(data)
        print(f"\n################ {data} ({nf} feature) ################")
        print(layer.drop(columns=[f"L{i}" for i in range(8)]).round(4).to_string(index=False))
        print("\n-- layer migliore per gruppo")
        print(bestg.pivot(index="gruppo", columns="modello", values="layer_migliore").loc[[g for g in GROUP_ORDER if g in set(bestg.gruppo)]].to_string())
        print("\n-- gruppi al layer migliore del modello (media | diff vs DiffuMamba in unità di std)")
        m = best.pivot(index="gruppo", columns="modello", values="media")
        z = best.pivot(index="gruppo", columns="modello", values="diff_su_std")
        order = [g for g in GROUP_ORDER if g in m.index] + ["ALL"]
        print(pd.concat({"media": m.loc[order].round(3), "z": z.loc[order].round(1)}, axis=1).to_string())
        print("\n-- step al 90% del valore finale (layer migliore)")
        print(dyn.pivot(index="gruppo", columns="modello", values="step_90pct").loc[order].round(0).to_string())
    print(f"\nCSV salvati in {OUT}/")
