# ============================================================
# plot_comparison.py — Confronto multilayer tra modelli
#
# Per ogni modello legge l'ultimo checkpoint di ciascun seed e
# disegna la ρ media per layer (media sui seed) con una banda di
# ±1 deviazione standard (BERT: 3 seed, 42/43/44, forniti da terzi).
#
# Uso:
#   python plot_comparison.py              → tutti i confronti
#   python plot_comparison.py <name> ...   → solo i confronti indicati
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from seeds import SEEDS
from runs import learning_curve_csv, bert_csvs

# ============================================================
# CONFIGURAZIONE
# ============================================================
# Convenzione: "DiffuMamba" e "DiffuTransformer" = probing con LayerNorm sugli
# hidden states, per un confronto corretto con gli hidden_states post-LN di
# BERT. La versione senza LN compare solo nel confronto con/senza LN.


def seed_csvs(model, data, ln=True):
    """CSV learning_curve di tutti i seed di un modello."""
    return [learning_curve_csv(model, data, s, ln=ln) for s in SEEDS]


MAMBA_FRASI_NOLN = {
    "label":    "DiffuMamba senza LN — frasi (128 tok)",
    "csvs":     seed_csvs("mamba", "sent", ln=False),
    "color":    "#2196F3",
    "linestyle": "-",
}
MAMBA_FRASI = {
    "label":    "DiffuMamba — frasi (128 tok)",
    "csvs":     seed_csvs("mamba", "sent"),
    "color":    "#3F51B5",
    "linestyle": ":",
}
MAMBA_FRASI_CON_LN = {**MAMBA_FRASI, "label": "DiffuMamba con LN — frasi (128 tok)"}
TRANSFORMER_FRASI = {
    "label":    "DiffuTransformer — frasi (128 tok)",
    "csvs":     seed_csvs("transformer", "sent"),
    "color":    "#9C27B0",
    "linestyle": "--",
}
AR_FRASI = {
    "label":    "Mamba AR — frasi (128 tok)",
    "csvs":     seed_csvs("ar", "sent"),
    "color":    "#00897B",
    "linestyle": "-",
}
BERT_FRASI = {
    "label":     "BERT — frasi",
    "csvs":      bert_csvs("sent"),        # formato lungo, un file per seed
    "color":     "#4CAF50",
    "linestyle": "-.",
}
MAMBA_BLOCCHI_NOLN = {
    "label":    "DiffuMamba senza LN — blocchi (512 tok)",
    "csvs":     seed_csvs("mamba", "blocks", ln=False),
    "color":    "#FF5722",
    "linestyle": "-",
}
MAMBA_BLOCCHI = {
    "label":    "DiffuMamba — blocchi (512 tok)",
    "csvs":     seed_csvs("mamba", "blocks"),
    "color":    "#BF360C",
    "linestyle": ":",
}
MAMBA_BLOCCHI_CON_LN = {**MAMBA_BLOCCHI, "label": "DiffuMamba con LN — blocchi (512 tok)"}
TRANSFORMER_BLOCCHI = {
    "label":    "DiffuTransformer — blocchi (512 tok)",
    "csvs":     seed_csvs("transformer", "blocks"),
    "color":    "#6A1B9A",
    "linestyle": "--",
}
AR_BLOCCHI = {
    "label":    "Mamba AR — blocchi (512 tok)",
    "csvs":     seed_csvs("ar", "blocks"),
    "color":    "#004D40",
    "linestyle": "-",
}
BERT_BLOCCHI = {
    "label":     "BERT — blocchi (512 tok)",
    "csvs":      bert_csvs("blocks"),      # formato lungo, un file per seed
    "color":     "#2E7D32",
    "linestyle": "-.",
}

OUT_SENT   = "../results/comparisons/sentences/per_layer"
OUT_BLOCKS = "../results/comparisons/blocks/per_layer"

COMPARISONS = [
    # --- frasi ---
    {
        "name":   "bert_mamba",
        "runs":   [MAMBA_FRASI, BERT_FRASI],
        "output": f"{OUT_SENT}/comparison_bert_mamba_sentences.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs BERT (frasi)",
    },
    {
        "name":   "bert_mamba_ln_vs_noln",
        "runs":   [MAMBA_FRASI_NOLN, MAMBA_FRASI_CON_LN, BERT_FRASI],
        "output": f"{OUT_SENT}/comparison_bert_mamba_ln_vs_noln_sentences.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba con/senza LN vs BERT (frasi)",
    },
    {
        "name":   "mamba_transformer",
        "runs":   [MAMBA_FRASI, TRANSFORMER_FRASI],
        "output": f"{OUT_SENT}/comparison_mamba_transformer_sentences.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs DiffuTransformer (frasi)",
    },
    {
        "name":   "bert_mamba_transformer",
        "runs":   [MAMBA_FRASI, TRANSFORMER_FRASI, BERT_FRASI],
        "output": f"{OUT_SENT}/comparison_bert_mamba_transformer_sentences.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs DiffuTransformer vs BERT (frasi)",
    },
    {
        "name":   "mamba_ar",
        "runs":   [MAMBA_FRASI, AR_FRASI],
        "output": f"{OUT_SENT}/comparison_mamba_ar_sentences.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs Mamba AR (frasi)",
    },
    {
        "name":   "mamba_ar_transformer",
        "runs":   [MAMBA_FRASI, AR_FRASI, TRANSFORMER_FRASI],
        "output": f"{OUT_SENT}/comparison_mamba_ar_transformer_sentences.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs Mamba AR vs DiffuTransformer (frasi)",
    },
    {
        "name":   "bert_mamba_ar",
        "runs":   [MAMBA_FRASI, AR_FRASI, BERT_FRASI],
        "output": f"{OUT_SENT}/comparison_bert_mamba_ar_sentences.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs Mamba AR vs BERT (frasi)",
    },
    # --- blocchi ---
    {
        "name":   "bert_mamba_blocks",
        "runs":   [MAMBA_BLOCCHI, BERT_BLOCCHI],
        "output": f"{OUT_BLOCKS}/comparison_bert_mamba_blocks.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs BERT (blocchi)",
    },
    {
        "name":   "bert_mamba_ln_vs_noln_blocks",
        "runs":   [MAMBA_BLOCCHI_NOLN, MAMBA_BLOCCHI_CON_LN, BERT_BLOCCHI],
        "output": f"{OUT_BLOCKS}/comparison_bert_mamba_ln_vs_noln_blocks.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba con/senza LN vs BERT (blocchi)",
    },
    {
        "name":   "mamba_transformer_blocks",
        "runs":   [MAMBA_BLOCCHI, TRANSFORMER_BLOCCHI],
        "output": f"{OUT_BLOCKS}/comparison_mamba_transformer_blocks.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs DiffuTransformer (blocchi)",
    },
    {
        "name":   "bert_mamba_transformer_blocks",
        "runs":   [MAMBA_BLOCCHI, TRANSFORMER_BLOCCHI, BERT_BLOCCHI],
        "output": f"{OUT_BLOCKS}/comparison_bert_mamba_transformer_blocks.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs DiffuTransformer vs BERT (blocchi)",
    },
    {
        "name":   "mamba_ar_blocks",
        "runs":   [MAMBA_BLOCCHI, AR_BLOCCHI],
        "output": f"{OUT_BLOCKS}/comparison_mamba_ar_blocks.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs Mamba AR (blocchi)",
    },
    {
        "name":   "mamba_ar_transformer_blocks",
        "runs":   [MAMBA_BLOCCHI, AR_BLOCCHI, TRANSFORMER_BLOCCHI],
        "output": f"{OUT_BLOCKS}/comparison_mamba_ar_transformer_blocks.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs Mamba AR vs DiffuTransformer (blocchi)",
    },
    {
        "name":   "bert_mamba_ar_blocks",
        "runs":   [MAMBA_BLOCCHI, AR_BLOCCHI, BERT_BLOCCHI],
        "output": f"{OUT_BLOCKS}/comparison_bert_mamba_ar_blocks.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs Mamba AR vs BERT (blocchi)",
    },
]

# ============================================================
# CARICAMENTO
# ============================================================

def load_curve(csv_path):
    """
    Restituisce un DataFrame con colonne step, layer_0..layer_N ordinato per step.
    Accetta il formato learning_curve (step, layer_*) e il formato lungo
    del probing BERT (checkpoint, layer, feature, score): in quel caso
    calcola il ρ medio su tutte le feature per checkpoint × layer.
    """
    df = pd.read_csv(csv_path)
    if {"checkpoint", "layer", "feature", "score"} <= set(df.columns):
        df = df.pivot_table(index="checkpoint", columns="layer",
                            values="score", aggfunc="mean")
        df.columns = [f"layer_{c}" for c in df.columns]
        df = df.reset_index().rename(columns={"checkpoint": "step"})
    return df.sort_values("step").reset_index(drop=True)


def last_checkpoint_per_layer(run):
    """
    ρ per layer all'ultimo checkpoint, per ogni seed disponibile.
    Ritorna (matrice seed × layer, step, n_seed) oppure None se manca tutto.
    """
    found = [p for p in run["csvs"] if os.path.exists(p)]
    if not found:
        return None
    if len(found) < len(run["csvs"]):
        print(f"[WARN] {run['label']}: {len(found)}/{len(run['csvs'])} seed disponibili")
    rows, steps = [], set()
    for p in found:
        df = load_curve(p)
        last = df.iloc[-1]
        steps.add(int(last["step"]))
        rows.append([last[c] for c in df.columns if c.startswith("layer_")])
    if len(steps) > 1:
        print(f"[WARN] {run['label']}: ultimo step diverso tra i seed {sorted(steps)}")
    return np.array(rows), max(steps), len(found)

# ============================================================
# PLOT
# ============================================================

def plot_comparison(runs, output_file, title):
    data = [(run, last_checkpoint_per_layer(run)) for run in runs]
    if any(d is None for _, d in data):
        missing = [run["label"] for run, d in data if d is None]
        print(f"[WARN] '{title}': nessun CSV per {missing}, skip.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    for run, (values, step, n) in data:
        mean = values.mean(axis=0)
        x = np.arange(len(mean))
        seeds_txt = f"media su {n} seed, " if n > 1 else ""
        ax.plot(x, mean,
                label=f"{run['label']} ({seeds_txt}step {step:,})",
                color=run["color"],
                linestyle=run["linestyle"],
                marker="o", markersize=5, linewidth=2)
        if n > 1:
            std = values.std(axis=0, ddof=1)
            ax.fill_between(x, mean - std, mean + std, color=run["color"], alpha=0.15, linewidth=0)

        # etichetta valore massimo (della media)
        peak_idx = int(mean.argmax())
        ax.annotate(f"{mean[peak_idx]:.3f}",
                    xy=(peak_idx, mean[peak_idx]),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color=run["color"])

    n_layers = len(mean)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Avg Spearman ρ", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150)
    print(f"Salvato: {output_file}")
    plt.close(fig)


if __name__ == "__main__":
    only = sys.argv[1:]
    for comp in COMPARISONS:
        if only and comp["name"] not in only:
            continue
        plot_comparison(comp["runs"], comp["output"], comp["title"])
