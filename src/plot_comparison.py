# ============================================================
# plot_comparison.py — Confronto multilayer tra run diverse
#
# Uso:
#   Lo script legge l'ultimo checkpoint di ciascuna run e
#   plotta ρ per layer, una linea per run.
#   python plot_comparison.py              → tutti i confronti
#   python plot_comparison.py <name> ...   → solo i confronti indicati
# ============================================================

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# CONFIGURAZIONE
# ============================================================

MAMBA_FRASI = {
    "label":    "Run 4 — frasi (128 tok)",
    "csv":      "../results/learning_curve_run4.csv",       # path al CSV
    "color":    "#2196F3",
    "linestyle": "-",
}
MAMBA_FRASI_LN = {
    "label":    "Run 4 — frasi + LN (128 tok)",
    "csv":      "../results/learning_curve_run4_ln.csv",    # LayerNorm sugli hidden states
    "color":    "#3F51B5",
    "linestyle": ":",
}
BERT_FRASI = {
    "label":     "BERT — frasi",
    "csv":       "../results/learning_curve_bert_frasi.csv",
    "color":     "#4CAF50",
    "linestyle": "-.",
}
MAMBA_BLOCCHI = {
    "label":    "Run 5 — blocchi (512 tok)",
    "csv":      "../results/learning_curve_run5.csv",
    "color":    "#FF5722",
    "linestyle": "-",
}
MAMBA_BLOCCHI_LN = {
    "label":    "Run 5 — blocchi + LN (512 tok)",
    "csv":      "../results/learning_curve_run5_ln.csv",    # LayerNorm sugli hidden states
    "color":    "#BF360C",
    "linestyle": ":",
}
BERT_BLOCCHI = {
    "label":     "BERT — blocchi (512 tok)",
    "csv":       "../results/bert_medium_dbmdz_blocks_512_probing.csv",   # formato lungo
    "color":     "#2E7D32",
    "linestyle": "-.",
}

COMPARISONS = [
    # --- frasi ---
    {
        "name":   "bert_mamba",
        "runs":   [MAMBA_FRASI, BERT_FRASI],
        "output": "../results/comparisons/sentences/comparison_bert_mamba_sentences.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs BERT",
    },
    {
        "name":   "bert_mamba_ln",
        "runs":   [MAMBA_FRASI_LN, BERT_FRASI],
        "output": "../results/comparisons/sentences/comparison_bert_mamba_ln_sentences.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba + LN vs BERT",
    },
    {
        "name":   "bert_mamba_ln_vs_noln",
        "runs":   [MAMBA_FRASI, MAMBA_FRASI_LN, BERT_FRASI],
        "output": "../results/comparisons/sentences/comparison_bert_mamba_ln_vs_noln_sentences.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba con/senza LN vs BERT",
    },
    # --- blocchi ---
    {
        "name":   "bert_mamba_blocks",
        "runs":   [MAMBA_BLOCCHI, BERT_BLOCCHI],
        "output": "../results/comparisons/blocks/comparison_bert_mamba_blocks.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba vs BERT (blocchi)",
    },
    {
        "name":   "bert_mamba_ln_blocks",
        "runs":   [MAMBA_BLOCCHI_LN, BERT_BLOCCHI],
        "output": "../results/comparisons/blocks/comparison_bert_mamba_ln_blocks.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba + LN vs BERT (blocchi)",
    },
    {
        "name":   "bert_mamba_ln_vs_noln_blocks",
        "runs":   [MAMBA_BLOCCHI, MAMBA_BLOCCHI_LN, BERT_BLOCCHI],
        "output": "../results/comparisons/blocks/comparison_bert_mamba_ln_vs_noln_blocks.pdf",
        "title":  "Confronto probing multilayer — DiffuMamba con/senza LN vs BERT (blocchi)",
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

# ============================================================
# PLOT
# ============================================================

def plot_comparison(runs, output_file, title):
    fig, ax = plt.subplots(figsize=(9, 5))

    for run in runs:
        df = load_curve(run["csv"])

        # prende l'ultimo checkpoint disponibile
        last_row = df.iloc[-1]
        layer_cols = [c for c in df.columns if c.startswith("layer_")]
        n_layers = len(layer_cols)
        values = [last_row[c] for c in layer_cols]
        x = list(range(n_layers))

        ax.plot(x, values,
                label=f"{run['label']} (step {int(last_row['step']):,})",
                color=run["color"],
                linestyle=run["linestyle"],
                marker="o", markersize=5, linewidth=2)

        # etichetta valore massimo
        peak_idx = values.index(max(values))
        ax.annotate(f"{max(values):.3f}",
                    xy=(peak_idx, max(values)),
                    xytext=(4, 6), textcoords="offset points",
                    fontsize=8, color=run["color"])

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
