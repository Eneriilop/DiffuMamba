# ============================================================
# plot_comparison.py — Confronto multilayer tra run diverse
#
# Uso:
#   Lo script legge l'ultimo checkpoint di ciascuna run e
#   plotta ρ per layer, una linea per run.
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# CONFIGURAZIONE
# ============================================================

RUNS = [
    {
        "label":    "Run 4 — frasi (128 tok)",
        "csv":      "learning_curve_multilayer.csv",       # path al CSV
        "color":    "#2196F3",
        "linestyle": "-",
    },
    {
        "label":    "Run 5 — blocchi (512 tok)",
        "csv":      "learning_curve_blocks_run5.csv",
        "color":    "#FF5722",
        "linestyle": "--",
    },
    {
    "label":     "BERT — frasi",
    "csv":       "learning_curve_bert_frasi.csv",
    "color":     "#4CAF50",
    "linestyle": "-.",
    },
    # Per aggiungere BERT o altre run:
    # {
    #     "label":    "BERT base italian",
    #     "csv":      "learning_curve_bert.csv",
    #     "color":    "#4CAF50",
    #     "linestyle": "-.",
    # },
]

REFERENCE_LINES = [
    {"y": 0.70, "label": "BERT (Üstün et al.)", "color": "gray", "linestyle": ":"},
]

OUTPUT_FILE = "comparison_runs.pdf"
TITLE       = "Confronto probing multilayer — avg Spearman ρ per layer"

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

for run in RUNS:
    df = pd.read_csv(run["csv"])

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

# linee di riferimento
for ref in REFERENCE_LINES:
    ax.axhline(ref["y"], color=ref["color"], linestyle=ref["linestyle"],
               linewidth=1.2, label=ref["label"])

ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Avg Spearman ρ", fontsize=12)
ax.set_title(TITLE, fontsize=13)
ax.set_xticks(range(n_layers))
ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
print(f"Salvato: {OUTPUT_FILE}")