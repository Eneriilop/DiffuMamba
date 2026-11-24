# ============================================================
# plot_groups.py — Analisi per gruppi di feature (Miaschi et al. 2020)
#
# Per ogni confronto in COMPARISONS produce:
# (in ../results/comparisons/<sentences|blocks>/, una sottocartella per tipo)
#   1. group_final/group_final_<name>.pdf           — barchart avg ρ per gruppo (ultimo ckpt, layer migliore)
#   2. group_curves/group_curves_<name>.pdf         — curve durante il training per gruppo (layer migliore)
#   3. group_peak_steps/group_peak_steps_<name>.csv — step in cui ogni gruppo raggiunge X% del picco
#   4. group_features/group_features_<name>.txt     — elenco delle feature escluse dal confronto
#
# Il confronto usa SOLO le feature della Table 1 del paper che sono
# presenti in tutte le run del confronto. Le altre (extra rispetto al
# paper, non condivise tra i modelli, del paper ma assenti nei CSV)
# vengono elencate nel file group_features_<name>.txt.
#
# Uso: python plot_groups.py              → tutti i confronti
#      python plot_groups.py <name> ...   → solo i confronti indicati
#      (modificare COMPARISONS e THRESHOLD_PERCS nella sezione CONFIGURAZIONE)
#
# Seed: per DiffuMamba e DiffuTransformer le medie per gruppo sono calcolate
# per ogni seed e poi mediate; barre d'errore e bande = ±1 dev. std sui seed.
# Layer migliore e tabella step-al-picco sono calcolati sulla media.
# BERT: 3 seed (42/43/44, forniti da terzi), trattati come gli altri modelli.
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import sys
from seeds import SEEDS
from runs import all_metrics_csv, bert_csvs

# ============================================================
# CONFIGURAZIONE
# ============================================================

# Convenzione: "DiffuMamba" e "DiffuTransformer" = probing con LayerNorm sugli
# hidden states, per confronto corretto con gli hidden_states post-LN di BERT.
# La versione senza LN compare solo nel confronto con/senza LN.


def seed_csvs(model, data, ln=True):
    """CSV all_metrics di tutti i seed di un modello."""
    return [all_metrics_csv(model, data, s, ln=ln) for s in SEEDS]


MAMBA_FRASI_NOLN = {
    "label":     "DiffuMamba senza LN — frasi (128 tok)",
    "csvs":      seed_csvs("mamba", "sent", ln=False),
    "color":     "#2196F3",
    "linestyle": "-",
}
MAMBA_FRASI = {
    "label":     "DiffuMamba — frasi (128 tok)",
    "csvs":      seed_csvs("mamba", "sent"),
    "color":     "#3F51B5",
    "linestyle": ":",
}
MAMBA_FRASI_CON_LN = {**MAMBA_FRASI, "label": "DiffuMamba con LN — frasi (128 tok)"}
TRANSFORMER_FRASI = {
    "label":     "DiffuTransformer — frasi (128 tok)",
    "csvs":      seed_csvs("transformer", "sent"),
    "color":     "#9C27B0",
    "linestyle": "--",
}
MAMBA_BLOCCHI_NOLN = {
    "label":     "DiffuMamba senza LN — blocchi (512 tok)",
    "csvs":      seed_csvs("mamba", "blocks", ln=False),
    "color":     "#FF5722",
    "linestyle": "--",
}
MAMBA_BLOCCHI = {
    "label":     "DiffuMamba — blocchi (512 tok)",
    "csvs":      seed_csvs("mamba", "blocks"),
    "color":     "#BF360C",
    "linestyle": ":",
}
MAMBA_BLOCCHI_CON_LN = {**MAMBA_BLOCCHI, "label": "DiffuMamba con LN — blocchi (512 tok)"}
TRANSFORMER_BLOCCHI = {
    "label":     "DiffuTransformer — blocchi (512 tok)",
    "csvs":      seed_csvs("transformer", "blocks"),
    "color":     "#6A1B9A",
    "linestyle": "--",
}
AR_FRASI = {
    "label":     "Mamba AR — frasi (128 tok)",
    "csvs":      seed_csvs("ar", "sent"),
    "color":     "#00897B",
    "linestyle": "-",
}
AR_BLOCCHI = {
    "label":     "Mamba AR — blocchi (512 tok)",
    "csvs":      seed_csvs("ar", "blocks"),
    "color":     "#004D40",
    "linestyle": "-",
}
BERT_FRASI = {
    "label":     "BERT medium — frasi",
    "csvs":      bert_csvs("sent"),        # formato lungo, un file per seed
    "color":     "#4CAF50",
    "linestyle": "-.",
    "layer_offset": -1,   # layer 1..8 → 0..7, come DiffuMamba
}
BERT_BLOCCHI = {
    "label":     "BERT medium — blocchi (512 tok)",
    "csvs":      bert_csvs("blocks"),      # formato lungo, un file per seed
    "color":     "#2E7D32",
    "linestyle": "-.",
    "layer_offset": -1,
}

# Ogni confronto produce i propri file (suffisso = "name") in OUT_DIR/<subdir>
COMPARISONS = [
    # --- frasi ---
    {
        "name":   "bert_vs_mamba_frasi",
        "title":  "BERT vs DiffuMamba — frasi",
        "runs":   [MAMBA_FRASI, BERT_FRASI],
        "subdir": "sentences",
    },
    {
        "name":   "bert_vs_mamba_frasi_ln_vs_noln",
        "title":  "BERT vs DiffuMamba con/senza LN — frasi",
        "runs":   [MAMBA_FRASI_NOLN, MAMBA_FRASI_CON_LN, BERT_FRASI],
        "subdir": "sentences",
    },
    {
        "name":   "mamba_vs_transformer_frasi",
        "title":  "DiffuMamba vs DiffuTransformer — frasi",
        "runs":   [MAMBA_FRASI, TRANSFORMER_FRASI],
        "subdir": "sentences",
    },
    {
        "name":   "bert_vs_mamba_vs_transformer_frasi",
        "title":  "BERT vs DiffuMamba vs DiffuTransformer — frasi",
        "runs":   [MAMBA_FRASI, TRANSFORMER_FRASI, BERT_FRASI],
        "subdir": "sentences",
    },
    {
        "name":   "mamba_vs_ar_frasi",
        "title":  "DiffuMamba vs Mamba AR — frasi",
        "runs":   [MAMBA_FRASI, AR_FRASI],
        "subdir": "sentences",
    },
    {
        "name":   "mamba_vs_transformer_vs_ar_frasi",
        "title":  "DiffuMamba vs DiffuTransformer vs Mamba AR — frasi",
        "runs":   [MAMBA_FRASI, TRANSFORMER_FRASI, AR_FRASI],
        "subdir": "sentences",
    },
    {
        "name":   "bert_vs_mamba_vs_ar_frasi",
        "title":  "BERT vs DiffuMamba vs Mamba AR — frasi",
        "runs":   [MAMBA_FRASI, AR_FRASI, BERT_FRASI],
        "subdir": "sentences",
    },
    # --- blocchi ---
    {
        "name":   "mamba_frasi_vs_blocchi",
        "title":  "DiffuMamba — frasi vs blocchi",
        "runs":   [MAMBA_FRASI, MAMBA_BLOCCHI],   # entrambi con LN
        "subdir": "blocks",
    },
    {
        "name":   "bert_vs_mamba_blocchi",
        "title":  "BERT vs DiffuMamba — blocchi",
        "runs":   [MAMBA_BLOCCHI, BERT_BLOCCHI],
        "subdir": "blocks",
    },
    {
        "name":   "bert_vs_mamba_blocchi_ln_vs_noln",
        "title":  "BERT vs DiffuMamba con/senza LN — blocchi",
        "runs":   [MAMBA_BLOCCHI_NOLN, MAMBA_BLOCCHI_CON_LN, BERT_BLOCCHI],
        "subdir": "blocks",
    },
    {
        "name":   "mamba_vs_transformer_blocchi",
        "title":  "DiffuMamba vs DiffuTransformer — blocchi",
        "runs":   [MAMBA_BLOCCHI, TRANSFORMER_BLOCCHI],
        "subdir": "blocks",
    },
    {
        "name":   "bert_vs_mamba_vs_transformer_blocchi",
        "title":  "BERT vs DiffuMamba vs DiffuTransformer — blocchi",
        "runs":   [MAMBA_BLOCCHI, TRANSFORMER_BLOCCHI, BERT_BLOCCHI],
        "subdir": "blocks",
    },
    {
        "name":   "mamba_vs_ar_blocchi",
        "title":  "DiffuMamba vs Mamba AR — blocchi",
        "runs":   [MAMBA_BLOCCHI, AR_BLOCCHI],
        "subdir": "blocks",
    },
    {
        "name":   "mamba_vs_transformer_vs_ar_blocchi",
        "title":  "DiffuMamba vs DiffuTransformer vs Mamba AR — blocchi",
        "runs":   [MAMBA_BLOCCHI, TRANSFORMER_BLOCCHI, AR_BLOCCHI],
        "subdir": "blocks",
    },
    {
        "name":   "bert_vs_mamba_vs_ar_blocchi",
        "title":  "BERT vs DiffuMamba vs Mamba AR — blocchi",
        "runs":   [MAMBA_BLOCCHI, AR_BLOCCHI, BERT_BLOCCHI],
        "subdir": "blocks",
    },
]

# Layer su cui calcolare il "migliore" per le learning curve
# (None = usa il layer con avg ρ massima all'ultimo checkpoint)
BEST_LAYER = None

# Percentuali di picco per la tabella step-at-X%
THRESHOLD_PERCS = [0.85, 0.90, 0.95]

OUT_DIR = "../results/comparisons"

# ============================================================
# FEATURE DEL PAPER (Miaschi et al. 2020, Table 1)
# ============================================================
# Nomi come prodotti da Profiling-UD sull'italiano (quelli dei CSV).
# Un pattern che termina con "*" è un prefisso.
# Corrispondenze con i nomi della Table 1:
#   sent_length            → n_tokens
#   xpos_VB-VBD-VBP-VBZ    → verbs_*   (morfologia dei verbi lessicali, versione italiana)
#   verbal_arity_*         → verb_edges_dist_*
#   avg_prep_chain_len     → avg_prepositional_chain_len
#   principal_prop_dist    → principal_proposition_dist
#   subordinate_prop_dist  → subordinate_proposition_dist
#   avg_subord_chain_len   → avg_subordinate_chain_len

PAPER_FEATURES = {
    "RawText":        ["n_tokens", "char_per_tok"],
    "Vocabulary":     ["ttr_form", "ttr_lemma"],
    "POS":            ["upos_dist_*", "xpos_dist_*", "lexical_density"],
    "VerbInflection": ["verbs_*", "aux_*"],
    "VerbPredicate":  ["verbal_head_dist", "verbal_root_perc",
                       "avg_verb_edges", "verb_edges_dist_*"],
    "TreeStructure":  ["parse_depth", "avg_links_len", "max_links_len",
                       "avg_prepositional_chain_len", "prep_dist_*",
                       "avg_token_per_clause"],
    "Order":          ["subj_pre", "obj_post"],
    "SyntacticDep":   ["dep_dist_*"],
    "Subord":         ["principal_proposition_dist", "subordinate_proposition_dist",
                       "avg_subordinate_chain_len", "subordinate_dist_1",
                       "subordinate_post"],
}

# Ordine canonico dei gruppi (dal paper)
GROUP_ORDER = list(PAPER_FEATURES)


def _matches(feature, pattern):
    if pattern.endswith("*"):
        return feature.startswith(pattern[:-1])
    return feature == pattern


def build_group_map(feature_cols):
    """
    Dato l'elenco di colonne-feature, restituisce {feature_name: group_name}
    per le sole feature presenti nella Table 1 del paper.
    """
    groups = {}
    for f in feature_cols:
        for grp, patterns in PAPER_FEATURES.items():
            if any(_matches(f, p) for p in patterns):
                groups[f] = grp
                break
    return groups


# ============================================================
# UTILITY
# ============================================================

def load_run(csv_path, layer_offset=0):
    """
    Accetta due formati:
      - largo (evaluate_multilayer.py): step, layer, avg_rho, <una colonna per feature>
      - lungo (probing BERT): checkpoint, layer, feature, score[, model]
        → convertito nel formato largo.
    layer_offset viene sommato all'indice di layer (BERT: 1..8 → -1 per avere 0..7
    come DiffuMamba).
    """
    df = pd.read_csv(csv_path)
    if {"checkpoint", "layer", "feature", "score"} <= set(df.columns):
        df = (df.pivot_table(index=["checkpoint", "layer"], columns="feature",
                             values="score", aggfunc="first")
                .reset_index()
                .rename(columns={"checkpoint": "step"}))
        df.columns.name = None
        feats = [c for c in df.columns if c not in ("step", "layer")]
        df.insert(2, "avg_rho", df[feats].mean(axis=1))
    df["layer"] = df["layer"] + layer_offset
    meta_cols = {"step", "layer", "avg_rho"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return df, feature_cols


def best_layer_for_run(df, feature_cols, group_map):
    """
    Restituisce il layer con avg ρ massima sulle feature con gruppo assegnato,
    calcolato sull'ultimo checkpoint.
    """
    last_step = df["step"].max()
    df_last = df[df["step"] == last_step]
    known_feats = [f for f in feature_cols if f in group_map]
    layer_avgs = df_last.groupby("layer")[known_feats].mean().mean(axis=1)
    return int(layer_avgs.idxmax())


def group_avg_by_step_layer(df, feature_cols, group_map):
    """
    Restituisce un DataFrame con colonne: step, layer, + un gruppo per colonna,
    con avg ρ calcolata sui feature del gruppo.
    """
    meta = df[["step", "layer"]].copy()
    for grp in GROUP_ORDER:
        feats = [f for f in feature_cols if group_map.get(f) == grp]
        if feats:
            meta[grp] = df[feats].mean(axis=1)
        else:
            meta[grp] = np.nan
    return meta


# ============================================================
# REPORT FEATURE — elenco separato delle feature escluse
# ============================================================

def feature_report(comp, loaded, out_path):
    """
    loaded: lista di (run_cfg, feature_cols).
    Restituisce (feature usate nel confronto, group_map) e scrive su
    out_path l'elenco delle feature escluse, divise per motivo.
    """
    feat_sets = {cfg["label"]: set(fc) for cfg, fc in loaded}
    all_feats = set.union(*feat_sets.values())
    shared    = set.intersection(*feat_sets.values())
    group_map = build_group_map(sorted(all_feats))

    used         = sorted(f for f in shared if f in group_map)
    extra        = sorted(f for f in all_feats if f not in group_map)
    paper_unshared = sorted(f for f in all_feats - shared if f in group_map)

    # pattern del paper senza alcuna corrispondenza nei CSV
    missing = [(grp, p) for grp, patterns in PAPER_FEATURES.items()
               for p in patterns
               if not any(_matches(f, p) for f in all_feats)]

    def owners(f):
        return ", ".join(lbl for lbl, s in feat_sets.items() if f in s)

    lines = [f"Confronto: {comp['title']}", ""]
    lines.append(f"[USATE] Feature del paper condivise da tutte le run: {len(used)}")
    for grp in GROUP_ORDER:
        feats = [f for f in used if group_map[f] == grp]
        lines.append(f"  {grp} ({len(feats)}): {', '.join(feats) if feats else '—'}")

    lines += ["", f"[ESCLUSE] Feature non presenti nel paper: {len(extra)}"]
    lines += [f"  {f}   (in: {owners(f)})" for f in extra] or ["  —"]

    lines += ["", f"[ESCLUSE] Feature del paper non condivise tra le run: {len(paper_unshared)}"]
    lines += [f"  {f} [{group_map[f]}]   (solo in: {owners(f)})"
              for f in paper_unshared] or ["  —"]

    lines += ["", f"[ASSENTI] Feature del paper assenti in tutti i CSV: {len(missing)}"]
    lines += [f"  {p} [{grp}]" for grp, p in missing] or ["  —"]

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nSalvato: {out_path}")

    return used, group_map


# ============================================================
# PLOT 1 — Barchart finale per gruppo (tutti i layer)
# ============================================================

def plot_final_barchart(runs_data, out_path, title):
    """
    Un gruppo di barre per ogni gruppo, una barra per ogni run,
    al layer migliore di ciascuna run.
    """
    present_groups = [g for g in GROUP_ORDER
                      if any(not np.isnan(rd["group_df"].iloc[-1].get(g, np.nan))
                             for rd in runs_data)]

    n_groups = len(present_groups)
    n_runs   = len(runs_data)
    x = np.arange(n_groups)
    width = 0.7 / n_runs

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, rd in enumerate(runs_data):
        last_step = rd["df"]["step"].max()
        best_lay  = rd["best_layer"]

        def point(gdf):
            return gdf[(gdf["step"] == last_step) & (gdf["layer"] == best_lay)]

        df_point, std_point = point(rd["group_df"]), point(rd["group_std"])
        vals = [float(df_point[g].values[0]) if g in df_point.columns else np.nan
                for g in present_groups]
        errs = [float(std_point[g].values[0]) if g in std_point.columns else np.nan
                for g in present_groups]
        errs = [0.0 if np.isnan(e) else e for e in errs]

        offset = (i - (n_runs - 1) / 2) * width
        seeds_txt = f", media su {rd['n']} seed" if rd["n"] > 1 else ""
        bars = ax.bar(x + offset, vals, width,
                      yerr=errs if rd["n"] > 1 else None, capsize=2,
                      error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
                      label=f"{rd['label']} (L{best_lay}{seeds_txt})",
                      color=rd["color"], alpha=0.85, edgecolor="white")

        # etichetta valore (media) sopra ogni barra, oltre la barra d'errore
        for bar, v, e in zip(bars, vals, errs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(present_groups, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Avg Spearman ρ", fontsize=12)
    ax.set_title(f"{title} — per gruppi di feature, ultimo checkpoint (layer migliore)",
                 fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Salvato: {out_path}")
    plt.close()


# ============================================================
# PLOT 2 — Learning curves per gruppo (layer migliore)
# ============================================================

def plot_learning_curves(runs_data, out_path, title):
    present_groups = [g for g in GROUP_ORDER
                      if any(g in rd["group_df"].columns and
                             not rd["group_df"][g].isna().all()
                             for rd in runs_data)]

    n_groups = len(present_groups)
    ncols = 3
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             sharex=False, sharey=False)
    axes = axes.flatten()

    for gi, grp in enumerate(present_groups):
        ax = axes[gi]
        for rd in runs_data:
            best_lay = rd["best_layer"]
            df_grp = rd["group_df"][rd["group_df"]["layer"] == best_lay].sort_values("step")
            df_std = rd["group_std"][rd["group_std"]["layer"] == best_lay].sort_values("step")
            if grp in df_grp.columns:
                ax.plot(df_grp["step"], df_grp[grp],
                        label=rd["label"],
                        color=rd["color"],
                        linestyle=rd["linestyle"],
                        linewidth=1.8, marker=".", markersize=4)
                if rd["n"] > 1:
                    m, s = df_grp[grp].values, df_std[grp].values
                    ax.fill_between(df_grp["step"], m - s, m + s,
                                    color=rd["color"], alpha=0.15, linewidth=0)

        ax.set_title(grp, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Avg ρ", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))

    # nascondi subplot vuoti
    for gi in range(n_groups, len(axes)):
        axes[gi].set_visible(False)

    fig.suptitle(f"{title} — learning curve per gruppi (layer migliore di ciascuna run)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Salvato: {out_path}")
    plt.close()


# ============================================================
# TABELLA — Step al X% del picco per gruppo
# ============================================================

def compute_peak_steps(runs_data, out_path):
    rows = []
    for rd in runs_data:
        best_lay = rd["best_layer"]
        df_grp = rd["group_df"][rd["group_df"]["layer"] == best_lay].copy()
        df_grp = df_grp.sort_values("step")

        present_groups = [g for g in GROUP_ORDER
                          if g in df_grp.columns and not df_grp[g].isna().all()]

        for grp in present_groups:
            series = df_grp[["step", grp]].dropna()
            peak_val = series[grp].max()
            for pct in THRESHOLD_PERCS:
                threshold = pct * peak_val
                reached = series[series[grp] >= threshold]
                step_at = int(reached["step"].min()) if len(reached) > 0 else None
                rows.append({
                    "run":       rd["label"],
                    "group":     grp,
                    "threshold": f"{int(pct*100)}%",
                    "peak_rho":  round(peak_val, 4),
                    "step_at":   step_at,
                })

    result = pd.DataFrame(rows)
    # pivot: threshold come colonne
    pivot = result.pivot_table(
        index=["run", "group"],
        columns="threshold",
        values="step_at",
        aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"run": "Run", "group": "Gruppo"})

    # aggiungi peak_rho
    peak_lookup = result[["run", "group", "peak_rho"]].drop_duplicates()
    peak_lookup = peak_lookup.rename(columns={"run": "Run", "group": "Gruppo"})
    pivot = pivot.merge(peak_lookup, on=["Run", "Gruppo"], how="left")

    # ordina per gruppo canonico
    pivot["_g_order"] = pivot["Gruppo"].map({g: i for i, g in enumerate(GROUP_ORDER)})
    pivot = pivot.sort_values(["Run", "_g_order"]).drop(columns=["_g_order"])

    pivot.to_csv(out_path, index=False)
    print(f"\nSalvato: {out_path}")
    print(pivot.to_string(index=False))
    return pivot


# ============================================================
# MAIN
# ============================================================

def prepare_run(run_cfg, dfs, used, group_map):
    """
    dfs: un DataFrame per seed. Medie per gruppo calcolate per seed e poi
    mediate (media e dev. std sui seed), sulle sole feature usate e sui soli
    (step, layer) presenti in tutti i seed. Il layer migliore è scelto sulla media.
    """
    keys = ["step", "layer"]
    common = set.intersection(*[set(map(tuple, d[keys].values)) for d in dfs])
    dfs = [d[[tuple(r) in common for r in d[keys].values]] for d in dfs]

    per_seed = pd.concat([group_avg_by_step_layer(d, used, group_map) for d in dfs])
    group_df  = per_seed.groupby(keys).mean().reset_index()
    group_std = (per_seed.groupby(keys).std(ddof=1).reset_index() if len(dfs) > 1
                 else group_df.assign(**{g: np.nan for g in GROUP_ORDER}))
    mean_df = pd.concat([d[keys + used] for d in dfs]).groupby(keys).mean().reset_index()

    best_lay = BEST_LAYER
    if best_lay is None:
        best_lay = best_layer_for_run(mean_df, used, group_map)
    print(f"{run_cfg['label']} — seed: {len(dfs)} — layer migliore: {best_lay}")

    return {
        "label":      run_cfg["label"],
        "color":      run_cfg["color"],
        "linestyle":  run_cfg["linestyle"],
        "df":         mean_df,
        "group_df":   group_df,
        "group_std":  group_std,
        "n":          len(dfs),
        "best_layer": best_lay,
    }


def main(only=None):
    """only: lista di "name" da eseguire (None = tutti i confronti)."""
    for comp in COMPARISONS:
        if only and comp["name"] not in only:
            continue
        name = comp["name"]
        print(f"\n################ {comp['title']} ################")

        loaded = []
        for run_cfg in comp["runs"]:
            found = [p for p in run_cfg["csvs"] if os.path.exists(p)]
            if not found:
                print(f"[WARN] nessun CSV per {run_cfg['label']}")
                continue
            if len(found) < len(run_cfg["csvs"]):
                print(f"[WARN] {run_cfg['label']}: {len(found)}/{len(run_cfg['csvs'])} seed disponibili")
            runs = [load_run(p, run_cfg.get("layer_offset", 0)) for p in found]
            feats = sorted(set.intersection(*[set(fc) for _, fc in runs]))
            loaded.append((run_cfg, [df for df, _ in runs], feats))

        if len(loaded) < len(comp["runs"]):
            print(f"[WARN] '{name}': non tutte le run sono disponibili, skip.")
            continue

        out_dir = os.path.join(OUT_DIR, comp.get("subdir", ""))
        for sub in ("group_final", "group_curves", "group_peak_steps", "group_features"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

        print("\n--- Feature ---")
        used, group_map = feature_report(
            comp, [(cfg, fc) for cfg, _, fc in loaded],
            f"{out_dir}/group_features/group_features_{name}.txt")

        print()
        runs_data = [prepare_run(cfg, dfs, used, group_map) for cfg, dfs, _ in loaded]

        print("\n--- Plot 1: barchart finale ---")
        plot_final_barchart(runs_data, f"{out_dir}/group_final/group_final_{name}.pdf", comp["title"])

        print("--- Plot 2: learning curves per gruppo ---")
        plot_learning_curves(runs_data, f"{out_dir}/group_curves/group_curves_{name}.pdf", comp["title"])

        print("--- Tabella: step al picco ---")
        compute_peak_steps(runs_data, f"{out_dir}/group_peak_steps/group_peak_steps_{name}.csv")


if __name__ == "__main__":
    # python plot_groups.py [name ...]  → solo i confronti indicati
    main(sys.argv[1:] or None)
