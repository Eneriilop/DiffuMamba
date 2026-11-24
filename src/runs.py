# ============================================================
# runs.py — nomi delle run, config e percorsi dei risultati
# ============================================================
# Unico punto in cui sono definiti i nomi, usato da training,
# probing, valutazione, aggregazione e grafici.
#
#   data  : "sent" (frasi, 128 tok) | "blocks" (blocchi, 512 tok)
#   model : "mamba" (DiffuMamba)    | "transformer" (DiffuTransformer)
#           | "ar" (Mamba AR)
#
# Nome della run: <data>[_<model>]_seed<N>;
#   sent_seed0, sent_transformer_seed0, sent_ar_seed0, blocks_seed0, ...
# Risultati del probing senza LayerNorm: suffisso "_noln"
#
# ============================================================

import importlib

MODELS = ("mamba", "transformer", "ar")
DATAS  = ("sent", "blocks")

_CONFIGS = {
    ("mamba",       "sent"):   "config",
    ("mamba",       "blocks"): "config_blocks",
    ("transformer", "sent"):   "config_transformer_sentences",
    ("transformer", "blocks"): "config_transformer",
    ("ar",          "sent"):   "config_ar_sentences",
    ("ar",          "blocks"): "config_ar_blocks",
}


def get_config(model: str, data: str):
    return importlib.import_module(_CONFIGS[(model, data)])


def base_name(model: str, data: str) -> str:
    """Nome senza seed, es. "sent" o "blocks_transformer"."""
    return data if model == "mamba" else f"{data}_{model}"


def run_name(model: str, data: str, seed: int) -> str:
    return f"{base_name(model, data)}_seed{seed}"


def result_name(model: str, data: str, seed: int, ln: bool = True) -> str:
    return run_name(model, data, seed) + ("" if ln else "_noln")


def checkpoints_dir(model, data, seed):
    return f"../checkpoints/{run_name(model, data, seed)}"


def probing_dir(model, data, seed, ln=True):
    return f"../probing_results/{result_name(model, data, seed, ln)}"


def learning_curve_csv(model, data, seed, ln=True):
    return f"../results/learning_curves/learning_curve_{result_name(model, data, seed, ln)}.csv"


def all_metrics_csv(model, data, seed, ln=True):
    return f"../results/all_metrics/all_metrics_{result_name(model, data, seed, ln)}.csv"


# BERT medium: addestrato da altri, solo CSV di probing in formato lungo
# (checkpoint, layer 1..8, feature, score, model), un file per seed.
# Hidden states post-LayerNorm (HuggingFace).
BERT_SEEDS = (42, 43, 44)
_BERT_DATA = {"sent": "frasi", "blocks": "blocchi"}


def bert_csvs(data):
    """CSV di probing BERT di tutti i seed per "sent" o "blocks"."""
    return [f"../results/bert/bert_{_BERT_DATA[data]}_s{s}_probing.csv" for s in BERT_SEEDS]
