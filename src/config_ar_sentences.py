# ============================================================
# config_ar_sentences.py — Mamba AR su frasi (ablation dell'obiettivo)
# ============================================================
# Identico a config.py (DiffuMamba su frasi) tranne EXPAND = 4:
# un solo Mamba per blocco (unidirezionale) invece di due, quindi
# expand raddoppiato per avere gli stessi parametri
# (Mamba AR 40.910.080 vs DiffuMamba 40.911.072).

# --- Tokenizer (da HuggingFace) ---
TOKENIZER_NAME = "dbmdz/bert-base-italian-cased"   # vocabolario 31102 token
MASK_TOKEN_ID  = 104                   # non usato dall'obiettivo AR
PAD_TOKEN_ID   = 0                     # [PAD]
MAX_SEQ_LEN    = 128

# --- Modello ---
HIDDEN_SIZE = 496
NUM_LAYERS  = 8
STATE_SIZE  = 16
EXPAND      = 4       # DiffuMamba: 2 (ma due Mamba per blocco)
CONV_KERNEL = 4

# --- Dati ---
DATA_PATH   = "../data/train_shuffled.csv"
DATA_FORMAT = "csv"
VAL_PATH = "../data/eval_3.csv"
VAL_FORMAT = "csv"

# --- Training ---
MAX_SAMPLES  = None
BATCH_SIZE   = 256
TOTAL_STEPS  = 117_188  # 10M frasi / 256 * 3 epoche (39_063 step per epoca)
WARMUP_STEPS = 1_000
LR           = 1e-4

# --- Checkpoint ---
# OUTPUT_DIR: deciso da runs.py (checkpoints/<data>[_<model>]_seed<N>)
LOG_EVERY    = 50
