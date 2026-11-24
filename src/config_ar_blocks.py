# ============================================================
# config_ar_blocks.py — Mamba AR su blocchi da 512 token (ablation dell'obiettivo)
# ============================================================
# Identico a config_blocks.py (DiffuMamba su blocchi) tranne EXPAND = 4:
# un solo Mamba per blocco (unidirezionale) invece di due, quindi
# expand raddoppiato per avere gli stessi parametri.

# --- Tokenizer ---
TOKENIZER_NAME = "dbmdz/bert-base-italian-cased"
MASK_TOKEN_ID  = 104                   # non usato dall'obiettivo AR
PAD_TOKEN_ID   = 0
MAX_SEQ_LEN    = 512

# --- Modello ---
HIDDEN_SIZE = 496
NUM_LAYERS  = 8
STATE_SIZE  = 16
EXPAND      = 4       # DiffuMamba: 2 (ma due Mamba per blocco)
CONV_KERNEL = 4

# --- Dati ---
DATA_PATH   = "../data/blocks_512"   # cartella Arrow dataset HuggingFace
DATA_FORMAT = "blocks_pretok"     # blocchi pre-tokenizzati
VAL_PATH   = "../data/eval_blocks_512"
VAL_FORMAT = "blocks_pretok"

# --- Training ---
MAX_SAMPLES  = None
BATCH_SIZE   = 64
TOTAL_STEPS  = 80_994   # 1.727.869 / 64 * 3 epoche
WARMUP_STEPS = 1_000
LR           = 1e-4

# --- Checkpoint ---
# OUTPUT_DIR: deciso da runs.py (checkpoints/<data>[_<model>]_seed<N>)
LOG_EVERY  = 50
