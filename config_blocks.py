# ============================================================
# config.py — Run 5 (blocchi 512 token)
# ============================================================

# --- Tokenizer ---
TOKENIZER_NAME = "dbmdz/bert-base-italian-cased"
MASK_TOKEN_ID  = 104
PAD_TOKEN_ID   = 0
MAX_SEQ_LEN    = 512

# --- Modello ---
HIDDEN_SIZE = 496
NUM_LAYERS  = 8
STATE_SIZE  = 16
EXPAND      = 2
CONV_KERNEL = 4

# --- Dati ---
DATA_PATH   = "data/blocks_512"   # cartella Arrow dataset HuggingFace
DATA_FORMAT = "blocks_pretok"     # blocchi pre-tokenizzati
VAL_PATH   = "data/eval_blocks_512"
VAL_FORMAT = "blocks_pretok"

# --- Training ---
MAX_SAMPLES  = None
BATCH_SIZE   = 64
TOTAL_STEPS  = 80_994   # 1.727.869 / 64 * 3 epoche
WARMUP_STEPS = 1_000
LR           = 1e-4

# --- Checkpoint ---
OUTPUT_DIR = "checkpoints_blocks_run5"
LOG_EVERY  = 50