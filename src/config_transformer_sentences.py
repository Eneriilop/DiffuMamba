# ============================================================
# config_transformer_sentences.py — ablation Transformer su frasi (come run4)
# ============================================================
# Identico a config.py (run4) tranne i blocchi del modello:
# BiMamba → Transformer encoder.

# --- Tokenizer (da HuggingFace) ---
TOKENIZER_NAME = "dbmdz/bert-base-italian-cased"   # vocabolario 31102 token
MASK_TOKEN_ID  = 104
PAD_TOKEN_ID   = 0
MAX_SEQ_LEN    = 128   # anche dimensione del positional embedding

# --- Modello ---
HIDDEN_SIZE = 496
NUM_LAYERS  = 8
NUM_HEADS   = 8       # 496 / 8 = 62 dimensioni per head
FFN_SIZE    = 2212    # parametri per blocco ≈ BiMambaBlock (vedi config_transformer.py)
DROPOUT     = 0.0     # DiffuMamba non usa dropout

# --- Dati ---
DATA_PATH   = "../data/train_shuffled.csv"
DATA_FORMAT = "csv"          # "csv" | "sentences" | "blocks"
VAL_PATH = "../data/eval_3.csv"
VAL_FORMAT = "csv"

# --- Training ---
MAX_SAMPLES  = None
BATCH_SIZE   = 256
TOTAL_STEPS  = 117_188  # 10M frasi / 256 * 3 epoche (39_063 step per epoca)
WARMUP_STEPS = 1_000
LR           = 1e-4
MASK_RATE    = 0.15

# --- Checkpoint ---
# OUTPUT_DIR: deciso da runs.py (checkpoints/<data>[_<model>]_seed<N>)
LOG_EVERY    = 50
