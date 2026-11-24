# ============================================================
# config_transformer.py — ablation Transformer su blocchi 512 token (come run5)
# ============================================================
# Identico a config_blocks.py (run5) tranne i blocchi del modello:
# BiMamba → Transformer encoder.

# --- Tokenizer ---
TOKENIZER_NAME = "dbmdz/bert-base-italian-cased"
MASK_TOKEN_ID  = 104
PAD_TOKEN_ID   = 0
MAX_SEQ_LEN    = 512   # anche dimensione del positional embedding

# --- Modello ---
HIDDEN_SIZE = 496
NUM_LAYERS  = 8
NUM_HEADS   = 8       # 496 / 8 = 62 dimensioni per head
FFN_SIZE    = 2212    # parametri per blocco ≈ BiMambaBlock (3.185.044 vs 3.185.312)
                      # con lo standard 4 × HIDDEN_SIZE = 1984 sarebbero 2.958.640
DROPOUT     = 0.0     # DiffuMamba non usa dropout

# --- Dati ---
DATA_PATH   = "../data/blocks_512"
DATA_FORMAT = "blocks_pretok"
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
