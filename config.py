# ============================================================
# config.py — tutti gli iperparametri in un unico posto
# ============================================================

# --- Tokenizer (da HuggingFace) ---
TOKENIZER_NAME = "dbmdz/bert-base-italian-uncased"   # vocabolario 31102 token
MASK_TOKEN_ID  = 103                   # [MASK] — stesso valore in tutti i BERT
PAD_TOKEN_ID   = 0                     # [PAD]
MAX_SEQ_LEN    = 128

# --- Modello ---
HIDDEN_SIZE = 512    # dimensione hidden (BERT medio - paper On the impact of pretraining data ...)
NUM_LAYERS  = 8       # numero di blocchi Bi-Mamba
STATE_SIZE  = 16      # dimensione stato SSM interno di Mamba (default libreria)
EXPAND      = 4       # expansion factor interno Mamba (default libreria 2)
CONV_KERNEL = 4       # kernel conv 1D interna Mamba (default libreria)

# --- Dati ---
#   "csv"       → un file csv con colonna "text" che contiene le frasi
#   "sentences" → un file con una frase per riga
#   "blocks"    → un file con blocchi di frasi separati da riga vuota
DATA_PATH   = "data/train_1_orig.csv"
DATA_FORMAT = "csv"          # "csv" | "sentences" | "blocks"
MIN_TOKENS  = 6             # filtra sequenze con meno di N token (spazio-split)

# --- Training ---
MAX_SAMPLES  = None             #None         # 500 usato per test rapidi
BATCH_SIZE   = 32        #32        # 8 usato per test rapidi
TOTAL_STEPS  = 624_000         #50_000        # 100 usato per test rapidi
WARMUP_STEPS = 6_000
LR           = 1e-4
MASK_RATE    = 0.15   # probabilità base di mascherare un token (come BERT)
                      # nel diffusion viene campionato t e usato come rate variabile

# --- Checkpoint ---
OUTPUT_DIR   = "checkpoints"
SAVE_EVERY   = 25_000   #salva checkpoint ogni N step (incluso ultimo)
LOG_EVERY    = 50
EVAL_EVERY   = 500
