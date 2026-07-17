# ============================================================
# config.py — tutti gli iperparametri in un unico posto
# ============================================================

# --- Tokenizer (da HuggingFace) ---
TOKENIZER_NAME = "dbmdz/bert-base-italian-cased"   # vocabolario 31102 token
MASK_TOKEN_ID  = 104                   # [MASK] — stesso valore in tutti i BERT - con uncased è 103
PAD_TOKEN_ID   = 0                     # [PAD]
MAX_SEQ_LEN    = 128

# --- Modello ---
HIDDEN_SIZE = 496    # dimensione hidden (BERT dim=512 - paper On the impact of pretraining data ...)
NUM_LAYERS  = 8       # numero di blocchi Bi-Mamba
STATE_SIZE  = 16      # dimensione stato SSM interno di Mamba (default libreria)
EXPAND      = 2       # expansion factor interno Mamba (default libreria)
CONV_KERNEL = 4       # kernel conv 1D interna Mamba (default libreria)

# --- Dati ---
#   "csv"       → un file csv con colonna "text" che contiene le frasi
#   "sentences" → un file con una frase per riga
#   "blocks"    → un file con blocchi di frasi separati da riga vuota
DATA_PATH   = "data/train_shuffled.csv"
DATA_FORMAT = "csv"          # "csv" | "sentences" | "blocks"
VAL_PATH = "data/eval_3.csv"
VAL_FORMAT = "csv"

# --- Training ---
MAX_SAMPLES  = None            
BATCH_SIZE   = 256        
TOTAL_STEPS  = 117_188  # 10M frasi / 256 * 3 epoche (39_063 step per epoca)
WARMUP_STEPS = 1_000  # ~8% degli step totali 
LR           = 1e-4
MASK_RATE    = 0.15   # probabilità base di mascherare un token (come BERT)
                      # nel diffusion viene campionato t e usato come rate variabile

# --- Checkpoint ---
OUTPUT_DIR   = "checkpoints_sent_run4"
# SAVE_EVERY   = 25_000   #salva checkpoint ogni N step (incluso ultimo)
LOG_EVERY    = 50
# EVAL_EVERY   = 10_000
