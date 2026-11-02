# ============================================================
# train.py — data loading + training loop
# ============================================================
# Dipendenze:
#   pip install transformers torch datasets
#
# Uso:
#   python train.py
#
# Supporta due formati di input (impostare in config.py):
#   DATA_FORMAT = "sentences"  → un file con una frase per riga
#   DATA_FORMAT = "blocks"     → blocchi di frasi separati da riga vuota
# ============================================================

import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from model import DiffuMamba
import config
import time
from torch.utils.data import SequentialSampler

# ── device ──────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ============================================================
# 1. DATASET
# ============================================================

class TextDataset(Dataset):
    """
    Lazy loading: carica solo le stringhe di testo in memoria,
    tokenizza ogni frase on-demand quando il DataLoader la richiede.
    Questo evita di tokenizzare milioni di frasi in anticipo.
    """

    def __init__(self, path: str, tokenizer, fmt: str):
        self.tokenizer = tokenizer

        # Legge solo le stringhe — niente tokenizzazione qui
        print("Caricamento testi...")
        if fmt == "csv":
            texts = []
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in tqdm(reader, desc="Lettura CSV"):
                    text = row.get("text") or row.get("texr") or ""
                    texts.append(text.strip())

        elif fmt == "sentences":
            with open(path, encoding="utf-8") as f:
                texts = [line.strip() for line in tqdm(f, desc="Lettura frasi")]

        elif fmt == "blocks":
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            texts = [b.strip() for b in raw.split("\n\n")]

        else:
            raise ValueError(f"DATA_FORMAT non riconosciuto: {fmt}")

        # filtra frasi troppo corte (split su spazio)
        #self.texts = [t for t in texts if len(t.split()) >= config.MIN_TOKENS]

        #Nessun filtro
        self.texts = texts

        # limita il numero di campioni (per test rapidi, None = tutti)
        if config.MAX_SAMPLES is not None:
            self.texts = self.texts[:config.MAX_SAMPLES]

        print(f"Dataset pronto: {len(self.texts)} sequenze")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # tokenizzazione on-demand: avviene solo quando il DataLoader
        # richiede questo specifico campione durante il training
        # Per ogni frase il tokenizer di HuggingFace produce due tensori:
        #   input_ids:      (L,)  sequenza di token ID (con padding a MAX_SEQ_LEN)
        #   attention_mask: (L,)  1 dove c'è testo, 0 dove c'è padding
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=config.MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ============================================================
# 2. DIFFUSION FORWARD PROCESS 
# ============================================================

def mask_tokens(input_ids: torch.Tensor, t: torch.Tensor, attention_mask: torch.Tensor):
    """
    Maschera ogni token con probabilità t (il timestep è direttamente il mask rate).
    I token di padding non vengono mai mascherati.

    input_ids      : (B, L)
    t              : (B,)  float in [0, 1]
    attention_mask : (B, L)

    Ritorna:
        x_t        : (B, L) sequenza corrotta
        masked     : (B, L) bool — True dove è stato applicato MASK
    """
    # probabilità di mask per ogni posizione: (B, 1) broadcast → (B, L)
    mask_prob = t.unsqueeze(1).expand_as(input_ids)
    # bernoulli lancia un dado per ogni token: True con probabilità t, False altrimenti
    # quindi mediamente il 15% dei token sarà mascherato se t=0.15, ma ogni token ha probabilità indipendente
    masked = torch.bernoulli(mask_prob).bool()

    # non maschera padding
    masked = masked & (attention_mask == 1)

    # clona la sequenza originale e inserisce MASK_TOKEN_ID dove masked è True (ID 103)
    # la maschera servirà dopo per calcolare la loss solo sui token mascherati
    x_t = input_ids.clone()
    # token 103 come BERT per i token mascherati
    x_t[masked] = config.MASK_TOKEN_ID
    return x_t, masked


# ============================================================
# 3. LOSS
# ============================================================

def compute_loss(logits: torch.Tensor, input_ids: torch.Tensor, masked: torch.Tensor):
    """
    Cross-entropy solo sui token mascherati.
    Equivalente al MLM loss di BERT, ma con mask rate variabile.
    """
    # con un t molto piccolo è possibile che non venga mascherato nessun token: in questo caso ritorna 0.0 (evita NaN)
    if not masked.any():
        return logits.sum() * 0.0

    # seleziona solo i token masked per calcolare la loss
    # contiene la distribuzione predetta dal modello solo sui token mascherati (N, vocab_size)
    logits_masked  = logits[masked]     # (N, vocab_size)
    # contiene il token ID originale solo dei token mascherati (N,)
    targets_masked = input_ids[masked]  # (N,)
    # calcola cross-entropy solo sui token mascherati, identica a MLM di BERT ma con mask rate variabile (t continuo)
    return F.cross_entropy(logits_masked, targets_masked)


# ============================================================
# 4. TRAINING LOOP
# ============================================================

def train():

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # tokenizer da HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

    # dataset e split train/val 90/10
    # dataset = TextDataset(config.DATA_PATH, tokenizer, config.DATA_FORMAT)
    # val_size   = max(1, int(0.1 * len(dataset)))
    # train_size = len(dataset) - val_size
    # train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Train sul dataset e validation su eval_3 
    train_ds = TextDataset(config.DATA_PATH, tokenizer, config.DATA_FORMAT)
    val_ds   = TextDataset(config.VAL_PATH, tokenizer, config.VAL_FORMAT)

    # dataloader assembla i batch con shuffle=true per il training
    # num_workers: tokenizzazione in parallelo su CPU mentre la GPU lavora
    # pin_memory: trasferimento CPU→GPU più veloce
    num_workers = min(4, os.cpu_count() or 1)
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=SequentialSampler(train_ds), num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)} sequenze | Val: {len(val_ds)} sequenze")

    # modello e ottimizzatore
    model = DiffuMamba().to(device)
    print(f"Parametri: {model.count_params():,}")
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=0.01)
    scaler = GradScaler('cuda')  # AMP: mantiene stabilità numerica con float16

    # resume da checkpoint se esiste
    start_step = 0
    # latest_ckpt = os.path.join(config.OUTPUT_DIR, "ckpt_step001000.pt")
    # if os.path.exists(latest_ckpt):
    #     ckpt = torch.load(latest_ckpt, map_location=device)
    #     model.load_state_dict(ckpt["model"])
    #     start_step = ckpt["step"]
    #     print(f"Ripreso da checkpoint: step {start_step}")

    # training per step, non per epoche
    model.train()
    step = start_step
    running_loss = 0.0
    epoch = 0
    train_iter = iter(train_dl)
    t0 = time.time()

    # salva checkpoint step 0 (modello non trainato)
    path = os.path.join(config.OUTPUT_DIR, "ckpt_step000000.pt")
    torch.save({"step": 0, "model": model.state_dict(), "loss": None}, path)
    print(f"[0] Checkpoint step 0 salvato: {path}")

    while step < config.TOTAL_STEPS:

        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            val_loss = evaluate(model, val_dl)
            print(f"[step {step} | fine epoca {epoch}] VAL loss={val_loss:.4f}")
            model.train()
            train_iter = iter(train_dl)
            continue   # riparte dal while per prendere il primo batch della nuova epoca

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        B = input_ids.size(0)

        if step < config.WARMUP_STEPS:
            for pg in optimizer.param_groups:
                pg["lr"] = config.LR * (step + 1) / config.WARMUP_STEPS

        t = torch.rand(B, device=device)
        x_t, masked = mask_tokens(input_ids, t, attention_mask)

        with autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(x_t, t)
            loss = compute_loss(logits, input_ids, masked)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        running_loss += loss.item()

        if step % config.LOG_EVERY == 0:
            elapsed = time.time() - t0
            sec_per_step = elapsed / (step - start_step)
            eta_h = (config.TOTAL_STEPS - step) * sec_per_step / 3600
            print(f"[{step}/{config.TOTAL_STEPS}] loss={running_loss / config.LOG_EVERY:.4f} | {sec_per_step:.2f}s/step | ETA {eta_h:.1f}h")
            running_loss = 0.0

        if (step <= 4000 and step % 400 == 0) or \
           (step > 4000 and step % 4000 == 0) or \
           step == config.TOTAL_STEPS:
            path = os.path.join(config.OUTPUT_DIR, f"ckpt_step{step:06d}.pt")
            torch.save({"step": step, "model": model.state_dict(), "loss": loss.item()}, path)
            print(f"[{step}] Checkpoint salvato: {path}")


def evaluate(model, val_dl, max_batches=200):
    # identico al training loop ma senza aggiornamento dei pesi e con torch.no_grad() per disabilitare la backprop e risparmiare memoria
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_dl:
            if n >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            B = input_ids.size(0)
            t = torch.rand(B, device=device)
            x_t, masked = mask_tokens(input_ids, t, attention_mask)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x_t, t)
                total += compute_loss(logits, input_ids, masked).item()
            n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    train()
