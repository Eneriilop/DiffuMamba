# ============================================================
# train.py — data loading + training loop
# ============================================================
# Dipendenze:
#   pip install transformers torch datasets
#
# Uso (dalla cartella src):
#   python train.py --model mamba --seed 0         → checkpoints/sent_seed0
#   python train.py --model transformer --seed 0   → checkpoints/sent_transformer_seed0
#   python train.py --model ar --seed 0            → checkpoints/sent_ar_seed0 (Mamba AR)
#   python train.py --model mamba --seed 0 --test-steps 100   → prova breve in checkpoints/_test/
#
# Training su frasi (128 token). Il seed controlla inizializzazione e
# rumore della diffusion; l'ordine dei dati è fisso (vedi seeds.py).
# ============================================================

import os
import csv
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import time
from torch.utils.data import SequentialSampler
from seeds import set_seed
from runs import get_config, run_name, checkpoints_dir

DATA = "sent"

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["mamba", "transformer", "ar"], required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--test-steps", type=int, default=None,
                    help="prova breve: N step, checkpoint in ../checkpoints/_test/")
args = parser.parse_args()

config = get_config(args.model, DATA)
if args.model == "mamba":
    from model import DiffuMamba
elif args.model == "ar":
    from model import MambaAR
else:
    from model_transformer import DiffuTransformer

# obiettivo: masked diffusion (DiffuMamba, DiffuTransformer) o autoregressivo (Mamba AR)
AR = args.model == "ar"

TEST = args.test_steps is not None
TOTAL_STEPS = args.test_steps if TEST else config.TOTAL_STEPS
RUN = run_name(args.model, DATA, args.seed)
OUTPUT_DIR = f"../checkpoints/_test/{RUN}" if TEST else checkpoints_dir(args.model, DATA, args.seed)

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


def compute_ar_loss(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    Loss autoregressiva (Mamba AR): la posizione i predice il token i+1.
    Cross-entropy solo dove il token da predire è reale (non padding).
    """
    pred    = logits[:, :-1]                    # (B, L-1, vocab_size)
    targets = input_ids[:, 1:]                  # (B, L-1)
    valid   = attention_mask[:, 1:] == 1        # (B, L-1)
    return F.cross_entropy(pred[valid], targets[valid])


# ============================================================
# 4. TRAINING LOOP
# ============================================================

def train():

    print(f"Run: {RUN} | modello: {args.model} | dati: {DATA} | seed: {args.seed} | "
          f"step: {TOTAL_STEPS} | output: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # tokenizer da HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)


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

    # seed: subito prima di creare il modello, così controlla l'inizializzazione
    # dei pesi e poi il rumore della diffusion (t e posizioni mascherate)
    set_seed(args.seed)

    # modello e ottimizzatore
    if args.model == "mamba":
        model = DiffuMamba().to(device)
    elif args.model == "ar":
        model = MambaAR(config).to(device)
    else:
        model = DiffuTransformer(config).to(device)
    print(f"Parametri: {model.count_params():,}")
    if TEST:
        print(f"[test] somma dei pesi iniziali: {sum(p.double().sum().item() for p in model.parameters()):.6f}")
 
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
    skipped_steps = 0   # step saltati dal GradScaler (gradienti inf/NaN in fp16)
    train_iter = iter(train_dl)
    t0 = time.time()
    meta = {"model_name": args.model, "data": DATA, "seed": args.seed}

    # salva checkpoint step 0 (modello non trainato)
    path = os.path.join(OUTPUT_DIR, "ckpt_step000000.pt")
    torch.save({"step": 0, "model": model.state_dict(), "loss": None, **meta}, path)
    print(f"[0] Checkpoint step 0 salvato: {path}")

    while step < TOTAL_STEPS:

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

        if AR:
            # Mamba AR: nessun rumore, il modello legge la sequenza originale
            if TEST and step < 3:
                print(f"[test] step {step}: somma input_ids={input_ids.sum().item()} | "
                      f"token da predire={(attention_mask[:, 1:] == 1).sum().item()}")
            with autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(input_ids)
                loss = compute_ar_loss(logits, input_ids, attention_mask)
        else:
            t = torch.rand(B, device=device)
            x_t, masked = mask_tokens(input_ids, t, attention_mask)

            if TEST and step < 3:
                # impronte per verificare che i dati siano gli stessi per tutti i modelli
                # e che t e maschere dipendano solo dal seed
                print(f"[test] step {step}: somma input_ids={input_ids.sum().item()} | "
                      f"t[:3]={[round(v, 6) for v in t[:3].tolist()]} | token mascherati={masked.sum().item()}")

            with autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x_t, t, attention_mask)
                loss = compute_loss(logits, input_ids, masked)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)   # salta lo step se i gradienti contengono inf/NaN
        scaler.update()          # in quel caso riduce la scala
        if scaler.get_scale() < scale_before:
            skipped_steps += 1

        step += 1
        running_loss += loss.item()

        if step % config.LOG_EVERY == 0:
            elapsed = time.time() - t0
            sec_per_step = elapsed / (step - start_step)
            eta_h = (TOTAL_STEPS - step) * sec_per_step / 3600
            print(f"[{step}/{TOTAL_STEPS}] loss={running_loss / config.LOG_EVERY:.4f} | {sec_per_step:.2f}s/step | "
                  f"ETA {eta_h:.1f}h | step saltati fp16: {skipped_steps}")
            running_loss = 0.0

        if (step <= 4000 and step % 400 == 0) or \
           (step > 4000 and step % 4000 == 0) or \
           step == TOTAL_STEPS:
            extra = {}
            if step == TOTAL_STEPS:
                # val loss finale (l'ultima epoca non arriva al controllo di fine epoca)
                val_loss = evaluate(model, val_dl, max_batches=5 if TEST else 200)
                model.train()
                print(f"[step {step} | fine training] VAL loss={val_loss:.4f} | step saltati fp16: {skipped_steps}")
                extra = {"val_loss": val_loss, "skipped_steps": skipped_steps}
            path = os.path.join(OUTPUT_DIR, f"ckpt_step{step:06d}.pt")
            torch.save({"step": step, "model": model.state_dict(), "loss": loss.item(), **meta, **extra}, path)
            print(f"[{step}] Checkpoint salvato: {path}")

    # segnala che il training è completo (run_all_seeds.sh salta le run con DONE)
    with open(os.path.join(OUTPUT_DIR, "DONE"), "w") as f:
        f.write(f"step={step}\nval_loss={val_loss:.4f}\nskipped_steps={skipped_steps}\n")


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
            if AR:
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(input_ids)
                    total += compute_ar_loss(logits, input_ids, attention_mask).item()
                n += 1
                continue
            t = torch.rand(B, device=device)
            x_t, masked = mask_tokens(input_ids, t, attention_mask)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x_t, t, attention_mask)
                total += compute_loss(logits, input_ids, masked).item()
            n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    train()
