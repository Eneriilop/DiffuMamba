# ============================================================
# train.py — Run 5 (blocchi 512 token pre-tokenizzati)
# ============================================================

import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from model import DiffuMamba
import config_blocks as config
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ============================================================
# 1. DATASET
# ============================================================

class BlockDataset(Dataset):
    """
    Carica blocchi pre-tokenizzati da disco (formato Arrow HuggingFace).
    Ogni blocco è esattamente MAX_SEQ_LEN token, senza padding.
    """
    def __init__(self, path: str):
        print(f"Caricamento blocchi da {path}...")
        self.data = load_from_disk(path)
        print(f"Dataset pronto: {len(self.data)} blocchi x {config.MAX_SEQ_LEN} token")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)
        # nessun padding: attention_mask tutto 1
        mask = torch.ones(len(ids), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}


class TextDataset(Dataset):
    """
    Dataset per file CSV (usato per validation su eval_3.csv).
    """
    def __init__(self, path: str, tokenizer, fmt: str):
        self.tokenizer = tokenizer
        print("Caricamento testi...")
        if fmt == "csv":
            texts = []
            with open(path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in tqdm(reader, desc="Lettura CSV"):
                    text = row.get("text") or row.get("texr") or ""
                    texts.append(text.strip())
        else:
            raise ValueError(f"DATA_FORMAT non riconosciuto: {fmt}")

        self.texts = texts
        if config.MAX_SAMPLES is not None:
            self.texts = self.texts[:config.MAX_SAMPLES]
        print(f"Dataset pronto: {len(self.texts)} sequenze")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
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

def mask_tokens(input_ids, t, attention_mask):
    """
    Processo forward della diffusione discreta (masked diffusion):
    ogni token viene sostituito con [MASK] indipendentemente con probabilità t.
    t ha shape (B,): un livello di rumore per sequenza, in [0, 1).
    Ritorna la sequenza corrotta x_t e la maschera booleana dei token mascherati.
    """
    # (B,) -> (B, L): stessa probabilità di mascheramento per tutti i token della sequenza
    mask_prob = t.unsqueeze(1).expand_as(input_ids)
    masked = torch.bernoulli(mask_prob).bool()
    # non mascherare le posizioni di padding
    masked = masked & (attention_mask == 1)
    x_t = input_ids.clone()
    x_t[masked] = config.MASK_TOKEN_ID
    return x_t, masked


# ============================================================
# 3. LOSS
# ============================================================

def compute_loss(logits, input_ids, masked):
    """
    Cross-entropy calcolata solo sui token mascherati: il modello deve
    ricostruire i token originali a partire da x_t.
    """
    # caso limite (t molto piccolo): nessun token mascherato.
    # Si ritorna uno zero collegato al grafo, così backward() non fallisce.
    if not masked.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[masked], input_ids[masked])


# ============================================================
# 4. EVALUATE
# ============================================================

def evaluate(model, val_dl, max_batches=200):
    """
    Loss media di validation su al più max_batches batch.
    t è campionato a caso, quindi il valore è una stima
    (varia leggermente tra una chiamata e l'altra).
    """
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_dl:
            if n >= max_batches:
                break
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            B = input_ids.size(0)
            t = torch.rand(B, device=device)
            x_t, masked = mask_tokens(input_ids, t, attention_mask)
            with autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x_t, t)
                total += compute_loss(logits, input_ids, masked).item()
            n += 1
    return total / max(n, 1)


# ============================================================
# 5. TRAINING LOOP
# ============================================================

def train():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)

    # train: blocchi pre-tokenizzati da Arrow dataset
    train_ds = BlockDataset(config.DATA_PATH)
    # val: blocchi pre-tokenizzati (TextDataset non è più usato)
    val_ds = BlockDataset(config.VAL_PATH)

    num_workers = min(4, os.cpu_count() or 1)
    # SequentialSampler: i blocchi vengono letti nell'ordine su disco, senza shuffle
    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                          sampler=SequentialSampler(train_ds),
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                          shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    model = DiffuMamba().to(device)
    print(f"Parametri: {model.count_params():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=0.01)
    # GradScaler per il mixed precision (fp16): scala la loss per evitare underflow dei gradienti
    scaler = GradScaler('cuda')

    # stato del training (start_step > 0 servirebbe per riprendere da checkpoint)
    start_step = 0
    model.train()
    step = start_step
    epoch = 0
    running_loss = 0.0
    train_iter = iter(train_dl)
    t0 = time.time()

    # checkpoint step 0 (modello non trainato)
    path = os.path.join(config.OUTPUT_DIR, "ckpt_step000000.pt")
    torch.save({"step": 0, "model": model.state_dict(), "loss": None}, path)
    print(f"[0] Checkpoint step 0 salvato: {path}")

    while step < config.TOTAL_STEPS:

        # il loop è guidato dagli step, non dalle epoche: quando il DataLoader
        # si esaurisce si fa una validation e si ricomincia dal primo blocco
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            val_loss = evaluate(model, val_dl)
            print(f"[step {step} | fine epoca {epoch}] VAL loss={val_loss:.4f}")
            model.train()
            train_iter = iter(train_dl)
            continue

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        B = input_ids.size(0)

        # warmup lineare
        if step < config.WARMUP_STEPS:
            for pg in optimizer.param_groups:
                pg["lr"] = config.LR * (step + 1) / config.WARMUP_STEPS

        # livello di rumore t ~ U(0,1) per ogni sequenza, poi corruzione
        t = torch.rand(B, device=device)
        x_t, masked = mask_tokens(input_ids, t, attention_mask)

        # forward in mixed precision: il modello è condizionato su t
        with autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(x_t, t)
            loss = compute_loss(logits, input_ids, masked)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # unscale prima del clipping, così la soglia 1.0 si applica ai gradienti reali
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)  # salta lo step se i gradienti contengono inf/NaN
        scaler.update()

        step += 1
        running_loss += loss.item()

        # log periodico: loss media sugli ultimi LOG_EVERY step + stima tempo residuo
        if step % config.LOG_EVERY == 0:
            elapsed = time.time() - t0
            sec_per_step = elapsed / (step - start_step)
            eta_h = (config.TOTAL_STEPS - step) * sec_per_step / 3600
            print(f"[{step}/{config.TOTAL_STEPS}] loss={running_loss / config.LOG_EVERY:.4f} "
                  f"| {sec_per_step:.2f}s/step | ETA {eta_h:.1f}h")
            running_loss = 0.0

        # checkpoint fitti all'inizio (ogni 400 step fino a 4000), poi ogni 4000 step,
        # più quello finale: servono per il probing lungo la traiettoria di training
        if (step <= 4000 and step % 400 == 0) or \
           (step > 4000 and step % 4000 == 0) or \
           step == config.TOTAL_STEPS:
            path = os.path.join(config.OUTPUT_DIR, f"ckpt_step{step:06d}.pt")
            torch.save({"step": step, "model": model.state_dict(), "loss": loss.item()}, path)
            print(f"[{step}] Checkpoint salvato: {path}")


if __name__ == "__main__":
    train()