# ============================================================
# train_blocks.py — training su blocchi da 512 token pre-tokenizzati
# ============================================================
# Uso (dalla cartella src):
#   python train_blocks.py --model mamba --seed 0         → checkpoints/blocks_seed0
#   python train_blocks.py --model transformer --seed 0   → checkpoints/blocks_transformer_seed0
#   python train_blocks.py --model ar --seed 0            → checkpoints/blocks_ar_seed0 (Mamba AR)
#   python train_blocks.py --model mamba --seed 0 --test-steps 100   → prova breve in checkpoints/_test/
#
# Il seed controlla inizializzazione e rumore della diffusion;
# l'ordine dei dati è fisso (vedi seeds.py).
# ============================================================

import os
import csv
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import time
from seeds import set_seed
from runs import get_config, run_name, checkpoints_dir

DATA = "blocks"

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


def compute_ar_loss(logits, input_ids, attention_mask):
    """
    Loss autoregressiva (Mamba AR): la posizione i predice il token i+1.
    Cross-entropy solo dove il token da predire è reale (sui blocchi lo sono tutti).
    """
    pred    = logits[:, :-1]
    targets = input_ids[:, 1:]
    valid   = attention_mask[:, 1:] == 1
    return F.cross_entropy(pred[valid], targets[valid])


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
            if AR:
                with autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(input_ids)
                    total += compute_ar_loss(logits, input_ids, attention_mask).item()
                n += 1
                continue
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
    print(f"Run: {RUN} | modello: {args.model} | dati: {DATA} | seed: {args.seed} | "
          f"step: {TOTAL_STEPS} | output: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # seed: subito prima di creare il modello, così controlla l'inizializzazione
    # dei pesi e poi il rumore della diffusion (t e posizioni mascherate)
    set_seed(args.seed)

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
    # GradScaler per il mixed precision (fp16): scala la loss per evitare underflow dei gradienti
    scaler = GradScaler('cuda')

    # stato del training (start_step > 0 servirebbe per riprendere da checkpoint)
    start_step = 0
    model.train()
    step = start_step
    epoch = 0
    running_loss = 0.0
    skipped_steps = 0   # step saltati dal GradScaler (gradienti inf/NaN in fp16)
    train_iter = iter(train_dl)
    t0 = time.time()
    meta = {"model_name": args.model, "data": DATA, "seed": args.seed}

    # checkpoint step 0 (modello non trainato)
    path = os.path.join(OUTPUT_DIR, "ckpt_step000000.pt")
    torch.save({"step": 0, "model": model.state_dict(), "loss": None, **meta}, path)
    print(f"[0] Checkpoint step 0 salvato: {path}")

    while step < TOTAL_STEPS:

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

            # forward in mixed precision: il modello è condizionato su t
            with autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x_t, t)
                loss = compute_loss(logits, input_ids, masked)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        # unscale prima del clipping, così la soglia 1.0 si applica ai gradienti reali
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)  # salta lo step se i gradienti contengono inf/NaN
        scaler.update()         # in quel caso riduce la scala
        if scaler.get_scale() < scale_before:
            skipped_steps += 1

        step += 1
        running_loss += loss.item()

        # log periodico: loss media sugli ultimi LOG_EVERY step + stima tempo residuo
        if step % config.LOG_EVERY == 0:
            elapsed = time.time() - t0
            sec_per_step = elapsed / (step - start_step)
            eta_h = (TOTAL_STEPS - step) * sec_per_step / 3600
            print(f"[{step}/{TOTAL_STEPS}] loss={running_loss / config.LOG_EVERY:.4f} "
                  f"| {sec_per_step:.2f}s/step | ETA {eta_h:.1f}h | step saltati fp16: {skipped_steps}")
            running_loss = 0.0

        # checkpoint fitti all'inizio (ogni 400 step fino a 4000), poi ogni 4000 step,
        # più quello finale: servono per il probing lungo la traiettoria di training
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


if __name__ == "__main__":
    train()