# DiffuMamba

Modello di linguaggio **diffusivo a mascheramento** (masked diffusion) con backbone **Mamba bidirezionale** al posto del Transformer, addestrato da zero su Wikipedia italiana. Il progetto studia, tramite **probing linguistico** sui checkpoint intermedi, quanto rapidamente e in che misura il modello apprende proprietà linguistiche rispetto a un Transformer di taglia comparabile.

> **Obiettivo di tesi:** valutare se un Diffusion Model con backbone Bidirectional Mamba apprende rappresentazioni linguistiche più rapidamente o meglio rispetto al Transformer, come passo intermedio verso un eventuale Flow Matching con backbone Mamba.

---

## Architettura

```
input_ids (B, L)          t ~ U(0,1)  (B,)
      │                         │
 Token Embedding          Linear(1 → D)   ← timestep embedding
      └──────────── + ──────────┘
                    │
     N × BiMambaBlock:   x + Mamba_fwd(LN(x)) + flip(Mamba_bwd(flip(LN(x))))
                    │
               LayerNorm
                    │
        LM head (weight tying con l'embedding) → logits (B, L, 31102)
```

- **BiMambaBlock** ([model.py](model.py)): due `mamba_ssm.Mamba` con pesi separati, uno sulla sequenza e uno sulla sequenza invertita (approccio di Vision Mamba); le uscite si sommano al residuo.
- **Processo diffusivo** ([train.py](train.py)): per ogni sequenza si campiona `t ~ U(0,1)` e ogni token (non di padding) viene sostituito con `[MASK]` con probabilità `t`. La loss è la cross-entropy **solo sui token mascherati** (come la MLM di BERT, ma con mask rate continuo).
- **Configurazione di default** ([config.py](config.py)): `HIDDEN_SIZE=496`, `NUM_LAYERS=8`, `d_state=16`, `expand=2`, `d_conv=4`, scelta per avere un numero di parametri confrontabile con il Transformer di riferimento (~40M). Per provare altre dimensioni c'è [count_params.py](count_params.py).
- **Tokenizer**: `dbmdz/bert-base-italian-cased` (vocabolario 31102, `[MASK]=104`, `[PAD]=0`).

## Requisiti

`mamba-ssm` e `causal-conv1d` richiedono **Linux + GPU NVIDIA con CUDA ≥ 11.6**: training e probing vanno eseguiti su un server GPU. Gli script di plot girano ovunque (anche nel `venv` Windows locale).

```bash
pip install torch transformers datasets tqdm
pip install mamba-ssm causal-conv1d          # solo su Linux + CUDA
pip install scikit-learn scipy pandas matplotlib
```

## Struttura del repository

| File | Scopo |
|---|---|
| [config.py](config.py) | Iperparametri per il training su **frasi** (128 token, batch 256) |
| [config_blocks.py](config_blocks.py) | Iperparametri per il training su **blocchi** (512 token, batch 64) |
| [model.py](model.py) | `BiMambaBlock` e `DiffuMamba` |
| [train.py](train.py) | Training su frasi (CSV / una frase per riga / blocchi di testo) |
| [prepare_blocks.py](prepare_blocks.py), [prepare_blocks_eval.py](prepare_blocks_eval.py) | Tokenizzano i documenti JSONL, li concatenano separati da `[SEP]` e li tagliano in blocchi da 512 token (dataset Arrow) |
| [train_blocks.py](train_blocks.py) | Training sui blocchi pre-tokenizzati |
| [probing.py](probing.py) | Probing su un singolo checkpoint (ultimo layer) |
| [probing_all_checkpoints.py](probing_all_checkpoints.py) | Probing sull'ultimo layer di tutti i checkpoint di una run |
| [probing_multilayer.py](probing_multilayer.py) | Probing su **ogni layer** di tutti i checkpoint |
| [evaluate_probing.py](evaluate_probing.py), [evaluate_all_checkpoints.py](evaluate_all_checkpoints.py), [evaluate_multilayer.py](evaluate_multilayer.py) | Calcolano la ρ di Spearman dalle predizioni salvate dagli script di probing |
| [plot_learning_curve.py](plot_learning_curve.py) | Curva ρ media vs step (con fasce per epoca e riferimento BERT) |
| [plot_heatmap.py](plot_heatmap.py) | Heatmap feature × checkpoint |
| [graph.py](graph.py) | Curva di apprendimento per layer |
| [plot_comparison.py](plot_comparison.py) | Confronto tra run: ρ per layer all'ultimo checkpoint |
| [test_tokenizer.py](test_tokenizer.py) | Controlla vocabolario e ID di `[MASK]` e `[PAD]` |
| `run*/` | Risultati (CSV, grafici, log) delle run già eseguite |

## Dati

La cartella `data/` e i checkpoint **non sono versionati**. Gli script si aspettano:

| Percorso | Contenuto |
|---|---|
| `data/train_shuffled.csv` | Frasi di training, colonna `text` (già mescolate: il DataLoader usa un `SequentialSampler`) |
| `data/eval_3.csv` | Frasi di validazione, colonna `text` |
| `data/training_blocks.json`, `data/eval_blocks.json` | Documenti in JSON Lines con campo `text`, input di `prepare_blocks*.py` |
| `data/train_probe.tsv`, `data/test_probe.tsv` | Dataset di probing: colonne `identifier`, `text`, più una colonna numerica per ogni feature linguistica (profiling UD) |

## Utilizzo

Gli script **non accettano argomenti da riga di comando**: i percorsi (checkpoint, input e output) si impostano nelle costanti in cima a ciascun file.

### 1. Training

**Su frasi (128 token):**
```bash
python train.py                 # legge config.py
```

**Su blocchi (512 token):**
```bash
python prepare_blocks.py        # → data/blocks_512
python prepare_blocks_eval.py   # → data/eval_blocks_512
python train_blocks.py          # legge config_blocks.py
```

I checkpoint vengono salvati in `OUTPUT_DIR` come `ckpt_stepXXXXXX.pt` (`{"step", "model", "loss"}`): allo step 0, ogni 400 step fino a 4000, poi ogni 4000 e all'ultimo step. Alla fine di ogni epoca viene stampata la loss di validazione. Il learning rate ha un warmup lineare di `WARMUP_STEPS` step e poi resta costante.

> ⚠️ `model.py` fa `import config`. `train_blocks.py` reindirizza quell'import a `config_blocks.py` (`sys.modules["config"]`) prima di importare il modello, quindi le dimensioni del modello vengono lette da `config_blocks.py`. Gli script di probing invece usano sempre `config.py`: se le dimensioni nei due file sono diverse, per caricare i checkpoint della Run 5 serve lo stesso reindirizzamento anche lì.

### 2. Probing

La metodologia segue Miaschi et al. / Brunato et al. (ACL 2020):

1. rappresentazione della frase = **mean pooling** dei token non di padding (Mamba non ha un token `[CLS]`), con input non mascherato e `t = 0`;
2. `MinMaxScaler` sulle rappresentazioni;
3. un **Ridge Regressor** (`alpha=1.0`) per ogni feature linguistica;
4. le predizioni grezze (`y_pred`, `y_true`) vengono salvate in un TSV per ogni feature.

```bash
python probing_multilayer.py     # imposta CHECKPOINTS_DIR e OUTPUT_BASE_DIR
python evaluate_multilayer.py    # → learning_curve_multilayer.csv, all_metrics_multilayer.csv
```

Varianti limitate all'ultimo layer: `probing.py` + `evaluate_probing.py` (un singolo checkpoint) e `probing_all_checkpoints.py` + `evaluate_all_checkpoints.py` (tutti i checkpoint). Gli script di probing saltano i checkpoint già elaborati, quindi un'esecuzione interrotta si può riprendere.

Nota: `probing_multilayer.py` estrae le rappresentazioni all'uscita di ciascun blocco, prima della `LayerNorm` finale; gli script sull'ultimo layer la applicano.

### 3. Grafici

```bash
python plot_learning_curve.py
python plot_heatmap.py
python graph.py
python plot_comparison.py        # configura la lista RUNS con i CSV da confrontare
```

Come riferimento viene usata la ρ ≈ 0.70 di BERT.

## Run e risultati

| Cartella | Contenuto |
|---|---|
| [run0_results/](run0_results/) | Prima run: log di training, metriche di probing sull'ultimo layer, curva di apprendimento, heatmap |
| [run_mamba_sentences/](run_mamba_sentences/) | Run 3 (frasi, backend `mamba_ssm`): probing sull'ultimo layer di tutti i checkpoint |
| [run4/](run4/) | Run 4 (frasi, 128 token, 117k step ≈ 3 epoche): probing multilayer |
| Run 5 | Blocchi da 512 token (`config_blocks.py`, `train_blocks.py`) |

In Run 4, all'ultimo checkpoint, la ρ media arriva a **0.72 al layer 4** e i layer centrali (2–6) superano il riferimento BERT di 0.70. I primi e gli ultimi layer restano un po' sotto.

---

## Piano di lavoro

**Fase 1: probing linguistico su BERT.** Ridge Regressor + MinMaxScaler sulle feature UD (8000 train / 2000 test), rappresentazione tramite `[CLS]`, 63 feature sull'ultimo layer. Rif.: ACL 2020 (Contextual Embeddings, Linguistic Profiling).

**Fase 2: studio teorico, Diffusion e Flow Matching.** Come si addestrano i modelli di linguaggio basati su diffusione; Flow Matching come alternativa.

**Fase 3: dataset e ambiente.** Dataset Wikipedia (frasi singole o blocchi di frasi); filtro delle frasi con meno di 6 token; ambiente HuggingFace con un modello base equivalente a un BERT piccolo.

**Fase 4: Diffusion + Bidirectional Mamba.** Sostituzione della backbone Transformer con Bi-Mamba, training da zero partendo da un modello piccolo, checkpoint a granularità variabile (frequenti all'inizio, più radi alla fine).

**Fase 5: analisi comparativa.** Probing ai vari checkpoint e confronto Bi-Mamba vs Transformer (~40M parametri) allo stesso punto di addestramento: quale backbone apprende prima le strutture linguistiche?
