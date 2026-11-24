# DiffuMamba

Codice del progetto DiffuMamba, con script per addestramento, valutazione e probing.

## Struttura

- `src/`: codice attuale
- `src/archive/`: script non più in uso
- `results/`: risultati della valutazione
  - `all_metrics/`: ρ per feature × layer × step (`all_metrics_<run>.csv`, medie `*_mean.csv`)
  - `learning_curves/`: ρ media per layer × step (`learning_curve_<run>.csv`, medie `*_mean.csv`)
  - `bert/`: CSV di probing di BERT medium (3 seed, forniti da terzi)
  - `analysis/`: analisi per layer e gruppi di feature
  - `comparisons/{sentences,blocks}/`: grafici di confronto, una sottocartella per tipo
  - `old/`: risultati single-seed precedenti
- `probing_results/`: risultati del probing inclusi nella repo
- `v0/`: versione precedente, non più mantenuta

## Installazione

Testato con Python 3.10.10.

```bash
python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## File non inclusi

Per limiti di dimensione, non sono inclusi:

- `data/`: dataset e blocchi preprocessati
- `checkpoints/`: pesi dei modelli
- `logs/`: log di esecuzione
- `probing_results/**/*.tsv`: risultati intermedi del probing

Le configurazioni si trovano in `src/config.py` e `src/config_blocks.py`.
