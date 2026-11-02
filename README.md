# DiffuMamba

Codice del progetto DiffuMamba, con script per addestramento, valutazione e probing.

## Struttura

- `src/`: codice attuale
- `src/archive/`: script non più in uso
- `results/`: metriche e curve di apprendimento
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
