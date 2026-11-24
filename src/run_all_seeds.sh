#!/usr/bin/env bash
# ============================================================
# run_all_seeds.sh — training, probing e valutazione per tutti i seed
# ============================================================
# Per ogni seed (seeds.py) e per ognuno dei modelli (DiffuMamba, DiffuTransformer,
# Mamba AR su frasi e su blocchi), in sequenza:
#   1. training        (saltato se checkpoints/<run>/DONE esiste)
#   2. probing con LN  (+ senza LN per DiffuMamba); i checkpoint già fatti vengono saltati
#   3. valutazione     (Spearman ρ → results/learning_curves/learning_curve_*.csv, results/all_metrics/all_metrics_*.csv)
# Alla fine: media sui seed (aggregate_seeds.py) e grafici.
#
# Ordine per seed: prima i 4 modelli con il seed 0, poi seed 1, poi seed 2,
# così dopo il primo giro c'è già un set completo e confrontabile.
# Si può rilanciare dopo un'interruzione: riprende da dove era arrivato
# (una run di training interrotta riparte da capo).
#
# Uso (dalla cartella src), staccato da VS Code e dal terminale:
#   setsid nohup bash run_all_seeds.sh > ../logs/run_all_seeds.log 2>&1 < /dev/null &
# Avanzamento: tail -f ../logs/run_all_seeds.log
# ============================================================

set -u
cd "$(dirname "$0")"
source ../env/bin/activate
mkdir -p ../logs

SEEDS=$(python -c "from seeds import SEEDS; print(*SEEDS)")
# Mamba AR (ar) aggiunto per l'ablation sull'obiettivo; le run già complete vengono saltate
COMBOS="sent:mamba sent:transformer sent:ar blocks:mamba blocks:transformer blocks:ar"

log() { echo "$(date '+%F %T') $*"; }

for seed in $SEEDS; do
  for combo in $COMBOS; do
    data=${combo%%:*}
    model=${combo##*:}
    run=$(python -c "from runs import run_name; print(run_name('$model', '$data', $seed))")
    script=$([ "$data" = sent ] && echo train.py || echo train_blocks.py)

    # 1. training
    if [ -f "../checkpoints/$run/DONE" ]; then
      log "TRAIN    $run: già completato, skip"
    else
      log "TRAIN    $run: inizio"
      if ! python -u "$script" --model "$model" --seed "$seed" > "../logs/train_$run.log" 2>&1; then
        log "ERRORE   training $run (vedi logs/train_$run.log): passo alla run successiva"
        continue
      fi
      log "TRAIN    $run: fine ($(tr '\n' ' ' < "../checkpoints/$run/DONE"))"
    fi

    # 2-3. probing e valutazione (con LN; per DiffuMamba anche senza LN)
    variants="ln"
    [ "$model" = mamba ] && variants="ln noln"
    for v in $variants; do
      flag=""; suffix=""
      [ "$v" = noln ] && { flag="--no-ln"; suffix="_noln"; }
      log "PROBING  $run$suffix: inizio"
      if python -u probing_multilayer.py --model "$model" --data "$data" --seed "$seed" $flag \
           > "../logs/probing_$run$suffix.log" 2>&1 \
         && python evaluate_multilayer.py --model "$model" --data "$data" --seed "$seed" $flag \
           > "../logs/evaluate_$run$suffix.log" 2>&1; then
        log "PROBING  $run$suffix: fine"
      else
        log "ERRORE   probing/valutazione $run$suffix (vedi logs/probing_$run$suffix.log e logs/evaluate_$run$suffix.log)"
      fi
    done
  done
done

log "AGGREGAZIONE e grafici"
python aggregate_seeds.py > ../logs/aggregate_seeds.log 2>&1 || log "ERRORE   aggregate_seeds.py"
python plot_comparison.py > ../logs/plot_comparison.log 2>&1 || log "ERRORE   plot_comparison.py"
python plot_groups.py     > ../logs/plot_groups.log 2>&1     || log "ERRORE   plot_groups.py"
log "FINE"
