#!/usr/bin/env bash
# Fase C: sensibilidad a K y tau (un factor cada vez) con el lambda/beta elegido en la fase B.
# Uso: server/run_phaseC.sh LAMBDA BETA [procesos_por_problema]
source "$(dirname "$0")/common.sh"
if [ $# -lt 2 ]; then echo "Uso: $0 LAMBDA BETA [procesos_por_problema]" >&2; exit 2; fi
LAM="$1"; BET="$2"; N_JOBS="${3:-$N_JOBS}"
GRID="$RESULTS_ROOT/hyperparam_search"
N_DATA_SEEDS="${N_DATA_SEEDS:-10}"   # semillas de datos 400.. (variable de entorno); las ya hechas se reutilizan

activate_env; require_clean_tree; guard_commit "$GRID"
start_campaign "phaseC_lam${LAM}_bet${BET}" "$@"

run_problem() {
  local p="$1" kt
  for kt in "2 1" "5 1" "3 5" "3 10"; do
    set -- $kt
    echo "=== K=$1 tau=$2"
    $PYTHON -m examples.run_hyperparameter_search --problem "$p" --skip_phase1 --n_jobs "$N_JOBS" \
      --n_data_seeds "$N_DATA_SEEDS" --n_evo_seeds 3 --grid_eta 0.08 --grid_alpha 0.01 \
      --grid_lambda "$LAM" --grid_beta "$BET" --sa_k_views "$1" --sa_tau "$2" \
      --output_dir "$GRID/$p" || return 1
  done
}

pids=()
for p in nguyen5 nguyen1; do
  run_problem "$p" > "$META_DIR/$p.log" 2>&1 &
  pids+=($!)
done
rc=0; wait_all "${pids[@]}" || rc=$?

$PYTHON -m examples.analyze_grid_paired --grid_root "$GRID" --problems nguyen5 nguyen1 --n_data_seeds "$N_DATA_SEEDS" \
  --output "$META_DIR/paired_analysis.json" > "$META_DIR/analysis.txt" 2>&1 || rc=1
cp "$META_DIR/paired_analysis.json" "$GRID/paired_analysis.json" 2>/dev/null || true
finish_campaign "$rc"
echo "Análisis: $META_DIR/analysis.txt"
exit "$rc"
