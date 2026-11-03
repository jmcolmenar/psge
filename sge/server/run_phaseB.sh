#!/usr/bin/env bash
# Fase B: lambda x beta con eta = 0.08, alpha = 0.01 (piloto, nguyen5 + nguyen1, solo validación).
# Uso: server/run_phaseB.sh [procesos_por_problema]
source "$(dirname "$0")/common.sh"
N_JOBS="${1:-$N_JOBS}"
GRID="$RESULTS_ROOT/hyperparam_search"

activate_env; require_clean_tree; guard_commit "$GRID"
start_campaign phaseB "$@"

pids=()
for p in nguyen5 nguyen1; do
  $PYTHON -m examples.run_hyperparameter_search --problem "$p" --skip_phase1 --n_jobs "$N_JOBS" \
    --n_data_seeds 10 --n_evo_seeds 3 --grid_eta 0.08 --grid_alpha 0.01 \
    --grid_lambda 0 0.25 0.5 1 --grid_beta 0.5 1 2 \
    --output_dir "$GRID/$p" > "$META_DIR/$p.log" 2>&1 &
  pids+=($!)
done
rc=0; wait_all "${pids[@]}" || rc=$?

$PYTHON -m examples.analyze_grid_paired --grid_root "$GRID" --problems nguyen5 nguyen1 \
  --output "$META_DIR/paired_analysis.json" > "$META_DIR/analysis.txt" 2>&1 || rc=1
cp "$META_DIR/paired_analysis.json" "$GRID/paired_analysis.json" 2>/dev/null || true
finish_campaign "$rc"
echo "Análisis: $META_DIR/analysis.txt"
exit "$rc"
