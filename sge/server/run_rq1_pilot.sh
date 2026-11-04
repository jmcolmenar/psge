#!/usr/bin/env bash
# Piloto de RQ1: SGE frente a PSGE (nguyen1 + nguyen5), pop 200 x gen 50, solo validación.
# Uso: server/run_rq1_pilot.sh [procesos]   (N_DATA_SEEDS y N_EVO_SEEDS por entorno; 20 x 5 por defecto)
source "$(dirname "$0")/common.sh"
N_JOBS="${1:-$(( NCPU ))}"
N_DATA_SEEDS="${N_DATA_SEEDS:-20}"; N_EVO_SEEDS="${N_EVO_SEEDS:-5}"
OUT="$RESULTS_ROOT/rq1_pilot"

activate_env; require_clean_tree; guard_commit "$OUT"
start_campaign rq1_pilot "$@"

rc=0
$PYTHON -m examples.run_rq1_pilot --problems nguyen1 nguyen5 --n_jobs "$N_JOBS" \
  --n_data_seeds "$N_DATA_SEEDS" --n_evo_seeds "$N_EVO_SEEDS" \
  --output_dir "$OUT" > "$META_DIR/rq1_pilot.log" 2>&1 || rc=$?

$PYTHON -m examples.analyze_rq1 --root "$OUT" --problems nguyen1 nguyen5 \
  --output "$META_DIR/rq1_analysis.json" > "$META_DIR/analysis.txt" 2>&1 || rc=1
cp "$META_DIR/rq1_analysis.json" "$OUT/rq1_analysis.json" 2>/dev/null || true
finish_campaign "$rc"
echo "Análisis: $META_DIR/analysis.txt"
exit "$rc"
