#!/usr/bin/env bash
# Confirmación del estudio de re-mapeo (fase 7 del plan; aquí se abre el test). Solo arranca si
# parameters/remap_study.yml tiene FROZEN: true (tras el piloto y el prerregistro) y el código está en un
# commit sin cambios. Se puede interrumpir y relanzar: reutiliza lo hecho.
# 10 benchmarks x 4 regímenes x 11 condiciones x 20 x 5 semillas = 44 000 ejecuciones emparejadas.
# Al terminar ejecuta el análisis prerregistrado (examples/analyze_remap_study.py); se puede repetir con
# server/run_remap_analysis.sh.
# Uso: server/run_remap_confirmation.sh [procesos]
source "$(dirname "$0")/common.sh"
N_JOBS="${1:-$NCPU}"
OUT="$RESULTS_ROOT/remap_confirmation"      # JSON de resumen con trayectorias (~2 GB)
RUNS="$RUNS_ROOT/remap_confirmation"

activate_env; require_clean_tree; guard_commit "$OUT"; require_free_gb "$OUT" "${MIN_FREE_GB:-5}"
start_campaign remap_confirmation "$@"
cp parameters/remap_study.yml "$META_DIR/remap_study.yml"
rc=0
$PYTHON -m examples.run_remap_study --phase confirm --output_dir "$OUT" --runs_dir "$RUNS" --n_jobs "$N_JOBS" \
  > "$META_DIR/remap_confirmation.log" 2>&1 || rc=$?
$PYTHON -m examples.analyze_remap_study --root "$OUT" --split test --output "$META_DIR/remap_analysis" \
  > "$META_DIR/analysis.txt" 2>&1 || rc=1
finish_campaign "$rc"
echo "Análisis: $META_DIR/analysis.txt"
exit "$rc"
