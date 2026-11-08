#!/usr/bin/env bash
# Comprobación mínima del estudio de re-mapeo: las 11 condiciones en nguyen1 (1 x 1 semillas,
# población 30 x 5 generaciones). Unos segundos. Condición de la fase 2 del plan: 11 ok, 0 errores,
# tasa de re-mapeo 0 en todas las psge_pp.
# Uso: server/run_remap_smoke.sh [procesos]
source "$(dirname "$0")/common.sh"
N_JOBS="${1:-4}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$RESULTS_ROOT/remap_smoke/$STAMP"
RUNS="$RUNS_ROOT/remap_smoke/$STAMP"

activate_env
start_campaign remap_smoke "$@"
rc=0
$PYTHON -m examples.run_remap_study --phase smoke --output_dir "$OUT" --runs_dir "$RUNS" --n_jobs "$N_JOBS" \
  > "$META_DIR/remap_smoke.log" 2>&1 || rc=$?
$PYTHON -m examples.summarize_remap_pilot --root "$OUT" > "$META_DIR/summary.txt" 2>&1 || rc=1
finish_campaign "$rc"
tail -2 "$META_DIR/remap_smoke.log"
cat "$META_DIR/summary.txt"
exit "$rc"
