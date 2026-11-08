#!/usr/bin/env bash
# Piloto del estudio de re-mapeo (fase 5 del plan): nguyen1 y nguyen5, regímenes m20/ruido 5 % y
# m100/sin ruido, 11 condiciones, 10 x 3 semillas (3000.. / 30000..), 500 x 100. Solo VALIDACIÓN:
# el test no se genera. 1320 ejecuciones emparejadas, ~2 h con 32 procesos.
# Al terminar deja el resumen descriptivo en meta/remap_pilot_*/summary.txt (+ CSV).
# Uso: server/run_remap_pilot.sh [procesos]
source "$(dirname "$0")/common.sh"
N_JOBS="${1:-$NCPU}"
OUT="$RESULTS_ROOT/remap_pilot"             # JSON de resumen con trayectorias (~70 MB)
RUNS="$RUNS_ROOT/remap_pilot"               # solo parameters.json por brazo

activate_env; require_clean_tree; guard_commit "$OUT"
start_campaign remap_pilot "$@"
cp parameters/remap_study.yml "$META_DIR/remap_study.yml"
rc=0
$PYTHON -m examples.run_remap_study --phase pilot --output_dir "$OUT" --runs_dir "$RUNS" --n_jobs "$N_JOBS" \
  > "$META_DIR/remap_pilot.log" 2>&1 || rc=$?
$PYTHON -m examples.summarize_remap_pilot --root "$OUT" --csv "$META_DIR/pilot_summary.csv" \
  > "$META_DIR/summary.txt" 2>&1 || rc=1
finish_campaign "$rc"
echo "Resumen: $META_DIR/summary.txt"
exit "$rc"
