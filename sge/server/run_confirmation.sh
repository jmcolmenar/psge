#!/usr/bin/env bash
# Estudio confirmatorio GECCO completo (aquí se abre el test). Exige parameters/gecco_frozen.yml con
# KL_DELTA numérico y código sin cambios. Se puede interrumpir y relanzar: reutiliza lo hecho.
# Uso: server/run_confirmation.sh [procesos]
source "$(dirname "$0")/common.sh"
N_JOBS="${1:-$NCPU}"
OUT="$RESULTS_ROOT/confirmation"

activate_env; require_clean_tree; guard_commit "$OUT"
start_campaign confirmation "$@"
cp parameters/gecco_frozen.yml "$META_DIR/gecco_frozen.yml"
rc=0
$PYTHON -m examples.run_confirmation --output_dir "$OUT" --n_jobs "$N_JOBS" \
  > "$META_DIR/confirmation.log" 2>&1 || rc=$?
# Análisis fijado de antemano (opción C: principal = benchmark unilateral; robustez = celdas bilateral)
$PYTHON -m examples.analyze_confirmation --root "$OUT" --global_unit benchmark --alternative less \
  --output "$META_DIR/confirmation_analysis" > "$META_DIR/analysis.txt" 2>&1 || rc=1
finish_campaign "$rc"
echo "Análisis: $META_DIR/analysis.txt"
exit "$rc"
