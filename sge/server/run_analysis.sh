#!/usr/bin/env bash
# Análisis confirmatorio EN EL SERVIDOR (opción C fijada de antemano). Lee los JSON de resumen de
# $RESULTS_ROOT/confirmation y deja el informe y las tablas en $RESULTS_ROOT/meta/analysis_<fecha>/.
# run_confirmation.sh ya lo ejecuta al terminar; este script sirve para repetirlo (p. ej., a mitad del
# estudio para revisar la integridad, o tras una actualización del análisis).
# Uso: server/run_analysis.sh
source "$(dirname "$0")/common.sh"
OUT="$RESULTS_ROOT/confirmation"
if [ ! -d "$OUT" ]; then echo "ERROR: no existe $OUT" >&2; exit 1; fi
activate_env; require_clean_tree
start_campaign analysis "$@"
rc=0
$PYTHON -m examples.analyze_confirmation --root "$OUT" --global_unit benchmark --alternative less \
  --output "$META_DIR/confirmation_analysis" > "$META_DIR/analysis.txt" 2>&1 || rc=$?
finish_campaign "$rc"
echo "Informe: $META_DIR/analysis.txt"
exit "$rc"
