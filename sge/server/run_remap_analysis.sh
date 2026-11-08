#!/usr/bin/env bash
# Repite en el servidor el análisis prerregistrado del estudio de re-mapeo sobre lo que haya en
# RESULTS_ROOT/remap_confirmation. Informe en meta/remap_analysis_*/analysis.txt (+ JSON y CSV por celda).
# Uso: server/run_remap_analysis.sh
source "$(dirname "$0")/common.sh"
OUT="$RESULTS_ROOT/remap_confirmation"
activate_env; require_clean_tree
start_campaign remap_analysis "$@"
rc=0
$PYTHON -m examples.analyze_remap_study --root "$OUT" --split test --output "$META_DIR/remap_analysis" \
  > "$META_DIR/analysis.txt" 2>&1 || rc=1
finish_campaign "$rc"
cat "$META_DIR/analysis.txt"
exit "$rc"
