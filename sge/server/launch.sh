#!/usr/bin/env bash
# Lanzador genérico: ejecuta cualquier comando con las mismas comprobaciones y metadatos.
# Uso: server/launch.sh NOMBRE comando args...
# En el comando, usa $RESULTS_ROOT entre comillas simples para que se expanda aquí:
#   server/launch.sh rq1_pilot python -m examples.run_pilot_campaign \
#       --problems nguyen1 nguyen5 --algorithms sge psge --output_dir '$RESULTS_ROOT/rq1_pilot'
source "$(dirname "$0")/common.sh"
if [ $# -lt 2 ]; then echo "Uso: $0 NOMBRE comando args..." >&2; exit 2; fi
NAME="$1"; shift
activate_env; require_clean_tree
start_campaign "$NAME" "$@"
args=(); for a in "$@"; do args+=("${a//\$RESULTS_ROOT/$RESULTS_ROOT}"); done
[ "${args[0]}" = "python" ] && args[0]="$PYTHON"
rc=0; "${args[@]}" > "$META_DIR/$NAME.log" 2>&1 || rc=$?
finish_campaign "$rc"
exit "$rc"
