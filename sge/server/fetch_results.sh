#!/usr/bin/env bash
# SE EJECUTA EN EL PORTÁTIL. Trae los resultados del servidor a un directorio fuera de OneDrive.
# Uso: server/fetch_results.sh [--full]
#   por defecto: JSON por ejecución, resúmenes, logs y metadatos (pocos MB)
#   --full:      además los directorios de cada ejecución (gramática por generación, progress_report)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
_env_RESULTS_ROOT="${RESULTS_ROOT:-}"
if [ -f "$HERE/local.env" ]; then source "$HERE/local.env"; fi
[ -n "$_env_RESULTS_ROOT" ] && RESULTS_ROOT="$_env_RESULTS_ROOT"
: "${SERVER:?Define SERVER en server/local.env}"
: "${SERVER_RESULTS:=psge-results}"
: "${LOCAL_RESULTS:=$HOME/psge-results}"

case "$LOCAL_RESULTS" in
  *OneDrive*|*CloudStorage*) echo "ERROR: LOCAL_RESULTS ($LOCAL_RESULTS) está en OneDrive." >&2; exit 1;;
esac
mkdir -p "$LOCAL_RESULTS"

filters=()
if [ "${1:-}" != "--full" ]; then
  # Los directorios de cada ejecución se llaman <config>_d<datos>_e<evo>/: se dejan en el servidor.
  filters=(--exclude='*_d[0-9]*_e[0-9]*/' --exclude='*_sData*_sEvo*/')
fi

rsync -avz --prune-empty-dirs "${filters[@]}" "$SERVER:$SERVER_RESULTS/" "$LOCAL_RESULTS/"
echo "Resultados en $LOCAL_RESULTS"
