#!/usr/bin/env bash
# Para un experimento lanzado con server/bg.sh, con todos sus procesos hijos.
# Uso: server/stop.sh [fichero.pid]   (sin argumento: el último lanzado)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
_env_RESULTS_ROOT="${RESULTS_ROOT:-}"
if [ -f "$HERE/local.env" ]; then source "$HERE/local.env"; fi
[ -n "$_env_RESULTS_ROOT" ] && RESULTS_ROOT="$_env_RESULTS_ROOT"
: "${RESULTS_ROOT:=$HOME/psge-results}"

pidfile="${1:-$(ls -t "$RESULTS_ROOT"/bg/*.pid 2>/dev/null | head -1)}"
if [ -z "$pidfile" ] || [ ! -f "$pidfile" ]; then echo "No hay ningún .pid en $RESULTS_ROOT/bg." >&2; exit 1; fi
pid="$(cat "$pidfile")"

if ! kill -0 "$pid" 2>/dev/null; then echo "$(basename "$pidfile" .pid) ya no está en marcha."; exit 0; fi
echo "Parando $(basename "$pidfile" .pid) (grupo $pid)..."
kill -TERM -- "-$pid"
for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "-$pid"; echo "Forzado con SIGKILL."; fi
echo "Parado. Las ejecuciones completadas se conservan; al relanzar se reutilizan."
