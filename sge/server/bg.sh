#!/usr/bin/env bash
# Lanza un script de server/ en segundo plano, desligado del terminal: sigue corriendo al cerrar ssh.
# Uso: server/bg.sh SCRIPT [args...]
#   server/bg.sh run_phaseB.sh 16
#   server/bg.sh run_phaseC.sh 0.5 1 16
#   server/bg.sh launch.sh rq1_pilot python -m examples.run_pilot_campaign ...
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$HERE/local.env" ]; then source "$HERE/local.env"; fi
: "${RESULTS_ROOT:=$HOME/psge-results}"

if [ $# -lt 1 ]; then echo "Uso: $0 SCRIPT [args...]   (p. ej. $0 run_phaseB.sh 16)" >&2; exit 2; fi
script="$(basename "$1")"; shift
if [ ! -x "$HERE/$script" ]; then echo "ERROR: no existe server/$script o no es ejecutable." >&2; exit 2; fi

mkdir -p "$RESULTS_ROOT/bg"
tag="${script%.sh}"; if [ "$script" = "launch.sh" ] && [ $# -ge 1 ]; then tag="$1"; fi
name="${tag}_$(date +%Y%m%d-%H%M%S)"
out="$RESULTS_ROOT/bg/$name.out"

# setsid: nueva sesión, sin terminal de control, así que cerrar ssh no le manda SIGHUP.
# nohup y </dev/null: por si algo intenta leer o escribir en el terminal.
setsid nohup "$HERE/$script" "$@" > "$out" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$RESULTS_ROOT/bg/$name.pid"

# Si falla al arrancar (cambios sin commit, conda, argumentos), mejor saberlo ahora.
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "El proceso terminó enseguida. Salida:" >&2
  cat "$out" >&2
  exit 1
fi

cat <<MSG
Lanzado en segundo plano: $script $*
  PID (grupo):  $pid
  Salida:       $out
Ya puedes cerrar el terminal.
  Seguir:       tail -f $out
  Progreso:     server/status.sh
  Parar:        server/stop.sh            (el último lanzado)
MSG
