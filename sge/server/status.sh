#!/usr/bin/env bash
# Estado de las campañas: procesos vivos, ejecuciones completadas por configuración y errores.
# Uso: server/status.sh [esperadas_por_config]   (por defecto 30 = 10 datos x 3 evolutivas)
source "$(dirname "$0")/common.sh"
EXPECTED="${1:-30}"
GRID="$RESULTS_ROOT/hyperparam_search"

echo "== Procesos en marcha"
pgrep -af "examples\.(run_|calibrate|analyze)" | grep -v pgrep || echo "(ninguno)"

for p in nguyen5 nguyen1; do
  [ -d "$GRID/$p" ] || continue
  echo; echo "== $p (completadas / $EXPECTED)"
  for d in "$GRID/$p"/*/; do
    n=$(find "$d" -maxdepth 1 -name '*_d*_e*.json' | wc -l | tr -d ' ')
    printf "  %-46s %3s %s\n" "$(basename "$d")" "$n" "$([ "$n" -ge "$EXPECTED" ] && echo ok)"
  done
done

last="$(ls -dt "$RESULTS_ROOT"/meta/*/ 2>/dev/null | head -1)"
if [ -n "$last" ]; then
  echo; echo "== Última campaña: $last"
  cat "$last/run_info.txt"
  grep -Hc "ERROR\|Traceback" "$last"/*.log 2>/dev/null || true
fi
