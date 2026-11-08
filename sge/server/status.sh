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

for d in "$RESULTS_ROOT"/remap_pilot "$RESULTS_ROOT"/remap_confirmation; do
  [ -d "$d" ] || continue
  n=$(find "$d" -name '*_d*_e*.json' | wc -l | tr -d ' ')
  echo; echo "== $(basename "$d"): $n ejecuciones emparejadas (piloto: 1320; confirmación: 44000)"
done

last="$(ls -dt "$RESULTS_ROOT"/meta/*/ 2>/dev/null | head -1)"
if [ -n "$last" ]; then
  echo; echo "== Última campaña: $last"
  cat "$last/run_info.txt"
  grep -Hc "ERROR\|Traceback" "$last"/*.log 2>/dev/null || true
fi

echo; echo "== Lanzados con server/bg.sh"
found=0
for f in $(ls -t "$RESULTS_ROOT"/bg/*.pid 2>/dev/null | head -5); do
  found=1; pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then st="en marcha"; else st="terminado"; fi
  printf "  %-40s %-10s %s\n" "$(basename "$f" .pid)" "$st" "${f%.pid}.out"
done
[ "$found" = 1 ] || echo "  (ninguno)"
