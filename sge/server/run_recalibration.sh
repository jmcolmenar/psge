#!/usr/bin/env bash
# Recalibración con profundidad 12 y presupuesto de confirmación (pop 500 x gen 100), en los pilotos.
#  (a) KL por actualización de PSGE (lf = 0.01) -> delta de PSGE-KL.
#      Regla fijada de antemano: delta = media de los P25 de nguyen1 y nguyen5, a 2 cifras significativas.
#  (b) KL por actualización de SA-PSGE con la configuración elegida (lambda 0.25, beta 2, alpha 0.01)
#      para eta en {0.04, 0.08, 0.16}, frente a PSGE. Solo se informa: eta = 0.08 queda congelado
#      salvo que su KL difiera de la de PSGE en más de un factor 3 (entonces se revisa).
# También deja los tiempos por ejecución a 500 x 100 para estimar el coste de la confirmación.
# Uso: server/run_recalibration.sh [procesos_por_problema]
source "$(dirname "$0")/common.sh"
N_JOBS="${1:-$N_JOBS}"
POP="${RECALIB_POP:-500}"; GEN="${RECALIB_GEN:-100}"   # solo para pruebas; los valores reales son 500 x 100
OUT="$RESULTS_ROOT/recalib_${POP}x${GEN}"

activate_env; require_clean_tree; guard_commit "$OUT"
start_campaign recalibration "$@"

run_problem() {
  local p="$1"
  echo "=== KL de PSGE ($p)"
  $PYTHON -m examples.run_hyperparameter_search --problem "$p" --skip_phase2 --n_calib_seeds 5 \
    --popsize "$POP" --generations "$GEN" --output_dir "$OUT/kl/$p" || return 1
  echo "=== KL de SA-PSGE ($p)"
  $PYTHON -m examples.calibrate_sa_eta --problem "$p" --popsize "$POP" --generations "$GEN" --n_seeds 5 \
    --etas 0.04 0.08 0.16 --alpha 0.01 --sa_lambda 0.25 --sa_beta 2 --n_jobs "$N_JOBS" \
    --output_dir "$OUT/eta" || return 1
}

pids=()
for p in nguyen5 nguyen1; do
  run_problem "$p" > "$META_DIR/$p.log" 2>&1 &
  pids+=($!)
done
rc=0; wait_all "${pids[@]}" || rc=$?

$PYTHON - "$OUT" > "$META_DIR/summary.txt" 2>&1 <<'PY' || rc=1
import json, os, sys, numpy as np
out = sys.argv[1]
p25 = {}
print("(a) KL por actualización de PSGE (presupuesto de la recalibración)")
for p in ("nguyen5", "nguyen1"):
    with open(os.path.join(out, "kl", p, "kl_calibration_summary.json")) as fh:
        c = json.load(fh)
    p25[p] = c["percentiles"]["P25"]
    print(f"  {p}: " + "  ".join(f"{k}={v:.2e}" for k, v in c["percentiles"].items()))
delta = float(f"{np.mean(list(p25.values())):.2g}")
print(f"  -> KL_DELTA = media de P25 = {delta:.2g}   (antes, a 200 x 50 y profundidad 10: 6e-05)")
print("\n(b) SA-PSGE (alpha 0.01, lambda 0.25, beta 2) frente a PSGE")
times = []
for p in ("nguyen5", "nguyen1"):
    with open(os.path.join(out, "eta", p, "calib_alpha0.01.json")) as fh:
        c = json.load(fh)
    for r in c["rows"]:
        flag = ""
        if r["name"] == "sa_eta0.08" and not (1 / 3 <= r["ratio"] <= 3):
            flag = "  <-- fuera del factor 3: REVISAR eta"
        print(f"  {p} {r['name']:<12} KL mediana={r['median_kl']:.2e}  ratio vs PSGE={r['ratio']:.2f}  "
              f"entropía final={r['final_entropy']:.3f}{flag}")
    times += [(r["name"], r["time_sec"]) for r in c["runs"]]
t = {n: np.median([s for m, s in times if m == n]) for n, _ in times}
print("\nTiempo mediano de UNA ejecución (un brazo), m = 20, en segundos:")
for n, s in sorted(t.items()):
    print(f"  {n:<12} {s:.0f}")
PY
finish_campaign "$rc"
echo "Resumen: $META_DIR/summary.txt"
exit "$rc"
