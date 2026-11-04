#!/usr/bin/env bash
# Mide el coste real de la confirmación en un PILOTO (nguyen5), con la configuración congelada:
# los 5 algoritmos, m = 20 y 100, sin ruido, 1 semilla de datos x 2 evolutivas. No toca los benchmarks
# confirmatorios. Al final imprime la estimación del estudio completo.
# Uso: server/run_timing.sh [procesos]
source "$(dirname "$0")/common.sh"
N_JOBS="${1:-$NCPU}"
OUT="$RESULTS_ROOT/timing_frozen"

activate_env; require_clean_tree; guard_commit "$OUT"
start_campaign timing "$@"
rc=0
$PYTHON -m examples.run_confirmation --benchmarks nguyen5 --m 20 100 --noise 0 --n_data_seeds 1 \
  --n_evo_seeds 2 --output_dir "$OUT" --n_jobs "$N_JOBS" > "$META_DIR/timing.log" 2>&1 || rc=$?
$PYTHON - "$OUT" > "$META_DIR/summary.txt" 2>&1 <<'PY' || rc=1
import glob, json, os, sys, numpy as np
out = sys.argv[1]
rows = [json.load(open(f)) for f in glob.glob(os.path.join(out, "*", "*", "*", "*.json"))]
print("Segundos de reloj por brazo y evaluaciones caso-individuo (mediana), nguyen5, 500 x 100:")
per_m = {}
for m in (20, 100):
    for alg in ("sge", "psge", "psge_kl", "sa_psge", "sa_psge_lam0"):
        r = [x for x in rows if x["m"] == m and x["algorithm"] == alg]
        if not r:
            continue
        w = np.median([x[a]["wall_sec"] for x in r for a in ("arm_S", "arm_Sz")])
        c = np.median([x[a]["n_case_evaluations"] for x in r for a in ("arm_S", "arm_Sz")])
        per_m.setdefault(m, []).append(w)
        print(f"  m={m:<4} {alg:<13} {w:7.1f} s   {c:12.0f} casos")
ncpu = os.cpu_count() or 1
total = sum(np.mean(v) * 2 * 6 * 2 * 5 * 100 for v in per_m.values())   # 2 brazos x 6 bench x 2 ruidos x 5 alg x 100
print(f"\nEstimación del estudio completo (si nguyen5 es representativo): {total / 3600:.0f} h de CPU "
      f"-> {total / 3600 / ncpu:.1f} h de reloj con {ncpu} procesos")
print("Los benchmarks de 2 y 5 variables y Keijzer-6 pueden costar distinto: tómalo como orden de magnitud.")
PY
finish_campaign "$rc"
echo "Resumen: $META_DIR/summary.txt"
exit "$rc"
