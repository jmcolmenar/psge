#!/usr/bin/env python3
"""
Resumen DESCRIPTIVO del piloto del estudio de re-mapeo (fase 5 del plan). No contrasta hipótesis.

Responde a las preguntas del piloto, por benchmark x régimen x condición:
  - ¿Baja la entropía final de forma gradual con lambda? ¿Colapsa con lambda = 0,1?
  - ¿Se disparan los modelos no evaluables (regla de inválidos)?
  - ¿La tasa de re-mapeo es ~0 en PSGE-PP y crece con lambda en PSGE? ¿Cuántos codones no se pudieron conservar?
  - ¿Cambia el tamaño efectivo de la mutación entre PSGE y PSGE-PP (fracción de mutaciones que cambian de regla)?
  - Coste por brazo y dispersión entre semillas de datos de log(APG+ATG), para decidir el número de réplicas.

Uso (desde sge/): python -m examples.summarize_remap_pilot --root ~/psge-results/remap_pilot [--csv salida.csv]
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from examples.invalid_rule import agg_evo, clean, is_invalid   # noqa: E402


def cond_key(name):
    if name == "sgef":
        return (0, 0.0)
    fam = 2 if name.startswith("psge_pp_") else 1
    return (fam, float(name.rsplit("lf", 1)[1]))


def load(root):
    rows = []
    for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        if os.sep + "_runs" + os.sep in path:
            continue
        try:
            with open(path) as fh:
                r = json.load(fh)
        except Exception:
            continue
        if "condition" in r:
            rows.append(r)
    return rows


def summarize(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["benchmark"], f"m{r['m']}_noise{r['noise']:g}", r["condition"])].append(r)
    out = []
    for (bench, regime, cond), rs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], cond_key(kv[0][2]))):
        by_data = defaultdict(list)
        for r in rs:
            by_data[r["seed_data"]].append(r)
        stab = [agg_evo([clean(r["apg_plus_atg"]) for r in v]) for v in by_data.values()]
        logs = [math.log(s) for s in stab if math.isfinite(s) and s > 0]
        arms = [r[a] for r in rs for a in ("arm_S", "arm_Sz")]
        trajs = [a["trajectory"] for a in arms]
        mut = [t["mut_rule_changes"][-1] / t["mut_codons"][-1] for t in trajs if t["mut_codons"][-1] > 0]
        pp_codons = sum(sum(t["pp_codons"]) for t in trajs)
        pp_fail = sum(sum(t["pp_failures"]) for t in trajs)
        out.append({
            "benchmark": bench, "regime": regime, "condition": cond, "n_runs": len(rs), "n_data_seeds": len(by_data),
            "median_apg_atg": float(np.median(stab)),
            "sd_log_apg_atg_between_data": float(np.std(logs, ddof=1)) if len(logs) > 1 else float("nan"),
            "invalid_rate_apg_atg": float(np.mean([is_invalid(r["apg_plus_atg"]) for r in rs])),
            "invalid_rate_nrmse_val": float(np.mean([is_invalid(r.get("nrmse_val_fS")) for r in rs])),
            "median_nrmse_val_fS": float(np.median([clean(r.get("nrmse_val_fS")) for r in rs])),
            "median_entropy_final": float(np.median([r["entropy_S"] for r in rs])),
            "min_entropy_final": float(np.min([r["entropy_S"] for r in rs])),
            "median_max_remap_rate": float(np.median([max(t["remap_rate"]) for t in trajs])),
            "median_mean_remap_rate": float(np.median([np.mean(t["remap_rate"]) for t in trajs])),
            "pp_failure_rate": pp_fail / pp_codons if pp_codons else 0.0,
            "median_frac_mutations_changing_rule": float(np.median(mut)) if mut else float("nan"),
            "median_final_invalid_frac_pop": float(np.median([t["invalid_frac"][-1] for t in trajs])),
            "median_final_unique_pct": float(np.median([t["unique_pct"][-1] for t in trajs])),
            "median_final_mean_depth": float(np.median([t["mean_depth"][-1] for t in trajs])),
            "median_wall_sec_per_arm": float(np.median([a["wall_sec"] for a in arms])),
            "median_case_evals_per_arm": float(np.median([a["n_case_evaluations"] for a in arms])),
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--csv", default=None)
    a = p.parse_args()
    rows = load(a.root)
    if not rows:
        raise SystemExit(f"no hay resultados en {a.root}")
    table = summarize(rows)
    cols = [("condition", "%-16s"), ("n_runs", "%5d"), ("median_apg_atg", "%9.4g"),
            ("sd_log_apg_atg_between_data", "%7.3f"), ("invalid_rate_apg_atg", "%6.3f"),
            ("median_entropy_final", "%6.3f"), ("min_entropy_final", "%6.3f"),
            ("median_max_remap_rate", "%6.3f"), ("pp_failure_rate", "%8.2e"),
            ("median_frac_mutations_changing_rule", "%6.3f"), ("median_wall_sec_per_arm", "%7.1f")]
    header = ["cond", "runs", "APG+ATG", "sdlog", "inval", "H_med", "H_min", "remap", "pp_fail", "mut_rule", "s/arm"]
    current = None
    for row in table:
        if (row["benchmark"], row["regime"]) != current:
            current = (row["benchmark"], row["regime"])
            print(f"\n== {current[0]}  {current[1]}")
            print("  " + "  ".join(f"{h:>8s}" for h in header))
        print("  " + "  ".join((fmt % row[c]).rjust(8) for c, fmt in cols))
    total_arm = sum(r["median_wall_sec_per_arm"] * r["n_runs"] * 2 for r in table)
    print(f"\nTiempo total aproximado (medianas): {total_arm / 3600:.1f} h de CPU en {len(rows)} ejecuciones emparejadas")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print(f"CSV: {a.csv}")


if __name__ == "__main__":
    main()
