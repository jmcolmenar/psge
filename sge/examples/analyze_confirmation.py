#!/usr/bin/env python3
"""
Análisis del estudio confirmatorio GECCO. FIJADO ANTES DE VER RESULTADOS (ver CHANGELOG_SA_PSGE.md).

Datos: <root>/<benchmark>/m<m>_noise<r>/<algoritmo>/<algoritmo>_d<datos>_e<evo>.json (run_confirmation.py).

Reglas previas
--------------
- Celda = benchmark x régimen (m, ruido). Réplica = semilla de datos. Se usan solo las semillas de datos
  COMPLETAS (todos los algoritmos con todas sus semillas evolutivas); las incompletas se listan.
- Dentro de cada semilla de datos, las semillas evolutivas se agregan con la mediana; los valores
  inválidos cuentan como +inf y nunca se descartan (examples/invalid_rule.py).
- Métricas: estabilidad primaria APG + ATG_hat (también APG y ATG); precisión NRMSE-test de f_S
  (modelo entrenado con las m observaciones; f_{S+z} como secundaria). Menor es mejor en todas.
- Efecto relativo de una celda (EMPAREJADO por semilla de datos): exp(mediana_i log((t_i + e) / (c_i + e))) - 1,
  con e = 1e-3 x mediana del control (evita log(0) cuando ambos modelos son exactos); IC BCa de la mediana de
  los log-ratios. Inválidos: inf frente a finito = +/-inf; ambos inf = 0.
- Umbrales (plan): mejora relevante de estabilidad = efecto <= -5 % e IC95 por debajo de 0;
  no inferioridad en NRMSE = extremo superior del IC90 bilateral (95 % unilateral) <= +2 %.
  Clasificación de cada celda: "éxito" (mejora relevante y no inferior), "trade-off" (mejora pero no
  no-inferior: sobre-regularización), "peor" (IC95 de estabilidad por encima de 0), "nulo" (resto).
- Nivel global: puntuación por bloque = media sobre las celdas del bloque de log(mediana de la celda).
  DECISIÓN FIJADA (2026-11-05): bloque = celda (24 bloques), pruebas bilaterales. El plan decía bloque =
  benchmark, pero con 6 bloques ninguna comparación global podía ser significativa bajo Holm.
  --global_unit benchmark queda como análisis descriptivo adicional.
  Friedman/Iman-Davenport entre psge, psge_kl, sa_psge y sa_psge_lam0. Comparaciones planificadas
  (Wilcoxon exacto sobre los bloques, log-ratio tratamiento/control): primarias SA-PSGE vs PSGE y
  SA-PSGE vs PSGE-KL con Holm; secundaria SA-PSGE vs SA-PSGE(λ = 0). --alternative solo afecta a la
  estabilidad ("less" = H1 de que SA-PSGE reduce APG+ATG; en RQ1, "greater" = PSGE la aumenta); NRMSE
  siempre bilateral.
- RQ1: PSGE frente a SGE con la misma estructura; Spearman JS vs APG+ATG por celda (asociación).

Salida: informe de texto por pantalla, <output>.json y tablas CSV (celdas, Pareto, coste).

Uso (desde sge/):
  python -m examples.analyze_confirmation --root ~/psge-results/confirmation --output informe
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from examples.invalid_rule import agg_evo, count_invalid, paired_diff  # noqa: E402
from examples.stats_utils import (bca_ci, friedman_iman_davenport, holm, rank_biserial,  # noqa: E402
                                  spearman_perm, wilcoxon_exact)

ALGS = ["sge", "psge", "psge_kl", "sa_psge", "sa_psge_lam0"]
RQ2_ALGS = ["psge", "psge_kl", "sa_psge", "sa_psge_lam0"]
METRICS = {"stab": "apg_plus_atg", "apg": "apg_norm", "atg": "atg_norm",
           "nrmse": "nrmse_test_fS", "nrmse_Sz": "nrmse_test_fSz"}
STAB_RELEVANT, NRMSE_MARGIN = -0.05, 0.02
RQ2_PRIMARY = [("sa_psge", "psge"), ("sa_psge", "psge_kl")]
RQ2_SECONDARY = [("sa_psge", "sa_psge_lam0")]
RQ1 = ("psge", "sge")


# ------------------------------------------------------------------ carga
def load(root):
    cells = collections.defaultdict(lambda: {a: collections.defaultdict(dict) for a in ALGS})
    for f in glob.glob(os.path.join(root, "*", "m*_noise*", "*", "*_d*_e*.json")):
        with open(f) as fh:
            r = json.load(fh)
        key = (r["benchmark"], f"m{r['m']}_noise{r['noise']:g}")
        if r["algorithm"] in ALGS:
            cells[key][r["algorithm"]][r["seed_data"]][r["seed_evo"]] = r
    return cells


def complete_seeds(cell, algs):
    evo = {a: set().union(*[set(v) for v in cell[a].values()]) if cell[a] else set() for a in algs}
    seeds = set.intersection(*[set(cell[a]) for a in algs]) if all(cell[a] for a in algs) else set()
    ok = sorted(s for s in seeds if all(set(cell[a][s]) == evo[a] and evo[a] for a in algs))
    return ok, sorted(set().union(*[set(cell[a]) for a in algs]) - set(ok))


def per_seed(cell, alg, key, seeds):
    return np.array([agg_evo([r[key] for r in cell[alg][s].values()]) for s in seeds])


# ------------------------------------------------------------------ comparaciones
def log_ratios(t, c):
    fin = c[np.isfinite(c)]
    eps = 1e-3 * float(np.median(fin)) if len(fin) else 1e-12
    eps = max(eps, 1e-12)
    out = []
    for x, y in zip(t, c):
        if math.isinf(x) and math.isinf(y):
            out.append(0.0)
        elif math.isinf(x):
            out.append(math.inf)
        elif math.isinf(y):
            out.append(-math.inf)
        else:
            out.append(math.log((x + eps) / (y + eps)))
    return np.array(out)


def _rel(v):
    return math.inf if v == math.inf else math.exp(v) - 1.0


def compare(t, c, rng, n_boot):
    d = np.array([paired_diff(x, y) for x, y in zip(t, c)])
    lr = log_ratios(t, c)
    lo95, hi95, _ = bca_ci(lr, n_boot=n_boot, rng=rng, level=0.95)
    lo90, hi90, _ = bca_ci(lr, n_boot=n_boot, rng=rng, level=0.90)
    return {"n": int(len(d)), "median_ctrl": float(np.median(c)), "median_treat": float(np.median(t)),
            "rel": _rel(float(np.median(lr))), "rel_ci95": [_rel(lo95), _rel(hi95)],
            "rel_ci90": [_rel(lo90), _rel(hi90)],
            "median_diff": float(np.median(d)), "wilcoxon_p": wilcoxon_exact(d)[1],
            "rank_biserial": rank_biserial(d), "treat_better": int((d < 0).sum()), "treat_worse": int((d > 0).sum())}


def classify(stab, nrmse, rq1=False):
    if rq1:   # PSGE (tratamiento) frente a SGE: aquí "positivo" es la hipótesis de RQ1
        if stab["rel_ci95"][0] > 0:
            return "PSGE más inestable", None
        if stab["rel_ci95"][1] < 0:
            return "PSGE más estable", None
        return "sin diferencia", None
    improve = stab["rel"] <= STAB_RELEVANT and stab["rel_ci95"][1] < 0
    worse = stab["rel_ci95"][0] > 0
    noninf = nrmse["rel_ci90"][1] <= NRMSE_MARGIN
    if improve and noninf:
        return "éxito", noninf
    if improve:
        return "trade-off", noninf
    if worse:
        return "peor", noninf
    return "nulo", noninf


def logscore(x):
    return math.inf if math.isinf(x) else math.log(max(x, 1e-12))


# ------------------------------------------------------------------ principal
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default="dumps/confirmation")
    # Decisión del investigador (2026-11-05, antes de lanzar): opción B = bloques = celdas, bilateral.
    p.add_argument("--global_unit", choices=["benchmark", "cell"], default="cell")
    p.add_argument("--alternative", choices=["two-sided", "less"], default="two-sided",
                   help="less = H1 pre-registrada de que el tratamiento REDUCE la métrica")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="prefijo de los ficheros de salida")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out_prefix = a.output or os.path.join(a.root, "confirmation_analysis")

    cells = load(a.root)
    if not cells:
        raise SystemExit(f"No hay resultados en {a.root}")
    report = {"settings": vars(a), "cells": {}, "integrity": {}}
    cell_rows, pareto_rows, cost_rows = [], [], []

    print(f"Análisis confirmatorio: {len(cells)} celdas en {a.root}")
    print(f"Unidad global: {a.global_unit}; alternativa: {a.alternative}\n")
    print("== Integridad")
    for key in sorted(cells):
        cell = cells[key]
        present = [x for x in ALGS if cell[x]]
        seeds, incomplete = complete_seeds(cell, present)
        inv = {x: sum(1 for s in cell[x] for r in cell[x][s].values()
                      if count_invalid([r["apg_plus_atg"], r["nrmse_test_fS"]])) for x in present}
        report["integrity"]["/".join(key)] = {"algorithms": present, "complete_data_seeds": len(seeds),
                                              "incomplete_data_seeds": incomplete, "invalid_runs": inv}
        print(f"   {key[0]:<15} {key[1]:<14} semillas completas {len(seeds):>2}"
              + (f"  incompletas {incomplete}" if incomplete else "") + f"  inválidas {inv}")

    # ---------------- por celda
    for key in sorted(cells):
        cell = cells[key]
        present = [x for x in ALGS if cell[x]]
        seeds, _ = complete_seeds(cell, present)
        if len(seeds) < 3:
            continue
        v = {x: {mname: per_seed(cell, x, mk, seeds) for mname, mk in METRICS.items()} for x in present}
        res = {"n_data_seeds": len(seeds), "medians": {x: {m: float(np.median(v[x][m])) for m in METRICS}
                                                       for x in present}, "comparisons": {}}
        for t, c in [RQ1] + RQ2_PRIMARY + RQ2_SECONDARY:
            if t in present and c in present:
                comp = {m: compare(v[t][m], v[c][m], rng, a.n_boot) for m in METRICS}
                cat, noninf = classify(comp["stab"], comp["nrmse"], rq1=((t, c) == RQ1))
                comp["category"], comp["nrmse_noninferior"] = cat, noninf
                res["comparisons"][f"{t}_vs_{c}"] = comp
                cell_rows.append({"benchmark": key[0], "regime": key[1], "comparison": f"{t}_vs_{c}",
                                  "n": comp["stab"]["n"],
                                  "stab_rel": comp["stab"]["rel"], "stab_ci95_lo": comp["stab"]["rel_ci95"][0],
                                  "stab_ci95_hi": comp["stab"]["rel_ci95"][1], "stab_p": comp["stab"]["wilcoxon_p"],
                                  "stab_rb": comp["stab"]["rank_biserial"],
                                  "nrmse_rel": comp["nrmse"]["rel"], "nrmse_ci90_hi": comp["nrmse"]["rel_ci90"][1],
                                  "category": cat})
        # mecanismo
        mech = {}
        for x in ("psge", "sa_psge"):
            if x in present:
                js = per_seed(cell, x, "js_divergence", seeds)
                rho, prho, n = spearman_perm(js, v[x]["stab"], rng=rng, n_perm=5000)
                mech[x] = {"rho": rho, "p_perm": prho, "n": n}
        res["mechanism"] = mech
        # Pareto (medianas de la celda)
        pts = {x: (res["medians"][x]["stab"], res["medians"][x]["nrmse"]) for x in present}
        for x, (s_, n_) in pts.items():
            dominated = any(o != x and pts[o][0] <= s_ and pts[o][1] <= n_ and pts[o] != (s_, n_) for o in pts)
            pareto_rows.append({"benchmark": key[0], "regime": key[1], "algorithm": x,
                                "median_apg_atg": s_, "median_nrmse_test": n_, "pareto_front": not dominated})
        # coste
        for x in present:
            runs = [r for s in seeds for r in cell[x][s].values()]
            cost_rows.append({"benchmark": key[0], "regime": key[1], "algorithm": x,
                              "median_case_evals_per_arm": float(np.median([r[arm]["n_case_evaluations"]
                                                                            for r in runs for arm in ("arm_S", "arm_Sz")])),
                              "median_wall_sec_per_pair": float(np.median([r["arm_S"]["wall_sec"] + r["arm_Sz"]["wall_sec"] for r in runs])),
                              "total_cpu_h": sum(r["arm_S"]["cpu_sec"] + r["arm_Sz"]["cpu_sec"] for r in runs) / 3600,
                              "max_peak_rss_mb": max(max(r["arm_S"]["peak_rss_kb_process"], r["arm_Sz"]["peak_rss_kb_process"]) for r in runs) / 1024})
        report["cells"]["/".join(key)] = res

    def print_cells(comp_name, title):
        print(f"\n== {title}  (efecto relativo de las medianas; IC95 BCa; p Wilcoxon exacto por celda)")
        print(f"   {'benchmark':<15} {'régimen':<14} {'Δ APG+ATG':>10} {'IC95':>18} {'p':>7} {'r_rb':>6} "
              f"{'Δ NRMSE':>8} {'IC90 sup':>9}  categoría")
        for r in cell_rows:
            if r["comparison"] == comp_name:
                print(f"   {r['benchmark']:<15} {r['regime']:<14} {100 * r['stab_rel']:>+9.1f}% "
                      f"[{100 * r['stab_ci95_lo']:>+6.1f}, {100 * r['stab_ci95_hi']:>+6.1f}]% {r['stab_p']:>7.4f} "
                      f"{r['stab_rb']:>+6.2f} {100 * r['nrmse_rel']:>+7.1f}% {100 * r['nrmse_ci90_hi']:>+8.1f}%  {r['category']}")
        cats = collections.Counter(r["category"] for r in cell_rows if r["comparison"] == comp_name)
        print(f"   resumen de celdas: {dict(cats)}")

    print_cells("psge_vs_sge", "RQ1: PSGE frente a SGE (positivo = PSGE más inestable)")
    for t, c in RQ2_PRIMARY + RQ2_SECONDARY:
        print_cells(f"{t}_vs_{c}", f"RQ2: {t} frente a {c} (negativo = {t} más estable)")

    # ---------------- nivel global
    def blocks_scores(alg, metric):
        groups = collections.defaultdict(list)
        for k, res in report["cells"].items():
            if alg in res["medians"]:
                bench, regime = k.split("/")
                groups[bench if a.global_unit == "benchmark" else k].append(logscore(res["medians"][alg][metric]))
        return {b: (math.inf if any(math.isinf(x) for x in xs) else float(np.mean(xs))) for b, xs in groups.items()}

    glob_res = {}
    print(f"\n== Nivel global (bloques = {a.global_unit}; puntuación = media de log(mediana de celda))")
    for metric, label in (("stab", "APG+ATG"), ("nrmse", "NRMSE-test")):
        sc = {x: blocks_scores(x, metric) for x in ALGS}
        common = sorted(set.intersection(*[set(sc[x]) for x in RQ2_ALGS])) if all(sc[x] for x in RQ2_ALGS) else []
        g = {"blocks": common}
        if len(common) >= 2:
            fr = friedman_iman_davenport([[sc[x][b] for x in RQ2_ALGS] for b in common])
            g["friedman"] = {**fr, "algorithms": RQ2_ALGS}
            print(f"   {label}: Friedman/Iman-Davenport F = {fr['F_iman_davenport']:.3f} (df {fr['df']}), "
                  f"p = {fr['p']:.4f}; rangos medios " +
                  ", ".join(f"{x} {r:.2f}" for x, r in zip(RQ2_ALGS, fr["mean_ranks"])))
        fam = []
        for t, c in [RQ1] + RQ2_PRIMARY + RQ2_SECONDARY:
            blocks = sorted(set(sc[t]) & set(sc[c]))
            if len(blocks) < 2:
                continue
            lr = np.array([paired_diff(sc[t][b], sc[c][b]) for b in blocks])
            if metric != "stab":
                alt = "two-sided"   # en precisión no hay hipótesis de dirección: la no inferioridad va por celda
            else:
                alt = a.alternative if (t, c) != RQ1 else ("greater" if a.alternative == "less" else "two-sided")
            pval = wilcoxon_exact(lr, alt)[1]
            fam.append({"comparison": f"{t}_vs_{c}", "n_blocks": len(blocks),
                        "median_log_ratio": float(np.median(lr)),
                        "geo_mean_ratio": float(math.exp(np.mean(lr))) if np.all(np.isfinite(lr)) else math.nan,
                        "treat_better_blocks": int((lr < 0).sum()), "p": pval, "alternative": alt,
                        "rank_biserial": rank_biserial(lr)})
        prim = [f for f in fam if f["comparison"] in [f"{t}_vs_{c}" for t, c in RQ2_PRIMARY]]
        for f, adj in zip(prim, holm([f["p"] for f in prim]) if prim else []):
            f["p_holm"] = adj
        g["comparisons"] = fam
        glob_res[metric] = g
        for f in fam:
            extra = f"  p Holm = {f['p_holm']:.4f}" if "p_holm" in f else ""
            print(f"   {label}: {f['comparison']:<26} razón geométrica = {f['geo_mean_ratio']:.3f}  "
                  f"mejor en {f['treat_better_blocks']}/{f['n_blocks']} bloques  p ({f['alternative']}) = {f['p']:.4f}{extra}")
    report["global"] = glob_res
    nb = len(set(k.split('/')[0] for k in report["cells"]))
    if a.global_unit == "benchmark" and nb <= 6 and a.alternative == "two-sided":
        print(f"\n   AVISO: con {nb} bloques el p bilateral mínimo de Wilcoxon es {2 / 2 ** nb:.4f}; con Holm sobre "
              f"2 comparaciones, el umbral de la menor es 0,025: ninguna comparación global puede ser significativa.")

    # ---------------- mecanismo y coste
    rhos = [r["mechanism"]["psge"]["rho"] for r in report["cells"].values() if "psge" in r.get("mechanism", {})]
    rhos = [x for x in rhos if np.isfinite(x)]
    if rhos:
        print(f"\n== Mecanismo (PSGE): Spearman JS vs APG+ATG por celda: mediana rho = {np.median(rhos):+.3f}, "
              f"positiva en {sum(x > 0 for x in rhos)}/{len(rhos)} celdas (asociación, no causalidad)")
    if cost_rows:
        print("\n== Coste (mediana de evaluaciones caso-individuo por brazo, relativa a PSGE en cada celda)")
        by = collections.defaultdict(list)
        base = {(r["benchmark"], r["regime"]): r["median_case_evals_per_arm"] for r in cost_rows if r["algorithm"] == "psge"}
        for r in cost_rows:
            b = base.get((r["benchmark"], r["regime"]))
            if b:
                by[r["algorithm"]].append(r["median_case_evals_per_arm"] / b)
        for x in ALGS:
            if by[x]:
                print(f"   {x:<13} {np.median(by[x]):.2f}x   CPU total {sum(r['total_cpu_h'] for r in cost_rows if r['algorithm'] == x):.1f} h")

    # ---------------- ficheros
    def write_csv(rows, suffix):
        if rows:
            with open(f"{out_prefix}_{suffix}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    write_csv(cell_rows, "cells")
    write_csv(pareto_rows, "pareto")
    write_csv(cost_rows, "cost")
    with open(f"{out_prefix}.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nGuardado: {out_prefix}.json, _cells.csv, _pareto.csv, _cost.csv")


if __name__ == "__main__":
    main()
