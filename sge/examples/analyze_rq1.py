#!/usr/bin/env python3
"""
Análisis del piloto de RQ1 (SGE frente a PSGE). Unidad de análisis: la semilla de datos.

Para cada problema y cada semilla de datos se agregan primero las semillas evolutivas con la
mediana (examples/invalid_rule.py: inválidos = +inf, nunca se descartan). Después:

1. Estabilidad de DATOS (métrica primaria, APG + ATG_hat; también APG y ATG por separado):
   diferencia emparejada PSGE - SGE por semilla de datos -> mediana, IC BCa 95 %, Wilcoxon exacto,
   rank-biserial y nº de semillas en que PSGE es más inestable.
2. Estabilidad de BÚSQUEDA (mismos datos S, distintas semillas evolutivas): para cada punto de
   validación, desviación típica entre semillas de las predicciones de f_S, dividida por sigma_y de
   validación; mediana sobre los puntos. Las predicciones no evaluables se excluyen punto a punto y
   se cuentan. Misma comparación emparejada.
3. Precisión: NRMSE de validación de f_{S+z}, misma comparación.
4. Mecanismo (solo PSGE): Spearman entre JS(P_S, P_{S+z}) y APG + ATG_hat por semilla de datos,
   con p por permutación. Es una ASOCIACIÓN, no una prueba causal. Además, entropía final de la PCFG.

Signo: diferencia > 0 = PSGE mayor (más inestable / más error) que SGE.

Uso (desde sge/):
  python -m examples.analyze_rq1 --root ~/psge-results/rq1_pilot --output informe.json
"""

import argparse
import collections
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from examples.invalid_rule import agg_evo, count_invalid, paired_diff  # noqa: E402
from examples.stats_utils import bca_ci, rank_biserial, spearman_perm, wilcoxon_exact  # noqa: E402

ALGS = ("sge", "psge")


def load(root, problem):
    runs = {a: collections.defaultdict(dict) for a in ALGS}
    for a in ALGS:
        for f in glob.glob(os.path.join(root, problem, a, f"{a}_d*_e*.json")):
            with open(f) as fh:
                r = json.load(fh)
            runs[a][r["seed_data"]][r["seed_evo"]] = r
    return runs


def search_instability(problem, seed_data, phenotypes, n_val):
    """Mediana sobre puntos de validación de SD(predicciones entre semillas) / sigma_y."""
    from sge.stability import evaluate_phenotype_on_point, generate_add_one_datasets
    val = generate_add_one_datasets(problem_name=problem, m=20, seed_data=seed_data,
                                    n_test=0, n_val=n_val)["val_set"]
    sigma = float(np.std([pt[-1] for pt in val])) or 1.0
    sds, n_bad = [], 0
    for pt in val:
        preds = [evaluate_phenotype_on_point(ph, pt) for ph in phenotypes]
        good = [v for v in preds if v is not None and math.isfinite(v) and abs(v) < 1e5]
        n_bad += len(preds) - len(good)
        if len(good) >= 2:
            sds.append(np.std(good) / sigma)
    return (float(np.median(sds)) if sds else math.inf), n_bad


def compare(diffs, n_boot, rng):
    d = np.asarray(diffs, dtype=float)
    lo, hi, method = bca_ci(d, n_boot=n_boot, rng=rng)
    _, p, n_nz = wilcoxon_exact(d)
    return {"n": int(len(d)), "median_diff": float(np.median(d)), "ci95": [lo, hi], "ci_method": method,
            "wilcoxon_p": p, "n_nonzero": n_nz, "rank_biserial": rank_biserial(d),
            "psge_higher": int((d > 0).sum()), "psge_lower": int((d < 0).sum())}


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default="dumps/rq1_pilot")
    p.add_argument("--problems", nargs="+", default=["nguyen1", "nguyen5"])
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_search", action="store_true", help="omitir la estabilidad de búsqueda (lenta)")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    report = {}
    for prob in args.problems:
        runs = load(args.root, prob)
        seeds = sorted(set(runs["sge"]) & set(runs["psge"]))
        evo = {a: sorted({e for s in seeds for e in runs[a][s]}) for a in ALGS}
        complete = [s for s in seeds if all(set(runs[a][s]) == set(evo[a]) for a in ALGS)]
        out = {"n_data_seeds": len(complete), "n_evo_seeds": len(evo["sge"]),
               "incomplete_data_seeds": sorted(set(seeds) - set(complete))}

        def per_seed(alg, key):
            return np.array([agg_evo([runs[alg][s][e][key] for e in evo[alg]]) for s in complete])

        levels, comps = {}, {}
        for key, label in (("apg_plus_atg", "APG+ATG"), ("apg_norm", "APG"), ("atg_norm", "ATG"),
                           ("nrmse_val", "NRMSE-val")):
            v = {a: per_seed(a, key) for a in ALGS}
            levels[label] = {a: float(np.median(v[a])) for a in ALGS}
            comps[label] = compare([paired_diff(x, y) for x, y in zip(v["psge"], v["sge"])], args.n_boot, rng)

        inv = {a: sum(count_invalid([r["apg_plus_atg"], r["nrmse_val"]]) > 0
                      for s in complete for r in runs[a][s].values()) for a in ALGS}

        if not args.no_search:
            si = {a: [] for a in ALGS}
            bad = {a: 0 for a in ALGS}
            for s in complete:
                for a in ALGS:
                    val, nb = search_instability(prob, s, [runs[a][s][e]["phenotype_S"] for e in evo[a]], args.n_val)
                    si[a].append(val)
                    bad[a] += nb
            levels["Busqueda"] = {a: float(np.median(si[a])) for a in ALGS}
            comps["Busqueda"] = compare([paired_diff(x, y) for x, y in zip(si["psge"], si["sge"])], args.n_boot, rng)
            out["search_invalid_predictions"] = bad

        js = per_seed("psge", "js_divergence")
        stab = per_seed("psge", "apg_plus_atg")
        rho, p_rho, n_rho = spearman_perm(js, stab, rng=rng)
        ent = {a: float(np.median([r.get("entropy_S", float("nan")) for s in complete
                                   for r in runs[a][s].values()])) for a in ALGS}
        out.update({"levels": levels, "comparisons": comps, "invalid_runs": inv,
                    "mechanism": {"spearman_js_vs_apg_atg": rho, "p_perm": p_rho, "n": n_rho,
                                  "median_js_psge": float(np.median(js))},
                    "pcfg_entropy_S_median": ent})
        report[prob] = out

        # ---- informe en texto ----
        print(f"\n== {prob}: {len(complete)} semillas de datos x {len(evo['sge'])} evolutivas "
              f"(mediana sobre evolutivas; diferencia = PSGE - SGE)")
        if out["incomplete_data_seeds"]:
            print(f"   semillas de datos incompletas (excluidas): {out['incomplete_data_seeds']}")
        print(f"   {'métrica':<10} {'SGE':>9} {'PSGE':>9} {'Δ mediana':>10} {'IC95 ' + comps['APG+ATG']['ci_method']:>22} "
              f"{'p Wilcoxon':>10} {'r_rb':>6} {'PSGE>SGE':>9}")
        for label, c in comps.items():
            lv = levels[label]
            print(f"   {label:<10} {lv['sge']:>9.4f} {lv['psge']:>9.4f} {c['median_diff']:>+10.4f} "
                  f"[{c['ci95'][0]:+9.4f}, {c['ci95'][1]:+9.4f}] {c['wilcoxon_p']:>10.4f} "
                  f"{c['rank_biserial']:>+6.2f} {c['psge_higher']:>4}/{c['n']}")
        print(f"   inválidas: SGE {inv['sge']}, PSGE {inv['psge']}"
              + (f"; predicciones no evaluables (búsqueda): {out['search_invalid_predictions']}"
                 if "search_invalid_predictions" in out else ""))
        print(f"   mecanismo (PSGE): Spearman JS vs APG+ATG rho = {rho:+.3f} (p perm = {p_rho:.4f}, n = {n_rho}); "
              f"JS mediana = {np.median(js):.5f}; entropía final PCFG: SGE {ent['sge']:.3f}, PSGE {ent['psge']:.3f}")

    path = args.output or os.path.join(args.root, "rq1_analysis.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nGuardado: {path}")


if __name__ == "__main__":
    main()
