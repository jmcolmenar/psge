#!/usr/bin/env python3
"""
Análisis PRERREGISTRADO del estudio de adaptación de la gramática y re-mapeo (plan del 2026-11-08, fase 6).
Fijado antes de lanzar la confirmación; cualquier cambio posterior es una desviación documentada.

Reglas comunes
  R1  Inválidos (examples/invalid_rule.py): valor no finito o >= 1e5 -> +inf; nunca se descartan.
  R2  Suelo de exactitud: APG+ATG, APG, ATG y brecha < EPS (1e-6, unidades normalizadas por Var(Y_S))
      se fijan a EPS antes de tomar logaritmos (soluciones exactas: el valor es error de redondeo).
      La NRMSE (= sqrt(MSE / Var)) se fija a sqrt(EPS) = 1e-3, la misma tolerancia en unidades de RMSE
      (sin ese suelo, una solución exacta da NRMSE ~1e-15 y el cociente con otra condición explota).
      Sensibilidad: EPS = 1e-4 y 1e-8 para los dos contrastes principales.
  R3  Réplica = semilla de datos; las semillas evolutivas se agregan con la mediana.
  R4  Solo semillas de datos completas (todas las condiciones con todas sus semillas evolutivas) en cada
      celda (benchmark x régimen); las incompletas se listan.
  R5  log-ratio emparejado por semilla de datos: log(t) - log(c); inf - inf = 0.
  R6  Puntuación de un benchmark para una condición: media sobre sus regímenes del log de la mediana
      (entre semillas de datos) de la celda. Efecto de un benchmark para un contraste: media sobre las
      celdas (y niveles de lambda) de la mediana de los log-ratios emparejados de cada celda.
      Si esa media mezcla +inf y -inf, se usa la mediana. Unidad de inferencia global: el benchmark.

Familia principal (Holm, alfa = 0,05)
  H-A1  Page unilateral (APG+ATG crece con lambda) sobre SGEF, PSGE 0,005 ... 0,1; benchmarks como bloques.
  H-C1  Wilcoxon exacto unilateral (PSGE-PP < PSGE) sobre los efectos por benchmark de log(PP / PSGE)
        promediados sobre los 5 niveles de lambda y los regímenes.

Contrastes secundarios (sin corrección; se declaran como tales)
  NR    (opción A, 2026-11-08) Coste en precisión de PSGE-PP: efecto por benchmark de log(NRMSE_test PP / PSGE)
        como en H-C1; no inferioridad si el extremo superior del IC BCa 90 % de la MEDIA entre benchmarks
        es < log(1,10) (margen del 10 %). No condiciona H-C1. Por celda: solo descriptivo.
  M     Comprobaciones de la manipulación: entropía final decreciente con lambda (Page); tasa de re-mapeo
        media de PSGE creciente con lambda (Page) y de PSGE-PP <= 0,05 en todas las celdas.
  H-A2  Spearman, dentro de cada celda y nivel de lambda de PSGE, entre entropía final y log APG+ATG a través
        de las semillas de datos: rho mediana y celdas con rho < 0.
  H-A3  Pendiente log APG+ATG (lambda 0,1 frente a SGEF) por benchmark y régimen; Wilcoxon unilateral
        m=20 > m=100 (media sobre ruido) y ruido 5 % > 0 % (media sobre m).
  H-C2  Page unilateral sobre la ventaja de PSGE-PP, -log(PP / PSGE), creciente a lo largo de los 5 lambdas.
  H-E1  Dentro de cada celda, Spearman entre APG+ATG y la brecha (medianas por condición) a través de las 11
        condiciones; media y desviación de rho sobre las celdas. Se confirma si la media es >= 0,6.
        También, dentro de cada condición, a través de las celdas.
  H-E2  La media de rho(APG+ATG) supera a la de rho(APG) y a la de rho(ATG); Wilcoxon emparejado sobre celdas.
  Robustez: celdas como bloques (Page y Wilcoxon bilaterales), Friedman / Iman-Davenport de las 11 condiciones
        con benchmarks como bloques, y EPS = 1e-4 / 1e-8.

Uso (desde sge/):
  python -m examples.analyze_remap_study --root ~/psge-results/remap_confirmation --output meta/x/remap_analysis
  python -m examples.analyze_remap_study --root ~/psge-results/remap_pilot --split val --output /tmp/pilot  # ensayo
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
from examples.invalid_rule import clean   # noqa: E402
from examples.stats_utils import (bca_ci, friedman_iman_davenport, holm, page_trend, rankdata,  # noqa: E402
                                  rank_biserial, wilcoxon_exact)

EPS = 1e-6
SENSITIVITY_EPS = (1e-4, 1e-8)
NI_MARGIN = math.log(1.10)
ALPHA = 0.05
REMAP_PP_MAX = 0.05
E1_THRESHOLD = 0.6


# ---------------------------------------------------------------------------- carga y agregación

def cond_lambda(name):
    return 0.0 if name == "sgef" else float(name.rsplit("lf", 1)[1])


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


def floor(v, eps):
    v = clean(v)
    return v if math.isinf(v) else max(v, eps)


def log_(v):
    return math.inf if math.isinf(v) else math.log(v)


def ldiff(a, b):
    """R5 sobre logaritmos."""
    if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
        return 0.0
    return a - b


def robust_mean(values):
    v = np.asarray(values, dtype=float)
    if np.any(v == math.inf) and np.any(v == -math.inf):
        return float(np.median(v))
    return float(np.mean(v))


def mean_remap(arm):
    t = arm.get("trajectory", {}).get("remap_rate", [])
    return float(np.mean(t)) if t else float("nan")


class Data:
    """Valores por (benchmark, régimen, condición, semilla de datos), ya con R1-R4."""

    METRICS = ("apg_plus_atg", "apg_norm", "atg_norm", "gap", "nrmse", "entropy", "remap")

    def __init__(self, rows, eps, split):
        self.eps = eps
        gap_key = "gen_gap_test_norm_fS" if split == "test" else "gen_gap_val_norm_fS"
        nrmse_key = "nrmse_test_fS" if split == "test" else "nrmse_val_fS"
        raw = defaultdict(lambda: defaultdict(list))
        evo = defaultdict(set)
        self.conditions = sorted({r["condition"] for r in rows}, key=lambda c: (c.startswith("psge_pp"), cond_lambda(c)))
        self.lambdas = sorted({cond_lambda(c) for c in self.conditions if c != "sgef"})
        for r in rows:
            key = (r["benchmark"], f"m{r['m']}_noise{r['noise']:g}", r["condition"], r["seed_data"])
            evo[key].add(r["seed_evo"])
            v = raw[key]
            v["apg_plus_atg"].append(floor(r["apg_plus_atg"], eps))
            v["apg_norm"].append(floor(r["apg_norm"], eps))
            v["atg_norm"].append(floor(r["atg_norm"], eps))
            v["gap"].append(floor(r.get(gap_key), eps))
            v["nrmse"].append(floor(r.get(nrmse_key), math.sqrt(eps)))
            v["entropy"].append(float(r["entropy_S"]))
            v["remap"].append(np.nanmean([mean_remap(r["arm_S"]), mean_remap(r["arm_Sz"])]))
        n_evo = max(len(s) for s in evo.values())
        self.cells = sorted({k[:2] for k in raw})
        self.benchmarks = sorted({k[0] for k in raw})
        self.regimes = sorted({k[1] for k in raw})
        self.seeds = {}
        self.incomplete = []
        self.val = {}
        for cell in self.cells:
            all_seeds = sorted({k[3] for k in raw if k[:2] == cell})
            ok = []
            for s in all_seeds:
                if all(len(evo.get(cell + (c, s), ())) == n_evo for c in self.conditions):
                    ok.append(s)
                else:
                    self.incomplete.append(cell + (s,))
            self.seeds[cell] = ok
            for c in self.conditions:
                for s in ok:
                    v = raw[cell + (c, s)]
                    self.val[cell + (c, s)] = {m: float(np.median(v[m])) for m in self.METRICS}
        self.n_runs = len(rows)
        self.n_evo = n_evo
        self.invalid = defaultdict(lambda: [0, 0])
        for r in rows:
            bad = math.isinf(clean(r["apg_plus_atg"])) or math.isinf(clean(r.get(nrmse_key)))
            self.invalid[r["condition"]][0] += int(bad)
            self.invalid[r["condition"]][1] += 1

    # mediana entre semillas de datos de una celda
    def cell_median(self, cell, cond, metric="apg_plus_atg"):
        return float(np.median([self.val[cell + (cond, s)][metric] for s in self.seeds[cell]]))

    def cell_log_median(self, cell, cond, metric="apg_plus_atg"):
        return log_(self.cell_median(cell, cond, metric))

    def paired_logratios(self, cell, t, c, metric="apg_plus_atg"):
        return [ldiff(log_(self.val[cell + (t, s)][metric]), log_(self.val[cell + (c, s)][metric]))
                for s in self.seeds[cell]]

    def cell_effect(self, cell, t, c, metric="apg_plus_atg"):
        return float(np.median(self.paired_logratios(cell, t, c, metric)))

    def bench_score(self, bench, cond, metric="apg_plus_atg"):
        return robust_mean([self.cell_log_median((bench, r), cond, metric) for r in self.regimes if (bench, r) in self.seeds])

    def bench_pp_effect(self, bench, metric="apg_plus_atg", lambdas=None):
        effs = [self.cell_effect((bench, r), f"psge_pp_lf{l:g}", f"psge_lf{l:g}", metric)
                for r in self.regimes if (bench, r) in self.seeds for l in (lambdas or self.lambdas)]
        return robust_mean(effs)


# ---------------------------------------------------------------------------- contrastes

def ordered_psge(d):
    return ["sgef"] + [f"psge_lf{l:g}" for l in d.lambdas]


def h_a1(d):
    order = ordered_psge(d)
    X = np.array([[d.bench_score(b, c) for c in order] for b in d.benchmarks])
    pg = page_trend(X)
    ratios = X[:, -1] - X[:, 0]
    lo, hi, meth = bca_ci(ratios, stat=np.median, n_boot=10000)
    return {"page_L": pg["L"], "p": pg["p"], "method": pg["method"], "order": order,
            "median_log_ratio_lmax_vs_sgef": float(np.median(ratios)), "ci95": [lo, hi], "ci_method": meth,
            "per_benchmark": dict(zip(d.benchmarks, X.tolist()))}


def h_c1(d):
    e = np.array([d.bench_pp_effect(b) for b in d.benchmarks])
    w, p, n = wilcoxon_exact(e, alternative="less")
    lo, hi, meth = bca_ci(e, stat=np.median, n_boot=10000)
    return {"W_plus": w, "p": p, "n_nonzero": n, "rank_biserial": rank_biserial(e),
            "median_log_ratio": float(np.median(e)), "ci95": [lo, hi], "ci_method": meth,
            "n_benchmarks_pp_lower": int(np.sum(e < 0)), "per_benchmark": dict(zip(d.benchmarks, e.tolist()))}


def nr_secondary(d):
    e = np.array([d.bench_pp_effect(b, metric="nrmse") for b in d.benchmarks])
    lo, hi, meth = bca_ci(e, stat=np.mean, n_boot=10000, level=0.90)
    per_cell = []
    for cell in d.cells:
        for l in d.lambdas:
            x = d.paired_logratios(cell, f"psge_pp_lf{l:g}", f"psge_lf{l:g}", "nrmse")
            per_cell.append({"cell": "/".join(cell), "lambda": l, "median_rel_pct": (math.exp(np.median(x)) - 1) * 100
                             if np.isfinite(np.median(x)) else float("inf")})
    return {"mean_log_ratio": robust_mean(e), "ci90": [lo, hi], "ci_method": meth,
            "margin_log": NI_MARGIN, "non_inferior": bool(hi < NI_MARGIN),
            "mean_rel_pct": (math.exp(robust_mean(e)) - 1) * 100 if np.isfinite(robust_mean(e)) else float("inf"),
            "per_benchmark": dict(zip(d.benchmarks, e.tolist())), "per_cell": per_cell}


def manipulation(d):
    order = ordered_psge(d)
    H = np.array([[d.bench_score(b, c, "entropy") for c in order] for b in d.benchmarks])
    ent = page_trend(-H)          # entropía DEcreciente con lambda
    psge = [f"psge_lf{l:g}" for l in d.lambdas]
    R = np.array([[np.median([d.val[(b, r, c, s)]["remap"] for r in d.regimes if (b, r) in d.seeds
                              for s in d.seeds[(b, r)]]) for c in psge] for b in d.benchmarks])
    rem = page_trend(R)
    pp_max = max(d.cell_median(cell, f"psge_pp_lf{l:g}", "remap") for cell in d.cells for l in d.lambdas)
    return {"entropy_page_p": ent["p"], "entropy_by_level_median": np.median(np.exp(H), axis=0).tolist(),
            "remap_psge_page_p": rem["p"], "remap_psge_by_level_median": np.median(R, axis=0).tolist(),
            "remap_pp_max_cell_median": pp_max, "remap_pp_ok": bool(pp_max <= REMAP_PP_MAX)}


def spearman_rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x) - (len(x) + 1) / 2, rankdata(y) - (len(y) + 1) / 2
    den = math.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    return float(np.sum(rx * ry) / den) if den > 0 else 0.0


def h_a2(d):
    rhos = []
    for cell in d.cells:
        for l in d.lambdas:
            c = f"psge_lf{l:g}"
            ent = [d.val[cell + (c, s)]["entropy"] for s in d.seeds[cell]]
            stab = [log_(d.val[cell + (c, s)]["apg_plus_atg"]) for s in d.seeds[cell]]
            rhos.append(spearman_rho(ent, stab))
    r = np.array([x for x in rhos if np.isfinite(x)])
    return {"median_rho": float(np.median(r)) if len(r) else float("nan"), "n_cells": int(len(r)),
            "n_negative": int(np.sum(r < 0))}


def h_a3(d):
    top, base = f"psge_lf{d.lambdas[-1]:g}", "sgef"
    ms = sorted({r.split("_")[0] for r in d.regimes}, key=lambda x: int(x[1:]))
    ns = sorted({r.split("_")[1] for r in d.regimes}, key=lambda x: float(x[5:]))
    slope = {(b, r): d.cell_log_median((b, r), top) - d.cell_log_median((b, r), base)
             for b in d.benchmarks for r in d.regimes if (b, r) in d.seeds}
    out = {}
    if any((b, f"{m}_{n}") not in slope for b in d.benchmarks for m in ms for n in ns):
        return {"skipped": "los regímenes no forman el diseño factorial m x ruido"}
    if len(ms) == 2:
        diff = [np.mean([slope[(b, f"{ms[0]}_{n}")] for n in ns]) - np.mean([slope[(b, f"{ms[1]}_{n}")] for n in ns])
                for b in d.benchmarks]
        out["small_m_minus_large_m"] = {"median": float(np.median(diff)), "p_greater": wilcoxon_exact(diff, "greater")[1]}
    if len(ns) == 2:
        diff = [np.mean([slope[(b, f"{m}_{ns[1]}")] for m in ms]) - np.mean([slope[(b, f"{m}_{ns[0]}")] for m in ms])
                for b in d.benchmarks]
        out["noise_minus_no_noise"] = {"median": float(np.median(diff)), "p_greater": wilcoxon_exact(diff, "greater")[1]}
    return out


def h_c2(d):
    A = np.array([[-robust_mean([d.cell_effect((b, r), f"psge_pp_lf{l:g}", f"psge_lf{l:g}")
                                 for r in d.regimes if (b, r) in d.seeds]) for l in d.lambdas] for b in d.benchmarks])
    pg = page_trend(A)
    return {"p": pg["p"], "median_advantage_by_lambda": np.median(A, axis=0).tolist()}


def h_e(d):
    within_cell = {"apg_plus_atg": [], "apg_norm": [], "atg_norm": []}
    for cell in d.cells:
        gap = [d.cell_log_median(cell, c, "gap") for c in d.conditions]
        for m in within_cell:
            within_cell[m].append(spearman_rho([d.cell_log_median(cell, c, m) for c in d.conditions], gap))
    within_cond = []
    for c in d.conditions:
        within_cond.append(spearman_rho([d.cell_log_median(cell, c) for cell in d.cells],
                                        [d.cell_log_median(cell, c, "gap") for cell in d.cells]))
    s = {m: np.array(v) for m, v in within_cell.items()}
    mean = {m: float(np.nanmean(v)) for m, v in s.items()}
    sd = {m: float(np.nanstd(v, ddof=1)) if np.sum(np.isfinite(v)) > 1 else float("nan") for m, v in s.items()}
    e2 = {}
    for m in ("apg_norm", "atg_norm"):
        diff = s["apg_plus_atg"] - s[m]
        diff = diff[np.isfinite(diff)]
        e2[m] = {"p_greater": wilcoxon_exact(diff, "greater")[1] if len(diff) else 1.0}
    return {"mean_rho": mean, "sd_rho": sd, "n_cells": len(d.cells),
            "E1_confirmed": bool(mean["apg_plus_atg"] >= E1_THRESHOLD),
            "E2_sum_beats_parts": bool(mean["apg_plus_atg"] > mean["apg_norm"] and mean["apg_plus_atg"] > mean["atg_norm"]),
            "E2_wilcoxon": e2, "within_condition_rho": dict(zip(d.conditions, within_cond))}


def robustness(d):
    order = ordered_psge(d)
    Xc = np.array([[d.cell_log_median(cell, c) for c in order] for cell in d.cells])
    up, down = page_trend(Xc)["p"], page_trend(-Xc)["p"]
    ec = [robust_mean([d.cell_effect(cell, f"psge_pp_lf{l:g}", f"psge_lf{l:g}") for l in d.lambdas]) for cell in d.cells]
    F = np.array([[d.bench_score(b, c) for c in d.conditions] for b in d.benchmarks])
    fr = friedman_iman_davenport(F)
    return {"cells_page_two_sided_p": min(1.0, 2 * min(up, down)),
            "cells_pp_wilcoxon_two_sided_p": wilcoxon_exact(ec, "two-sided")[1],
            "friedman": {"p": fr["p"], "F": fr["F_iman_davenport"], "mean_ranks": dict(zip(d.conditions, fr["mean_ranks"]))}}


# ---------------------------------------------------------------------------- informe

def analyze(rows, split="test", eps=EPS, sensitivity=True):
    d = Data(rows, eps, split)
    a1, c1 = h_a1(d), h_c1(d)
    p_adj = holm([a1["p"], c1["p"]])
    res = {
        "eps": eps, "split": split, "n_runs": d.n_runs, "n_evo_seeds": d.n_evo,
        "benchmarks": d.benchmarks, "regimes": d.regimes, "conditions": d.conditions,
        "data_seeds_per_cell": {"/".join(k): len(v) for k, v in d.seeds.items()},
        "incomplete_data_seeds": ["/".join(map(str, k)) for k in d.incomplete],
        "invalid": {c: {"n": v[0], "of": v[1], "rate": v[0] / v[1]} for c, v in d.invalid.items()},
        "primary": {
            "H-A1": {**a1, "p_holm": p_adj[0], "confirmed": bool(p_adj[0] < ALPHA)},
            "H-C1": {**c1, "p_holm": p_adj[1], "confirmed": bool(p_adj[1] < ALPHA)},
        },
        "secondary": {"NR": nr_secondary(d), "manipulation": manipulation(d), "H-A2": h_a2(d), "H-A3": h_a3(d),
                      "H-C2": h_c2(d), "H-E": h_e(d)},
        "robustness": robustness(d),
    }
    if sensitivity:
        res["sensitivity_eps"] = {}
        for e in SENSITIVITY_EPS:
            ds = Data(rows, e, split)
            a, c = h_a1(ds), h_c1(ds)
            pa = holm([a["p"], c["p"]])
            res["sensitivity_eps"][str(e)] = {"H-A1_p_holm": pa[0], "H-C1_p_holm": pa[1]}
    return res, d


def fmt_p(p):
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def report(res):
    P, S = res["primary"], res["secondary"]
    out = []
    w = out.append
    w("ANÁLISIS PRERREGISTRADO: adaptación de la gramática y re-mapeo en PSGE")
    w(f"Conjunto: {res['split']} | EPS = {res['eps']:g} | {res['n_runs']} ejecuciones emparejadas | "
      f"{len(res['benchmarks'])} benchmarks x {len(res['regimes'])} regímenes x {len(res['conditions'])} condiciones")
    w(f"Semillas de datos completas por celda: min {min(res['data_seeds_per_cell'].values())}, "
      f"max {max(res['data_seeds_per_cell'].values())}; incompletas: {len(res['incomplete_data_seeds'])}")
    inv = {c: v for c, v in res["invalid"].items() if v["n"]}
    w("Inválidos: " + (", ".join(f"{c} {v['n']}/{v['of']}" for c, v in inv.items()) if inv else "ninguno"))
    w("")
    w("== Familia principal (Holm, alfa 0,05)")
    a = P["H-A1"]
    w(f"H-A1 dosis-respuesta: Page L = {a['page_L']:.1f}, p = {fmt_p(a['p'])}, p Holm = {fmt_p(a['p_holm'])} -> "
      f"{'CONFIRMADA' if a['confirmed'] else 'no confirmada'}")
    w(f"     log-ratio mediano lambda max / SGEF = {a['median_log_ratio_lmax_vs_sgef']:+.3f} "
      f"(x{math.exp(a['median_log_ratio_lmax_vs_sgef']):.2f}), IC95 [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]")
    c = P["H-C1"]
    w(f"H-C1 re-mapeo: Wilcoxon unilateral W+ = {c['W_plus']:.1f}, p = {fmt_p(c['p'])}, p Holm = {fmt_p(c['p_holm'])} -> "
      f"{'CONFIRMADA' if c['confirmed'] else 'no confirmada'}")
    w(f"     log-ratio mediano PP/PSGE = {c['median_log_ratio']:+.3f} (x{math.exp(c['median_log_ratio']):.2f}), "
      f"IC95 [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], rank-biserial {c['rank_biserial']:+.2f}, "
      f"PP menor en {c['n_benchmarks_pp_lower']}/{len(res['benchmarks'])} benchmarks")
    for b, e in c["per_benchmark"].items():
        w(f"       {b:<16s} {e:+.3f}")
    w("")
    w("== Secundarios")
    nr = S["NR"]
    w(f"NR precisión de PSGE-PP: media log(NRMSE PP/PSGE) = {nr['mean_log_ratio']:+.3f} ({nr['mean_rel_pct']:+.1f} %), "
      f"IC90 [{nr['ci90'][0]:+.3f}, {nr['ci90'][1]:+.3f}], margen {nr['margin_log']:.3f} -> "
      f"{'no inferior' if nr['non_inferior'] else 'NO se demuestra no inferioridad'}")
    m = S["manipulation"]
    w(f"M  entropía decreciente con lambda: Page p = {fmt_p(m['entropy_page_p'])}; mediana por nivel "
      + " ".join(f"{x:.3f}" for x in m["entropy_by_level_median"]))
    w(f"M  re-mapeo de PSGE creciente con lambda: Page p = {fmt_p(m['remap_psge_page_p'])}; por nivel "
      + " ".join(f"{x:.3f}" for x in m["remap_psge_by_level_median"]))
    w(f"M  re-mapeo de PSGE-PP: máximo de las medianas por celda {m['remap_pp_max_cell_median']:.4f} "
      f"({'<=' if m['remap_pp_ok'] else '>'} {REMAP_PP_MAX})")
    h2 = S["H-A2"]
    w(f"H-A2 entropía vs APG+ATG: rho mediana {h2['median_rho']:+.3f}, negativas {h2['n_negative']}/{h2['n_cells']}")
    for k, v in S["H-A3"].items():
        if k == "skipped":
            w(f"H-A3 omitida: {v}")
        else:
            w(f"H-A3 {k}: mediana {v['median']:+.3f}, p unilateral {fmt_p(v['p_greater'])}")
    h = S["H-C2"]
    w(f"H-C2 ventaja de PSGE-PP creciente con lambda: Page p = {fmt_p(h['p'])}; por nivel "
      + " ".join(f"{x:+.3f}" for x in h["median_advantage_by_lambda"]))
    e = S["H-E"]
    w(f"H-E1 rho(APG+ATG, brecha) media {e['mean_rho']['apg_plus_atg']:.3f} (DE {e['sd_rho']['apg_plus_atg']:.3f}) "
      f"en {e['n_cells']} celdas -> {'CONFIRMADA' if e['E1_confirmed'] else 'no confirmada'} (umbral {E1_THRESHOLD})")
    w(f"H-E2 rho(APG) {e['mean_rho']['apg_norm']:.3f}, rho(ATG) {e['mean_rho']['atg_norm']:.3f} -> "
      f"{'la suma sigue mejor' if e['E2_sum_beats_parts'] else 'la suma NO sigue mejor'} "
      f"(p vs APG {fmt_p(e['E2_wilcoxon']['apg_norm']['p_greater'])}, vs ATG {fmt_p(e['E2_wilcoxon']['atg_norm']['p_greater'])})")
    w("")
    r = res["robustness"]
    w("== Robustez")
    w(f"Celdas como bloques: Page bilateral p = {fmt_p(r['cells_page_two_sided_p'])}; "
      f"Wilcoxon PP vs PSGE bilateral p = {fmt_p(r['cells_pp_wilcoxon_two_sided_p'])}")
    w(f"Friedman / Iman-Davenport (11 condiciones): F = {r['friedman']['F']:.3f}, p = {fmt_p(r['friedman']['p'])}")
    for k, v in res.get("sensitivity_eps", {}).items():
        w(f"EPS = {k}: H-A1 p Holm {fmt_p(v['H-A1_p_holm'])}, H-C1 p Holm {fmt_p(v['H-C1_p_holm'])}")
    return "\n".join(out)


def write_cells_csv(d, path):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["benchmark", "regime", "condition", "lambda", "n_data_seeds", "median_apg_plus_atg",
                     "median_nrmse", "median_gap", "median_entropy", "median_remap_rate"])
        for cell in d.cells:
            for c in d.conditions:
                wr.writerow([cell[0], cell[1], c, cond_lambda(c), len(d.seeds[cell])] +
                            [d.cell_median(cell, c, m) for m in ("apg_plus_atg", "nrmse", "gap", "entropy", "remap")])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="test en la confirmación; val solo para ensayar el análisis con el piloto")
    p.add_argument("--output", required=True, help="prefijo de salida: <prefijo>.json, <prefijo>_cells.csv")
    p.add_argument("--no_sensitivity", action="store_true")
    a = p.parse_args()
    rows = load(a.root)
    if not rows:
        raise SystemExit(f"no hay resultados en {a.root}")
    res, d = analyze(rows, split=a.split, sensitivity=not a.no_sensitivity)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output + ".json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    write_cells_csv(d, a.output + "_cells.csv")
    print(report(res))
    print(f"\nJSON: {a.output}.json\nCSV:  {a.output}_cells.csv")


if __name__ == "__main__":
    main()
