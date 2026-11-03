#!/usr/bin/env python3
"""
Calibrate SA_ETA (SA-PSGE step size) against PSGE's grammar movement.

Rationale
---------
PSGE (additive update, LEARNING_FACTOR) and SA-PSGE (multiplicative update
P*(r) ~ P_t(r) * exp(eta * C(r)) + uniform mixture alpha) have different
functional forms, so reusing LEARNING_FACTOR as eta does not give comparable
grammar movement. To compare *credit signals* rather than *step sizes*, we pick
eta so that the per-update KL(P_{t+1} || P_t) of SA-PSGE matches PSGE's
(median over updates and seeds), for a given alpha.

Only the training set S is used (n_test=0: no test data is generated).
Seeds here are disjoint from the grid-search and confirmatory seeds.

Usage (from sge/):
  python3 -m examples.calibrate_sa_eta --problem nguyen5 --popsize 200 --generations 50 \
      --n_seeds 3 --etas 0.1 0.3 1 3 10 --alpha 0.005 --n_jobs 4
"""
import argparse
import copy
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

CALIB_SEED_DATA_BASE = 500
CALIB_SEED_EVO_BASE = 5000


def _worker(task):
    import importlib
    import sge
    import sge.engine as eng
    import sge.grammar as gmod
    from sge.stability import generate_add_one_datasets, StabilityEvaluationFunction, calculate_pcfg_entropy
    sau = importlib.import_module("sge.operators.stability_aware_update")

    kls, ents = [], []
    orig = eng.perform_grammar_update

    def logged(population, best_gen, flag, eval_func, iteration):
        before = copy.deepcopy(gmod.get_pcfg())
        orig(population, best_gen, flag, eval_func, iteration)
        after = gmod.get_pcfg()
        kls.append(sau.compute_pcfg_kl_divergence(after, before, gmod.get_mask()))
        ents.append(calculate_pcfg_entropy(after, gmod.get_mask()))

    eng.perform_grammar_update = logged
    try:
        b = generate_add_one_datasets(task["problem"], m=task["m"], seed_data=task["seed_data"], n_test=0, n_val=0)
        gmod.reset()
        opts = {
            "POPSIZE": task["popsize"], "GENERATIONS": task["generations"],
            "ELITISM": max(1, int(round(0.01 * task["popsize"]))),
            "LEARNING_FACTOR": task["lf"], "SEED": task["seed_evo"],
            "GRAMMAR": task["grammar"], "RUN": 1, "VERBOSE": False, "SAVE_STEP": 10**9,
            "EXPERIMENT_NAME": os.path.join(task["out"], task["name"], f"s{task['seed_evo']}"),
            "ALGORITHM": task["algorithm"], **task["alg_opts"],
        }
        t0 = time.time()
        best, _ = sge.evolutionary_algorithm(
            evaluation_function=StabilityEvaluationFunction(dataset=b["S"]),
            parameters_file=task["parameters_file"], options_dict=opts)
        dt = time.time() - t0
    finally:
        eng.perform_grammar_update = orig
    k = np.array(kls)
    upd = k[k > 1e-15]
    return {
        "name": task["name"], "seed_evo": task["seed_evo"], "alg_opts": task["alg_opts"],
        "median_kl": float(np.median(upd)) if len(upd) else 0.0,
        "mean_kl": float(np.mean(upd)) if len(upd) else 0.0,
        "n_updates": int(len(upd)),
        "final_entropy": float(ents[-1]) if ents else 1.0,
        "best_train_nrmse": float(best["fitness"]),
        "best_phenotype": best["phenotype"],
        "time_sec": dt,
    }


def main():
    from sge.stability import BENCHMARK_SPECS
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--problem", default="nguyen5", choices=list(BENCHMARK_SPECS))
    p.add_argument("--m", type=int, default=20)
    p.add_argument("--popsize", type=int, default=200)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--lf", type=float, default=0.01, help="PSGE LEARNING_FACTOR (reference)")
    p.add_argument("--etas", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--alpha", type=float, default=0.005)
    p.add_argument("--sa_lambda", type=float, default=0.5)
    p.add_argument("--sa_beta", type=float, default=1.0)
    p.add_argument("--sa_k_views", type=int, default=3)
    p.add_argument("--parameters", default="parameters/standard.yml")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--output_dir", default="dumps/calib_sa_eta")
    a = p.parse_args()

    out = os.path.join(a.output_dir, a.problem)
    os.makedirs(out, exist_ok=True)
    grammar_file = BENCHMARK_SPECS[a.problem]["grammar"]
    base = dict(problem=a.problem, m=a.m, popsize=a.popsize, generations=a.generations, lf=a.lf,
                grammar=grammar_file, parameters_file=a.parameters, out=out)
    tasks = []
    for i in range(a.n_seeds):
        seeds = dict(seed_data=CALIB_SEED_DATA_BASE + i, seed_evo=CALIB_SEED_EVO_BASE + i)
        tasks.append({**base, **seeds, "name": "psge", "algorithm": "psge", "alg_opts": {}})
        for eta in a.etas:
            tasks.append({**base, **seeds, "name": f"sa_eta{eta:g}", "algorithm": "sa_psge",
                          "alg_opts": {"SA_ETA": eta, "SA_ALPHA": a.alpha, "SA_LAMBDA": a.sa_lambda,
                                       "SA_BETA": a.sa_beta, "SA_K_VIEWS": a.sa_k_views, "SA_TAU": 1}})

    print(f"{len(tasks)} runs ({a.problem}, pop={a.popsize}, gen={a.generations}, alpha={a.alpha})")
    results = []
    with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result(); results.append(r)
            print(f"  {r['name']:<12} seed={r['seed_evo']}  medKL={r['median_kl']:.2e}  "
                  f"H={r['final_entropy']:.3f}  trainNRMSE={r['best_train_nrmse']:.3f}  {r['time_sec']:.0f}s")

    from collections import defaultdict
    g = defaultdict(list)
    for r in results:
        g[r["name"]].append(r)
    ref = float(np.median([r["median_kl"] for r in g["psge"]]))
    print("\nconfig        median KL/update   ratio vs PSGE   final entropy   train NRMSE (median)")
    rows = []
    for name in ["psge"] + [f"sa_eta{e:g}" for e in a.etas]:
        rs = g[name]
        kl = float(np.median([r["median_kl"] for r in rs]))
        row = {"name": name, "median_kl": kl, "ratio": kl / ref if ref > 0 else float("nan"),
               "final_entropy": float(np.median([r["final_entropy"] for r in rs])),
               "train_nrmse": float(np.median([r["best_train_nrmse"] for r in rs]))}
        rows.append(row)
        print(f"{name:<13} {kl:>14.3e}   {row['ratio']:>12.2f}   {row['final_entropy']:>13.3f}   {row['train_nrmse']:>10.3f}")

    # log-linear interpolation of eta giving ratio == 1
    sa = [(e, r["ratio"]) for e, r in zip(a.etas, rows[1:]) if r["ratio"] > 0]
    suggested = None
    for (e0, r0), (e1, r1) in zip(sa, sa[1:]):
        if (r0 - 1) * (r1 - 1) <= 0 and r0 != r1:
            t = (np.log(1) - np.log(r0)) / (np.log(r1) - np.log(r0))
            suggested = float(np.exp(np.log(e0) + t * (np.log(e1) - np.log(e0))))
            break
    print(f"\nSuggested SA_ETA (KL-matched to PSGE, alpha={a.alpha}): "
          f"{suggested if suggested is not None else 'outside tested range'}")
    with open(os.path.join(out, f"calib_alpha{a.alpha:g}.json"), "w") as fh:
        json.dump({"args": vars(a), "psge_ref_median_kl": ref, "rows": rows,
                   "suggested_sa_eta": suggested, "runs": results}, fh, indent=2)


if __name__ == "__main__":
    main()
