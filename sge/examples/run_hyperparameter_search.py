#!/usr/bin/env python3
"""
Hyperparameter calibration and grid search for PSGE-KL and SA-PSGE.

Phase 1 – KL Calibration:
  Runs standard PSGE on the target problem (default: nguyen5) for
  N_CALIB_SEEDS seeds in the main process, logging per-generation
  KL(P_{t+1} || P_t) via a monkeypatch on sge.engine.
  Reports percentile table and suggests a sensible delta for PSGE-KL.

Phase 2 – SA-PSGE Grid Search:
  Cartesian grid over SA_ETA x SA_ALPHA x SA_LAMBDA x SA_BETA (CLI lists)
  plus a PSGE baseline on the target problem.
  Uses N_DATA_SEEDS x N_EVO_SEEDS paired experiments per config.
  Per-config summary: mean over evo seeds within each data seed first,
  then median across data seeds (seeds are NOT independent datasets).
  Evaluates H2a (APG+ATG < PSGE) and H2b (NRMSE-VALIDATION <= 102% of PSGE).

  TEST HYGIENE: this script never generates nor evaluates the test set
  (n_test=0). Tuning decisions use the validation set only.

Usage (from psge-forked-jmcolmenar/sge/ with psge-original env active):
  python3 -m examples.run_hyperparameter_search
  python3 -m examples.run_hyperparameter_search --skip_phase1 --n_jobs 6
  python3 -m examples.run_hyperparameter_search --skip_phase2  # only calibrate

Reference:
  Plan_investigacion_GECCO_estabilidad_SGE_PSGE_SA_PSGE.docx (Secciones 8.1-8.5, 9.2)
"""

import os
import sys
import json
import time
import copy
import argparse
import itertools
import numpy as np
from examples.invalid_rule import agg_evo, count_invalid
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import sge
import sge.engine as _engine_mod
import sge.grammar as _grammar_mod
from sge.operators.stability_aware_update import compute_pcfg_kl_divergence
from sge.stability import (
    BENCHMARK_SPECS,
    generate_add_one_datasets,
    calculate_stability_metrics,
    calculate_jensen_shannon,
    StabilityEvaluationFunction,
)

# Default grid axes (override from CLI: --grid_eta/--grid_alpha/--grid_lambda/--grid_beta)
ETA_VALUES    = [0.02, 0.04, 0.08]     # around the KL-matched value 0.04
ALPHA_VALUES  = [0.001, 0.005, 0.01]
LAMBDA_VALUES = [0.5]
BETA_VALUES   = [1.0]


# =====================================================================
# PHASE 1: KL CALIBRATION
# =====================================================================

def run_kl_calibration(problem, n_seeds, popsize, generations, output_dir):
    """
    Run standard PSGE for n_seeds on `problem`, capturing the per-generation
    KL divergence D_KL(P_{t+1} || P_t) via a monkeypatch on sge.engine.

    The monkeypatch replaces the module-level binding in sge.engine (which is
    looked up by name each call), records the KL, then delegates to the
    original function. It is restored in a finally block.

    Returns:
        np.ndarray: all per-step KL values across all seeds and generations.
    """
    spec = BENCHMARK_SPECS[problem]
    grammar_file = spec["grammar"]
    all_kl = []

    _orig_fn = _engine_mod.perform_grammar_update

    def _logging_update(population, best_gen, flag, eval_func, iteration):
        pcfg_before = copy.deepcopy(_grammar_mod.get_pcfg())
        _orig_fn(population, best_gen, flag, eval_func, iteration)
        pcfg_after = _grammar_mod.get_pcfg()
        mask = _grammar_mod.get_mask()
        kl = compute_pcfg_kl_divergence(pcfg_after, pcfg_before, mask)
        all_kl.append(kl)

    _engine_mod.perform_grammar_update = _logging_update

    try:
        for idx in range(n_seeds):
            seed_data = 300 + idx
            seed_evo  = 3000 + idx
            data_bundle = generate_add_one_datasets(
                problem_name=problem, m=20, noise_ratio=0.0,
                seed_data=seed_data, n_test=0, n_val=0,
            )
            S = data_bundle["S"]

            sge.grammar.reset()
            eval_func = StabilityEvaluationFunction(dataset=S)
            opts = {
                "POPSIZE": popsize, "GENERATIONS": generations,
                "ELITISM": max(1, int(round(0.01 * popsize))),
                "LEARNING_FACTOR": 0.01, "SEED": seed_evo,
                "GRAMMAR": grammar_file,
                "EXPERIMENT_NAME": os.path.join(output_dir, "calib", f"seed{idx}"),
                "RUN": 1, "VERBOSE": False, "SAVE_STEP": 99999,
                "ALGORITHM": "psge",
            }
            sge.evolutionary_algorithm(
                evaluation_function=eval_func,
                parameters_file="parameters/standard.yml",
                options_dict=opts,
            )
            print(f"  Calib seed {idx + 1}/{n_seeds} done  "
                  f"({len(all_kl)} KL samples accumulated)")
    finally:
        _engine_mod.perform_grammar_update = _orig_fn   # always restore

    return np.array(all_kl)


def report_kl_calibration(kl_values, output_dir):
    """
    Print percentile table of per-step KL values observed in PSGE runs.
    Recommends delta = P25 (constrains ~75 % of updates).
    Also saves the raw array and returns the suggested delta.
    """
    np.save(os.path.join(output_dir, "kl_calibration_values.npy"), kl_values)
    nonzero = kl_values[kl_values > 1e-15]

    print("\n" + "=" * 65)
    print("PHASE 1 – KL CALIBRATION RESULTS")
    print("=" * 65)
    print(f"  Total per-generation KL samples : {len(kl_values)}")
    print(f"  Non-zero KL samples             : {len(nonzero)}")

    if len(nonzero) == 0:
        print("  WARNING: Grammar never updated — check LEARNING_FACTOR.")
        suggested = 0.001
    else:
        print(f"\n  {'Percentile':>12}  {'KL value':>14}")
        print(f"  {'-'*12}  {'-'*14}")
        for pct in [5, 10, 25, 50, 75, 90, 95, 99]:
            val = float(np.percentile(nonzero, pct))
            print(f"  P{pct:<3}        :  {val:.8f}")
        suggested = float(np.percentile(nonzero, 25))
        print(f"\n  Suggested delta for PSGE-KL : {suggested:.6f}  (= P25)")
        print(f"  Rationale: ~75 % of PSGE updates exceed this threshold")
        print(f"  and would be projected back.  P10 = "
              f"{float(np.percentile(nonzero, 10)):.6f} for tighter control.")

    print("=" * 65 + "\n")

    # Save human-readable summary
    calib_summary = {
        "n_samples": int(len(kl_values)),
        "n_nonzero": int(len(nonzero)),
        "percentiles": {
            f"P{p}": float(np.percentile(nonzero, p)) if len(nonzero) > 0 else 0.0
            for p in [5, 10, 25, 50, 75, 90, 95, 99]
        },
        "suggested_delta": suggested,
    }
    with open(os.path.join(output_dir, "kl_calibration_summary.json"), "w") as f:
        json.dump(calib_summary, f, indent=2)

    return suggested


# =====================================================================
# PHASE 2: GRID SEARCH WORKER (runs in a subprocess)
# =====================================================================

def _grid_worker(task):
    """
    Worker: one paired Add-One stability experiment for a grid config.
    Fully self-contained for subprocess execution.
    """
    import os, sys, json, time
    import numpy as np

    # Ensure the project root is importable in the subprocess
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    import sge
    import sge.grammar as gmod
    from sge.stability import (
        generate_add_one_datasets,
        calculate_stability_metrics,
        calculate_jensen_shannon,
        calculate_pcfg_entropy,
        StabilityEvaluationFunction,
    )

    config_name     = task["config_name"]
    algorithm       = task["algorithm"]
    alg_opts        = task.get("alg_opts", {})
    problem         = task["problem"]
    m               = task.get("m", 20)
    seed_data       = task["seed_data"]
    seed_evo        = task["seed_evo"]
    lf              = task["learning_factor"]
    popsize         = task["popsize"]
    generations     = task["generations"]
    grammar_file    = task["grammar_file"]
    parameters_file = task["parameters_file"]
    output_dir      = task["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{config_name}_d{seed_data}_e{seed_evo}.json")

    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                return json.load(f), True
        except Exception:
            pass  # Corrupted → re-run

    data_bundle = generate_add_one_datasets(
        problem_name=problem, m=m, noise_ratio=0.0,
        seed_data=seed_data, n_test=0, n_val=500,   # NO test set during tuning
    )
    S        = data_bundle["S"]
    z        = data_bundle["z"]
    S_plus_z = data_bundle["S_plus_z"]
    val_set  = data_bundle["val_set"]
    var_y    = data_bundle["var_y_S"]

    def run_arm(dataset, arm_suffix):
        sge.grammar.reset()
        arm_dir = os.path.join(
            output_dir,
            f"{config_name}_d{seed_data}_e{seed_evo}",
            arm_suffix,
        )
        opts = {
            "POPSIZE": popsize, "GENERATIONS": generations,
            "ELITISM": max(1, int(round(0.01 * popsize))),
            "LEARNING_FACTOR": lf, "SEED": seed_evo,
            "GRAMMAR": grammar_file, "EXPERIMENT_NAME": arm_dir,
            "RUN": 1, "VERBOSE": False, "SAVE_STEP": 99999,
            "ALGORITHM": algorithm,
            **alg_opts,
        }
        eval_f = StabilityEvaluationFunction(dataset=dataset)
        return sge.evolutionary_algorithm(
            evaluation_function=eval_f,
            parameters_file=parameters_file,
            options_dict=opts,
        )

    t0 = time.time()
    best_S,  pcfg_S  = run_arm(S,        "arm_S")
    best_Sz, pcfg_Sz = run_arm(S_plus_z, "arm_Sz")
    elapsed = time.time() - t0

    mask    = gmod.get_mask()
    metrics = calculate_stability_metrics(
        model_S_phenotype=best_S["phenotype"],
        model_S_plus_z_phenotype=best_Sz["phenotype"],
        S=S, z=z, test_set=None, var_y=var_y, val_set=val_set,
    )
    js_div, _ = calculate_jensen_shannon(pcfg_S, pcfg_Sz, mask)

    result = {
        "config_name":  config_name,
        "algorithm":    algorithm,
        "alg_opts":     {k: float(v) if isinstance(v, (int, float)) else v
                         for k, v in alg_opts.items()},
        "problem":      problem,
        "seed_data":    seed_data,
        "seed_evo":     seed_evo,
        "learning_factor": lf,
        "apg_plus_atg": float(metrics["data_stability_norm"]),
        "nrmse_val":    float(metrics["nrmse_val_fSz"]),
        "nrmse_val_fS": float(metrics["nrmse_val_fS"]),
        "nrmse_train":  float(metrics["nrmse_train_fSz"]),
        "nrmse_train_fS": float(metrics["nrmse_train_fS"]),
        "apg_norm":     float(metrics["apg_norm"]),
        "atg_norm":     float(metrics["atg_norm"]),
        "entropy_S":    float(calculate_pcfg_entropy(pcfg_S, mask)),
        "entropy_Sz":   float(calculate_pcfg_entropy(pcfg_Sz, mask)),
        "m":            m,
        "phenotype_S":  best_S["phenotype"],
        "phenotype_Sz": best_Sz["phenotype"],
        "js_divergence": float(js_div),
        "time_sec":     elapsed,
    }
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    return result, False


# =====================================================================
# PHASE 2: TASK BUILDER
# =====================================================================

def build_grid_tasks(problem, n_data_seeds, n_evo_seeds, popsize, generations,
                     parameters_file, output_dir, fixed_sa_opts,
                     eta_values=None, alpha_values=None, lambda_values=None, beta_values=None):
    """
    Returns the full task list:
      - 1 PSGE baseline config
      - len(LAMBDA_VALUES) x len(BETA_VALUES) SA-PSGE configs
    each with n_data_seeds x n_evo_seeds seed combinations.
    """
    spec = BENCHMARK_SPECS[problem]
    grammar_file = spec["grammar"]

    seed_data_list = [400 + i for i in range(n_data_seeds)]
    seed_evo_list  = [4000 + i for i in range(n_evo_seeds)]

    tasks = []
    for sd, se in itertools.product(seed_data_list, seed_evo_list):

        # PSGE baseline (no SA opts needed)
        tasks.append({
            "config_name": "psge_baseline",
            "algorithm": "psge",
            "alg_opts": {},
            "problem": problem, "m": 20,
            "seed_data": sd, "seed_evo": se,
            "learning_factor": 0.01,
            "popsize": popsize, "generations": generations,
            "grammar_file": grammar_file,
            "parameters_file": parameters_file,
            "output_dir": os.path.join(output_dir, "psge_baseline"),
        })

        # SA-PSGE grid
        for eta, alpha, lam, beta in itertools.product(
                eta_values or ETA_VALUES, alpha_values or ALPHA_VALUES,
                lambda_values or LAMBDA_VALUES, beta_values or BETA_VALUES):
            tag = lambda v: f"{v:g}".replace(".", "p")
            config_name = f"sa_eta{tag(eta)}_alp{tag(alpha)}_lam{tag(lam)}_bet{tag(beta)}"
            # K/tau are part of the cache key only when non-default, so earlier
            # K=3/tau=1 results keep their names and stay reusable.
            k_views, tau = fixed_sa_opts["SA_K_VIEWS"], fixed_sa_opts["SA_TAU"]
            if (k_views, tau) != (3, 1):
                config_name += f"_k{k_views}_tau{tau}"
            sa_opts = {**fixed_sa_opts, "SA_ETA": eta, "SA_ALPHA": alpha,
                       "SA_LAMBDA": lam, "SA_BETA": beta}
            tasks.append({
                "config_name": config_name,
                "algorithm": "sa_psge",
                "alg_opts": sa_opts,
                "problem": problem, "m": 20,
                "seed_data": sd, "seed_evo": se,
                "learning_factor": 0.01,
                "popsize": popsize, "generations": generations,
                "grammar_file": grammar_file,
                "parameters_file": parameters_file,
                "output_dir": os.path.join(output_dir, config_name),
            })

    return tasks


# =====================================================================
# PHASE 2: ANALYSIS & REPORTING
# =====================================================================

def analyze_grid_results(all_results, output_dir):
    """
    Groups results by config_name, computes median/IQR, evaluates H2a+H2b,
    prints a ranked decision table, and saves a JSON summary.

    H2a: APG+ATG median < PSGE baseline median  (SA-PSGE is more stable)
    H2b: NRMSE-validation median <= 1.02 * PSGE baseline median  (cost <= 2%)

    Aggregation: for each config, metrics are first aggregated over evo seeds
    within each data seed with the MEDIAN; medians/IQRs are then taken across
    data seeds. Invalid runs follow examples/invalid_rule.py (+inf, never dropped,
    counted in "invalid_runs").
    "n" is the number of data seeds (independent replicas).
    The column labelled NRMSE-val in older outputs is now NRMSE-val.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for r in all_results:
        groups[r["config_name"]].append(r)

    def per_data_seed(recs, key):
        by_sd = defaultdict(list)
        for r in recs:
            by_sd[r["seed_data"]].append(r[key])
        if key == "js_divergence":
            return np.array([np.median(v) for _, v in sorted(by_sd.items())])
        return np.array([agg_evo(v) for _, v in sorted(by_sd.items())])

    summary = []
    for name, recs in groups.items():
        stab  = per_data_seed(recs, "apg_plus_atg")
        val   = per_data_seed(recs, "nrmse_val")   # validation, NOT test
        js    = per_data_seed(recs, "js_divergence")
        aopts = recs[0]["alg_opts"]
        summary.append({
            "config_name":    name,
            "algorithm":      recs[0]["algorithm"],
            "sa_eta":         float(aopts.get("SA_ETA", float("nan"))),
            "sa_alpha":       float(aopts.get("SA_ALPHA", float("nan"))),
            "sa_lambda":      float(aopts.get("SA_LAMBDA", float("nan"))),
            "sa_beta":        float(aopts.get("SA_BETA",   float("nan"))),
            "n":              len(stab),
            "n_runs":         len(recs),
            "med_apg_atg":   float(np.median(stab)),
            "iqr_apg_atg":   float(np.percentile(stab, 75) - np.percentile(stab, 25)),
            "med_nrmse_val": float(np.median(val)),
            "iqr_nrmse_val": float(np.percentile(val, 75) - np.percentile(val, 25)),
            "med_js":         float(np.median(js)),
            "invalid_runs":   sum(1 for r in recs
                                  if count_invalid([r["apg_plus_atg"], r["nrmse_val"]])),
        })

    # Locate PSGE baseline
    baseline = next((s for s in summary if s["config_name"] == "psge_baseline"), None)
    if baseline is None:
        print("WARNING: No PSGE baseline results — cannot evaluate H2a/H2b.")
        return summary, None

    b_stab = baseline["med_apg_atg"]
    b_val = baseline["med_nrmse_val"]

    # Sort SA-PSGE configs by APG+ATG ascending (more stable = better for H2a)
    sa_configs = sorted(
        [s for s in summary if s["algorithm"] == "sa_psge"],
        key=lambda x: x["med_apg_atg"],
    )

    # ---- Print decision table --------------------------------------- #
    COL = 32
    print("\n" + "=" * 90)
    print("PHASE 2 – SA-PSGE GRID SEARCH: DECISION TABLE")
    print(f"  Problem: {all_results[0]['problem']}  |  "
          f"PSGE baseline: APG+ATG={b_stab:.4f}  NRMSE-val={b_val:.4f}")
    print("=" * 90)
    COL = 40
    hdr = (f"{'Config':<{COL}} {'η':>5} {'α':>6} {'n':>3}"
           f"  {'APG+ATG':>9} {'Δ stab':>8}"
           f"  {'NRMSE-val':>9} {'Δ val%':>8}"
           f"  {'H2a':>3} {'H2b':>3}")
    print(hdr)
    print("-" * 90)

    # Print baseline row
    print(f"{'psge_baseline':<{COL}} {'—':>5} {'—':>6} {baseline['n']:>3}"
          f"  {b_stab:>9.4f} {'(ref)':>8}"
          f"  {b_val:>9.4f} {'(ref)':>8}"
          f"  {'ref':>3} {'ref':>3}  inv={baseline['invalid_runs']}/{baseline['n_runs']}")
    print("-" * 90)

    best_h2ab = None
    best_h2a_only = None

    for s in sa_configs:
        delta_stab     = s["med_apg_atg"] - b_stab
        delta_val_pct = (s["med_nrmse_val"] - b_val) / (b_val + 1e-12) * 100.0
        h2a = s["med_apg_atg"]   < b_stab
        h2b = s["med_nrmse_val"] <= b_val * 1.02

        h2a_str = "✓" if h2a else "✗"
        h2b_str = "✓" if h2b else "✗"

        print(f"{s['config_name']:<{COL}} {s['sa_eta']:>5.3f} {s['sa_alpha']:>6.3f} {s['n']:>3}"
              f"  {s['med_apg_atg']:>9.4f} {delta_stab:>+8.4f}"
              f"  {s['med_nrmse_val']:>9.4f} {delta_val_pct:>+7.1f}%"
              f"  {h2a_str:>3} {h2b_str:>3}  inv={s['invalid_runs']}/{s['n_runs']}")

        if h2a and h2b and best_h2ab is None:
            best_h2ab = s
        if h2a and best_h2a_only is None:
            best_h2a_only = s

    print("=" * 90)

    # ---- Verdict --------------------------------------------------- #
    print("\n── VERDICT ──")
    if best_h2ab:
        print(f"  ✅ Best config (H2a ∧ H2b):  {best_h2ab['config_name']}")
        print(f"     APG+ATG  = {best_h2ab['med_apg_atg']:.4f}  "
              f"(Δ = {best_h2ab['med_apg_atg'] - b_stab:+.4f} vs PSGE)")
        print(f"     NRMSE-val = {best_h2ab['med_nrmse_val']:.4f}  "
              f"(Δ = {(best_h2ab['med_nrmse_val']-b_val)/b_val*100:+.1f}% vs PSGE)")
    elif best_h2a_only:
        print(f"  ⚠ H2a satisfied but H2b violated — stability improves at cost of accuracy.")
        print(f"    Best H2a config: {best_h2a_only['config_name']}")
    else:
        print("  ✗ No SA-PSGE config satisfies H2a (stability does not improve over PSGE).")
        print("    Possible causes: too few generations, underpowered pilot n, "
              "or SA signal needs tuning.")

    print("=" * 90 + "\n")

    # ---- Save JSON summary ----------------------------------------- #
    summary_path = os.path.join(output_dir, "gridsearch_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "baseline": baseline,
            "sa_configs_ranked": sa_configs,
            "best_h2ab": best_h2ab,
            "best_h2a_only": best_h2a_only,
        }, f, indent=2)
    print(f"  Grid search summary saved: {summary_path}")

    return summary, best_h2ab


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="KL calibration (Phase 1) + SA-PSGE grid search (Phase 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--problem",        default="nguyen5",
                   choices=list(BENCHMARK_SPECS.keys()),
                   help="Benchmark problem to use for both phases")
    p.add_argument("--popsize",        type=int, default=200)
    p.add_argument("--generations",    type=int, default=50)
    p.add_argument("--n_calib_seeds",  type=int, default=3,
                   help="PSGE seeds for KL calibration (Phase 1)")
    p.add_argument("--n_data_seeds",   type=int, default=2,
                   help="Data seeds per grid config (Phase 2)")
    p.add_argument("--n_evo_seeds",    type=int, default=2,
                   help="Evo seeds per data seed (Phase 2)")
    p.add_argument("--n_jobs",         type=int, default=4,
                   help="Parallel workers for Phase 2")
    p.add_argument("--parameters",     default="parameters/standard.yml")
    p.add_argument("--output_dir",     default="dumps/hyperparam_search")
    p.add_argument("--skip_phase1",    action="store_true",
                   help="Skip KL calibration; use --fallback_delta as delta")
    p.add_argument("--skip_phase2",    action="store_true",
                   help="Run calibration only (skip grid search)")
    p.add_argument("--fallback_delta", type=float, default=6e-5,
                   help="delta to use for PSGE-KL when --skip_phase1 is set")
    # Fixed SA-PSGE hyperparams (not swept)
    p.add_argument("--sa_k_views",     type=int,   default=3)
    p.add_argument("--grid_eta",       type=float, nargs="+", default=ETA_VALUES)
    p.add_argument("--grid_alpha",     type=float, nargs="+", default=ALPHA_VALUES)
    p.add_argument("--grid_lambda",    type=float, nargs="+", default=LAMBDA_VALUES)
    p.add_argument("--grid_beta",      type=float, nargs="+", default=BETA_VALUES)
    p.add_argument("--sa_p_min",       type=float, default=0.001)
    p.add_argument("--sa_tau",         type=int,   default=1)
    p.add_argument("--sa_elite_ratio", type=float, default=0.10)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    n_grid_configs = 1 + (len(args.grid_eta) * len(args.grid_alpha)
                          * len(args.grid_lambda) * len(args.grid_beta))
    n_seeds_per    = args.n_data_seeds * args.n_evo_seeds

    print("=" * 75)
    print("HYPERPARAMETER CALIBRATION & GRID SEARCH")
    print("=" * 75)
    print(f"  Problem        : {args.problem}")
    print(f"  Pop / Gen      : {args.popsize} / {args.generations}")
    print(f"  Output dir     : {args.output_dir}")
    if not args.skip_phase1:
        print(f"  Phase 1        : KL calibration ({args.n_calib_seeds} seeds)")
    if not args.skip_phase2:
        print(f"  Phase 2        : {n_grid_configs} configs x {n_seeds_per} seeds "
              f"= {n_grid_configs * n_seeds_per} paired runs | {args.n_jobs} workers")
        print(f"  Grid           : η ∈ {args.grid_eta} × α ∈ {args.grid_alpha} × "
              f"λ ∈ {args.grid_lambda} × β ∈ {args.grid_beta}")
    print("=" * 75)

    # ---------------------------------------------------------------- #
    # PHASE 1 – KL Calibration                                         #
    # ---------------------------------------------------------------- #
    if not args.skip_phase1:
        print(f"\n>> PHASE 1: KL Calibration  ({args.n_calib_seeds} PSGE seeds, "
              f"pop={args.popsize}, gen={args.generations})")
        kl_vals = run_kl_calibration(
            problem=args.problem,
            n_seeds=args.n_calib_seeds,
            popsize=args.popsize,
            generations=args.generations,
            output_dir=args.output_dir,
        )
        suggested_delta = report_kl_calibration(kl_vals, args.output_dir)
    else:
        print(f"\n>> PHASE 1: Skipped  (using fallback delta = {args.fallback_delta})")
        suggested_delta = args.fallback_delta

    # ---------------------------------------------------------------- #
    # PHASE 2 – SA-PSGE Grid Search                                    #
    # ---------------------------------------------------------------- #
    if args.skip_phase2:
        print("\n>> PHASE 2: Skipped (--skip_phase2)")
        return

    # Persist the calibrated delta alongside the grid results
    with open(os.path.join(args.output_dir, "calibrated_kl_delta.json"), "w") as f:
        json.dump({"kl_delta_suggested": suggested_delta}, f, indent=2)

    fixed_sa_opts = {
        "SA_K_VIEWS":     args.sa_k_views,
        "SA_P_MIN":       args.sa_p_min,
        "SA_TAU":         args.sa_tau,
        "SA_ELITE_RATIO": args.sa_elite_ratio,
    }

    tasks = build_grid_tasks(
        problem=args.problem,
        n_data_seeds=args.n_data_seeds,
        n_evo_seeds=args.n_evo_seeds,
        popsize=args.popsize,
        generations=args.generations,
        parameters_file=args.parameters,
        output_dir=args.output_dir,
        fixed_sa_opts=fixed_sa_opts,
        eta_values=args.grid_eta, alpha_values=args.grid_alpha,
        lambda_values=args.grid_lambda, beta_values=args.grid_beta,
    )

    total = len(tasks)
    print(f"\n>> PHASE 2: Grid Search  ({total} paired experiments, {args.n_jobs} workers)")

    all_results = []
    completed   = 0
    t_start     = time.time()

    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(_grid_worker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                res, cached = future.result()
                all_results.append(res)
                completed += 1
                cache_tag = "cached" if cached else f"{res['time_sec']:.1f}s"
                print(f"  [{completed:>3}/{total}]  {res['config_name']:<32}  "
                      f"APG+ATG={res['apg_plus_atg']:.4f}  "
                      f"NRMSE-val={res['nrmse_val']:.4f}  [{cache_tag}]")
            except Exception as exc:
                completed += 1
                t = futures[future]
                print(f"  [{completed:>3}/{total}]  ERROR in {t['config_name']}: {exc}")

    elapsed = time.time() - t_start
    print(f"\n  All {total} tasks done in {elapsed:.1f}s.")

    analyze_grid_results(all_results, args.output_dir)


if __name__ == "__main__":
    main()
