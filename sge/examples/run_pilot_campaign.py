#!/usr/bin/env python3
"""
OBSOLETO para RQ1: usar examples/run_rq1_pilot.py + examples/analyze_rq1.py (agregación por semilla de
datos, regla de inválidos, solo validación y estadística correcta). Se conserva por historial.

Automated Pilot Campaign for SGE / PSGE / PSGE-KL / SA-PSGE Algorithmic Stability.

Executes paired Add-One stability experiments across:
  - Multiple benchmarks (e.g., Nguyen-1 and Nguyen-5)
  - Multiple data seeds (replicas of S and z)
  - Multiple evolutionary seeds (paired identically for A(S) and A(S U {z}))
  - All four algorithms: sge, psge, psge_kl, sa_psge

Features:
  - Multi-process parallel execution across CPU cores.
  - Automatic resume: skips already completed (seed, problem, algorithm) pairs.
  - Automated statistical summary: medians, IQRs, pairwise Wilcoxon signed-rank tests,
    Spearman correlation between JS and APG+ATG, and 4-way comparison tables.

Reference:
  Plan_investigacion_GECCO_estabilidad_SGE_PSGE_SA_PSGE.docx (Secciones 6, 7, 9, 11, 16)
"""

import os
import sys
import json
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import sge
from sge.stability import (
    BENCHMARK_SPECS,
    generate_add_one_datasets,
    calculate_stability_metrics,
    calculate_jensen_shannon,
    calculate_pcfg_entropy,
    StabilityEvaluationFunction,
)


def run_single_stability_experiment(task):
    """
    Worker task executing a single paired Add-One experiment:
      A(S; s) vs A(S U {z}; s)
    Runs in its own isolated process space.
    """
    problem = task["problem"]
    algorithm = task["algorithm"]
    m = task["m"]
    noise = task["noise"]
    seed_data = task["seed_data"]
    seed_evo = task["seed_evo"]
    learning_factor = task["learning_factor"]
    popsize = task["popsize"]
    generations = task["generations"]
    elitism = task["elitism"]
    grammar_file = task["grammar_file"]
    parameters_file = task["parameters_file"]
    output_dir = task["output_dir"]
    alg_opts = task.get("alg_opts", {})  # SA-PSGE / PSGE-KL hyperparams

    filename_base = f"{algorithm}_{problem}_m{m}_sData{seed_data}_sEvo{seed_evo}"
    json_path = os.path.join(output_dir, f"{filename_base}.json")

    # If already completed, load and return existing record
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                saved = json.load(f)
            return saved, True
        except Exception:
            pass  # Corrupted, re-run

    # Generate paired datasets
    data_bundle = generate_add_one_datasets(
        problem_name=problem,
        m=m,
        noise_ratio=noise,
        seed_data=seed_data,
        n_test=1000,
    )
    S = data_bundle["S"]
    z = data_bundle["z"]
    S_plus_z = data_bundle["S_plus_z"]
    test_set = data_bundle["test_set"]
    var_y_S = data_bundle["var_y_S"]

    # --- Arm A: A(S; seed_evo) ---
    sge.grammar.reset()
    arm_A_dir = os.path.join(output_dir, filename_base, "arm_S")
    custom_opts_A = {
        "POPSIZE": popsize,
        "GENERATIONS": generations,
        "ELITISM": elitism,
        "LEARNING_FACTOR": learning_factor,
        "SEED": seed_evo,
        "GRAMMAR": grammar_file,
        "EXPERIMENT_NAME": arm_A_dir,
        "RUN": 1,
        "VERBOSE": False,
        "SAVE_STEP": 10000,
        "ALGORITHM": algorithm,
        **alg_opts,
    }
    eval_func_A = StabilityEvaluationFunction(dataset=S, test_set=test_set)
    t0 = time.time()
    best_S, pcfg_S = sge.evolutionary_algorithm(
        evaluation_function=eval_func_A,
        parameters_file=parameters_file,
        options_dict=custom_opts_A,
    )
    time_S = time.time() - t0

    # --- Arm B: A(S U {z}; seed_evo) ---
    sge.grammar.reset()
    arm_B_dir = os.path.join(output_dir, filename_base, "arm_Sz")
    custom_opts_B = {
        "POPSIZE": popsize,
        "GENERATIONS": generations,
        "ELITISM": elitism,
        "LEARNING_FACTOR": learning_factor,
        "SEED": seed_evo,
        "GRAMMAR": grammar_file,
        "EXPERIMENT_NAME": arm_B_dir,
        "RUN": 1,
        "VERBOSE": False,
        "SAVE_STEP": 10000,
        "ALGORITHM": algorithm,
        **alg_opts,
    }
    eval_func_B = StabilityEvaluationFunction(dataset=S_plus_z, test_set=test_set)
    t1 = time.time()
    best_Sz, pcfg_Sz = sge.evolutionary_algorithm(
        evaluation_function=eval_func_B,
        parameters_file=parameters_file,
        options_dict=custom_opts_B,
    )
    time_Sz = time.time() - t1

    # Stability & grammar metrics
    metrics = calculate_stability_metrics(
        model_S_phenotype=best_S["phenotype"],
        model_S_plus_z_phenotype=best_Sz["phenotype"],
        S=S,
        z=z,
        test_set=test_set,
        var_y=var_y_S,
    )
    mask = sge.grammar.get_mask()
    js_divergence, _ = calculate_jensen_shannon(pcfg_S, pcfg_Sz, mask)
    entropy_S = calculate_pcfg_entropy(pcfg_S, mask)
    entropy_Sz = calculate_pcfg_entropy(pcfg_Sz, mask)

    result = {
        "metadata": {
            "problem": problem,
            "algorithm": algorithm,
            "m": m,
            "noise": noise,
            "seed_data": seed_data,
            "seed_evo": seed_evo,
            "learning_factor": learning_factor,
            "popsize": popsize,
            "generations": generations,
            "grammar": grammar_file,
            "time_S_sec": time_S,
            "time_Sz_sec": time_Sz,
            "time_total_sec": time_S + time_Sz,
        },
        "models": {
            "f_S_phenotype": best_S["phenotype"],
            "f_Sz_phenotype": best_Sz["phenotype"],
            "f_S_tree_depth": best_S.get("tree_depth"),
            "f_Sz_tree_depth": best_Sz.get("tree_depth"),
        },
        "stability_metrics": {
            "apg_norm": metrics["apg_norm"],
            "atg_norm": metrics["atg_norm"],
            "data_stability_norm": metrics["data_stability_norm"],
        },
        "grammar_dynamics": {
            "jensen_shannon_divergence": js_divergence,
            "entropy_norm_S": entropy_S,
            "entropy_norm_Sz": entropy_Sz,
        },
        "generalization_metrics": {
            "nrmse_train_fS": metrics["nrmse_train_fS"],
            "nrmse_test_fS": metrics["nrmse_test_fS"],
            "gen_gap_fS": metrics["gen_gap_fS"],
            "nrmse_train_fSz": metrics["nrmse_train_fSz"],
            "nrmse_test_fSz": metrics["nrmse_test_fSz"],
            "gen_gap_fSz": metrics["gen_gap_fSz"],
        },
    }

    # Save individual JSON
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    return result, False


# =====================================================================
# STATISTICAL ANALYSIS & REPORTING HELPERS
# =====================================================================

def compute_wilcoxon_signed_rank(x, y):
    """
    Computes Wilcoxon signed-rank test for paired samples.
    Uses scipy.stats if available, otherwise pure numpy fallback.
    """
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(x, y, zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        # Fallback implementation
        diffs = np.array(x) - np.array(y)
        diffs = diffs[diffs != 0]
        n = len(diffs)
        if n < 5:
            return float(np.sum(diffs > 0)), 1.0
        ranks = np.argsort(np.argsort(np.abs(diffs))) + 1
        pos_sum = np.sum(ranks[diffs > 0])
        neg_sum = np.sum(ranks[diffs < 0])
        t_stat = min(pos_sum, neg_sum)
        # Normal approximation
        mean_t = n * (n + 1) / 4.0
        var_t = n * (n + 1) * (2 * n + 1) / 24.0
        z = (t_stat - mean_t) / np.sqrt(var_t)
        # Approximate two-sided p-value
        from math import erf
        p_val = 2.0 * (0.5 * (1.0 + erf(z / np.sqrt(2.0))))
        return float(t_stat), float(p_val)


def compute_spearman_rank_correlation(x, y):
    """
    Computes Spearman rank correlation between two vectors.
    """
    try:
        from scipy.stats import spearmanr
        res = spearmanr(x, y)
        return float(res.correlation), float(res.pvalue)
    except Exception:
        x_rank = np.argsort(np.argsort(x)).astype(float)
        y_rank = np.argsort(np.argsort(y)).astype(float)
        corr = float(np.corrcoef(x_rank, y_rank)[0, 1])
        return corr, 1.0


def _alg_stats(records, metric_key, sub_key=None):
    """Extract a numpy array for a metric from a list of result records."""
    if sub_key:
        return np.array([r[metric_key][sub_key] for r in records])
    return np.array([r[metric_key] for r in records])


def _fmt(median, iqr):
    return f"{median:.4f} [{iqr:.4f}]"


def _median_iqr(arr):
    return np.median(arr), np.percentile(arr, 75) - np.percentile(arr, 25)


def analyze_campaign_results(results, output_dir):
    """
    Consolidates raw paired experimental runs into publication-ready tables:
      - Medians, IQRs for all four algorithms (sge, psge, psge_kl, sa_psge)
      - Pairwise Wilcoxon Signed-Rank Tests:
          * PSGE vs SGE         (RQ1: does grammar adaptation hurt stability?)
          * SA-PSGE vs PSGE     (RQ2a: does stability-aware signal help?)
          * PSGE-KL vs PSGE     (RQ2b: does KL constraint help?)
          * SA-PSGE vs PSGE-KL  (RQ2c: signal vs mere smoothing)
      - Spearman rho (JS vs APG+ATG) for PSGE and SA-PSGE separately (H1b, H2d)
    """
    summary_csv_path = os.path.join(output_dir, "pilot_campaign_summary.csv")
    report_md_path = os.path.join(output_dir, "pilot_analysis_report.md")

    # ------------------------------------------------------------------ #
    # 1. Write full tabular CSV                                           #
    # ------------------------------------------------------------------ #
    header = (
        "problem,algorithm,m,noise,seed_data,seed_evo,learning_factor,"
        "apg_norm,atg_norm,apg_plus_atg_norm,js_divergence,"
        "nrmse_test_fS,nrmse_test_fSz,gen_gap_fSz,time_total_sec\n"
    )
    with open(summary_csv_path, "w") as f:
        f.write(header)
        for r in results:
            meta = r["metadata"]
            stab = r["stability_metrics"]
            gram = r["grammar_dynamics"]
            gen = r["generalization_metrics"]
            row = (
                f"{meta['problem']},{meta['algorithm']},{meta['m']},{meta['noise']},"
                f"{meta['seed_data']},{meta['seed_evo']},{meta['learning_factor']},"
                f"{stab['apg_norm']:.6e},{stab['atg_norm']:.6e},{stab['data_stability_norm']:.6e},"
                f"{gram['jensen_shannon_divergence']:.6e},"
                f"{gen['nrmse_test_fS']:.6e},{gen['nrmse_test_fSz']:.6e},{gen['gen_gap_fSz']:.6e},"
                f"{meta['time_total_sec']:.2f}\n"
            )
            f.write(row)

    # ------------------------------------------------------------------ #
    # 2. Detect which algorithms are present in the results               #
    # ------------------------------------------------------------------ #
    present_algs = sorted(set(r["metadata"]["algorithm"] for r in results),
                          key=lambda a: ["sge", "psge", "psge_kl", "sa_psge"].index(a)
                          if a in ["sge", "psge", "psge_kl", "sa_psge"] else 99)

    ALG_LABELS = {
        "sge": "SGE",
        "psge": "PSGE",
        "psge_kl": "PSGE-KL",
        "sa_psge": "SA-PSGE",
    }

    # Pairwise comparisons to report (only if both algorithms present)
    COMPARISONS = [
        ("psge",     "sge",      "RQ1 – Grammar adaptation vs Baseline"),
        ("sa_psge",  "psge",     "RQ2a – SA-PSGE vs PSGE"),
        ("psge_kl",  "psge",     "RQ2b – PSGE-KL vs PSGE"),
        ("sa_psge",  "psge_kl",  "RQ2c – SA-PSGE vs PSGE-KL"),
    ]

    unique_conditions = sorted(
        set((r["metadata"]["problem"], r["metadata"]["m"], r["metadata"]["noise"])
            for r in results)
    )

    report_lines = []
    report_lines.append("# Informe de Resultados de la Campaña Piloto\n")
    report_lines.append(f"Generado automáticamente: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append(f"Total de ejecuciones analizadas: {len(results)}\n")
    report_lines.append(f"Algoritmos presentes: {', '.join(ALG_LABELS.get(a, a) for a in present_algs)}\n\n")

    for prob, m, noise in unique_conditions:
        report_lines.append(f"## Benchmark: `{prob}` (m={m}, ruido={noise*100:.0f}%)\n\n")

        def sub(alg):
            return [r for r in results
                    if r["metadata"]["problem"] == prob
                    and r["metadata"]["algorithm"] == alg
                    and r["metadata"]["m"] == m
                    and r["metadata"]["noise"] == noise]

        alg_records = {a: sub(a) for a in present_algs}

        # ---- Summary table (one row per algorithm) -------------------- #
        report_lines.append("### Resumen por algoritmo\n")
        report_lines.append("| Algoritmo | n | APG+ATG Mediana [IQR] | NRMSE-Test Mediana [IQR] | JS Mediana [IQR] |")
        report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for alg in present_algs:
            recs = alg_records[alg]
            if not recs:
                continue
            stab = _alg_stats(recs, "stability_metrics", "data_stability_norm")
            test = _alg_stats(recs, "generalization_metrics", "nrmse_test_fSz")
            js   = _alg_stats(recs, "grammar_dynamics", "jensen_shannon_divergence")
            report_lines.append(
                f"| **{ALG_LABELS.get(alg, alg)}** | {len(recs)} "
                f"| {_fmt(*_median_iqr(stab))} "
                f"| {_fmt(*_median_iqr(test))} "
                f"| {_fmt(*_median_iqr(js))} |"
            )
        report_lines.append("\n")

        # ---- Pairwise statistical tests ------------------------------- #
        report_lines.append("### Tests estadísticos por pares (Wilcoxon signed-rank)\n")
        report_lines.append("| Comparación | Hipótesis | Δ APG+ATG (trat−ctrl) | p-valor Estabilidad | p-valor NRMSE-Test | Interpretación |")
        report_lines.append("| :--- | :--- | :---: | :---: | :---: | :--- |")

        for trt_alg, ctrl_alg, label in COMPARISONS:
            if trt_alg not in present_algs or ctrl_alg not in present_algs:
                continue
            trt_recs  = alg_records[trt_alg]
            ctrl_recs = alg_records[ctrl_alg]
            if not trt_recs or not ctrl_recs:
                continue

            stab_trt  = _alg_stats(trt_recs,  "stability_metrics",      "data_stability_norm")
            stab_ctrl = _alg_stats(ctrl_recs,  "stability_metrics",      "data_stability_norm")
            test_trt  = _alg_stats(trt_recs,   "generalization_metrics", "nrmse_test_fSz")
            test_ctrl = _alg_stats(ctrl_recs,  "generalization_metrics", "nrmse_test_fSz")

            n_pairs = min(len(stab_trt), len(stab_ctrl))
            _, p_stab = compute_wilcoxon_signed_rank(stab_trt[:n_pairs], stab_ctrl[:n_pairs])
            _, p_test = compute_wilcoxon_signed_rank(test_trt[:n_pairs], test_ctrl[:n_pairs])

            delta_stab = np.median(stab_trt) - np.median(stab_ctrl)
            trt_label  = ALG_LABELS.get(trt_alg, trt_alg)
            ctrl_label = ALG_LABELS.get(ctrl_alg, ctrl_alg)

            # Interpret stability direction (negative = treatment MORE stable)
            if p_stab < 0.05:
                direction = "✓ Más estable" if delta_stab < 0 else "✗ Menos estable"
                interp = f"{direction} (significativo)"
            else:
                interp = "Sin diferencia significativa"

            report_lines.append(
                f"| {trt_label} vs {ctrl_label} | {label} "
                f"| {delta_stab:+.4f} "
                f"| p = {p_stab:.4e} "
                f"| p = {p_test:.4e} "
                f"| {interp} |"
            )
        report_lines.append("\n")

        # ---- Spearman H1b / H2d --------------------------------------- #
        report_lines.append("### Correlación JS vs APG+ATG (Spearman ρ)\n")
        for alg in present_algs:
            if alg == "sge":
                continue  # SGE grammar never changes; JS always 0
            recs = alg_records[alg]
            if len(recs) < 3:
                continue
            js   = _alg_stats(recs, "grammar_dynamics", "jensen_shannon_divergence")
            stab = _alg_stats(recs, "stability_metrics", "data_stability_norm")
            rho, p_rho = compute_spearman_rank_correlation(js, stab)
            hyp = "H1b" if alg == "psge" else "H2d"
            report_lines.append(
                f"- **{ALG_LABELS.get(alg, alg)}** ({hyp}): "
                f"ρ = {rho:.4f} (p = {p_rho:.4e}, n = {len(recs)})\n"
            )
        report_lines.append("\n")

    report_text = "".join(report_lines)
    with open(report_md_path, "w") as f:
        f.write(report_text)

    print("\n" + "=" * 70)
    print("CAMPAIGN ANALYSIS REPORT SUMMARY")
    print("=" * 70)
    print(report_text)
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Report MD:   {report_md_path}")
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run automated pilot campaign for SGE / PSGE / PSGE-KL / SA-PSGE stability."
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=["nguyen1", "nguyen5"],
        choices=list(BENCHMARK_SPECS.keys()),
        help="List of benchmark problems to run (default: nguyen1 nguyen5)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["sge", "psge"],
        choices=["sge", "psge", "psge_kl", "sa_psge"],
        help="Algorithms to evaluate (default: sge psge)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=20,
        help="Training dataset size m (default: 20)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Target noise ratio (default: 0.0)",
    )
    parser.add_argument(
        "--n_data_seeds",
        type=int,
        default=5,
        help="Number of independent data replicas (default: 5 for fast pilot; 20 in full Plan)",
    )
    parser.add_argument(
        "--n_evo_seeds",
        type=int,
        default=3,
        help="Number of paired evolutionary seeds per data replica (default: 3 for fast pilot; 5 in full Plan)",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=50,
        help="Population size (default: 50 for fast pilot, 500 in full Plan)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of generations (default: 20 for fast pilot, 100 in full Plan)",
    )
    parser.add_argument(
        "--elitism",
        type=int,
        default=2,
        help="Elitism (default: 2)",
    )
    parser.add_argument(
        "--parameters",
        type=str,
        default="parameters/standard.yml",
        help="Base YAML parameter file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dumps/pilot_campaign",
        help="Output directory for campaign dumps",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)",
    )
    # ---- PSGE-KL hyperparams ----------------------------------------- #
    parser.add_argument(
        "--kl_delta",
        type=float,
        default=6e-5,
        help="KL constraint threshold for PSGE-KL (default: 6e-5, P25 of PSGE per-update KL; see CHANGELOG_SA_PSGE.md)",
    )
    # ---- SA-PSGE hyperparams ----------------------------------------- #
    parser.add_argument(
        "--sa_k_views",
        type=int,
        default=3,
        help="SA-PSGE: number of subsampled views K (default: 3)",
    )
    parser.add_argument(
        "--sa_lambda",
        type=float,
        default=0.5,
        help="SA-PSGE: exponential update step size lambda (default: 0.5)",
    )
    parser.add_argument(
        "--sa_beta",
        type=float,
        default=1.0,
        help="SA-PSGE: inverse temperature for rank-based credit (default: 1.0)",
    )
    parser.add_argument(
        "--sa_alpha",
        type=float,
        default=0.005,
        help="SA-PSGE: uniform mixture weight alpha (default: 0.005; 0.05 anchors the PCFG at uniform)",
    )
    parser.add_argument(
        "--sa_eta",
        type=float,
        default=0.04,
        help="SA-PSGE: step size eta, decoupled from learning factor (default: 0.04, KL-matched to PSGE)",
    )
    parser.add_argument(
        "--sa_p_min",
        type=float,
        default=0.001,
        help="SA-PSGE: minimum probability floor p_min (default: 0.001)",
    )
    parser.add_argument(
        "--sa_tau",
        type=int,
        default=1,
        help="SA-PSGE: update period in generations (default: 1)",
    )
    parser.add_argument(
        "--sa_elite_ratio",
        type=float,
        default=0.10,
        help="SA-PSGE: elite fraction used for credit computation (default: 0.10)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Learning factor: 0.0 for SGE (no grammar update), 0.01 for all PSGE variants
    LF_MAP = {"sge": 0.0, "psge": 0.01, "psge_kl": 0.01, "sa_psge": 0.01}

    # Algorithm-specific hyperparameters (PSGE-KL and SA-PSGE)
    shared_alg_opts = {
        "KL_DELTA":       args.kl_delta,
        "SA_K_VIEWS":     args.sa_k_views,
        "SA_LAMBDA":      args.sa_lambda,
        "SA_BETA":        args.sa_beta,
        "SA_ALPHA":       args.sa_alpha,
        "SA_ETA":         args.sa_eta,
        "SA_P_MIN":       args.sa_p_min,
        "SA_TAU":         args.sa_tau,
        "SA_ELITE_RATIO": args.sa_elite_ratio,
    }

    # Build tasks grid
    tasks = []
    for problem in args.problems:
        spec = BENCHMARK_SPECS[problem]
        grammar_file = spec["grammar"]

        for alg in args.algorithms:
            lf = LF_MAP.get(alg, 0.01)

            for d_idx in range(args.n_data_seeds):
                seed_data = 100 + d_idx

                for e_idx in range(args.n_evo_seeds):
                    seed_evo = 1000 + e_idx

                    tasks.append({
                        "problem": problem,
                        "algorithm": alg,
                        "m": args.m,
                        "noise": args.noise,
                        "seed_data": seed_data,
                        "seed_evo": seed_evo,
                        "learning_factor": lf,
                        "popsize": args.popsize,
                        "generations": args.generations,
                        "elitism": args.elitism,
                        "grammar_file": grammar_file,
                        "parameters_file": args.parameters,
                        "output_dir": args.output_dir,
                        "alg_opts": shared_alg_opts,
                    })

    total_tasks = len(tasks)
    print("=" * 70)
    print("STARTING PILOT STABILITY CAMPAIGN")
    print("=" * 70)
    print(f" Problems:        {args.problems}")
    print(f" Algorithms:      {args.algorithms}")
    print(f" Data Seeds:      {args.n_data_seeds} (seed_data: 100..{100 + args.n_data_seeds - 1})")
    print(f" Evo Seeds/Data:  {args.n_evo_seeds} (paired identically: 1000..{1000 + args.n_evo_seeds - 1})")
    print(f" Total Runs:      {total_tasks} paired experiments")
    print(f" Workers:         {args.n_jobs} parallel processes")
    print(f" Output dir:      {args.output_dir}")
    print("=" * 70)

    results = []
    completed = 0
    cached = 0

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(run_single_stability_experiment, t): t for t in tasks}

        for future in as_completed(futures):
            res, was_cached = future.result()
            results.append(res)
            completed += 1
            if was_cached:
                cached += 1

            meta = res["metadata"]
            stab = res["stability_metrics"]
            time_str = "cached" if was_cached else f"{meta['time_total_sec']:.1f}s"
            print(
                f"[{completed}/{total_tasks}] "
                f"{meta['algorithm'].upper()} | {meta['problem']} | "
                f"d_seed={meta['seed_data']} | e_seed={meta['seed_evo']} | "
                f"APG+ATG={stab['data_stability_norm']:.4f} | "
                f"{time_str}"
            )

    elapsed_total = time.time() - t_start
    print(f"\nAll {total_tasks} runs finished ({cached} loaded from cache) in {elapsed_total:.1f}s.")

    # Run statistical analysis and generate reports
    analyze_campaign_results(results, args.output_dir)


if __name__ == "__main__":
    main()
