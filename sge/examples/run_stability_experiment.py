#!/usr/bin/env python3
"""
Experiment runner for Algorithmic Stability in SGE and PSGE.
Executes the paired Add-One protocol:
    Run A: A(S; s)
    Run B: A(S U {z}; s)
with aligned evolutionary seeds, and computes:
    - APG (Add-One Point-wise Gap)
    - ATG_hat (Empirical Add-One Training Gap)
    - APG + ATG_hat (Primary Data Stability Metric)
    - JS(P_S, P_{S+z}) (Jensen-Shannon Grammar Divergence)
    - NRMSE_test and Generalization Gap

Reference:
    Plan_investigacion_GECCO_estabilidad_SGE_PSGE_SA_PSGE.docx (Secciones 4, 5, 6, 7)
"""

import os
import sys
import json
import time
import argparse
import numpy as np

import sge
from sge.stability import (
    BENCHMARK_SPECS,
    generate_add_one_datasets,
    calculate_stability_metrics,
    calculate_jensen_shannon,
    calculate_pcfg_entropy,
    StabilityEvaluationFunction,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paired Add-One stability experiment for SGE / PSGE."
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="nguyen1",
        choices=list(BENCHMARK_SPECS.keys()),
        help="Benchmark problem (default: nguyen1)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="psge",
        choices=["sge", "psge", "psge_kl", "sa_psge"],
        help="Algorithm type: 'sge', 'psge', 'psge_kl', or 'sa_psge'",
    )
    parser.add_argument(
        "--kl_delta",
        type=float,
        default=6e-5,
        help="KL divergence constraint delta for PSGE-KL (default: 6e-5, P25 of PSGE per-update KL; see CHANGELOG_SA_PSGE.md)",
    )
    parser.add_argument(
        "--sa_lambda",
        type=float,
        default=0.5,
        help="Stability weight lambda for SA-PSGE (0.0 = ablation, default: 0.5)",
    )
    parser.add_argument(
        "--sa_k_views",
        type=int,
        default=3,
        help="Number of subsampled views K for SA-PSGE (default: 3)",
    )
    parser.add_argument(
        "--sa_beta",
        type=float,
        default=1.0,
        help="Concentration parameter beta for SA-PSGE (default: 1.0)",
    )
    parser.add_argument(
        "--sa_alpha",
        type=float,
        default=0.005,
        help="Uniform mixture fraction alpha for SA-PSGE (default: 0.005; 0.05 anchors the PCFG at uniform)",
    )
    parser.add_argument(
        "--sa_eta",
        type=float,
        default=0.04,
        help="Step size eta for SA-PSGE, decoupled from learning factor (default: 0.04, KL-matched to PSGE)",
    )
    parser.add_argument(
        "--sa_p_min",
        type=float,
        default=0.001,
        help="Minimum probability floor p_min for SA-PSGE (default: 0.001)",
    )
    parser.add_argument(
        "--sa_tau",
        type=int,
        default=1,
        help="Update interval in generations for SA-PSGE (default: 1)",
    )
    parser.add_argument(
        "--sa_elite_ratio",
        type=float,
        default=0.10,
        help="Elite fraction for SA-PSGE credit calculation (default: 0.10)",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=20,
        help="Number of training instances in base set S (default: 20; 100 also used in Plan)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Target noise ratio as fraction of sigma_y (default: 0.0, e.g. 0.05 for 5%%)",
    )
    parser.add_argument(
        "--seed_data",
        type=int,
        default=42,
        help="Random seed for dataset generation (S, z, test_set)",
    )
    parser.add_argument(
        "--seed_evo",
        type=int,
        default=1234,
        help="Evolutionary random seed s (paired identically for S and S U {z})",
    )
    parser.add_argument(
        "--learning_factor",
        type=float,
        default=None,
        help="Learning factor eta (default: 0.01 for psge, 0.0 for sge)",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=100,
        help="Population size (default: 100 for pilot testing, 500 in full Plan)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="Generations (default: 30 for pilot testing, 100 in full Plan)",
    )
    parser.add_argument(
        "--elitism",
        type=int,
        default=2,
        help="Number of elite individuals (default: 2)",
    )
    parser.add_argument(
        "--grammar",
        type=str,
        default=None,
        help="Grammar file path (default: auto-detected for problem)",
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
        default="dumps/stability_results",
        help="Directory to save experimental results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print generation logs from sge engine",
    )
    return parser.parse_args()


def run_single_arm(dataset, test_set, arm_name, args, grammar_file, learning_factor):
    """
    Executes a single evolutionary run on the specified dataset with fixed seed.
    Returns:
        best_ind: dict of best individual
        final_pcfg: numpy array of PCFG probabilities
        elapsed_sec: float
    """
    # 1. Reset grammar singleton to clean slate
    sge.grammar.reset()

    # 2. Configure parameters
    run_exp_name = os.path.join(
        args.output_dir,
        f"{args.algorithm}_{args.problem}_m{args.m}_sData{args.seed_data}_sEvo{args.seed_evo}",
        arm_name
    )

    custom_options = {
        "POPSIZE": args.popsize,
        "GENERATIONS": args.generations,
        "ELITISM": args.elitism,
        "LEARNING_FACTOR": learning_factor,
        "SEED": args.seed_evo,
        "GRAMMAR": grammar_file,
        "EXPERIMENT_NAME": run_exp_name,
        "RUN": 1,
        "VERBOSE": args.verbose,
        "SAVE_STEP": 1000,  # avoid unnecessary json dumps during pilot runs
        "ALGORITHM": args.algorithm,
        "KL_DELTA": args.kl_delta,
        "SA_K_VIEWS": args.sa_k_views,
        "SA_LAMBDA": args.sa_lambda,
        "SA_BETA": args.sa_beta,
        "SA_ALPHA": args.sa_alpha,
        "SA_ETA": args.sa_eta,
        "SA_P_MIN": args.sa_p_min,
        "SA_TAU": args.sa_tau,
        "SA_ELITE_RATIO": args.sa_elite_ratio,
    }

    # 3. Create evaluation adapter
    eval_func = StabilityEvaluationFunction(dataset=dataset, test_set=test_set)

    # 4. Run evolution
    start_time = time.time()
    best_ind, final_pcfg = sge.evolutionary_algorithm(
        evaluation_function=eval_func,
        parameters_file=args.parameters,
        options_dict=custom_options
    )
    elapsed = time.time() - start_time

    return best_ind, final_pcfg, elapsed


def main():
    args = parse_args()

    # Determine learning factor based on algorithm
    if args.learning_factor is not None:
        lf = args.learning_factor
    else:
        lf = 0.01 if args.algorithm == "psge" else 0.0

    # Determine grammar
    spec = BENCHMARK_SPECS[args.problem]
    grammar_file = args.grammar if args.grammar else spec["grammar"]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PAIRED ADD-ONE STABILITY EXPERIMENT")
    print("=" * 70)
    print(f" Problem:          {spec['description']}")
    print(f" Algorithm:        {args.algorithm.upper()} (learning_factor={lf})")
    print(f" Training size m:  {args.m} (perturbation S -> S U {{z}}, m={args.m} -> {args.m+1})")
    print(f" Noise level:      {args.noise * 100:.1f}%")
    print(f" Seed Data:        {args.seed_data}")
    print(f" Seed Evolution:   {args.seed_evo} (identically paired for A and B)")
    print(f" Population:       {args.popsize} | Generations: {args.generations}")
    print(f" Grammar:          {grammar_file}")
    print("=" * 70)

    # 1. Generate paired datasets
    data_bundle = generate_add_one_datasets(
        problem_name=args.problem,
        m=args.m,
        noise_ratio=args.noise,
        seed_data=args.seed_data,
        n_test=1000
    )
    S = data_bundle["S"]
    z = data_bundle["z"]
    S_plus_z = data_bundle["S_plus_z"]
    test_set = data_bundle["test_set"]
    var_y_S = data_bundle["var_y_S"]

    print(f"[1/3] Datasets generated: |S|={len(S)}, |S U {{z}}|={len(S_plus_z)}, |Test|={len(test_set)}")
    print(f"      Add-one point z = {z}")
    print(f"      Var(Y_S) = {var_y_S:.6e}")

    # 2. Run Arm A: A(S; s)
    print(f"\n[2/3] Running Arm A: A(S; seed={args.seed_evo})...")
    best_S, pcfg_S, time_S = run_single_arm(
        dataset=S,
        test_set=test_set,
        arm_name="arm_S",
        args=args,
        grammar_file=grammar_file,
        learning_factor=lf
    )
    phenotype_S = best_S["phenotype"]
    print(f"      Arm A finished in {time_S:.2f}s")
    print(f"      f_S phenotype: {phenotype_S}")
    print(f"      f_S train fitness (NRMSE): {best_S['fitness']:.6e}")

    # 3. Run Arm B: A(S U {z}; s)
    print(f"\n[3/3] Running Arm B: A(S U {{z}}; seed={args.seed_evo})...")
    best_Sz, pcfg_Sz, time_Sz = run_single_arm(
        dataset=S_plus_z,
        test_set=test_set,
        arm_name="arm_S_plus_z",
        args=args,
        grammar_file=grammar_file,
        learning_factor=lf
    )
    phenotype_Sz = best_Sz["phenotype"]
    print(f"      Arm B finished in {time_Sz:.2f}s")
    print(f"      f_{{S+z}} phenotype: {phenotype_Sz}")
    print(f"      f_{{S+z}} train fitness (NRMSE): {best_Sz['fitness']:.6e}")

    # 4. Compute Stability & Divergence Metrics
    metrics = calculate_stability_metrics(
        model_S_phenotype=phenotype_S,
        model_S_plus_z_phenotype=phenotype_Sz,
        S=S,
        z=z,
        test_set=test_set,
        var_y=var_y_S
    )

    mask = sge.grammar.get_mask()
    js_divergence, js_by_nt = calculate_jensen_shannon(pcfg_S, pcfg_Sz, mask)
    entropy_S = calculate_pcfg_entropy(pcfg_S, mask)
    entropy_Sz = calculate_pcfg_entropy(pcfg_Sz, mask)

    # 5. Compile full result record
    result = {
        "metadata": {
            "problem": args.problem,
            "algorithm": args.algorithm,
            "m": args.m,
            "noise": args.noise,
            "seed_data": args.seed_data,
            "seed_evo": args.seed_evo,
            "learning_factor": lf,
            "popsize": args.popsize,
            "generations": args.generations,
            "grammar": grammar_file,
            "time_S_sec": time_S,
            "time_Sz_sec": time_Sz,
        },
        "models": {
            "f_S_phenotype": phenotype_S,
            "f_Sz_phenotype": phenotype_Sz,
            "f_S_tree_depth": best_S.get("tree_depth"),
            "f_Sz_tree_depth": best_Sz.get("tree_depth"),
        },
        "losses_on_z": {
            "loss_fS_z_raw": metrics["loss_fS_z"],
            "loss_fSz_z_raw": metrics["loss_fSz_z"],
            "loss_fS_z_norm": metrics["loss_fS_z_norm"],
            "loss_fSz_z_norm": metrics["loss_fSz_z_norm"],
        },
        "stability_metrics": {
            "apg_raw": metrics["apg_raw"],
            "apg_norm": metrics["apg_norm"],
            "atg_raw": metrics["atg_raw"],
            "atg_norm": metrics["atg_norm"],
            "data_stability_raw (APG+ATG)": metrics["data_stability_raw"],
            "data_stability_norm (APG+ATG)": metrics["data_stability_norm"],
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
        }
    }

    # 6. Save JSON file
    filename_base = f"{args.algorithm}_{args.problem}_m{args.m}_sData{args.seed_data}_sEvo{args.seed_evo}"
    json_path = os.path.join(args.output_dir, f"{filename_base}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # 7. Append row to CSV summary
    csv_path = os.path.join(args.output_dir, "stability_summary.csv")
    csv_header = (
        "algorithm,problem,m,noise,seed_data,seed_evo,learning_factor,"
        "apg_norm,atg_norm,apg_plus_atg_norm,js_divergence,"
        "nrmse_train_fS,nrmse_test_fS,gen_gap_fS,"
        "nrmse_train_fSz,nrmse_test_fSz,gen_gap_fSz,time_total_sec\n"
    )
    csv_row = (
        f"{args.algorithm},{args.problem},{args.m},{args.noise},{args.seed_data},{args.seed_evo},{lf},"
        f"{metrics['apg_norm']:.6e},{metrics['atg_norm']:.6e},{metrics['data_stability_norm']:.6e},{js_divergence:.6e},"
        f"{metrics['nrmse_train_fS']:.6e},{metrics['nrmse_test_fS']:.6e},{metrics['gen_gap_fS']:.6e},"
        f"{metrics['nrmse_train_fSz']:.6e},{metrics['nrmse_test_fSz']:.6e},{metrics['gen_gap_fSz']:.6e},"
        f"{(time_S + time_Sz):.2f}\n"
    )

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_header:
            f.write(csv_header)
        f.write(csv_row)

    # 8. Print formatted report
    print("\n" + "=" * 70)
    print("STABILITY & GENERALIZATION RESULTS")
    print("=" * 70)
    print(f" APG (Add-One Point-wise Gap) [norm]:        {metrics['apg_norm']:.6e}")
    print(f" ATG (Empirical Add-One Training Gap) [norm]: {metrics['atg_norm']:.6e}")
    print(f" PRIMARY STABILITY (APG + ATG) [norm]:        {metrics['data_stability_norm']:.6e}")
    print(f" JS Divergence JS(P_S, P_{{S+z}}):              {js_divergence:.6e}")
    print("-" * 70)
    print(f" Model f_S:   Train NRMSE = {metrics['nrmse_train_fS']:.6f} | Test NRMSE = {metrics['nrmse_test_fS']:.6f} | Gen Gap = {metrics['gen_gap_fS']:.6f}")
    print(f" Model f_Sz:  Train NRMSE = {metrics['nrmse_train_fSz']:.6f} | Test NRMSE = {metrics['nrmse_test_fSz']:.6f} | Gen Gap = {metrics['gen_gap_fSz']:.6f}")
    print("-" * 70)
    print(f" JSON record saved to: {json_path}")
    print(f" CSV summary appended: {csv_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
