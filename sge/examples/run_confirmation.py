#!/usr/bin/env python3
"""
Estudio confirmatorio GECCO (RQ1 + RQ2). Aquí, y solo aquí, se abre el test.

Diseño (plan; configuración en parameters/gecco_frozen.yml, que se pasa tal cual al motor):
  - Benchmarks: nguyen7, nguyen9, keijzer6, pagie1, vladislavleva4, vladislavleva7.
  - Regímenes: m in {20, 100} x ruido en y in {0, 0.05 sigma_y}. Interpolación/extrapolación según el
    dominio de test de cada benchmark (Keijzer-6 y Vladislavleva extrapolan en parte).
  - Algoritmos: sge, psge, psge_kl, sa_psge y la ablación sa_psge_lam0 (SA_LAMBDA = 0).
  - Réplicas: N_DATA semillas de datos (1000..) x N_EVO semillas evolutivas emparejadas (10000..),
    distintas de todas las de los pilotos. La misma semilla evolutiva en los dos brazos (S, S+z).

Cada ejecución emparejada guarda un JSON (en --output_dir) con: APG, ATG, APG+ATG; NRMSE de entrenamiento y TEST de f_S y
f_{S+z}; JS entre las gramáticas finales y entropía; KL por actualización (mediana, máximo, nº); fenotipos;
y coste por brazo (evaluaciones, evaluaciones caso-individuo, tiempo de reloj y CPU, memoria pico).
Los directorios de cada brazo (en --runs_dir, que puede estar en otro disco) guardan la gramática de cada
generación.

Uso (desde sge/):
  python -m examples.run_confirmation --output_dir ~/psge-results/confirmation --n_jobs 32
  python -m examples.run_confirmation --dry_run --sec_per_arm 20     # solo cuenta y estima coste
"""

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

CONFIRM_BENCHMARKS = ["nguyen7", "nguyen9", "keijzer6", "pagie1", "vladislavleva4", "vladislavleva7"]
ALGORITHMS = {
    # nombre: (ALGORITHM, opciones que cambian respecto a gecco_frozen.yml)
    "sge":          ("sge",     {"LEARNING_FACTOR": 0.0}),
    "psge":         ("psge",    {}),
    "psge_kl":      ("psge_kl", {}),
    "sa_psge":      ("sa_psge", {}),
    "sa_psge_lam0": ("sa_psge", {"SA_LAMBDA": 0.0}),
}
DATA_SEED_BASE, EVO_SEED_BASE = 1000, 10000
N_TEST = 1000


def check_frozen(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg.get("KL_DELTA"), (int, float)):
        raise SystemExit(f"{path}: KL_DELTA = {cfg.get('KL_DELTA')!r}. Rellénalo con la recalibración antes de confirmar.")
    return cfg


def _arm(dataset, task, arm):
    """Ejecuta un brazo y devuelve (mejor, pcfg, info de dinámica y coste)."""
    import copy
    import resource
    import sge
    import sge.engine as eng
    import sge.grammar as gmod
    from sge.operators.stability_aware_update import compute_pcfg_kl_divergence
    from sge.stability import StabilityEvaluationFunction

    kls = []
    orig = eng.perform_grammar_update

    def logged(population, best_gen, flag, eval_func, iteration):
        before = copy.deepcopy(gmod.get_pcfg())
        orig(population, best_gen, flag, eval_func, iteration)
        kls.append(float(compute_pcfg_kl_divergence(gmod.get_pcfg(), before, gmod.get_mask())))

    algorithm, overrides = ALGORITHMS[task["alg"]]
    opts = {"SEED": task["seed_evo"], "GRAMMAR": task["grammar"], "RUN": 1, "VERBOSE": False,
            "EXPERIMENT_NAME": os.path.join(task["run_dir"], arm), "ALGORITHM": algorithm, **overrides}
    eval_f = StabilityEvaluationFunction(dataset=dataset)
    eng.perform_grammar_update = logged
    t0, c0 = time.time(), time.process_time()
    try:
        gmod.reset()
        best, pcfg = sge.evolutionary_algorithm(evaluation_function=eval_f,
                                                parameters_file=task["parameters_file"], options_dict=opts)
    finally:
        eng.perform_grammar_update = orig
    upd = [k for k in kls if k > 1e-15]
    import numpy as np
    info = {
        "wall_sec": time.time() - t0, "cpu_sec": time.process_time() - c0,
        "n_evaluations": eval_f.n_evaluations, "n_case_evaluations": eval_f.n_case_evaluations,
        "peak_rss_kb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "n_grammar_updates": len(upd),
        "kl_update_median": float(np.median(upd)) if upd else 0.0,
        "kl_update_max": float(np.max(upd)) if upd else 0.0,
    }
    return best, pcfg, info


def worker(task):
    import sge.grammar as gmod
    from sge.stability import (calculate_jensen_shannon, calculate_pcfg_entropy,
                               calculate_stability_metrics, generate_add_one_datasets)

    os.makedirs(os.path.dirname(task["json_path"]), exist_ok=True)
    if os.path.exists(task["json_path"]):
        try:
            with open(task["json_path"]) as fh:
                return json.load(fh), True
        except Exception:
            pass   # JSON corrupto: se repite

    b = generate_add_one_datasets(problem_name=task["benchmark"], m=task["m"], noise_ratio=task["noise"],
                                  seed_data=task["seed_data"], n_test=N_TEST, n_val=0)
    best_S, pcfg_S, info_S = _arm(b["S"], task, "arm_S")
    best_Sz, pcfg_Sz, info_Sz = _arm(b["S_plus_z"], task, "arm_Sz")
    mask = gmod.get_mask()
    mt = calculate_stability_metrics(model_S_phenotype=best_S["phenotype"],
                                     model_S_plus_z_phenotype=best_Sz["phenotype"],
                                     S=b["S"], z=b["z"], test_set=b["test_set"], var_y=b["var_y_S"])
    js, _ = calculate_jensen_shannon(pcfg_S, pcfg_Sz, mask)
    res = {
        "benchmark": task["benchmark"], "m": task["m"], "noise": task["noise"], "algorithm": task["alg"],
        "seed_data": task["seed_data"], "seed_evo": task["seed_evo"],
        "apg_norm": float(mt["apg_norm"]), "atg_norm": float(mt["atg_norm"]),
        "apg_plus_atg": float(mt["data_stability_norm"]),
        "nrmse_train_fS": float(mt["nrmse_train_fS"]), "nrmse_train_fSz": float(mt["nrmse_train_fSz"]),
        "nrmse_test_fS": float(mt["nrmse_test_fS"]), "nrmse_test_fSz": float(mt["nrmse_test_fSz"]),
        "js_divergence": float(js),
        "entropy_S": float(calculate_pcfg_entropy(pcfg_S, mask)),
        "entropy_Sz": float(calculate_pcfg_entropy(pcfg_Sz, mask)),
        "phenotype_S": best_S["phenotype"], "phenotype_Sz": best_Sz["phenotype"],
        "arm_S": info_S, "arm_Sz": info_Sz,
    }
    with open(task["json_path"], "w") as fh:
        json.dump(res, fh, indent=2)
    return res, False


def build_tasks(a):
    from sge.stability import BENCHMARK_SPECS
    tasks = []
    for bench, m, noise, alg, i, j in itertools.product(
            a.benchmarks, a.m, a.noise, a.algorithms, range(a.n_data_seeds), range(a.n_evo_seeds)):
        sd, se = DATA_SEED_BASE + i, EVO_SEED_BASE + j
        regime = f"m{m}_noise{noise:g}"
        base = os.path.join(a.output_dir, bench, regime, alg)
        runs_base = os.path.join(a.runs_dir or a.output_dir, bench, regime, alg)
        tasks.append({"benchmark": bench, "m": m, "noise": noise, "alg": alg, "seed_data": sd, "seed_evo": se,
                      "grammar": BENCHMARK_SPECS[bench]["grammar"], "parameters_file": a.parameters,
                      "run_dir": os.path.join(runs_base, f"{alg}_d{sd}_e{se}"),
                      "json_path": os.path.join(base, f"{alg}_d{sd}_e{se}.json")})
    return tasks


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--benchmarks", nargs="+", default=CONFIRM_BENCHMARKS)
    p.add_argument("--m", type=int, nargs="+", default=[20, 100])
    p.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05])
    p.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    p.add_argument("--n_data_seeds", type=int, default=20)
    p.add_argument("--n_evo_seeds", type=int, default=5)
    p.add_argument("--parameters", default="parameters/gecco_frozen.yml")
    p.add_argument("--output_dir", default="dumps/confirmation")
    p.add_argument("--runs_dir", default=None,
                   help="directorio para los brazos (gramática por generación, ~2 MB por brazo); "
                        "por defecto, el mismo que --output_dir. Los JSON de resumen van siempre a --output_dir")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--dry_run", action="store_true", help="solo contar tareas y estimar coste")
    p.add_argument("--sec_per_arm", type=float, default=None, help="segundos por brazo para la estimación")
    a = p.parse_args()

    if not a.dry_run:
        check_frozen(a.parameters)
    tasks = build_tasks(a)
    todo = [t for t in tasks if not os.path.exists(t["json_path"])]
    print(f"Confirmación: {len(tasks)} ejecuciones emparejadas ({len(tasks) - len(todo)} ya hechas), "
          f"{len(a.benchmarks)} benchmarks x {len(a.m) * len(a.noise)} regímenes x {len(a.algorithms)} algoritmos x "
          f"{a.n_data_seeds} x {a.n_evo_seeds} semillas")
    if a.sec_per_arm:
        h = len(todo) * 2 * a.sec_per_arm / 3600
        print(f"Estimación: {h:.0f} h de CPU -> {h / a.n_jobs:.1f} h de reloj con {a.n_jobs} procesos")
    if a.dry_run:
        return

    t0, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
        futs = {ex.submit(worker, t): t for t in todo}
        for f in as_completed(futs):
            t = futs[f]
            tag = f"{t['benchmark']} m{t['m']} n{t['noise']:g} {t['alg']} d{t['seed_data']} e{t['seed_evo']}"
            try:
                r, _ = f.result()
                done += 1
                print(f"[{done + failed}/{len(todo)}] {tag}: APG+ATG={r['apg_plus_atg']:.4f} "
                      f"wall={r['arm_S']['wall_sec'] + r['arm_Sz']['wall_sec']:.0f}s", flush=True)
            except Exception as e:
                failed += 1
                print(f"ERROR {tag}: {e!r}", flush=True)
    print(f"Terminado: {done} ok, {failed} con error, {time.time() - t0:.0f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
