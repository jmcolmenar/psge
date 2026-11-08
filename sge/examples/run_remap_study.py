#!/usr/bin/env python3
"""
Estudio de adaptación de la gramática y re-mapeo en PSGE (plan del 2026-11-08): RQ-A, RQ-C y RQ-E.

Condiciones (11), mismo motor, gramática, operadores y presupuesto (parameters/remap_study.yml):
  - sgef:            PSGE sin actualizar la gramática (lambda = 0; ALGORITHM = sge)
  - psge_lf<l>:      PSGE estándar, con re-mapeo, para cada l de REMAP_LAMBDAS
  - psge_pp_lf<l>:   PSGE-PP (re-codificación que conserva el fenotipo tras cada actualización)

Fases (--phase):
  - smoke:    comprobación mínima de las 11 condiciones (nguyen1, 1 x 1 semillas, población 30 x 5 generaciones)
  - pilot:    nguyen1 y nguyen5, solo VALIDACIÓN (el test no se genera), semillas 3000.. / 30000..
  - confirm:  REMAP_CONFIRM_BENCHMARKS (BENCHMARKS_REMAP_STUDY.md), test de 1000 puntos, semillas 2000.. / 20000..;
              exige FROZEN: true en el fichero de parámetros.

Cada ejecución emparejada (S y S+z con la misma semilla evolutiva) guarda un JSON en --output_dir con:
APG, ATG, APG+ATG; NRMSE de entrenamiento, validación/test; brecha de generalización en unidades de
APG+ATG; JS y entropía de las gramáticas finales; fenotipos; y por brazo, coste y la trayectoria compacta
por generación (entropía, KL, tasa de re-mapeo, re-codificación de PSGE-PP, tamaño, % únicos, mutación).

Uso (desde sge/):
  python -m examples.run_remap_study --phase smoke --output_dir /tmp/remap_smoke --n_jobs 4
  python -m examples.run_remap_study --phase pilot --output_dir ~/psge-results/remap_pilot --n_jobs 32
  python -m examples.run_remap_study --phase confirm --dry_run --sec_per_arm 70
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

PHASES = {
    #          benchmarks (None = de stability.py), regímenes (m, ruido), semillas datos/evo, n_val, n_test
    "smoke":   {"benchmarks": ["nguyen1"], "regimes": [(20, 0.0)], "n_data": 1, "n_evo": 1,
                "data_base": 3900, "evo_base": 39000, "n_val": 50, "n_test": 0,
                "overrides": {"POPSIZE": 30, "GENERATIONS": 5, "ELITISM": 1}},
    "pilot":   {"benchmarks": "pilot", "regimes": [(20, 0.05), (100, 0.0)], "n_data": 10, "n_evo": 3,
                "data_base": 3000, "evo_base": 30000, "n_val": 500, "n_test": 0, "overrides": {}},
    "confirm": {"benchmarks": "confirm", "regimes": [(20, 0.0), (20, 0.05), (100, 0.0), (100, 0.05)],
                "n_data": 20, "n_evo": 5, "data_base": 2000, "evo_base": 20000, "n_val": 0, "n_test": 1000,
                "overrides": {}},
}
SIG = 6   # cifras significativas de la trayectoria en el JSON


def conditions(lambdas):
    """nombre -> (ALGORITHM, LEARNING_FACTOR)"""
    out = {"sgef": ("sge", 0.0)}
    for lf in lambdas:
        out[f"psge_lf{lf:g}"] = ("psge", float(lf))
    for lf in lambdas:
        out[f"psge_pp_lf{lf:g}"] = ("psge_pp", float(lf))
    return out


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _round(v):
    if isinstance(v, float):
        return float(f"{v:.{SIG}g}")
    return v


def _arm(dataset, task, arm):
    """Ejecuta un brazo y devuelve (mejor, pcfg, info de coste y trayectoria)."""
    import resource
    import sge
    import sge.engine as eng
    import sge.grammar as gmod
    from sge.stability import StabilityEvaluationFunction

    opts = {"SEED": task["seed_evo"], "GRAMMAR": task["grammar"], "RUN": 1, "VERBOSE": False,
            "EXPERIMENT_NAME": os.path.join(task["run_dir"], arm), "ALGORITHM": task["algorithm"],
            "LEARNING_FACTOR": task["lf"], **task["overrides"]}
    eval_f = StabilityEvaluationFunction(dataset=dataset)
    t0, c0 = time.time(), time.process_time()
    gmod.reset()
    best, pcfg = sge.evolutionary_algorithm(evaluation_function=eval_f,
                                            parameters_file=task["parameters_file"], options_dict=opts)
    traj = eng.get_trajectory()
    n_gen = len(traj["gen"])
    expected = int(opts.get("GENERATIONS", load_config(task["parameters_file"])["GENERATIONS"])) + 1
    if n_gen != expected:
        raise RuntimeError(f"trayectoria con {n_gen} generaciones; se esperaban {expected}")
    info = {
        "wall_sec": time.time() - t0, "cpu_sec": time.process_time() - c0,
        "n_evaluations": eval_f.n_evaluations, "n_case_evaluations": eval_f.n_case_evaluations,
        "peak_rss_kb_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "best_fitness": float(best["fitness"]),
        "trajectory": {k: [_round(x) for x in v] for k, v in traj.items()},
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
                                  seed_data=task["seed_data"], n_test=task["n_test"], n_val=task["n_val"])
    best_S, pcfg_S, info_S = _arm(b["S"], task, "arm_S")
    best_Sz, pcfg_Sz, info_Sz = _arm(b["S_plus_z"], task, "arm_Sz")
    mask = gmod.get_mask()
    mt = calculate_stability_metrics(model_S_phenotype=best_S["phenotype"],
                                     model_S_plus_z_phenotype=best_Sz["phenotype"],
                                     S=b["S"], z=b["z"], test_set=b["test_set"] or None,
                                     var_y=b["var_y_S"], val_set=b["val_set"] or None)
    js, _ = calculate_jensen_shannon(pcfg_S, pcfg_Sz, mask)

    def f(key):
        v = mt.get(key)
        return None if v is None else float(v)

    res = {
        "phase": task["phase"], "benchmark": task["benchmark"], "m": task["m"], "noise": task["noise"],
        "condition": task["condition"], "algorithm": task["algorithm"], "learning_factor": task["lf"],
        "seed_data": task["seed_data"], "seed_evo": task["seed_evo"],
        "apg_norm": f("apg_norm"), "atg_norm": f("atg_norm"), "apg_plus_atg": f("data_stability_norm"),
        "mean_train_loss_norm_fS": f("mean_in_sample_loss_fS_norm"),
        "nrmse_train_fS": f("nrmse_train_fS"), "nrmse_train_fSz": f("nrmse_train_fSz"),
        "nrmse_val_fS": f("nrmse_val_fS"), "nrmse_val_fSz": f("nrmse_val_fSz"),
        "nrmse_test_fS": f("nrmse_test_fS"), "nrmse_test_fSz": f("nrmse_test_fSz"),
        "mse_val_norm_fS": f("mse_val_norm_fS"), "mse_test_norm_fS": f("mse_test_norm_fS"),
        "gen_gap_val_norm_fS": f("gen_gap_val_norm_fS"), "gen_gap_test_norm_fS": f("gen_gap_test_norm_fS"),
        "js_divergence": float(js),
        "entropy_S": float(calculate_pcfg_entropy(pcfg_S, mask)),
        "entropy_Sz": float(calculate_pcfg_entropy(pcfg_Sz, mask)),
        "phenotype_S": best_S["phenotype"], "phenotype_Sz": best_Sz["phenotype"],
        "pcfg_final_S": [[_round(float(p)) for p in row] for row in pcfg_S],
        "pcfg_final_Sz": [[_round(float(p)) for p in row] for row in pcfg_Sz],
        "arm_S": info_S, "arm_Sz": info_Sz,
    }
    tmp = task["json_path"] + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, separators=(",", ":"))
    os.replace(tmp, task["json_path"])
    return res, False


def build_tasks(a, cfg):
    from sge.stability import BENCHMARK_SPECS, REMAP_CONFIRM_BENCHMARKS, REMAP_PILOT_BENCHMARKS
    ph = PHASES[a.phase]
    named = {"pilot": REMAP_PILOT_BENCHMARKS, "confirm": REMAP_CONFIRM_BENCHMARKS}
    default = named[ph["benchmarks"]] if isinstance(ph["benchmarks"], str) else ph["benchmarks"]
    benchmarks = a.benchmarks or default
    regimes = ph["regimes"] if not a.regimes else [tuple(map(float, r.split(":"))) for r in a.regimes]
    n_data = a.n_data_seeds or ph["n_data"]
    n_evo = a.n_evo_seeds or ph["n_evo"]
    conds = conditions(cfg["REMAP_LAMBDAS"])
    names = a.conditions or list(conds)
    unknown = [c for c in names if c not in conds]
    if unknown:
        raise SystemExit(f"condiciones desconocidas: {unknown}; disponibles: {list(conds)}")
    tasks = []
    for bench, (m, noise), cond, i, j in itertools.product(benchmarks, regimes, names, range(n_data), range(n_evo)):
        m = int(m)
        sd, se = ph["data_base"] + i, ph["evo_base"] + j
        regime = f"m{m}_noise{noise:g}"
        alg, lf = conds[cond]
        tasks.append({
            "phase": a.phase, "benchmark": bench, "m": m, "noise": float(noise), "condition": cond,
            "algorithm": alg, "lf": lf, "seed_data": sd, "seed_evo": se,
            "grammar": BENCHMARK_SPECS[bench]["grammar"], "parameters_file": a.parameters,
            "overrides": ph["overrides"], "n_val": ph["n_val"], "n_test": ph["n_test"],
            "run_dir": os.path.join(a.runs_dir or os.path.join(a.output_dir, "_runs"), bench, regime, cond,
                                    f"d{sd}_e{se}"),
            "json_path": os.path.join(a.output_dir, bench, regime, cond, f"{cond}_d{sd}_e{se}.json"),
        })
    return tasks


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--phase", choices=list(PHASES), required=True)
    p.add_argument("--benchmarks", nargs="+", default=None, help="por defecto, los de la fase")
    p.add_argument("--regimes", nargs="+", default=None, help="m:ruido, p. ej. 20:0.05 100:0")
    p.add_argument("--conditions", nargs="+", default=None, help="por defecto, las 11")
    p.add_argument("--n_data_seeds", type=int, default=None)
    p.add_argument("--n_evo_seeds", type=int, default=None)
    p.add_argument("--parameters", default="parameters/remap_study.yml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--runs_dir", default=None,
                   help="directorio de cada brazo (solo parameters.json con la configuración de "
                        "remap_study.yml); por defecto, <output_dir>/_runs")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--dry_run", action="store_true", help="solo contar tareas y estimar coste")
    p.add_argument("--sec_per_arm", type=float, default=None, help="segundos por brazo para la estimación")
    a = p.parse_args()

    cfg = load_config(a.parameters)
    if a.phase == "confirm" and not a.dry_run and cfg.get("FROZEN") is not True:
        raise SystemExit(f"{a.parameters}: FROZEN no es true. La confirmación solo se lanza con la "
                         "configuración congelada tras el piloto y el prerregistro.")
    tasks = build_tasks(a, cfg)
    todo = [t for t in tasks if not os.path.exists(t["json_path"])]
    n_b = len({t["benchmark"] for t in tasks})
    n_r = len({(t["m"], t["noise"]) for t in tasks})
    n_c = len({t["condition"] for t in tasks})
    print(f"Fase {a.phase}: {len(tasks)} ejecuciones emparejadas ({len(tasks) - len(todo)} ya hechas): "
          f"{n_b} benchmarks x {n_r} regímenes x {n_c} condiciones x semillas")
    if a.sec_per_arm:
        h = len(todo) * 2 * a.sec_per_arm / 3600
        print(f"Estimación: {h:.0f} h de CPU -> {h / a.n_jobs:.1f} h de reloj con {a.n_jobs} procesos")
    if a.dry_run:
        return

    t0, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=a.n_jobs) as ex:
        futs = {ex.submit(worker, t): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            tag = f"{t['benchmark']} m{t['m']} n{t['noise']:g} {t['condition']} d{t['seed_data']} e{t['seed_evo']}"
            try:
                r, _ = fut.result()
                done += 1
                rr = max(r["arm_S"]["trajectory"]["remap_rate"])
                print(f"[{done + failed}/{len(todo)}] {tag}: APG+ATG={r['apg_plus_atg']:.4g} "
                      f"H={r['entropy_S']:.3f} remap_max={rr:.3f} "
                      f"wall={r['arm_S']['wall_sec'] + r['arm_Sz']['wall_sec']:.0f}s", flush=True)
            except Exception as e:
                failed += 1
                print(f"ERROR {tag}: {e!r}", flush=True)
    print(f"Terminado: {done} ok, {failed} con error, {time.time() - t0:.0f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
