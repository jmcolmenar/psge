#!/usr/bin/env python3
"""
Piloto de RQ1: SGE frente a PSGE estándar en estabilidad add-one (S -> S U {z}).

Usa el mismo worker emparejado que la búsqueda de hiperparámetros
(run_hyperparameter_search._grid_worker), así que SGE y PSGE pasan por el mismo código,
la misma gramática, el mismo presupuesto y las mismas semillas:
  - SGE:  ALGORITHM = sge,  LEARNING_FACTOR = 0     (gramática fija)
  - PSGE: ALGORITHM = psge, LEARNING_FACTOR = 0.01  (actualización estándar de la PCFG)

Diseño (plan, "Diseño recomendado de réplicas"): N_DATA_SEEDS réplicas de datos x N_EVO_SEEDS semillas
evolutivas emparejadas. Semillas de datos 500.. y evolutivas 5000.., distintas de las del ajuste
(400../4000..). Solo validación: el test no se genera.

Salida: <output_dir>/<problema>/<algoritmo>/<algoritmo>_d<datos>_e<evo>.json (+ un directorio por
ejecución con la gramática de cada generación). Si el JSON existe, se reutiliza.
El análisis está en examples/analyze_rq1.py.

Uso (desde sge/):
  python -m examples.run_rq1_pilot --output_dir ~/psge-results/rq1_pilot --n_jobs 16
"""

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from examples.run_hyperparameter_search import _grid_worker  # noqa: E402
from sge.stability import BENCHMARK_SPECS  # noqa: E402

ALGORITHMS = {"sge": 0.0, "psge": 0.01}   # algoritmo -> LEARNING_FACTOR


def build_tasks(problems, n_data_seeds, n_evo_seeds, popsize, generations, m, parameters_file, output_dir):
    tasks = []
    for problem in problems:
        grammar_file = BENCHMARK_SPECS[problem]["grammar"]
        for sd, se, alg in itertools.product(range(500, 500 + n_data_seeds),
                                             range(5000, 5000 + n_evo_seeds), ALGORITHMS):
            tasks.append({
                "config_name": alg, "algorithm": alg, "alg_opts": {},
                "problem": problem, "m": m, "seed_data": sd, "seed_evo": se,
                "learning_factor": ALGORITHMS[alg],
                "popsize": popsize, "generations": generations,
                "grammar_file": grammar_file, "parameters_file": parameters_file,
                "output_dir": os.path.join(output_dir, problem, alg),
            })
    return tasks


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--problems", nargs="+", default=["nguyen1", "nguyen5"])
    p.add_argument("--n_data_seeds", type=int, default=20)
    p.add_argument("--n_evo_seeds", type=int, default=5)
    p.add_argument("--popsize", type=int, default=200)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--m", type=int, default=20)
    p.add_argument("--parameters", default="parameters/standard.yml")
    p.add_argument("--output_dir", default="dumps/rq1_pilot")
    p.add_argument("--n_jobs", type=int, default=4)
    args = p.parse_args()

    tasks = build_tasks(args.problems, args.n_data_seeds, args.n_evo_seeds, args.popsize,
                        args.generations, args.m, args.parameters, args.output_dir)
    print(f"RQ1 piloto: {len(tasks)} ejecuciones emparejadas "
          f"({len(args.problems)} problemas x {args.n_data_seeds} datos x {args.n_evo_seeds} evo x "
          f"{len(ALGORITHMS)} algoritmos), pop {args.popsize} x gen {args.generations}, m = {args.m}")
    t0, done, cached, failed = time.time(), 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
        futures = {ex.submit(_grid_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            tag = f"{t['problem']}/{t['algorithm']} d{t['seed_data']} e{t['seed_evo']}"
            try:
                res, was_cached = fut.result()
                done += 1
                cached += int(was_cached)
                print(f"[{done + failed}/{len(tasks)}] {tag}: APG+ATG={res['apg_plus_atg']:.4f} "
                      f"NRMSE-val={res['nrmse_val']:.4f} JS={res['js_divergence']:.5f}"
                      f"{' [cache]' if was_cached else ''}", flush=True)
            except Exception as e:  # se registra y se sigue; el análisis cuenta las ausencias
                failed += 1
                print(f"ERROR {tag}: {e!r}", flush=True)
    print(f"Terminado: {done} ok ({cached} de caché), {failed} con error, {time.time() - t0:.0f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
