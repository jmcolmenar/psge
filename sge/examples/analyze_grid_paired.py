#!/usr/bin/env python3
"""
Paired analysis of SA-PSGE grid results against the PSGE baseline.

For every config found under <grid_root>/<problem>/, and for each data seed:
  1. mean over evo seeds (seeds are NOT independent datasets);
  2. paired difference vs psge_baseline on the same data seed.
Reports per problem the median paired Δ(APG+ATG) with a percentile-bootstrap
95 % CI of the median, the number of data seeds where SA-PSGE beats PSGE, and
the median relative Δ NRMSE-val (%). Only validation results are used.

Usage (from psge-forked-jmcolmenar/sge/):
  python3 -m examples.analyze_grid_paired --problems nguyen5 nguyen1
"""

import os
import glob
import json
import argparse
import collections
import numpy as np


def load_results(problem_dir, data_seeds, evo_seeds):
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(os.path.join(problem_dir, "*", "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        if "config_name" not in r:
            continue
        if r["seed_data"] in data_seeds and r["seed_evo"] in evo_seeds:
            res[r["config_name"]][r["seed_data"]].append((r["apg_plus_atg"], r["nrmse_val"]))
    return res


def paired_rows(res, n_evo, n_boot, rng):
    base = res["psge_baseline"]
    rows = []
    for cfg, per_seed in res.items():
        if cfg == "psge_baseline":
            continue
        # Only complete data seeds (all evo seeds present in both arms)
        seeds = sorted(s for s in per_seed
                       if len(per_seed[s]) == n_evo and len(base.get(s, [])) == n_evo)
        if len(seeds) < 2:
            continue
        stab = np.array([np.mean([x[0] for x in per_seed[s]]) - np.mean([x[0] for x in base[s]])
                         for s in seeds])
        val_sa = np.array([np.mean([x[1] for x in per_seed[s]]) for s in seeds])
        val_b = np.array([np.mean([x[1] for x in base[s]]) for s in seeds])
        ok = val_b > 1e-9
        dval = (val_sa[ok] / val_b[ok] - 1.0) * 100.0
        boot = [np.median(rng.choice(stab, len(stab))) for _ in range(n_boot)]
        rows.append({
            "config": cfg, "n": len(seeds),
            "d_stab_median": float(np.median(stab)),
            "d_stab_ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
            "stab_wins": int((stab < 0).sum()),
            "d_val_pct_median": float(np.median(dval)) if len(dval) else 0.0,
            "val_wins": int((dval < 0).sum()),
        })
    return sorted(rows, key=lambda r: r["d_stab_median"])


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--grid_root", default="dumps/hyperparam_search")
    p.add_argument("--problems", nargs="+", default=["nguyen5", "nguyen1"])
    p.add_argument("--n_data_seeds", type=int, default=10)
    p.add_argument("--n_evo_seeds", type=int, default=3)
    p.add_argument("--n_boot", type=int, default=5000)
    p.add_argument("--boot_seed", type=int, default=0)
    p.add_argument("--output", default=None, help="JSON output (default: <grid_root>/paired_analysis.json)")
    args = p.parse_args()

    rng = np.random.default_rng(args.boot_seed)
    data_seeds = set(400 + i for i in range(args.n_data_seeds))
    evo_seeds = set(4000 + i for i in range(args.n_evo_seeds))

    out = {}
    ranks = collections.defaultdict(dict)
    print(f"Paired diffs vs PSGE, {args.n_data_seeds} data seeds x {args.n_evo_seeds} evo seeds "
          f"(mean over evo seeds first). Δ = median(SA - PSGE); CI = bootstrap 95 % of median.")
    for prob in args.problems:
        res = load_results(os.path.join(args.grid_root, prob), data_seeds, evo_seeds)
        rows = paired_rows(res, args.n_evo_seeds, args.n_boot, rng)
        out[prob] = rows
        print(f"\n== {prob}")
        for k, r in enumerate(rows):
            ranks[r["config"]][prob] = k + 1
            lo, hi = r["d_stab_ci95"]
            print(f"{r['config']:44s} n={r['n']:>2}  Δstab={r['d_stab_median']:+.4f} "
                  f"CI[{lo:+.3f},{hi:+.3f}]  wins={r['stab_wins']}/{r['n']}  "
                  f"Δval%={r['d_val_pct_median']:+6.1f}  val-wins={r['val_wins']}/{r['n']}")

    common = [c for c, rk in ranks.items() if len(rk) == len(args.problems)]
    print("\nMean stability rank across problems (configs present in all):")
    for c in sorted(common, key=lambda c: np.mean(list(ranks[c].values()))):
        print(f"  {c:44s} {[ranks[c][pb] for pb in args.problems]}  "
              f"mean={np.mean(list(ranks[c].values())):.1f}")

    path = args.output or os.path.join(args.grid_root, "paired_analysis.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
