# Running the SA-PSGE pilot grid on a server

Pilot / tuning only (nguyen5, nguyen1). Validation set only: these scripts never generate the test set.
Background and results so far: `CHANGELOG_SA_PSGE.md`.

## 1. Get the code onto the server

Most of the SA-PSGE work is **not committed yet** (see `git status`). Commit and push it first, so the
server runs a known commit, then clone it there. Do not commit `dumps/` (≈1 GB) or `.idea/`.

```bash
git clone https://github.com/jmcolmenar/psge.git && cd psge/sge
git rev-parse HEAD      # record this hash with the results
```

## 2. Environment (same versions as the laptop runs)

```bash
conda create -y -n psge-original python=3.8
conda activate psge-original
pip install -r requirements.txt        # numpy==1.19.2, PyYAML==5.3.1, tqdm==4.51.0
python -m unittest tests.test_stability tests.test_sa_psge   # expect 12 tests OK
```

Pin BLAS to one thread per process, otherwise the worker processes fight over the cores:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## 3. Reuse the runs already done (optional)

Each paired run writes `dumps/hyperparam_search/<problem>/<config>/<config>_d<data>_e<evo>.json`. If that file
exists, it is reused and the run is skipped. Only the `.json` files are needed; the run subdirectories are
not. From the laptop:

```bash
rsync -av --include='*/' --include='*.json' --exclude='*' dumps/hyperparam_search/ SERVER:psge/sge/dumps/hyperparam_search/
```

**Determinism check before mixing machines:** delete one cached file on the server, for example
`dumps/hyperparam_search/nguyen5/psge_baseline/psge_baseline_d400_e4000.json`. Rerun step 4 for a moment
and compare `apg_plus_atg` / `nrmse_val` with the laptop copy. If they differ, start with no cached runs and
rerun everything on the server, so that all results come from the same machine.

## 4. Commands

Run from `psge/sge/` inside `tmux` or `nohup`. Each paired run takes about 35 s on one core. `--n_jobs` is
the number of worker processes **per command**. With two problems in parallel, use about half the cores
for each.

```bash
J=16    # workers per problem
```

### Phase B: λ × β at η = 0.08, α = 0.01 (interrupted, resume)
12 configs × 30 runs per problem (~30 already done per problem, plus the cached baseline and λ=0.5/β=1).
λ = 0 is the planned "no stability signal" ablation.

```bash
for p in nguyen5 nguyen1; do
  nohup python -m examples.run_hyperparameter_search --problem $p --skip_phase1 --n_jobs $J \
    --n_data_seeds 10 --n_evo_seeds 3 --grid_eta 0.08 --grid_alpha 0.01 \
    --grid_lambda 0 0.25 0.5 1 --grid_beta 0.5 1 2 \
    --output_dir dumps/hyperparam_search/$p > dumps/grid_sa_${p}_phaseB.log 2>&1 &
done; wait
python -m examples.analyze_grid_paired --problems nguyen5 nguyen1
```

Choose `LAM`/`BET`: the config with the best mean stability rank across both problems whose median
Δ NRMSE-val is ≤ +2 % in both. If none qualifies, stop and review instead of forcing a choice.

### Phase C: sensitivity to K and τ (one factor at a time, at the chosen λ/β)
New config names get a `_k<K>_tau<tau>` suffix, so they cannot collide with the K=3/τ=1 runs.

```bash
LAM=0.5; BET=1     # <- from phase B
for p in nguyen5 nguyen1; do
  ( for kt in "2 1" "5 1" "3 5" "3 10"; do set -- $kt
      python -m examples.run_hyperparameter_search --problem $p --skip_phase1 --n_jobs $J \
        --n_data_seeds 10 --n_evo_seeds 3 --grid_eta 0.08 --grid_alpha 0.01 \
        --grid_lambda $LAM --grid_beta $BET --sa_k_views $1 --sa_tau $2 \
        --output_dir dumps/hyperparam_search/$p
    done ) > dumps/grid_sa_${p}_phaseC.log 2>&1 &
done; wait
python -m examples.analyze_grid_paired --problems nguyen5 nguyen1
```

The plan treats K/τ as **sensitivity only**: report them, and do not pick a new K/τ from them post hoc unless
the K=3/τ=1 choice is clearly dominated.

## 5. What to bring back

- `dumps/hyperparam_search/paired_analysis.json` and the console output of `analyze_grid_paired`.
  This analysis covers **all** configs found. The decision table printed by `run_hyperparameter_search`
  (`gridsearch_summary.json`) covers only the grid of the **last** call and is overwritten each time.
- `dumps/grid_sa_*_phase{B,C}.log`: check them for `ERROR` lines.
- All `*.json` per-run files (small; rsync with the filter above).
- Commit hash, `pip freeze`, CPU model / core count, wall time.

## 6. Still open before freezing (not addressed by these runs)

- `parameters/standard.yml` has MIN_TREE_DEPTH = MAX_TREE_DEPTH = 10; the plan says max depth 12.
  Changing it invalidates all cached runs (the cache key does not include it), so decide **before** running
  phase B on the server. If you change it, start with no cached runs.
- The cache key also ignores `SA_ELITE_RATIO`, `SA_P_MIN`, pop/gen and `standard.yml`. If any of these
  change, use a new `--output_dir`.
