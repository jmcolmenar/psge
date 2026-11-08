"""
Module: stability_aware_update.py
Implements:
  1. SA-PSGE: Stability-Aware PCFG update using subsampled views,
     relative sensitivity proxy, normalized rank credit, uniform mixture,
     and minimum probability floor.
  2. PSGE-KL: Ablation of PSGE that constrains the KL-divergence per update
     to separate dynamic smoothing from stability awareness.
  3. Dispatcher `perform_grammar_update`: dispatches between SGE, PSGE, PSGE-KL,
     and SA-PSGE based on params['ALGORITHM'].

References:
  Plan_investigacion_GECCO_estabilidad_SGE_PSGE_SA_PSGE.docx (Secciones 8.1 - 8.5)
"""

import copy
from math import log2
import numpy as np

import sge.grammar as grammar
from sge.parameters import params
from sge.operators.update import independent_update, get_grammar_counter
from sge.operators.phenotype_preserving import reencode_population


# ---------------------------------------------------------------------
# Dedicated RNG for SA-PSGE subsampled views.
# It is independent of the global np.random stream used by the
# evolutionary operators, so drawing views does NOT shift the evolutionary
# trajectory relative to PSGE / PSGE-KL runs with the same evo seed.
# Re-seeded from params['SEED'] in engine.setup() via reset_sa_rng().
# ---------------------------------------------------------------------
_SA_RNG_STREAM_TAG = 0x5A9E  # fixed tag to decorrelate from other seed uses
_SA_RNG = np.random.default_rng([0, _SA_RNG_STREAM_TAG])


def reset_sa_rng(seed):
    """Re-seed the SA-PSGE view RNG deterministically from the evo seed."""
    global _SA_RNG
    _SA_RNG = np.random.default_rng([int(seed), _SA_RNG_STREAM_TAG])


def get_sa_rng():
    return _SA_RNG


def rank01(values):
    """
    Computes normalized ranks in [0, 1] in ascending order (smallest value gets 0.0).
    Handles ties using the average rank.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=float)

    # Sort and compute rank
    sorted_indices = np.argsort(values)
    ranks = np.empty(n, dtype=float)
    ranks[sorted_indices] = np.arange(n, dtype=float)

    # Handle ties by averaging
    unique_vals, inverse_indices, counts = np.unique(values, return_inverse=True, return_counts=True)
    if len(unique_vals) < n:
        for idx_u, count in enumerate(counts):
            if count > 1:
                matching_mask = (inverse_indices == idx_u)
                ranks[matching_mask] = np.mean(ranks[matching_mask])

    return ranks / (n - 1.0)


def compute_pcfg_kl_divergence(pcfg_new, pcfg_old, mask):
    """
    Computes mean KL-divergence D_KL(P_new || P_old) across all choice non-terminals.
    """
    rows, cols = pcfg_new.shape
    kl_values = []
    for i in range(rows):
        valid = [j for j in range(cols) if mask[i, j]]
        if len(valid) <= 1:
            continue
        p = pcfg_new[i, valid].astype(float)
        q = pcfg_old[i, valid].astype(float)

        p = np.clip(p, 1e-12, 1.0)
        p = p / np.sum(p)
        q = np.clip(q, 1e-12, 1.0)
        q = q / np.sum(q)

        kl = np.sum(p * np.log2(p / q))
        kl_values.append(max(0.0, float(kl)))

    return float(np.mean(kl_values)) if len(kl_values) > 0 else 0.0


def kl_constrained_update(individuals, lf, n_best, delta=6e-5):
    """
    PSGE-KL (Sección 8.5):
    Applies standard PSGE update, but projects the tentative PCFG P* back toward P_t
    so that D_KL(P_{t+1} || P_t) <= delta.
    """
    pcfg_current = copy.deepcopy(grammar.get_pcfg())
    mask = grammar.get_mask()

    # Tentative standard update
    independent_update(individuals, lf, n_best)
    pcfg_tentative = copy.deepcopy(grammar.get_pcfg())

    current_kl = compute_pcfg_kl_divergence(pcfg_tentative, pcfg_current, mask)

    # If within limit, accept update
    if current_kl <= delta:
        return

    # Otherwise, project using bisection on gamma in [0, 1]
    # P(gamma) = (1 - gamma) * P_t + gamma * P*
    low = 0.0
    high = 1.0
    best_gamma = 0.0

    for _ in range(12):
        mid = 0.5 * (low + high)
        pcfg_mid = (1.0 - mid) * pcfg_current + mid * pcfg_tentative

        # Normalize rows
        rows, cols = pcfg_mid.shape
        for i in range(rows):
            valid = [j for j in range(cols) if mask[i, j]]
            if len(valid) > 1:
                pcfg_mid[i, valid] = pcfg_mid[i, valid] / np.sum(pcfg_mid[i, valid])

        kl_mid = compute_pcfg_kl_divergence(pcfg_mid, pcfg_current, mask)
        if kl_mid <= delta:
            best_gamma = mid
            low = mid
        else:
            high = mid

    # Apply projected distribution
    pcfg_projected = (1.0 - best_gamma) * pcfg_current + best_gamma * pcfg_tentative
    rows, cols = pcfg_projected.shape
    gram = grammar.get_pcfg()
    for i in range(rows):
        valid = [j for j in range(cols) if mask[i, j]]
        if len(valid) > 1:
            gram[i, valid] = pcfg_projected[i, valid] / np.sum(pcfg_projected[i, valid])


def extract_train_dataset(eval_func):
    """Safely extracts the training dataset list from the evaluation function."""
    if hasattr(eval_func, "get_train_set"):
        return eval_func.get_train_set()
    if hasattr(eval_func, "_StabilityEvaluationFunction__train_set"):
        return getattr(eval_func, "_StabilityEvaluationFunction__train_set")
    if hasattr(eval_func, "_SymbolicRegression__train_set"):
        return getattr(eval_func, "_SymbolicRegression__train_set")
    return None


def stability_aware_update(
    population,
    eval_func,
    eta,
    k_views=3,
    lambda_param=0.5,
    beta=1.0,
    alpha=0.005,
    p_min=1e-3,
    elite_ratio=0.10,
    rng=None,
    fallback_lf=0.01,
):
    """
    SA-PSGE Update Mechanism (Secciones 8.1 - 8.4 del Plan):
      1. Generates K subsampled views (80% of S without replacement).
      2. For each elite individual i, computes:
           mu_i = mean loss across K views
           sigma_i = std loss across K views
           s_i = sigma_i / (mu_i + eps) (relative sensitivity proxy)
      3. Ranks: q_i = rank01(mu_i) + lambda * rank01(s_i)
      4. Individual weights: w_i = exp(-beta * q_i)
      5. Rule credit C(r) weighted by w_i and freq_norm(r, i)
      6. Exponential update: P*_{t+1}(r) proportional to P_t(r) * exp(eta * C(r))
      7. Safeguards: mixture with uniform distribution U (fraction alpha)
         and probability floor p_min.

    Notes:
      - `eta` (params['SA_ETA']) is the SA-PSGE step size. It is deliberately
        decoupled from PSGE's LEARNING_FACTOR: the two update rules have
        different functional forms, so the same numeric value does not give
        comparable grammar movement. Calibrate eta so that the per-update
        KL(P_{t+1}||P_t) is comparable to PSGE's (see
        examples/calibrate_sa_eta.py).
      - `alpha` must be small relative to the update magnitude; otherwise the
        uniform mixture anchors the PCFG at the uniform distribution and
        SA-PSGE degenerates to SGE (observed with alpha=0.05, eta=0.01).
      - Views are drawn from a dedicated RNG (see reset_sa_rng), never from
        the global np.random stream.
    """
    dataset_S = extract_train_dataset(eval_func)
    if dataset_S is None or len(dataset_S) < 4:
        # Fallback to standard update if no dataset is accessible
        independent_update(population, fallback_lf, params.get("N_BEST", 1))
        return

    m = len(dataset_S)
    sub_size = max(2, int(0.8 * m))

    # 1. Select elite individuals (sorted by fitness ascending)
    pop_sorted = sorted(population, key=lambda ind: ind["fitness"])
    n_elite = max(2, min(len(pop_sorted), int(elite_ratio * len(pop_sorted))))
    elite = pop_sorted[:n_elite]

    # 2. Sample K views without replacement from S
    if rng is None:
        rng = get_sa_rng()
    views = []
    for _ in range(k_views):
        indices = rng.choice(m, size=sub_size, replace=False)
        views.append([dataset_S[idx] for idx in indices])

    # 3. Evaluate each elite individual on the K views
    mu_list = []
    sigma_list = []
    sens_list = []

    for ind in elite:
        phenotype = ind["phenotype"]
        losses = []
        for v in views:
            if hasattr(eval_func, "evaluate_on_dataset"):
                err = eval_func.evaluate_on_dataset(phenotype, v)
            else:
                err = eval_func.evaluate(phenotype)[0]
            if err is None or np.isnan(err) or np.isinf(err):
                err = 1e6
            losses.append(float(err))

        mu = float(np.mean(losses))
        sigma = float(np.std(losses))
        sens = sigma / (mu + 1e-6)

        mu_list.append(mu)
        sigma_list.append(sigma)
        sens_list.append(sens)

    # 4. Normalized ranks & individual weights
    rank_mu = rank01(mu_list)
    rank_sens = rank01(sens_list)
    q = rank_mu + lambda_param * rank_sens
    w = np.exp(-beta * q)
    sum_w = np.sum(w)
    if sum_w > 0:
        w = w / sum_w

    # 5. Rule credit accumulation
    # elite[i]['grammar_counter'][nt_idx][rule_idx]
    gram = grammar.get_pcfg()
    rows, columns = gram.shape
    mask = grammar.get_mask()

    credit = np.zeros_like(gram, dtype=float)

    for i_idx, ind in enumerate(elite):
        ind_weight = w[i_idx]
        counter = ind.get("grammar_counter")
        if counter is None:
            continue

        for nt_idx in range(rows):
            valid = [j for j in range(columns) if mask[nt_idx, j]]
            if len(valid) <= 1:
                continue

            total_used_nt = sum(counter[nt_idx])
            if total_used_nt > 0:
                for r_idx in valid:
                    freq_norm = counter[nt_idx][r_idx] / float(total_used_nt)
                    credit[nt_idx, r_idx] += ind_weight * freq_norm

    # Normalize credit per non-terminal
    for nt_idx in range(rows):
        valid = [j for j in range(columns) if mask[nt_idx, j]]
        if len(valid) <= 1:
            continue
        sum_c = np.sum(credit[nt_idx, valid])
        if sum_c > 0:
            credit[nt_idx, valid] = credit[nt_idx, valid] / sum_c

    # 6. Tentative update P*_{t+1}(r) proportional to P_t(r) * exp(eta * C(r))
    p_star = np.copy(gram)
    for nt_idx in range(rows):
        valid = [j for j in range(columns) if mask[nt_idx, j]]
        if len(valid) <= 1:
            continue

        old_p = gram[nt_idx, valid]
        exp_c = np.exp(eta * credit[nt_idx, valid])
        updated_p = old_p * exp_c
        p_star[nt_idx, valid] = updated_p / np.sum(updated_p)

    # 7. Safeguards: Uniform mixture + Minimum probability floor
    for nt_idx in range(rows):
        valid = [j for j in range(columns) if mask[nt_idx, j]]
        k_rules = len(valid)
        if k_rules <= 1:
            continue

        # Mixture with uniform distribution U
        u = 1.0 / k_rules
        mixed = (1.0 - alpha) * p_star[nt_idx, valid] + alpha * u

        # Apply minimum floor p_min
        floored = np.maximum(mixed, p_min)
        final_probs = floored / np.sum(floored)

        gram[nt_idx, valid] = final_probs


def perform_grammar_update(population, best_gen, flag, eval_func, iteration):
    """
    Central dispatcher called each generation by engine.py.
    Selects the update algorithm according to params['ALGORITHM']:
      - 'sge': No update (static uniform grammar).
      - 'psge': Standard PSGE update (Mégane et al., 2022).
      - 'psge_kl': PSGE with KL-divergence constraint ablation.
      - 'sa_psge': Stability-Aware PSGE.
      - 'psge_pp': PSGE update followed by phenotype-preserving re-encoding of the population
                   (see phenotype_preserving.py). Same update, same RNG use as 'psge'.
    """
    alg = params.get("ALGORITHM", "psge").lower()
    lf = params.get("LEARNING_FACTOR", 0.01)

    if alg == "sge":
        return

    if alg in ("psge", "psge_kl", "psge_pp") and lf == 0.0:
        return

    if alg in ("psge", "psge_pp"):
        pcfg_old = copy.deepcopy(grammar.get_pcfg()) if alg == "psge_pp" else None
        if not flag:
            independent_update(population, lf, params.get("N_BEST", 1))
        else:
            independent_update([best_gen] + population, lf, params.get("N_BEST", 1))
        if alg == "psge_pp":
            reencode_population(population, pcfg_old, grammar.get_pcfg())

    elif alg == "psge_kl":
        delta = params.get("KL_DELTA", 6e-5)
        inds = population if not flag else [best_gen] + population
        kl_constrained_update(inds, lf, params.get("N_BEST", 1), delta=delta)

    elif alg == "sa_psge":
        tau = params.get("SA_TAU", 1)
        if iteration % tau == 0:
            stability_aware_update(
                population=population,
                eval_func=eval_func,
                eta=params.get("SA_ETA", 0.04),
                k_views=params.get("SA_K_VIEWS", 3),
                lambda_param=params.get("SA_LAMBDA", 0.5),
                beta=params.get("SA_BETA", 1.0),
                alpha=params.get("SA_ALPHA", 0.005),
                p_min=params.get("SA_P_MIN", 1e-3),
                elite_ratio=params.get("SA_ELITE_RATIO", 0.10),
                fallback_lf=lf,
            )
    else:
        raise ValueError(f"Unknown ALGORITHM '{alg}'")
