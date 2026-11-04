"""
Module: stability.py
Implements the algorithmic stability metrics (APG, ATG, APG+ATG),
grammar divergence (Jensen-Shannon, Entropy, KL), dataset generation under the
add-one protocol, and benchmark problem definitions for SGE, PSGE, and SA-PSGE.

References:
- Fong & Motani (GECCO 2026): "Generalization Analysis of Symbolic Regression Algorithms via Stability Theory"
- Plan de investigacion GECCO (Secciones 4, 5, 6)
"""

import numpy as np
from math import log2
import copy

# Import protected mathematical operators used by SGE
from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, _sig_, _pow_, protdiv


# =====================================================================
# 1. BENCHMARK PROBLEMS DEFINITIONS (Sección 6.1 del Plan)
# =====================================================================

def nguyen1(x):
    """Nguyen-1: y = x^3 + x^2 + x, x in [-1, 1]"""
    x0 = x[0] if isinstance(x, (list, tuple, np.ndarray)) else x
    return x0**3 + x0**2 + x0


def nguyen5(x):
    """Nguyen-5: y = sin(x^2)*cos(x) - 1, x in [-1, 1]"""
    x0 = x[0] if isinstance(x, (list, tuple, np.ndarray)) else x
    return np.sin(x0**2) * np.cos(x0) - 1.0


def nguyen7(x):
    """Nguyen-7: y = ln(x+1) + ln(x^2+1), x in [0, 2]"""
    x0 = x[0] if isinstance(x, (list, tuple, np.ndarray)) else x
    return np.log(x0 + 1.0) + np.log(x0**2 + 1.0)


def nguyen9(x):
    """Nguyen-9: y = sin(x) + sin(y^2), x, y in [0, 1]"""
    x0, x1 = x[0], x[1]
    return np.sin(x0) + np.sin(x1**2)


EULER_GAMMA = 0.57721566490153286061


def _digamma(x):
    """Digamma psi(x) para x > 0 (recurrencia hasta x >= 6 y serie asintótica; error < 1e-12)."""
    x = float(x)
    acc = 0.0
    while x < 6.0:
        acc -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    series = inv2 * (1.0 / 12 - inv2 * (1.0 / 120 - inv2 * (1.0 / 252 - inv2 * (1.0 / 240 - inv2 * (1.0 / 132)))))
    return acc + np.log(x) - 0.5 * inv - series


def keijzer6(x):
    """Keijzer-6: número armónico H(x) = sum_{i=1}^{x} 1/i (Keijzer 2003; McDermott et al. 2012).

    El original usa x ENTERO: entrenamiento E[1, 50, 1], test E[1, 120, 1]. Con enteros, m = 100 puntos
    distintos es imposible (solo hay 50), así que se usa la extensión continua exacta
    H(x) = psi(x + 1) + gamma, que coincide con la suma en cada entero y es suave entre ellos
    (desviación documentada, decisión del 2026-11-04). Entrenamiento U[1, 50]; validación y test U[1, 120]
    (interpolación + extrapolación, como el test original).
    """
    x0 = x[0] if isinstance(x, (list, tuple, np.ndarray)) else x
    return _digamma(float(x0) + 1.0) + EULER_GAMMA


def pagie1(x):
    """Pagie-1: y = 1/(1 + x^-4) + 1/(1 + y^-4), x, y in [-5, 5].
    El original entrena en la rejilla E[-5, 5, 0.4] (676 puntos) sin test; el protocolo add-one
    necesita S de m puntos, así que se muestrea U[-5, 5]^2 (desviación documentada)."""
    x0, x1 = x[0], x[1]
    denom0 = 1.0 + (x0**(-4) if x0 != 0 else 1e12)
    denom1 = 1.0 + (x1**(-4) if x1 != 0 else 1e12)
    return (1.0 / denom0) + (1.0 / denom1)


def vladislavleva4(x):
    """Vladislavleva-4 (UBall5D): 10 / (5 + sum_i (x_i - 3)^2).
    Entrenamiento U[0.05, 6.05]^5, test U[-0.25, 6.35]^5 (Vladislavleva et al. 2009; McDermott et al. 2012)."""
    denom = 5.0 + sum((x[i] - 3.0)**2 for i in range(5))
    return 10.0 / denom


def vladislavleva7(x):
    """Vladislavleva-7: (x0 - 3)(x1 - 3) + 2 sin((x0 - 4)(x1 - 4)).
    Entrenamiento U[0.05, 6.05]^2, test U[-0.25, 6.35]^2 (Vladislavleva et al. 2009; McDermott et al. 2012)."""
    x0, x1 = x[0], x[1]
    return (x0 - 3.0) * (x1 - 3.0) + 2.0 * np.sin((x0 - 4.0) * (x1 - 4.0))


BENCHMARK_SPECS = {
    "nguyen1": {
        "function": nguyen1,
        "n_vars": 1,
        "domain": [(-1.0, 1.0)],
        "grammar": "grammars/regression_1v.pybnf",
        "description": "Nguyen-1: x^3 + x^2 + x (Pilot)"
    },
    "nguyen5": {
        "function": nguyen5,
        "n_vars": 1,
        "domain": [(-1.0, 1.0)],
        "grammar": "grammars/regression_1v.pybnf",
        "description": "Nguyen-5: sin(x^2)cos(x) - 1 (Pilot)"
    },
    "nguyen7": {
        "function": nguyen7,
        "n_vars": 1,
        "domain": [(0.0, 2.0)],
        "grammar": "grammars/regression_1v.pybnf",
        "description": "Nguyen-7: ln(x+1) + ln(x^2+1)"
    },
    "nguyen9": {
        "function": nguyen9,
        "n_vars": 2,
        "domain": [(0.0, 1.0), (0.0, 1.0)],
        "grammar": "grammars/regression.pybnf",
        "description": "Nguyen-9: sin(x0) + sin(x1^2)"
    },
    "keijzer6": {
        "function": keijzer6,
        "n_vars": 1,
        "train_domain": [(1.0, 50.0)],
        "test_domain": [(1.0, 120.0)],
        "grammar": "grammars/regression_1v.pybnf",
        "description": "Keijzer-6: Harmonic number, continuous extension psi(x+1)+gamma"
    },
    "pagie1": {
        "function": pagie1,
        "n_vars": 2,
        "domain": [(-5.0, 5.0), (-5.0, 5.0)],
        "grammar": "grammars/regression.pybnf",
        "description": "Pagie-1: 1/(1+x^-4) + 1/(1+y^-4)"
    },
    "vladislavleva4": {
        "function": vladislavleva4,
        "n_vars": 5,
        "train_domain": [(0.05, 6.05)] * 5,
        "test_domain": [(-0.25, 6.35)] * 5,
        "grammar": "grammars/regression_5v.pybnf",
        "description": "Vladislavleva-4: 5D radial"
    },
    "vladislavleva7": {
        "function": vladislavleva7,
        "n_vars": 2,
        "train_domain": [(0.05, 6.05), (0.05, 6.05)],
        "test_domain": [(-0.25, 6.35), (-0.25, 6.35)],
        "grammar": "grammars/regression.pybnf",
        "description": "Vladislavleva-7: 2D interaction"
    }
}


# =====================================================================
# 2. DATASET GENERATION UNDER ADD-ONE PROTOCOL (Sección 5.2 del Plan)
# =====================================================================

VALIDATION_SEED_OFFSET = 55555
TEST_SEED_OFFSET = 99999


def _sample_noise_free_set(func, domains, n, seed):
    """Draws n noise-free points uniformly from `domains` with its own RNG."""
    if n <= 0:
        return []
    rng = np.random.RandomState(seed)
    xs = np.column_stack([rng.uniform(low, high, size=n) for low, high in domains])
    ys = np.array([func(row) for row in xs], dtype=float)
    return [list(xs[i]) + [float(ys[i])] for i in range(n)]


def _target_variance(dataset):
    if len(dataset) <= 1:
        return None
    v = float(np.var([pt[-1] for pt in dataset]))
    return v if v > 1e-12 else 1.0


def generate_add_one_datasets(problem_name, m=20, noise_ratio=0.0, seed_data=42, n_test=1000, n_val=500):
    """
    Generates paired datasets under the Add-One protocol:
      - S: Base dataset with m instances [(x, y), ...]
      - z: A single independent instance (x_z, y_z) drawn from the same domain
      - S_plus_z: Perturbed dataset S U {z} with m+1 instances
      - val_set: Validation set (n_val instances) for hyperparameter tuning /
        model selection. Independent RNG stream (seed_data + VALIDATION_SEED_OFFSET),
        same input distribution as the test set, noise-free like the test set.
      - test_set: Isolated test set with n_test instances. Must NOT be used
        during tuning. Pass n_test=0 to not generate it at all (tuning scripts).

    The S/z/test streams are unchanged with respect to previous versions,
    so existing results remain reproducible.

    Returns:
      dict with keys: 'S', 'z', 'S_plus_z', 'val_set', 'test_set',
                      'var_y_S', 'var_y_val', 'var_y_test', 'spec'
    """
    if problem_name not in BENCHMARK_SPECS:
        raise ValueError(f"Unknown benchmark problem '{problem_name}'. Available: {list(BENCHMARK_SPECS.keys())}")

    spec = BENCHMARK_SPECS[problem_name]
    func = spec["function"]
    n_vars = spec["n_vars"]
    rng = np.random.RandomState(seed_data)

    domains = spec.get("domain") or spec.get("train_domain")

    # Sample m + 1 development points
    # Points 0 to m-1 form S, point m is z
    dev_points_x = []
    for low, high in domains:
        dev_points_x.append(rng.uniform(low, high, size=m + 1))
    dev_points_x = np.column_stack(dev_points_x)  # shape (m+1, n_vars)

    dev_points_y = np.array([func(row) for row in dev_points_x], dtype=float)

    # Add Gaussian noise if requested: noise_ratio * sigma_y
    if noise_ratio > 0.0:
        sigma_y = float(np.std(dev_points_y))
        noise = rng.normal(0.0, noise_ratio * sigma_y, size=m + 1)
        dev_points_y += noise

    # Split into S and z
    S = [list(dev_points_x[i]) + [float(dev_points_y[i])] for i in range(m)]
    z = list(dev_points_x[m]) + [float(dev_points_y[m])]
    S_plus_z = copy.deepcopy(S)
    S_plus_z.append(copy.deepcopy(z))

    # Held-out sets (independent streams; same distribution for val and test)
    test_domains = spec.get("test_domain") or domains
    test_set = _sample_noise_free_set(func, test_domains, n_test, seed_data + TEST_SEED_OFFSET)
    val_set = _sample_noise_free_set(func, test_domains, n_val, seed_data + VALIDATION_SEED_OFFSET)

    # Compute variance of targets
    targets_S = np.array([pt[-1] for pt in S])
    var_y_S = float(np.var(targets_S)) if len(targets_S) > 1 else 1.0
    if var_y_S <= 1e-12:
        var_y_S = 1.0

    var_y_test = _target_variance(test_set)
    var_y_val = _target_variance(val_set)

    return {
        "problem_name": problem_name,
        "m": m,
        "seed_data": seed_data,
        "noise_ratio": noise_ratio,
        "S": S,
        "z": z,
        "S_plus_z": S_plus_z,
        "val_set": val_set,
        "test_set": test_set,
        "var_y_S": var_y_S,
        "var_y_val": var_y_val,
        "var_y_test": var_y_test,
        "spec": spec
    }


# =====================================================================
# 3. SAFE EXPRESSION EVALUATION
# =====================================================================

def evaluate_phenotype_on_point(phenotype_str, point, n_vars=1):
    """
    Evaluates an evolved phenotype string on a single data point.
    Returns predicted float y_hat, or None if evaluation raises an arithmetic error.
    """
    if phenotype_str is None:
        return None

    # Handle input vector x
    x_input = point[:-1] if len(point) > 1 else [point[0]]
    # Ensure x has at least 2 elements to prevent IndexError if grammar generates x[1]
    if len(x_input) < 2:
        x_input = list(x_input) + [0.0] * (2 - len(x_input))

    env_globals = {
        "sin": np.sin,
        "cos": np.cos,
        "_log_": _log_,
        "_div_": _div_,
        "_exp_": _exp_,
        "_inv_": _inv_,
        "_sqrt_": _sqrt_,
        "_sig_": _sig_,
        "_pow_": _pow_,
        "protdiv": protdiv
    }
    env_locals = {"x": x_input}

    try:
        y_hat = eval(phenotype_str, env_globals, env_locals)
        if isinstance(y_hat, np.ndarray):
            y_hat = float(y_hat.item()) if y_hat.size == 1 else float(y_hat[0])
        else:
            y_hat = float(y_hat)
        if np.isnan(y_hat) or np.isinf(y_hat):
            return None
        return y_hat
    except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError, IndexError, TypeError):
        return None


def calculate_point_squared_loss(phenotype_str, point, invalid_loss=1e6, var_y=None):
    """
    Calculates the squared error loss V(f, point) = (f(x) - y)^2.
    If var_y is provided, normalizes the loss: V_norm(f, point) = (f(x) - y)^2 / var_y.
    """
    y_target = point[-1]
    y_hat = evaluate_phenotype_on_point(phenotype_str, point)
    if y_hat is None:
        loss = invalid_loss
    else:
        loss = (y_hat - y_target)**2
        if np.isnan(loss) or np.isinf(loss) or loss > invalid_loss:
            loss = invalid_loss

    if var_y is not None and var_y > 1e-12:
        return loss / var_y
    return loss


def calculate_dataset_nrmse(phenotype_str, dataset, invalid_fitness=9999999):
    """
    Calculates NRMSE = sqrt(MSE / Var(Y)) = RMSE / sigma_y over a dataset.
    """
    if phenotype_str is None or len(dataset) == 0:
        return invalid_fitness

    targets = [pt[-1] for pt in dataset]
    mean_y = sum(targets) / len(targets)
    denom = sum((y - mean_y)**2 for y in targets)
    if denom <= 1e-12:
        denom = 1.0

    sse = 0.0
    for pt in dataset:
        y_target = pt[-1]
        y_hat = evaluate_phenotype_on_point(phenotype_str, pt)
        if y_hat is None:
            return invalid_fitness
        diff = y_hat - y_target
        sse += diff * diff

    nrmse = np.sqrt(sse / denom)
    return float(nrmse) if not (np.isnan(nrmse) or np.isinf(nrmse)) else invalid_fitness


# =====================================================================
# 4. STABILITY METRICS (APG, ATG, APG+ATG, JENSEN-SHANNON, ENTROPY)
# =====================================================================

def calculate_stability_metrics(model_S_phenotype, model_S_plus_z_phenotype, S, z, test_set=None, var_y=None, val_set=None):
    """
    Computes stability and generalization metrics for a paired run A(S) and A(S U {z}):
      - APG: Add-One Point-wise Gap = |V(f_S, z) - V(f_{S+z}, z)|
      - ATG_hat: Empirical Add-One Training Gap = |V(f_{S+z}, z) - (1/m) sum_{z_i in S} V(f_S, z_i)|
      - Data Stability: APG + ATG_hat
      - Training NRMSE for both models
      - Validation NRMSE for both models (if val_set is given)
      - Test NRMSE and generalization gap (only if test_set is given;
        tuning code must pass test_set=None)
    """
    m = len(S)

    # 1. Losses on the add-one point z
    loss_fS_z_raw = calculate_point_squared_loss(model_S_phenotype, z)
    loss_fSz_z_raw = calculate_point_squared_loss(model_S_plus_z_phenotype, z)

    # Normalized losses on z (by var_y)
    loss_fS_z_norm = calculate_point_squared_loss(model_S_phenotype, z, var_y=var_y)
    loss_fSz_z_norm = calculate_point_squared_loss(model_S_plus_z_phenotype, z, var_y=var_y)

    # 2. In-sample losses on S under model f_S
    losses_fS_on_S_raw = [calculate_point_squared_loss(model_S_phenotype, pt) for pt in S]
    losses_fS_on_S_norm = [calculate_point_squared_loss(model_S_phenotype, pt, var_y=var_y) for pt in S]

    mean_in_sample_loss_fS_raw = float(np.mean(losses_fS_on_S_raw))
    mean_in_sample_loss_fS_norm = float(np.mean(losses_fS_on_S_norm))

    # 3. APG (raw and normalized)
    apg_raw = abs(loss_fS_z_raw - loss_fSz_z_raw)
    apg_norm = abs(loss_fS_z_norm - loss_fSz_z_norm)

    # 4. ATG_hat (empirical ATG comparing in-sample loss of z in f_{S+z} with mean in-sample loss in f_S)
    atg_raw = abs(loss_fSz_z_raw - mean_in_sample_loss_fS_raw)
    atg_norm = abs(loss_fSz_z_norm - mean_in_sample_loss_fS_norm)

    # 5. Combined Data Stability
    data_stability_raw = apg_raw + atg_raw
    data_stability_norm = apg_norm + atg_norm

    # 6. Generalization metrics (NRMSE)
    nrmse_train_fS = calculate_dataset_nrmse(model_S_phenotype, S)
    nrmse_train_fSz = calculate_dataset_nrmse(model_S_plus_z_phenotype, S)

    if val_set:
        nrmse_val_fS = calculate_dataset_nrmse(model_S_phenotype, val_set)
        nrmse_val_fSz = calculate_dataset_nrmse(model_S_plus_z_phenotype, val_set)
    else:
        nrmse_val_fS = nrmse_val_fSz = None

    if test_set:
        nrmse_test_fS = calculate_dataset_nrmse(model_S_phenotype, test_set)
        nrmse_test_fSz = calculate_dataset_nrmse(model_S_plus_z_phenotype, test_set)
        gen_gap_fS = abs(nrmse_test_fS - nrmse_train_fS)
        gen_gap_fSz = abs(nrmse_test_fSz - nrmse_train_fSz)
    else:
        nrmse_test_fS = nrmse_test_fSz = gen_gap_fS = gen_gap_fSz = None

    return {
        "loss_fS_z": loss_fS_z_raw,
        "loss_fSz_z": loss_fSz_z_raw,
        "loss_fS_z_norm": loss_fS_z_norm,
        "loss_fSz_z_norm": loss_fSz_z_norm,
        "mean_in_sample_loss_fS": mean_in_sample_loss_fS_raw,
        "mean_in_sample_loss_fS_norm": mean_in_sample_loss_fS_norm,
        "apg_raw": apg_raw,
        "apg_norm": apg_norm,
        "atg_raw": atg_raw,
        "atg_norm": atg_norm,
        "data_stability_raw": data_stability_raw,
        "data_stability_norm": data_stability_norm,
        "nrmse_train_fS": nrmse_train_fS,
        "nrmse_val_fS": nrmse_val_fS,
        "nrmse_val_fSz": nrmse_val_fSz,
        "nrmse_test_fS": nrmse_test_fS,
        "gen_gap_fS": gen_gap_fS,
        "nrmse_train_fSz": nrmse_train_fSz,
        "nrmse_test_fSz": nrmse_test_fSz,
        "gen_gap_fSz": gen_gap_fSz,
    }


def calculate_jensen_shannon(pcfg_S, pcfg_S_plus_z, mask):
    """
    Calculates the Jensen-Shannon divergence JS(P_S, P_{S+z}) between two PCFG probability tables.
    Returns:
      - mean_js: average JS divergence across all non-terminals with > 1 production rules
      - js_by_nt: list of JS values for each non-terminal
    """
    rows, cols = pcfg_S.shape
    js_values = []

    for i in range(rows):
        valid_indices = [j for j in range(cols) if mask[i, j]]
        if len(valid_indices) <= 1:
            continue

        p = pcfg_S[i, valid_indices].astype(float)
        q = pcfg_S_plus_z[i, valid_indices].astype(float)

        # Normalize to ensure valid discrete distribution
        sum_p = np.sum(p)
        sum_q = np.sum(q)
        p = p / sum_p if sum_p > 0 else np.ones_like(p) / len(p)
        q = q / sum_q if sum_q > 0 else np.ones_like(q) / len(q)

        m = 0.5 * (p + q)

        # KL divergences KL(P||M) and KL(Q||M)
        kl_p_m = 0.0
        kl_q_m = 0.0
        for idx in range(len(valid_indices)):
            if p[idx] > 1e-12 and m[idx] > 1e-12:
                kl_p_m += p[idx] * log2(p[idx] / m[idx])
            if q[idx] > 1e-12 and m[idx] > 1e-12:
                kl_q_m += q[idx] * log2(q[idx] / m[idx])

        js = 0.5 * kl_p_m + 0.5 * kl_q_m
        js_values.append(max(0.0, float(js)))

    mean_js = float(np.mean(js_values)) if len(js_values) > 0 else 0.0
    return mean_js, js_values


def calculate_pcfg_entropy(pcfg, mask):
    """
    Computes normalized Shannon entropy for each non-terminal and average across all choice non-terminals.
    H_norm(P) = H(P) / log2(K)
    """
    rows, cols = pcfg.shape
    entropies = []

    for i in range(rows):
        valid_indices = [j for j in range(cols) if mask[i, j]]
        k = len(valid_indices)
        if k <= 1:
            continue

        p = pcfg[i, valid_indices].astype(float)
        sum_p = np.sum(p)
        p = p / sum_p if sum_p > 0 else np.ones_like(p) / k

        h = 0.0
        for prob in p:
            if prob > 1e-12:
                h -= prob * log2(prob)
        h_norm = h / log2(k) if k > 1 else 1.0
        entropies.append(float(h_norm))

    return float(np.mean(entropies)) if len(entropies) > 0 else 1.0


# =====================================================================
# 5. SGE EVALUATION FUNCTION ADAPTER
# =====================================================================

class StabilityEvaluationFunction:
    """
    Evaluation adapter compatible with SGE evolutionary_algorithm.
    Computes fitness as NRMSE (RRSE) on the given dataset.

    track_test_online (default False): if True, every evaluated individual is
    also scored on test_set (logged as other_info['test_error']). This is only
    for monitoring, never for selection, but it is very expensive (n_test
    evaluations per individual, ~25x the cost of fitness with m=20, n_test=500)
    and keeps the test set "in the loop". Leave it off for pilots/tuning;
    compute test metrics once on the final models instead.
    """
    def __init__(self, dataset, test_set=None, invalid_fitness=9999999, track_test_online=False):
        self.__train_set = dataset
        self.__test_set = test_set if track_test_online else None
        self.__invalid_fitness = invalid_fitness
        # Contadores de coste (plan: contabilizar individuos y evaluaciones caso-individuo)
        self.n_evaluations = 0          # llamadas de evaluación (individuo completo o submuestra)
        self.n_case_evaluations = 0     # casos evaluados = suma de len(conjunto) en cada llamada
        self.calculate_denominators()

    def calculate_denominators(self):
        train_outputs = [entry[-1] for entry in self.__train_set]
        train_output_mean = float(sum(train_outputs)) / len(train_outputs)
        self.__denominator_train = sum([(i - train_output_mean)**2 for i in train_outputs])
        if self.__denominator_train <= 1e-12:
            self.__denominator_train = 1.0

        if self.__test_set:
            test_outputs = [entry[-1] for entry in self.__test_set]
            test_output_mean = float(sum(test_outputs)) / len(test_outputs)
            self.__denominator_test = sum([(i - test_output_mean)**2 for i in test_outputs])
            if self.__denominator_test <= 1e-12:
                self.__denominator_test = 1.0

    def evaluate(self, individual):
        if individual is None:
            return self.__invalid_fitness, {"generation": 0, "evals": 1, "test_error": self.__invalid_fitness}

        # Train error (NRMSE)
        self.n_evaluations += 1
        self.n_case_evaluations += len(self.__train_set)
        error = calculate_dataset_nrmse(individual, self.__train_set, invalid_fitness=self.__invalid_fitness)

        # Test error
        test_error = float("nan")
        if self.__test_set is not None:
            test_error = calculate_dataset_nrmse(individual, self.__test_set, invalid_fitness=self.__invalid_fitness)

        return error, {"generation": 0, "evals": 1, "test_error": test_error}

    def get_train_set(self):
        return self.__train_set

    def evaluate_on_dataset(self, individual, subset):
        self.n_evaluations += 1
        self.n_case_evaluations += len(subset)
        return calculate_dataset_nrmse(individual, subset, invalid_fitness=self.__invalid_fitness)
