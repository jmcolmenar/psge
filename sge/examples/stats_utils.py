"""
Estadística del plan (sección "Estadística prevista") sin depender de scipy.

  - bca_ci:            IC bootstrap BCa de un estadístico (por defecto la mediana).
  - wilcoxon_exact:    prueba de rangos con signo de Wilcoxon, p exacto (distribución por
                       programación dinámica sobre los rangos, válida con empates), bilateral.
  - rank_biserial:     tamaño de efecto para datos emparejados (Kerby): (W+ - W-) / (W+ + W-).
  - spearman_perm:     rho de Spearman con p por permutación (bilateral).
  - holm:              ajuste de Holm-Bonferroni.
  - friedman_iman_davenport: Friedman y su corrección F de Iman-Davenport (p con la F de Snedecor).

Convenciones:
  - Las diferencias exactamente 0 se descartan en Wilcoxon y rank-biserial (zero_method="wilcox").
  - Los valores +/-inf (ejecuciones inválidas, ver invalid_rule.py) se ordenan como los más extremos
    en las pruebas de rangos. bca_ci con valores no finitos cae a IC percentil y lo indica.
"""

import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def rankdata(x):
    """Rangos medios (1..n) con empates, como scipy.stats.rankdata(method='average')."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and (xs[j + 1] == xs[i]):
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _signed_ranks(d):
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return d, np.array([])
    return d, rankdata(np.abs(d))


def wilcoxon_exact(d, alternative="two-sided"):
    """Devuelve (W+, p exacto, n sin ceros).
    alternative: "two-sided"; "less" (H1: las diferencias tienden a ser negativas);
    "greater" (H1: tienden a ser positivas)."""
    d, r = _signed_ranks(d)
    n = len(d)
    if n == 0:
        return 0.0, 1.0, 0
    w_plus = float(r[d > 0].sum())
    # Rangos dobles para que sean enteros incluso con empates (x.5).
    r2 = np.rint(2 * r).astype(int)
    total = int(r2.sum())
    dist = np.zeros(total + 1, dtype=float)
    dist[0] = 1.0
    for k in r2:
        dist[k:] = dist[k:] + dist[:total + 1 - k]   # cada rango entra con signo + o -
    dist /= dist.sum()
    w2 = int(round(2 * w_plus))
    p_low = float(dist[:w2 + 1].sum())     # P(W+ <= w): pocas diferencias positivas
    p_high = float(dist[w2:].sum())
    if alternative == "less":
        return w_plus, min(1.0, p_low), n
    if alternative == "greater":
        return w_plus, min(1.0, p_high), n
    return w_plus, float(min(1.0, 2 * min(p_low, p_high))), n


def rank_biserial(d):
    d, r = _signed_ranks(d)
    if len(d) == 0:
        return 0.0
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def bca_ci(x, stat=np.median, n_boot=10000, level=0.95, rng=None):
    """IC BCa. Devuelve (lo, hi, metodo) con metodo 'bca' o 'percentile' (si hay no finitos)."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = len(x)
    theta = stat(x)
    boot = np.array([stat(x[rng.integers(0, n, n)]) for _ in range(n_boot)])
    alpha = (1 - level) / 2
    jack = None
    if np.all(np.isfinite(x)) and n >= 3 and np.all(np.isfinite(boot)) and np.isfinite(theta):
        jack = np.array([stat(np.delete(x, i)) for i in range(n)])
    if jack is None or not np.all(np.isfinite(jack)):
        lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
        return float(lo), float(hi), "percentile"
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / n_boot
    prop = min(max(prop, 1.0 / n_boot), 1 - 1.0 / n_boot)
    z0 = _N.inv_cdf(prop)
    dj = jack.mean() - jack
    den = 6.0 * (np.sum(dj ** 2) ** 1.5)
    a = float(np.sum(dj ** 3) / den) if den > 0 else 0.0
    qs = []
    for z_a in (_N.inv_cdf(alpha), _N.inv_cdf(1 - alpha)):
        qs.append(_N.cdf(z0 + (z0 + z_a) / (1 - a * (z0 + z_a))))
    lo, hi = np.percentile(boot, [100 * qs[0], 100 * qs[1]])
    return float(lo), float(hi), "bca"


def spearman_perm(x, y, n_perm=20000, rng=None):
    """Devuelve (rho, p bilateral por permutación, n con pares finitos)."""
    rng = rng or np.random.default_rng(0)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3:
        return float("nan"), 1.0, n
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2))
    if den == 0:
        return 0.0, 1.0, n
    rho = float(np.sum(rx * ry) / den)
    perms = np.array([np.sum(rx * rng.permutation(ry)) / den for _ in range(n_perm)])
    p = (np.sum(np.abs(perms) >= abs(rho) - 1e-12) + 1) / (n_perm + 1)
    return rho, float(p), n


def holm(pvalues):
    """p ajustados de Holm-Bonferroni (mismo orden que la entrada)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def _betacf(a, b, x, itmax=500, eps=3e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc_reg(a, b, x):
    """Beta incompleta regularizada I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def f_sf(f, d1, d2):
    """P(F >= f) para una F de Snedecor con (d1, d2) grados de libertad."""
    if f <= 0:
        return 1.0
    return betainc_reg(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def friedman_iman_davenport(scores):
    """scores: matriz bloques x algoritmos (menor = mejor; +inf permitido).
    Devuelve dict con rangos medios, chi2 de Friedman, F de Iman-Davenport y su p."""
    X = np.asarray(scores, dtype=float)
    N, k = X.shape
    R = np.array([rankdata(row) for row in X])
    Rj = R.mean(axis=0)
    chi2 = 12.0 * N / (k * (k + 1)) * (np.sum(Rj ** 2) - k * (k + 1) ** 2 / 4.0)
    denom = N * (k - 1) - chi2
    F = (N - 1) * chi2 / denom if denom > 0 else math.inf
    d1, d2 = k - 1, (k - 1) * (N - 1)
    p = 0.0 if math.isinf(F) else f_sf(F, d1, d2)
    return {"mean_ranks": Rj.tolist(), "chi2_friedman": float(chi2), "F_iman_davenport": float(F),
            "df": [d1, d2], "p": float(p), "n_blocks": N, "k": k}
