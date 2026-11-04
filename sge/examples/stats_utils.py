"""
Estadística del plan (sección "Estadística prevista") sin depender de scipy.

  - bca_ci:            IC bootstrap BCa de un estadístico (por defecto la mediana).
  - wilcoxon_exact:    prueba de rangos con signo de Wilcoxon, p exacto (distribución por
                       programación dinámica sobre los rangos, válida con empates), bilateral.
  - rank_biserial:     tamaño de efecto para datos emparejados (Kerby): (W+ - W-) / (W+ + W-).
  - spearman_perm:     rho de Spearman con p por permutación (bilateral).

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


def wilcoxon_exact(d):
    """Devuelve (W+, p bilateral exacto, n sin ceros)."""
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
    p_low = dist[:w2 + 1].sum()
    p_high = dist[w2:].sum()
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
    if not np.all(np.isfinite(x)) or n < 3:
        lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
        return float(lo), float(hi), "percentile"
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / n_boot
    prop = min(max(prop, 1.0 / n_boot), 1 - 1.0 / n_boot)
    z0 = _N.inv_cdf(prop)
    jack = np.array([stat(np.delete(x, i)) for i in range(n)])
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
