"""
Regla de ejecuciones inválidas (fijada el 2026-11-04, antes de la fase C y de la confirmación).

Un modelo puede no ser evaluable en algún punto (desbordamiento de exp, polo de una división
protegida solo en 0...). Entonces:
  - calculate_dataset_nrmse devuelve el centinela 9 999 999 para TODO el conjunto;
  - calculate_point_squared_loss devuelve una pérdida tope de 1e6 (/ Var(Y)) para ese punto.
Promediar esos centinelas como si fueran errores reales distorsiona cualquier agregado.

Regla:
  R1. Un valor de una ejecución es INVÁLIDO si no es finito o es >= INVALID_THRESHOLD (1e5).
      Valores válidos observados en 1 782 ejecuciones piloto: NRMSE-val <= 24, APG+ATG <= 15.
  R2. Un valor inválido se sustituye por +inf (el peor resultado posible). NUNCA se descarta:
      descartar favorecería al algoritmo que falla.
  R3. Dentro de cada semilla de datos, las semillas evolutivas se agregan con la MEDIANA
      (un fallo entre 3 semillas no domina el agregado).
  R4. Diferencias emparejadas: inf - inf = 0 (fallan ambos); inf - x = +inf; x - inf = -inf.
  R5. Diferencia relativa (%): ambos inf -> 0; solo el tratamiento inf -> +inf;
      solo el control inf -> -100.
  R6. Se informa siempre del número y la tasa de ejecuciones inválidas por configuración.
"""

import math
import numpy as np

INVALID_THRESHOLD = 1e5


def is_invalid(value):
    """R1."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return True
    return (not math.isfinite(v)) or v >= INVALID_THRESHOLD


def clean(value):
    """R1 + R2: valor válido tal cual; inválido -> +inf."""
    return math.inf if is_invalid(value) else float(value)


def agg_evo(values):
    """R3: mediana de las semillas evolutivas de una semilla de datos (inválidos ya como +inf)."""
    return float(np.median([clean(v) for v in values]))


def paired_diff(treat, ctrl):
    """R4."""
    if math.isinf(treat) and math.isinf(ctrl):
        return 0.0
    return treat - ctrl


def rel_diff_pct(treat, ctrl):
    """R5. Devuelve None si el control es ~0 y ambos son finitos (no hay porcentaje definido)."""
    t_inf, c_inf = math.isinf(treat), math.isinf(ctrl)
    if t_inf and c_inf:
        return 0.0
    if t_inf:
        return math.inf
    if c_inf:
        return -100.0
    if ctrl <= 1e-9:
        return None
    return (treat / ctrl - 1.0) * 100.0


def count_invalid(values):
    """R6."""
    return sum(1 for v in values if is_invalid(v))
