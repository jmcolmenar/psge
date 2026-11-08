# Estudio de adaptación de la gramática y re-mapeo en PSGE: registro de cambios

Plan: «Plan de investigación: ritmo de adaptación de la gramática, re-mapeo y estabilidad en PSGE»
(2026-11-08). Rama `remap-adaptation-study`, creada desde `stability-sa-psge` (6bde6d2). El registro del
estudio anterior sigue en `CHANGELOG_SA_PSGE.md`.

## 2026-11-08: infraestructura (fase 2) y benchmarks confirmatorios (fase 3)

### Compatibilidad con el estudio anterior
Los valores por defecto nuevos (`TRACK_TRAJECTORY: False`, `DUMP_*: True`) dejan el motor como estaba:
`sge`, `psge`, `psge_kl` y `sa_psge` siguen la misma trayectoria con la misma semilla. Los únicos cambios
en código existente son contadores (mutación) y condiciones sobre volcados (logger). Los 37 tests
anteriores siguen en verde.

### PSGE-PP (`ALGORITHM: psge_pp`; `sge/operators/phenotype_preserving.py`)
- Misma actualización de la PCFG que PSGE (`independent_update`, misma alternancia con `best_gen`) y,
  justo después, re-codificación de la población: cada codon usado pasa a la misma posición relativa
  del intervalo de su regla en la gramática nueva.
- Los intervalos replican exactamente la selección de `grammar._recursive_mapping`: umbrales
  `round(probabilidad acumulada, 3)`, y en el régimen de límite de profundidad solo las reglas del
  camino más corto, renormalizadas. El régimen lo decide la profundidad guardada en el codon.
- Las entradas del genotipo se sustituyen, nunca se modifican: los descendientes comparten entradas con
  sus padres.
- Si una regla queda con intervalo vacío en la gramática nueva (su probabilidad se redondea a 0), el
  codon no se puede conservar: se deja igual y se cuenta como fallo (`pp_failures`).
- No usa el generador aleatorio: con la misma semilla, PSGE y PSGE-PP hacen la misma primera
  actualización (test).

### Instrumentación (`sge/engine.py`, `TRACK_TRAJECTORY: True`)
Trayectoria por generación, tras la actualización de la gramática, accesible con `engine.get_trajectory()`:
`best_fitness`, `best_gen_fitness`, `median_fitness`, `invalid_frac`, `mean_depth`, `mean_used_codons`,
`unique_pct`, `entropy`, `kl_update`, `remap_rate`, `pp_codons`, `pp_moved`, `pp_failures`,
`mut_codons` y `mut_rule_changes` (acumulados).
- **Tasa de re-mapeo:** fracción de la población cuyo fenotipo guardado difiere del que da su genotipo,
  sin cambios, con la PCFG nueva. Trabaja sobre copias y guarda y restaura el estado de `np.random`
  (el mapeo sortea un codon si el genotipo se queda corto), así que medir no cambia la búsqueda (test).
  Con `REMAP_RATE_SAMPLE: 100` se mide sobre 100 individuos a rangos equiespaciados de la población
  ordenada, sin azar (sobre los 500 costaba ~0,6 s por generación, unos 60 s por brazo).
- **Tamaño efectivo de la mutación** (riesgo señalado en el plan): codones mutados y cuántos cambian de
  regla (`sge/operators/mutation.py`, solo contadores).
- **Volcados opcionales** (`sge/logger.py`): `DUMP_GRAMMAR`, `DUMP_POPULATION` (la población de la
  generación 0 era el 98 % del disco del estudio anterior) y `DUMP_PROGRESS`.

### Métricas (`sge/stability.py`)
- Brecha de generalización en las unidades de APG+ATG (RQ-E):
  `gen_gap_test_norm_fS = |media V(f_S, test) − media V(f_S, S)|`, con V normalizada por Var(Y\_S), y la
  misma con validación (`gen_gap_val_norm_fS`) para el piloto. También `mse_test_norm_fS` y `mse_val_norm_fS`.

### Estadística (`examples/stats_utils.py`)
- `page_trend`: prueba de tendencia de Page, unilateral (valores crecientes con el orden de las columnas),
  p exacto por convolución de las permutaciones de cada bloque (válido con empates y +inf); aproximación
  normal si k > 8. Coincide con `scipy.stats.page_trend_test(method="exact")` (test con valores de referencia).

### Benchmarks (`BENCHMARKS_REMAP_STUDY.md`)
- Regla escrita antes de ejecutar nada: Nguyen y Keijzer menos los ya usados, los que necesitan π, los
  polinomios de una variable (familia del piloto), los duplicados, la primitiva trivial y el de 3 variables.
- Confirmación: Nguyen-6, -8, -10, -11, -12 y Keijzer-4, -9, -11, -14, -15 (10). Piloto: Nguyen-1 y -5.

### Lanzadores
- `parameters/remap_study.yml`: presupuesto del estudio anterior, `REMAP_LAMBDAS` = [0,005, 0,01, 0,02,
  0,05, 0,1], trayectoria activada y volcados desactivados. **`FROZEN: false`** hasta el prerregistro.
- `examples/run_remap_study.py --phase smoke|pilot|confirm`: 11 condiciones (`sgef`, `psge_lf<λ>`,
  `psge_pp_lf<λ>`). Piloto: solo validación, semillas 3000.. / 30000..; confirmación: test de 1000 puntos,
  semillas 2000.. / 20000.., y se niega a lanzar sin `FROZEN: true`. JSON compacto con trayectorias y
  gramática final; escritura atómica.
- `examples/summarize_remap_pilot.py`: resumen descriptivo del piloto (entropía, inválidos, re-mapeo,
  fallos de PSGE-PP, mutación, coste y dispersión entre semillas de datos). No contrasta hipótesis.
- `server/run_remap_smoke.sh`, `server/run_remap_pilot.sh`, `server/run_remap_confirmation.sh`; `status.sh`
  cuenta también las ejecuciones de estas campañas.

### Tests (`tests/test_remap.py`, 18 nuevos; 55 en total)
Selección de reglas idéntica a la del mapeo (en los dos regímenes); la re-codificación conserva todos
los fenotipos salvo fallos contados; identidad si la gramática no cambia; entradas compartidas intactas;
la medición no toca el RNG; con y sin trayectoria la búsqueda es idéntica (psge, psge_pp, sge);
PSGE-PP sin re-mapeo y PSGE con él; SGEF plano; brecha; benchmarks; Page.

### Coste previsto (a revisar con el piloto)
- Piloto: 1320 ejecuciones emparejadas. Confirmación: 10 × 4 × 11 × 20 × 5 = 44 000 (el plan preveía
  35 200 con 8 benchmarks); con ~70 s por brazo, ~1700 h de CPU, ~53 h con 32 procesos. Recorte previsto
  si hiciera falta: 3 semillas evolutivas.
- Disco: ~50 KB por ejecución emparejada (trayectorias de 101 generaciones × 2 brazos) → ~2 GB en la
  confirmación, en `RESULTS_ROOT`; `RUNS_ROOT` solo guarda `parameters.json` de cada brazo.
