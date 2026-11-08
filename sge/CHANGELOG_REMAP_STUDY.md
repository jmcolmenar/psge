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

## 2026-11-08: piloto (fase 5)

Campaña `meta/remap_pilot_20261108-204010` (servidor, commit 85628b5; el «dirty» del registro es solo el
bit de ejecución de `server/archive_runs.sh`). 1320 ejecuciones emparejadas, 0 errores, 0 modelos no
evaluables, 1 h 35 min con 32 procesos. La prueba mínima da en el servidor exactamente los mismos valores
que en el portátil. Solo validación; el test no se generó.

| Pregunta del piloto | Resultado | Decisión |
| --- | --- | --- |
| ¿La entropía baja de forma gradual con λ? | Mediana final 1,0 (SGEF), ~0,93, ~0,8, ~0,65, 0,35–0,54, 0,30–0,51; mínimo 0,19 | Rejilla de λ sin cambios |
| ¿λ = 0,1 colapsa la gramática o dispara los inválidos? | No: entropía mínima 0,19, inválidos 0 | — |
| ¿Funciona la manipulación de C? | Máximo por actualización: PSGE 26–77 % de la población; PSGE-PP 0 (λ ≤ 0,02) y 2–5 % (λ ≥ 0,05) por codones imposibles de conservar (≤ 1,1e-4) | — |
| ¿Cambia el tamaño efectivo de la mutación? | Fracción de mutaciones que cambian de regla menor en PSGE (p. ej., 0,29 frente a 0,33 en Nguyen-1, λ = 0,1) | Se informa como diferencia entre variantes |
| Coste | ~96 s por par con m = 20 y ~176 s con m = 100; PSGE-PP ~10 % más lento | Confirmación ~52 h con 32 procesos |
| Réplicas | DE entre semillas de datos del log-ratio emparejado 0,5–1,7 (celdas sin soluciones exactas) | 20 × 5 sin cambios |

Dos problemas del plan de análisis, resueltos antes de congelar (decisiones del 2026-11-08):

1. **Soluciones exactas.** En Nguyen-1, m = 100, sin ruido, el 24 % de las ejecuciones encuentra la función
   y APG+ATG baja hasta 1e-34 (redondeo); la DE del log entre semillas sube de ~1 a 5–33.
   **Decisión:** suelo EPS = 1e-6 (unidades normalizadas) para APG+ATG, APG, ATG y la brecha antes de
   tomar logaritmos, y sqrt(EPS) = 1e-3 para la NRMSE (el ensayo del análisis con el piloto mostró que
   sin ese suelo el cociente de NRMSE también explota). Sensibilidad con 1e-4 y 1e-8.
2. **No inferioridad en NRMSE inalcanzable.** Con 20 semillas de datos, la semianchura del IC90 por celda
   del cociente de NRMSE es del 21–45 % en celdas sin soluciones exactas, frente a un margen del 2 %: H-C1
   no podría confirmarse nunca. **Decisión (opción A):** la NRMSE deja de condicionar H-C1 y pasa a un
   contraste secundario propio a nivel de benchmark: no inferioridad si el extremo superior del IC BCa 90 %
   de la media entre benchmarks de log(NRMSE PP / PSGE) es < log(1,10). La tabla por celda y el Pareto
   quedan como descripción.

## 2026-11-08: prerregistro (fase 6)

- `examples/analyze_remap_study.py`: análisis completo fijado (reglas, familia principal H-A1 + H-C1 con
  Holm, secundarios NR, comprobaciones de la manipulación, H-A2, H-A3, H-C2, H-E1, H-E2, robustez y
  sensibilidad a EPS). Ensayado sobre el piloto con `--split val`.
- `tests/test_analyze_remap.py` (9 tests) con datos simulados de efecto conocido: el nulo no se confirma;
  efectos de λ (+0,3 por nivel) y de PSGE-PP (−0,5) se confirman con el tamaño correcto; Holm; no
  inferioridad (NRMSE igual → sí; +30 % → no, sin afectar a H-C1); seguimiento de la brecha; interacción
  de H-A3; soluciones exactas e inválidos; semillas incompletas. Calibración del error de tipo I con 300
  simulaciones nulas: H-A1 1,7 % y H-C1 2,3 % con α = 0,025; 2,3 % y 3,0 % con α = 0,05.
- `parameters/remap_study.yml`: **`FROZEN: true`**. `server/run_remap_confirmation.sh` ejecuta el análisis
  al terminar; `server/run_remap_analysis.sh` lo repite.
- Tag `remap-confirmation-v1`.
