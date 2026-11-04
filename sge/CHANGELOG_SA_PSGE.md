# SA-PSGE: correcciones del 2026-11-03 (pasos 1, 2, 4 y 6)

## Diagnóstico que motiva los cambios (piloto `dumps/four_alg_test`, pop 50 × 20 gen)
- SA-PSGE acabó con entropía normalizada 0,999–1,000, igual que SGE: la gramática nunca se movía.
  La mezcla uniforme (alpha = 0,05 por generación) pesaba más que el paso multiplicativo
  (eta = LEARNING_FACTOR = 0,01). Lo reproduce `tests/test_sa_psge.py::test_old_defaults_anchor_at_uniform`.
- Con ese presupuesto, la mitad de las ejecuciones (de todos los algoritmos) convergen a la constante
  `_log_(cos(1.0))`, cuyo NRMSE de entrenamiento es > 1. Las comparaciones de NRMSE de ese piloto no son
  informativas.
- El grid de hiperparámetros elegía usando NRMSE-test (leakage respecto al protocolo).

## Cambios
1. **SA_ETA desacoplado de LEARNING_FACTOR** (`parameters.py`, `stability_aware_update.py`, `--sa_eta`).
   Calibrado con `examples/calibrate_sa_eta.py` para que la KL(P_{t+1}||P_t) mediana por actualización
   iguale a la de PSGE (lf = 0,01), con pop 200 × gen 50 y 3 semillas de calibración (500x/5000x):

   | problema | alpha | eta KL-equivalente | entropía final (PSGE / SA eta≈0,04) |
   |---|---|---|---|
   | nguyen5 | 0,005 | 0,044 | 0,92 / ~0,92 |
   | nguyen1 | 0,005 | 0,038 | 0,92 / ~0,90 |
   | nguyen5 | 0,001 | 0,041 | 0,92 / 0,90 |

   Valor por defecto **SA_ETA = 0,04**. Resultados crudos en `dumps/calib_sa_eta/`.
2. **SA_ALPHA por defecto: 0,05 → 0,005**, también en `run_pilot_campaign.py` y
   `run_stability_experiment.py`, que antes pasaban 0,05 explícitamente.
4. **Conjunto de validación** en `generate_add_one_datasets` (`val_set`, n_val = 500, semilla
   `seed_data + 55555`, misma distribución que el test). Los flujos de S, z y test no cambian, así que los
   resultados anteriores siguen siendo reproducibles. `n_test=0` no genera test.
   `calculate_stability_metrics(test_set=None, val_set=...)` devuelve `nrmse_val_*`.
   `run_hyperparameter_search.py` ya no genera ni evalúa el test: H2b y el ranking usan NRMSE-val; se
   agrega primero por semilla de datos (media de las semillas evolutivas) y luego mediana entre semillas
   de datos (n = nº de semillas de datos). Grid configurable:
   `--grid_eta --grid_alpha --grid_lambda --grid_beta`; por defecto pop 200 / gen 50; elitismo = 1 % de la población.
   `StabilityEvaluationFunction` ya **no evalúa el test en cada individuo** por defecto
   (`track_test_online=False`). Solo servía para el log, mantenía el test dentro del bucle y era
   unas 25 veces más caro que el fitness.
6. **RNG propio para las vistas de SA-PSGE** (`reset_sa_rng(seed)`, llamado en `engine.setup`). Ya no
   consume `np.random` global, así que el emparejamiento de semillas con PSGE/PSGE-KL se mantiene.

3. **KL_DELTA de PSGE-KL calibrado: 0,05 → 6e-5** (`parameters.py`, `stability_aware_update.py`,
   `run_pilot_campaign.py`, `run_stability_experiment.py`, `--fallback_delta` de `run_hyperparameter_search.py`).
   Fase 1 de `run_hyperparameter_search.py --skip_phase2`, PSGE lf = 0,01, pop 200 × gen 50, m = 20, sin ruido,
   3 semillas de calibración (300+i / 3000+i), 153 actualizaciones por problema:

   | problema | P10 | P25 | P50 | P75 | P90 |
   |---|---|---|---|---|---|
   | nguyen5 | 5,0e-5 | 5,9e-5 | 7,6e-5 | 1,2e-4 | 1,5e-4 |
   | nguyen1 | 4,3e-5 | 5,9e-5 | 7,7e-5 | 1,0e-4 | 1,8e-4 |

   El valor anterior (0,05) era ~800 veces mayor que el P99, así que **PSGE-KL nunca proyectaba y era idéntico a
   PSGE**: cualquier resultado previo de PSGE-KL no sirve como control. Se fija delta = P25 (restringe ~75 % de
   las actualizaciones). Sanity: nguyen5, pop 200 × gen 20 → KL máx 5,9997e-5 ≤ delta, 76 % de las
   actualizaciones en la cota. Resultados crudos en `dumps/calib_kl_delta/`.

5. **Grid de SA-PSGE (piloto, NO congelado)** en nguyen5 y nguyen1. `run_hyperparameter_search.py --skip_phase1`,
   pop 200 × gen 50, m = 20, sin ruido, eta ∈ {0,02, 0,04, 0,08} × alpha ∈ {0,001, 0,005, 0,01}, lambda = 0,5,
   beta = 1, K = 3, tau = 1, élite 10 %, p_min = 0,001; **10 semillas de datos (400–409) × 3 evolutivas
   (4000–4002)**; solo validación (sin test). Logs: `dumps/grid_sa_<p>_10x3.log`; crudos en
   `dumps/hyperparam_search/<p>/`; análisis emparejado en `dumps/hyperparam_search/paired_analysis.txt`.
   - Un primer grid con 2 × 2 semillas era puro ruido (APG+ATG de una misma configuración entre 0,02 y 4,6);
     no usarlo.
   - Tabla del script (mediana entre semillas de datos): solo **eta = 0,08, alpha = 0,01** cumple H2a ∧ H2b en
     ambos problemas (nguyen5: APG+ATG 0,485 vs 0,570, NRMSE-val +0,9 %; nguyen1: 0,026 vs 0,044,
     NRMSE-val −53 %). Mejor rango medio de estabilidad entre problemas (2.º y 4.º de 9).
   - **Pero**, con diferencias emparejadas por semilla de datos, ningún IC bootstrap 95 % de la mediana de
     ΔAPG+ATG excluye 0 (candidata: nguyen5 −0,090 [−0,376, +0,078], 7/10 victorias; nguyen1 ≈ 0
     [−0,055, +0,045], 5/10). Las mejoras de NRMSE-val de nguyen1 parecen reales (7/10 victorias, −53 %).
   - Lectura: en el piloto, SA-PSGE no se distingue de PSGE en estabilidad con n = 10; la elección de eta/alpha
     es débil. eta = 0,08 está en el borde del grid.

   - **Fase A (bordes)**: eta ∈ {0,08, 0,16} × alpha ∈ {0,01, 0,02}. Todos peores que eta = 0,08 / alpha = 0,01
     en ambos problemas; eta = 0,16 / alpha = 0,02 empeora de forma significativa en nguyen5 (ΔAPG+ATG +0,83,
     IC [+0,50, +1,11], 0/10 victorias). La candidata queda en el interior del grid de eta/alpha.
   - **Fase B (lambda × beta en eta = 0,08 / alpha = 0,01)**: interrumpida (≈30 de 330 ejecuciones nuevas por
     problema). Se continúa en servidor: ver `SERVER_RUNS.md` (fases B y C).
   - Código: `examples/analyze_grid_paired.py` (análisis emparejado reproducible sobre todas las configuraciones
     del directorio; JSON en `dumps/hyperparam_search/paired_analysis.json`). `run_hyperparameter_search.py`
     añade `_k<K>_tau<tau>` al nombre de configuración cuando K ≠ 3 o tau ≠ 1, para que la caché no confunda
     ejecuciones con distinto K/tau.

## Tests
`python3 -m unittest tests.test_stability tests.test_sa_psge` → 12 tests OK.

## Pendiente (no hecho)
- Paso 5 (congelación): decidir si se congela eta = 0,08 / alpha = 0,01 o se amplía (eta = 0,16; lambda, beta, K, tau
  sin explorar). Decisión del investigador, pendiente.
- ~~`parameters/standard.yml` usa MIN_TREE_DEPTH = MAX_TREE_DEPTH = 10~~. **Resuelto (2026-11-04): 12/12.** MAX_TREE_DEPTH = 12 sigue el plan.
  MIN_TREE_DEPTH es, en el código, la profundidad máxima de inicialización (`grammar.max_init_depth`): a partir de
  ella solo se usan reglas no recursivas. 12/12 mantiene la convención original de PSGE (10/10). Todas las
  ejecuciones anteriores (portátil, 10/10) quedan fuera de las comparaciones; el servidor parte de cero.
