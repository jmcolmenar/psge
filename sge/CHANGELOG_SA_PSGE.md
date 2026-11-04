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

## 2026-11-04: fase B en el servidor y regla de ejecuciones inválidas
- Fase B completa en el servidor (commit `0268719`, profundidad 12/12, 13 configs × 30 ejecuciones × 2 problemas,
  535 s, sin errores). Resultados en `~/psge-results/hyperparam_search` (fuera del repositorio).
- Divergencias investigadas en Nguyen-1: (a) una ejecución (lam1_bet2, d402 e4002) con un modelo no evaluable en un
  punto de validación (1/x dentro de exp, x = 0,00092): `calculate_dataset_nrmse` devuelve el centinela 9 999 999,
  que el análisis promediaba como un error; (b) la semilla de datos 403 tiene el entrenamiento en [-0,90; 0,55] y
  99/500 puntos de validación fuera: modelos con polo en x = 1 (`x/(1-x)`) dan NRMSE-val 23,8. No es específico de
  SA-PSGE: la división y `_inv_` solo se protegen en 0 exacto. Operadores sin polos: no se cambian (decisión
  pendiente, cambiaría la representación de todos los algoritmos).
- **Regla de inválidos** (`examples/invalid_rule.py`, tests en `tests/test_invalid_rule.py`): valor no finito o
  >= 1e5 = inválido; se cuenta como +inf, nunca se descarta; semillas evolutivas agregadas con la **mediana**;
  tasa de inválidos por configuración en la salida. Aplicada en `analyze_grid_paired.py` y en la tabla de decisión
  de `run_hyperparameter_search.py` (solo análisis: las ejecuciones no cambian).
- Efecto en la fase B: con la media, la única configuración con ΔNRMSE-val <= +2 % en ambos problemas era
  λ = 1, β = 2; con la mediana es **λ = 0,25, β = 0,5**. En ambos casos ΔAPG+ATG ≈ 0 con IC que incluye 0.
  La selección depende de la agregación: con 10 semillas de datos la fase B no discrimina entre configuraciones.
  La mayoría de configuraciones empeora mucho la NRMSE-val de Nguyen-1 (sobre-regularización en el problema fácil).
- Lanzadores: `N_DATA_SEEDS` permite ampliar a 20 semillas de datos reutilizando las 10 ya hechas.

## 2026-11-04 (tarde): fase B con 20 semillas y piloto de RQ1
- Fase B ampliada a 20 semillas de datos (commit `6bf7313`): la regla elige **λ = 0,25, β = 2** (única con
  ΔNRMSE-val <= +2 % en ambos problemas). ΔAPG+ATG frente a PSGE: nguyen5 −0,068 [−0,19; +0,34], 12/20;
  nguyen1 +0,002 [−0,015; +0,010], 10/20. La elección cambió tres veces (10 semillas/media, 10/mediana,
  20/mediana): con este diseño la fase B no distingue configuraciones y SA-PSGE no mejora la estabilidad de PSGE.
  Fase C con λ = 0,25, β = 2 en marcha en el servidor.
- `examples/stats_utils.py` (sin scipy): IC BCa, Wilcoxon exacto con empates, rank-biserial, Spearman con p por
  permutación. 7 tests (Wilcoxon contrastado con enumeración completa).
- Piloto de RQ1: `examples/run_rq1_pilot.py` (SGE lf = 0 frente a PSGE lf = 0,01 con el mismo worker emparejado;
  semillas 500../5000.., 20 × 5, 200 × 50, solo validación), `examples/analyze_rq1.py` (estabilidad de datos
  APG+ATG, APG, ATG; estabilidad de búsqueda = dispersión de las predicciones de f_S entre semillas;
  NRMSE-val; Spearman JS vs APG+ATG en PSGE; entropía de la PCFG) y `server/run_rq1_pilot.sh`.
- El worker emparejado guarda además `apg_norm`, `atg_norm`, `nrmse_train_fS`, `entropy_S`, `entropy_Sz` y `m`
  (campos nuevos; los existentes no cambian).
- `run_pilot_campaign.py` queda obsoleto para RQ1.

## 2026-11-04 (noche): resultados de fase C y RQ1, preparación de la congelación
- Fase C (λ = 0,25, β = 2; 10 semillas de datos): ninguna variante de K/τ domina a K = 3, τ = 1. τ = 10 y K = 2
  mejoran la estabilidad en nguyen5 con IC que excluye 0, pero empeoran NRMSE-val (> +2 % en nguyen5, +75 % en
  nguyen1). Se mantiene K = 3, τ = 1.
- Piloto RQ1 (commit `77e1634`, 20 × 5, 200 × 50): nguyen5 ΔAPG+ATG (PSGE − SGE) = +0,166, IC BCa [+0,006; +0,29],
  Wilcoxon p = 0,083, r = +0,45, 14/20; impulsado por ATG (p = 0,015). nguyen1 sin efecto (−0,007; ATG en sentido
  contrario). Estabilidad de búsqueda y NRMSE-val equivalentes. Spearman JS vs APG+ATG ≈ +0,3, no significativo.
  Decisión: seguir el protocolo hasta la confirmación sin rediseñar SA-PSGE.
- `parameters/gecco_frozen.yml`: configuración congelada (500 × 100, profundidad 12, SA-PSGE η = 0,08, α = 0,01,
  λ = 0,25, β = 2, K = 3, τ = 1). KL_DELTA = PENDIENTE hasta la recalibración.
- `server/run_recalibration.sh`: KL de PSGE y de SA-PSGE a 500 × 100 en los pilotos. Reglas fijadas antes de ver
  resultados: KL_DELTA = media de los P25 (2 cifras); η = 0,08 se mantiene salvo que su KL difiera de la de PSGE
  en más de un factor 3. También mide tiempos para estimar el coste de la confirmación.
- Benchmarks confirmatorios contrastados con McDermott et al. (2012) y la implementación de HeuristicLab:
  - **Vladislavleva-4: la gramática `regression.pybnf` solo tenía x[0] y x[1]** (3 de 5 variables inaccesibles).
    Nueva `grammars/regression_5v.pybnf` (mismas reglas, 5 variables).
  - Vladislavleva-4 y -7: dominios corregidos a entrenamiento U[0,05; 6,05] y test/validación U[−0,25; 6,35]
    (antes [0; 6] en ambos).
  - Pagie-1: el original usa la rejilla E[−5; 5; 0,4] sin test; se mantiene U[−5; 5]² por el protocolo add-one
    (desviación documentada).
  - Nguyen-7, Nguyen-9, Nguyen-1, Nguyen-5: coinciden.
  - Keijzer-6 (decisión del investigador, opción B): extensión continua exacta H(x) = ψ(x+1) + γ, que coincide con
    la suma armónica en cada entero. Entrenamiento U[1; 50]; validación y test U[1; 120]. El original usa enteros
    (E[1; 50; 1] / E[1; 120; 1]), con los que m = 100 sería imposible; el código anterior truncaba x real (función
    escalonada) y validaba solo en [51; 120].
  - `tests/test_benchmarks.py` (7 tests): cada gramática cubre todas las variables de su problema y comparte el
    resto de reglas con las demás; dominios de Vladislavleva; valores de referencia.
- `server/common.sh`: las variables puestas al lanzar tienen prioridad sobre `local.env`.

- `examples/run_confirmation.py` + `server/run_confirmation.sh`: estudio confirmatorio (6 benchmarks × m {20, 100} ×
  ruido {0; 0,05} × 5 algoritmos [sge, psge, psge_kl, sa_psge, sa_psge_lam0] × 20 × 5 semillas = 12 000 ejecuciones
  emparejadas; semillas 1000../10000..; test de 1000 puntos). Se niega a arrancar con KL_DELTA = PENDIENTE.
  Guarda APG/ATG, NRMSE de entrenamiento y test, JS, entropía, KL por actualización y coste por brazo.
- `StabilityEvaluationFunction` cuenta evaluaciones y evaluaciones caso-individuo. En una prueba pequeña,
  SA-PSGE evalúa ~20 % más casos que el resto (submuestras de la élite): hay que informarlo en el artículo.
- `server/run_timing.sh`: coste real a 500 × 100 medido en nguyen5 (piloto), con estimación del estudio completo.

## Tests
`python3 -m unittest tests.test_stability tests.test_sa_psge tests.test_invalid_rule tests.test_stats_utils tests.test_benchmarks` → 32 tests OK.

## Pendiente (no hecho)
- Paso 5 (congelación): decidir si se congela eta = 0,08 / alpha = 0,01 o se amplía (eta = 0,16; lambda, beta, K, tau
  sin explorar). Decisión del investigador, pendiente.
- ~~`parameters/standard.yml` usa MIN_TREE_DEPTH = MAX_TREE_DEPTH = 10~~. **Resuelto (2026-11-04): 12/12.** MAX_TREE_DEPTH = 12 sigue el plan.
  MIN_TREE_DEPTH es, en el código, la profundidad máxima de inicialización (`grammar.max_init_depth`): a partir de
  ella solo se usan reglas no recursivas. 12/12 mantiene la convención original de PSGE (10/10). Todas las
  ejecuciones anteriores (portátil, 10/10) quedan fuera de las comparaciones; el servidor parte de cero.
