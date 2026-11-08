# Benchmarks del estudio de adaptación y re-mapeo (fase 3 del plan)

Regla fijada el 2026-11-08, **antes de ejecutar nada sobre estos problemas**. Ningún resultado de estos
benchmarks se ha visto al elegirlos; la elección solo usa las especificaciones publicadas.

## Regla de selección

**Conjunto de partida:** los problemas de regresión de Nguyen (Nguyen-1 a -12; Uy et al., 2011) y de
Keijzer (Keijzer-1 a -15; Keijzer, 2003), con la numeración de McDermott et al. (GECCO 2012). Son los
benchmarks clásicos de GP más usados en la literatura de GE y PSGE.

Se excluyen, en este orden:

| Criterio | Motivo | Excluidos |
| --- | --- | --- |
| E1. Usado ya en este proyecto | Pilotos o confirmación del estudio anterior: no pueden confirmar | Nguyen-1, -5, -7, -9; Keijzer-6 |
| E2. Necesita constantes que la gramática no puede construir con 1,0 | π en sin(2πx) | Keijzer-1, -2, -3 |
| E3. Polinomio de una variable | Misma familia que Nguyen-1, el problema piloto | Nguyen-2, -3, -4 |
| E4. Misma forma funcional que otro problema del conjunto (salvo constante o dominio) | Duplicaría un bloque de inferencia; se conserva la versión de Nguyen, la de los pilotos | Keijzer-8 (= Nguyen-8), Keijzer-10 (= Nguyen-11), Keijzer-12 (= Nguyen-12), Keijzer-13 (∝ Nguyen-10) |
| E5. El objetivo es una primitiva de la gramática | ln x es `_log_(x[0])`: trivial | Keijzer-7 |
| E6. Más de 2 variables y 1000 puntos de entrenamiento | Necesitaría una gramática de 3 variables que no se usa en ningún otro problema | Keijzer-5 |

**Resultado: 10 benchmarks** (el plan preveía unos 8; la regla da 10 y se mantienen todos: más bloques
dan más potencia a los contrastes por benchmark, a cambio de ~25 % más de cómputo).

## Especificaciones

| Benchmark | Función | Variables | Entrenamiento (S y z) | Test | Fuente de la especificación |
| --- | --- | --- | --- | --- | --- |
| Nguyen-6 | sin(x) + sin(x + x²) | 1 | U[−1, 1] | U[−1, 1] | Uy et al. 2011, vía McDermott et al. 2012 |
| Nguyen-8 | √x | 1 | U[0, 4] | U[0, 4] | ídem |
| Nguyen-10 | 2 sin(x) cos(y) | 2 | U[0, 1]² | U[0, 1]² | Uy et al. 2011, vía HeuristicLab |
| Nguyen-11 | x^y | 2 | U[0, 1]² | U[0, 1]² | ídem |
| Nguyen-12 | x⁴ − x³ + y²/2 − y | 2 | U[0, 1]² | U[0, 1]² | ídem |
| Keijzer-4 | x³ e^−x cos x sin x (sin² x cos x − 1) | 1 | U[0, 10] | U[0,05, 10,05] | Keijzer 2003, vía McDermott et al. 2012 |
| Keijzer-9 | arcsinh x = ln(x + √(x² + 1)) | 1 | U[0, 100] | U[0, 100] | ídem |
| Keijzer-11 | xy + sin((x − 1)(y − 1)) | 2 | U[−3, 3]² | U[−3, 3]² | ídem |
| Keijzer-14 | 8 / (2 + x² + y²) | 2 | U[−3, 3]² | U[−3, 3]² | ídem |
| Keijzer-15 | x³/5 + y³/2 − y − x | 2 | U[−3, 3]² | U[−3, 3]² | ídem |

Comprobación de fuentes:
- Tabla de McDermott et al. (2012) reproducida en «Analysing Symbolic Regression Benchmarks
  under a Meta-Learning Approach» (arXiv:1805.10365, tabla 3).
- Descripciones de HeuristicLab, que citan el artículo original de Uy et al. (2011): Nguyen-10 y -12 con
  «20 random points in [0, 1]x[0, 1]».
- El artículo de McDermott et al. no está en acceso abierto; antes de redactar el artículo conviene
  contrastar la tabla con el original de la ACM.

## Desviaciones respecto a las especificaciones originales (las mismas que en el estudio anterior)

- **Muestreo uniforme en lugar de rejillas.** Keijzer usa rejillas E[a, b, paso] y Nguyen 20 puntos; el
  protocolo add-one necesita S de m ∈ {20, 100} puntos y un punto z de la misma distribución, así que S y
  z se muestrean uniformes en el dominio de entrenamiento y el test (1000 puntos), uniforme en el de test.
- **Ruido** en y del 0 % o el 5 % de σ\_y, como en el estudio anterior.
- **Gramática:** la común del proyecto (`regression_1v.pybnf` o `regression.pybnf`), con la constante 1,0
  y sin raíz cuadrada ni potencia: √x y x^y solo se pueden aproximar o construir con exp y log. Es la
  misma para las 11 condiciones, así que no sesga la comparación.

## Código

`sge/stability.py`: funciones `nguyen6` … `keijzer15`, entradas en `BENCHMARK_SPECS` y las listas
`REMAP_PILOT_BENCHMARKS` (nguyen1, nguyen5) y `REMAP_CONFIRM_BENCHMARKS`. Tests en
`tests/test_remap.py` (valores, dominios, que ninguno se usó antes) y `tests/test_benchmarks.py`
(cada gramática nombra todas las variables y comparte las reglas).
