# Flujo de trabajo portátil ↔ servidor

- **Portátil:** se edita el código, se ejecutan los tests y se hace commit + push.
- **Servidor:** `git pull` y se lanzan los scripts de `server/` a mano. Los resultados se escriben en
  `RESULTS_ROOT` (por defecto `~/psge-results`), fuera del repositorio.
- **Portátil:** `server/fetch_results.sh` trae los resultados a `~/psge-results`, fuera de OneDrive.

```
portátil: editar → tests → commit → push
servidor: git pull → server/run_*.sh → ~/psge-results/
portátil: server/fetch_results.sh → ~/psge-results/ (fuera de OneDrive)
```

## Configuración (una vez por máquina)

```bash
cp server/local.env.example server/local.env   # no se versiona
# Servidor: RESULTS_ROOT, CONDA_ENV, N_JOBS.  Portátil: SERVER, SERVER_RESULTS, LOCAL_RESULTS.
```

## Lanzar en el servidor

Desde `sge/`, con `server/bg.sh` delante del script. El experimento queda desligado del terminal y
**sigue corriendo al cerrar ssh**:

```bash
git pull && python -m unittest tests.test_stability tests.test_sa_psge
server/bg.sh run_phaseB.sh 16    # 16 procesos por problema (nguyen5 y nguyen1 en paralelo)
# ya puedes cerrar el terminal
server/status.sh                 # al volver: qué sigue en marcha y cuántas ejecuciones lleva cada config
tail -f ~/psge-results/bg/run_phaseB_*.out
server/stop.sh                   # parar el último lanzado (las ejecuciones hechas se conservan)
```

Si el script falla al arrancar (cambios sin commit, conda, argumentos), `bg.sh` lo dice en los primeros
segundos y muestra el error. Los scripts también se pueden lanzar sin `bg.sh`, en primer plano.

| Script | Qué hace |
|---|---|
| `run_phaseB.sh [J]` | λ ∈ {0, 0.25, 0.5, 1} × β ∈ {0.5, 1, 2} con η = 0.08, α = 0.01; 10 × 3 semillas; análisis emparejado al final |
| `run_phaseC.sh LAM BET [J]` | Sensibilidad a K ∈ {2, 5} y τ ∈ {5, 10} (un factor cada vez) con el λ/β elegido |
| `launch.sh NOMBRE cmd…` | Cualquier otro experimento con las mismas comprobaciones y metadatos |
| `bg.sh SCRIPT args…` | Lanza cualquiera de los anteriores en segundo plano, desligado del terminal |
| `stop.sh [pid]` | Para el último lanzado con `bg.sh` (o el indicado), con todos sus procesos |
| `status.sh [n]` | Procesos vivos, ejecuciones completadas por configuración (sobre n esperadas), errores |
| `fetch_results.sh [--full]` | **En el portátil.** rsync del servidor a `LOCAL_RESULTS` |

Todos los lanzadores:
- **se niegan a ejecutar con cambios sin commit** en ficheros versionados (`ALLOW_DIRTY=1` para forzar);
- **fijan el commit de cada directorio de resultados** (`.commit`). La caché reutiliza un JSON si ya existe
  sin mirar el código, así que relanzar en el mismo directorio con otro commit mezclaría resultados.
  Si el cambio no afecta a los resultados: `ALLOW_COMMIT_CHANGE=1`; si los afecta, usa otro `RESULTS_ROOT`;
- guardan en `RESULTS_ROOT/meta/<campaña>_<fecha>/`: commit, tag, rama, `pip freeze`, CPU, memoria,
  `standard.yml`, tiempo de reloj, logs y `errors.txt`;
- fijan un hilo BLAS por proceso.

Relanzar una campaña interrumpida reutiliza las ejecuciones ya hechas: basta con lanzar el mismo script otra vez.

## Estructura de resultados

```
~/psge-results/
  hyperparam_search/            fases B y C (mismo directorio: comparten el baseline PSGE)
    .commit
    nguyen5/<config>/<config>_d<datos>_e<evo>.json    resumen de cada ejecución emparejada
    nguyen5/<config>/<config>_d<datos>_e<evo>/        gramática por generación, progress_report
    paired_analysis.json
  meta/phaseB_20261104-101500/  run_info.txt, analysis.txt, logs, pip_freeze.txt, …
```

`fetch_results.sh` sin `--full` trae solo los JSON de resumen, los análisis y `meta/` (pocos MB). Los
directorios por ejecución (gramática por generación) se quedan en el servidor salvo que se pidan con `--full`.

## Antes de la fase B

- La profundidad (`MIN/MAX_TREE_DEPTH = 10` en `standard.yml`, el plan dice máx. 12) sigue abierta. Decídela
  **antes** de lanzar: cambiarla después invalida toda la caché.
- Se recomienda **no copiar al servidor las ejecuciones del portátil**: con ~35 s por ejecución emparejada,
  rehacerlo todo cuesta poco y garantiza que todos los resultados salen de la misma máquina y del mismo commit.
- La caché tampoco distingue `SA_ELITE_RATIO`, `SA_P_MIN`, población/generaciones ni `standard.yml`. Si cambias
  alguno, usa otro `RESULTS_ROOT`.
