#!/usr/bin/env bash
# Funciones comunes de los lanzadores del servidor. Se carga con `source`; no se ejecuta.
set -euo pipefail

SGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SGE_DIR"

# Configuración local de cada máquina (no versionada). Ver server/local.env.example.
# Las variables puestas al lanzar (RESULTS_ROOT=... server/x.sh) tienen prioridad sobre local.env.
_env_RESULTS_ROOT="${RESULTS_ROOT:-}"; _env_CONDA_ENV="${CONDA_ENV:-}"; _env_N_JOBS="${N_JOBS:-}"; _env_PYTHON="${PYTHON:-}"
if [ -f "$SGE_DIR/server/local.env" ]; then source "$SGE_DIR/server/local.env"; fi
[ -n "$_env_RESULTS_ROOT" ] && RESULTS_ROOT="$_env_RESULTS_ROOT"
[ -n "$_env_CONDA_ENV" ] && CONDA_ENV="$_env_CONDA_ENV"
[ -n "$_env_N_JOBS" ] && N_JOBS="$_env_N_JOBS"
[ -n "$_env_PYTHON" ] && PYTHON="$_env_PYTHON"
: "${RESULTS_ROOT:=$HOME/psge-results}"   # fuera del repositorio y fuera de OneDrive
: "${CONDA_ENV:=psge-original}"
NCPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
: "${N_JOBS:=$(( NCPU / 2 > 0 ? NCPU / 2 : 1 ))}"   # procesos por problema (dos problemas en paralelo)

# Un hilo BLAS por proceso; si no, los workers compiten por los núcleos.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

activate_env() {
  if [ -n "${PYTHON:-}" ]; then return; fi
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    PYTHON=python
  else
    echo "ERROR: conda no está en el PATH. Define PYTHON=/ruta/a/python en server/local.env." >&2
    exit 1
  fi
}

# Solo se lanza código que está en un commit: si no, los resultados no son reproducibles.
require_clean_tree() {
  if [ -n "$(git status --porcelain --untracked-files=no)" ] && [ "${ALLOW_DIRTY:-0}" != "1" ]; then
    echo "ERROR: hay cambios sin commit en ficheros versionados:" >&2
    git status --short --untracked-files=no >&2
    echo "Haz commit (o git pull) antes de lanzar. Para forzar: ALLOW_DIRTY=1." >&2
    exit 1
  fi
}

# La caché de run_hyperparameter_search reutiliza un JSON si ya existe, sin mirar el código.
# Para no mezclar resultados de commits distintos en el mismo directorio, se fija el commit
# la primera vez y después se exige el mismo.
guard_commit() {
  local dir="$1" head; head="$(git rev-parse HEAD)"
  mkdir -p "$dir"
  if [ ! -f "$dir/.commit" ]; then
    echo "$head" > "$dir/.commit"
  elif [ "$(cat "$dir/.commit")" != "$head" ] && [ "${ALLOW_COMMIT_CHANGE:-0}" != "1" ]; then
    echo "ERROR: $dir contiene resultados del commit $(cat "$dir/.commit")" >&2
    echo "       y el código actual es $head." >&2
    echo "Usa otro RESULTS_ROOT/directorio, o ALLOW_COMMIT_CHANGE=1 si el cambio no afecta a los resultados." >&2
    exit 1
  fi
}

# Crea $META_DIR con todo lo necesario para reproducir la ejecución.
start_campaign() {
  local name="$1"; shift
  META_DIR="$RESULTS_ROOT/meta/${name}_$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$META_DIR"
  {
    echo "campaign:   $name"
    echo "command:    $0 $*"
    echo "start:      $(date -Iseconds)"
    echo "host:       $(hostname)"
    echo "commit:     $(git rev-parse HEAD)"
    echo "describe:   $(git describe --tags --always --dirty)"
    echo "branch:     $(git rev-parse --abbrev-ref HEAD)"
    echo "n_jobs:     $N_JOBS (por problema), cpus: $NCPU"
    echo "python:     $($PYTHON -V 2>&1)"
  } > "$META_DIR/run_info.txt"
  git diff HEAD > "$META_DIR/git_diff.patch" || true
  $PYTHON -m pip freeze > "$META_DIR/pip_freeze.txt" 2>/dev/null || true
  (lscpu 2>/dev/null || sysctl -a 2>/dev/null | grep -i cpu) > "$META_DIR/cpu.txt" || true
  (free -h 2>/dev/null || true) > "$META_DIR/memory.txt"
  cp parameters/standard.yml "$META_DIR/standard.yml"
  START_SECONDS=$SECONDS
  echo "Campaña $name -> logs y metadatos en $META_DIR"
}

finish_campaign() {
  local status="${1:-0}"
  {
    echo "end:        $(date -Iseconds)"
    echo "wall_sec:   $(( SECONDS - START_SECONDS ))"
    echo "exit:       $status"
  } >> "$META_DIR/run_info.txt"
  grep -Hn "ERROR\|Traceback" "$META_DIR"/*.log > "$META_DIR/errors.txt" 2>/dev/null || true
  if [ -s "$META_DIR/errors.txt" ]; then
    echo "ATENCIÓN: hay errores en los logs, ver $META_DIR/errors.txt"
  fi
  echo "Terminado (exit $status). Tiempo: $(( SECONDS - START_SECONDS )) s."
}

# Lanza un proceso por problema en paralelo y espera a todos; devuelve != 0 si alguno falla.
wait_all() {
  local rc=0 pid
  for pid in "$@"; do wait "$pid" || rc=1; done
  return $rc
}
