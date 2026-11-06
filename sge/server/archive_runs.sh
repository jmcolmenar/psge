#!/usr/bin/env bash
# Archiva de forma permanente los datos pesados de una campaña ($RUNS_ROOT/<campaña>, p. ej. en /tmp).
#
# Por defecto (modo esencial) guarda lo que pide el plan: gramática de cada generación
# (grammar_probabilities.json), progress_report.csv y parameters.json de cada brazo. Omite iteration_*.json
# (volcado de la población inicial: ~98 % del tamaño, ~2,2 MB por brazo y regenerable con la semilla).
#   Confirmación: ~1,3 GB sin comprimir -> ~0,4 GB comprimido.
# Con --all guarda todo (~50 GB -> ~17 GB comprimido).
#
# NUNCA borra el origen. Verifica el archivo (lo lee entero y cuenta los ficheros) y deja un manifiesto con
# sha256 junto al archivo y una copia del manifiesto en $RESULTS_ROOT/meta/archive_<fecha>/.
#
# Uso: server/archive_runs.sh [--all] DESTINO [CAMPAÑA]     (CAMPAÑA por defecto: confirmation)
#   server/bg.sh archive_runs.sh ~/psge-archive               # en segundo plano
source "$(dirname "$0")/common.sh"

MODE=essential
if [ "${1:-}" = "--all" ]; then MODE=all; shift; fi
DEST="${1:?Uso: $0 [--all] DESTINO [CAMPAÑA]}"
CAMPAIGN="${2:-confirmation}"
SRC="$RUNS_ROOT/$CAMPAIGN"
[ -d "$SRC" ] || { echo "ERROR: no existe $SRC" >&2; exit 1; }
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"

case "$DEST/" in "$SRC/"*) echo "ERROR: el destino está dentro del origen" >&2; exit 1;; esac

if command -v zstd >/dev/null 2>&1; then COMP=(zstd -q -T0 -10); EXT=zst; DECOMP=(zstd -dc)
elif command -v pigz >/dev/null 2>&1; then COMP=(pigz -6); EXT=gz; DECOMP=(pigz -dc)
else COMP=(gzip -6); EXT=gz; DECOMP=(gzip -dc); fi

STAMP="$(date +%Y%m%d-%H%M%S)"
NAME="psge-runs_${CAMPAIGN}_${MODE}_${STAMP}"
ARCHIVE="$DEST/$NAME.tar.$EXT"
LIST="$(mktemp)"; trap 'rm -f "$LIST"' EXIT

echo "Origen: $SRC  ->  $ARCHIVE  (modo $MODE, ${COMP[0]})"
cd "$RUNS_ROOT"
if [ "$MODE" = essential ]; then
  find "$CAMPAIGN" -type f \( -name 'grammar_probabilities.json' -o -name 'progress_report.csv' \
       -o -name 'parameters.json' \) | sort > "$LIST"
else
  find "$CAMPAIGN" -type f | sort > "$LIST"
fi
NFILES=$(wc -l < "$LIST")
BYTES=$(tr '\n' '\0' < "$LIST" | xargs -0 stat -c %s | awk '{s+=$1} END {print s+0}')
NEED_GB=$(( BYTES / 2 / 1024 / 1024 / 1024 + 1 ))      # con compresión ~3x basta la mitad, con margen
require_free_gb "$DEST" "$NEED_GB"
echo "Ficheros: $NFILES  ($(( BYTES / 1024 / 1024 )) MB sin comprimir)"

T0=$SECONDS
tar -cf - -T "$LIST" | "${COMP[@]}" > "$ARCHIVE"
echo "Comprimido en $(( SECONDS - T0 )) s: $(du -h "$ARCHIVE" | cut -f1)"

echo "Verificando (lectura completa del archivo)..."
NCHECK=$("${DECOMP[@]}" "$ARCHIVE" | tar -tf - | grep -vc '/$')
if [ "$NCHECK" != "$NFILES" ]; then
  echo "ERROR: el archivo tiene $NCHECK ficheros y se esperaban $NFILES. No borres el origen." >&2; exit 1
fi
SHA=$(sha256sum "$ARCHIVE" | cut -d' ' -f1)
COMMIT="$(cat "$RESULTS_ROOT/$CAMPAIGN/.commit" 2>/dev/null || echo desconocido)"
MANIFEST="$DEST/$NAME.manifest.txt"
{
  echo "archivo:          $(basename "$ARCHIVE")"
  echo "sha256:           $SHA"
  echo "modo:             $MODE"
  echo "origen:           $SRC  (host $(hostname))"
  echo "ficheros:         $NFILES"
  echo "bytes_originales: $BYTES"
  echo "commit_campaña:   $COMMIT"
  echo "creado:           $(date -Iseconds)"
  echo "restaurar:        ${DECOMP[*]} $(basename "$ARCHIVE") | tar -xf -"
} > "$MANIFEST"
mkdir -p "$RESULTS_ROOT/meta/archive_$STAMP"; cp "$MANIFEST" "$RESULTS_ROOT/meta/archive_$STAMP/"
cat "$MANIFEST"
echo
echo "Archivo verificado. El origen NO se ha borrado. Cuando hayas copiado el archivo a su sitio definitivo:"
echo "  rm -rf \"$SRC\""
