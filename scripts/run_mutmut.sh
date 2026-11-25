#!/usr/bin/env bash
# Run the GIG/Bessel mutmut baseline configured in [tool.mutmut] (pyproject.toml).
#
# Usage: scripts/run_mutmut.sh [OUT_DIR] [MUTANT_PATTERN ...]
#   OUT_DIR defaults to mutmut-artifacts/<UTC timestamp>.
#   MUTANT_PATTERN values are passed to `mutmut run` (fnmatch globs) for pilots.
# Env: MUTMUT_MAX_CHILDREN (default 2), PYTHON (default .venv/bin/python).
# Requires mutmut 3.8.1 in the project venv:
#   uv sync --frozen && uv pip install --python .venv/bin/python 'mutmut==3.8.1'
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="${1:-mutmut-artifacts/$(date -u +%Y%m%dT%H%M%SZ)}"
shift || true
PY="${PYTHON:-.venv/bin/python}"
CHILDREN="${MUTMUT_MAX_CHILDREN:-2}"
mkdir -p "$OUT_DIR"

# CPU-only JAX with float64; one thread per worker so workers do not oversubscribe.
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/normix-mutmut-jax-cache}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1

{
  echo "git_sha=$(git rev-parse HEAD)"
  echo "mutmut=$("$PY" -m mutmut --version)"
  "$PY" -c 'import sys,jax;print("python="+sys.version.split()[0]);print("jax="+jax.__version__)'
  echo "nproc=$(OMP_NUM_THREADS= nproc --all) max_children=$CHILDREN patterns=$*"
  echo "start=$(date -u +%FT%TZ)"
} | tee "$OUT_DIR/run-info.txt"

start=$(date +%s)
set +e
"$PY" -m mutmut run --max-children "$CHILDREN" "$@" 2>&1 | tr '\r' '\n' > "$OUT_DIR/mutmut-run.log"
rc=${PIPESTATUS[0]}
set -e
end=$(date +%s)
{ echo "end=$(date -u +%FT%TZ)"; echo "exit_code=$rc"; echo "wall_seconds=$((end - start))"; } \
  | tee -a "$OUT_DIR/run-info.txt"

"$PY" -m mutmut results --all true > "$OUT_DIR/mutmut-results.txt" || true
"$PY" -m mutmut export-cicd-stats >/dev/null 2>&1 && cp mutants/mutmut-cicd-stats.json "$OUT_DIR/" || true
# Per-mutant status list as "name<TAB>status", sorted by name.
grep -E '^ +[^ ]+: ' "$OUT_DIR/mutmut-results.txt" | sed -E 's/^ +//; s/: /\t/' | sort > "$OUT_DIR/status.tsv" || true
cut -f2 "$OUT_DIR/status.tsv" | sort | uniq -c | tee "$OUT_DIR/summary.txt"
exit "$rc"
