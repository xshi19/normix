#!/usr/bin/env bash
# Collate committed asv_bench/results/ JSON into the Sphinx HTML tree at
# docs/_build/html/benchmarks/. Run after sphinx-build (Makefile html /
# docs CI). Does not re-time the suite.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/docs/_build/html/benchmarks}"
if [[ "$DEST" != /* ]]; then
  DEST="$ROOT/$DEST"
fi

case "$DEST" in
  */benchmarks) ;;
  */benchmarks/) DEST="${DEST%/}" ;;
  *)
    echo "ERROR: destination must be a 'benchmarks' directory, not '$DEST'." >&2
    echo "asv publish rmtree's html_dir; refusing to point it at the Sphinx root." >&2
    exit 1
    ;;
esac

cd "$ROOT"

jsons=(asv_bench/results/wukong/*.json)
if [[ ! -e "${jsons[0]}" ]]; then
  echo "ERROR: asv_bench/results/wukong/ has no JSON; cannot publish the dashboard." >&2
  exit 1
fi

cd "$ROOT/asv_bench"
uv run asv publish --no-pull

rm -rf "$DEST"
mkdir -p "$DEST"
cp -a "$ROOT/asv_bench/.asv/html/." "$DEST/"
echo "ASV dashboard → $DEST"
