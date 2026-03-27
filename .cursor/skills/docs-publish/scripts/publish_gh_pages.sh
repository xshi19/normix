#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PUBLISH_DIR="${TMPDIR:-/tmp}/normix-gh-pages"

cd "$REPO_ROOT"

if [[ ! -d docs/_build/html ]]; then
  echo "docs/_build/html not found. Build docs first:" >&2
  echo "  . .venv/bin/activate && cd docs && make clean && make html" >&2
  exit 1
fi

rm -rf "$PUBLISH_DIR"
git worktree prune || true

if git show-ref --verify --quiet refs/heads/gh-pages; then
  git worktree remove "$PUBLISH_DIR" --force 2>/dev/null || true
  git branch -D gh-pages 2>/dev/null || true
fi

git worktree add -B gh-pages "$PUBLISH_DIR" origin/gh-pages

cleanup() {
  cd "$REPO_ROOT" || true
  git worktree remove "$PUBLISH_DIR" --force 2>/dev/null || true
}
trap cleanup EXIT

cd "$PUBLISH_DIR"
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -a "$REPO_ROOT/docs/_build/html/." "$PUBLISH_DIR/"
: > .nojekyll

git add -A
if git diff --cached --quiet; then
  echo "No gh-pages changes to publish."
  exit 0
fi

git commit -m "Deploy updated Sphinx docs"
git push origin gh-pages

echo "Published docs/_build/html to gh-pages."
