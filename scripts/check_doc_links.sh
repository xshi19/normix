#!/usr/bin/env bash
# Enforce cross-link discipline between published docs and dev-notes/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0

# Paths under docs/ that are not published (Phase 2 moves these to dev-notes/).
INTERNAL_DOC_PREFIXES=(
  "docs/plans/"
  "docs/tech_notes/"
  "docs/investigations/"
  "docs/reviews/"
  "docs/archive/"
  "docs/references/"
  "docs/design/"
)

is_internal_doc() {
  local file="$1"
  local prefix
  for prefix in "${INTERNAL_DOC_PREFIXES[@]}"; do
    if [[ "$file" == "$prefix"* ]]; then
      return 0
    fi
  done
  return 1
}

check_absent_in_published_docs() {
  local pattern="$1"
  local message="$2"
  local hits=""
  local file

  if [[ ! -d docs ]]; then
    return 0
  fi

  while IFS= read -r file; do
    if is_internal_doc "$file"; then
      continue
    fi
    if rg -n "$pattern" "$file" >/dev/null 2>&1; then
      hits+=$(rg -n "$pattern" "$file" || true)
      hits+=$'\n'
    fi
  done < <(find docs -type f \( -name '*.md' -o -name '*.rst' \))

  if [[ -n "$hits" ]]; then
    echo "ERROR: $message"
    printf '%s\n' "$hits"
    fail=1
  fi
}

check_absent() {
  local pattern="$1"
  local path="$2"
  local message="$3"
  if [[ -e "$path" ]] && rg -n "$pattern" "$path" >/dev/null 2>&1; then
    echo "ERROR: $message"
    rg -n "$pattern" "$path" || true
    fail=1
  fi
}

check_absent_in_published_docs 'dev-notes/' \
  'published docs/ must not reference dev-notes/ (excluding internal-only folders until Phase 2)'

check_absent 'dev-notes/' normix/ 'normix/ docstrings must not reference dev-notes/'
check_absent 'dev-notes/' README.md 'README.md must not reference dev-notes/'

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "Cross-link checks: OK"
