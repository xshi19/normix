#!/usr/bin/env bash
# Enforce cross-link discipline between published docs and dev-notes/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0

check_absent_in_published_docs() {
  local pattern="$1"
  local message="$2"
  local hits=""

  if [[ ! -d docs ]]; then
    return 0
  fi

  if hits=$(rg -n "$pattern" docs/ --glob '*.md' --glob '*.rst' 2>/dev/null || true); then
    if [[ -n "$hits" ]]; then
      echo "ERROR: $message"
      printf '%s\n' "$hits"
      fail=1
    fi
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

# Structural invariant: internal folder names must not exist under docs/.
if forbidden=$(find docs -type d \( \
  -name plans -o -name investigations -o -name reviews \
  -o -name tech_notes -o -name archive -o -name references \
\) 2>/dev/null); then
  if [[ -n "$forbidden" ]]; then
    echo "ERROR: docs/ must not contain internal-only directories:"
    printf '%s\n' "$forbidden"
    fail=1
  fi
fi

check_absent_in_published_docs 'dev-notes/' \
  'published docs/ must not reference dev-notes/'

check_absent 'dev-notes/' normix/ 'normix/ docstrings must not reference dev-notes/'
check_absent 'dev-notes/' README.md 'README.md must not reference dev-notes/'

# Legacy internal paths must not appear in tracked agent-facing files.
LEGACY_PATTERN='docs/(plans|tech_notes|investigations|reviews|archive|references)/'
for path in AGENTS.md .cursor/rules .cursor/skills; do
  if [[ -e "$path" ]] && rg -n "$LEGACY_PATTERN" "$path" >/dev/null 2>&1; then
    echo "ERROR: $path still references legacy docs/ internal paths"
    rg -n "$LEGACY_PATTERN" "$path" || true
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "Cross-link checks: OK"
