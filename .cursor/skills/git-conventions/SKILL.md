---
name: git-conventions
description: >-
  Git commit conventions, the pre-commit checklist, and release-please
  for normix. Use when committing, opening a PR, preparing a release,
  or when the user asks about commit format or git workflow.
---

# Git Conventions

> High-level principles: `dev-notes/design/agent_instructions_design.md`

## Commit Message Format

Conventional commits: `type(scope): description`

### Types

| Type | When |
|---|---|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `chore` | Build, CI, tooling changes |

### Scopes

Use the most specific scope that applies:

| Scope | When |
|---|---|
| `gamma`, `gig`, `nig`, ... | Changes to a specific distribution |
| `bessel` | Bessel function changes |
| `em` | EM algorithm / fitting |
| `solver` | η→θ solvers |
| `arch` | ARCHITECTURE.md updates |
| `rules` | .cursor/rules/ updates |
| `design` | `docs/design/` or `dev-notes/design/` updates |
| `tech` | `dev-notes/tech_notes/` updates |
| `theory` | docs/theory/ updates |
| `skills` | `.cursor/skills/` updates |
| `notebook` | Notebook changes |

### Examples

```
feat(gig): add analytical Hessian for GIG log-partition
fix(bessel): handle overflow for large z in Hankel regime
docs(arch): add new bessel regime to ARCHITECTURE.md
docs(rules): add gotcha about jnp.where vs lax.cond
docs(design): record decision on Cholesky naming convention
docs(tech): add tech note on EM convergence profiling
docs(theory): add NIG distribution derivation
docs(skills): add show-me visual-explanation skill
test(gamma): add edge case tests for alpha near zero
refactor(solver): extract eta-rescaling into shared helper
perf(em): switch E-step to CPU backend for large N
```

## Pre-Commit Checklist

Before committing, verify:

- [ ] `uv run pytest tests/` passes
- [ ] No debug `print` statements left in code
- [ ] Type hints on all public methods
- [ ] If changes advance a plan phase → update the relevant plan in `dev-notes/plans/`
- [ ] If new modules were added → update `dev-notes/ARCHITECTURE.md`
- [ ] If new distribution was added → update `.cursor/rules/project-overview.mdc`
- [ ] If design decisions were made → update `dev-notes/design/design.md`

## Release workflow (release-please)

Versioning and PyPI releases are automated via `googleapis/release-please-action`.
**Never edit `version` in `pyproject.toml` or `__version__` in `normix/__init__.py` by hand.**

Config and workflows:

- `release-please-config.json`
- `.release-please-manifest.json`
- `.github/workflows/release-please.yml`
- `.github/workflows/publish.yml`
- `.github/workflows/docs-full.yml`

How it works:

1. A commit on `master` whose type is a visible changelog section opens or updates a Release PR.
   The title pattern is `chore: release X.Y.Z`.
2. Merging the Release PR bumps `pyproject.toml` and `normix/__init__.py`, updates
   `CHANGELOG.md`, and pushes a `vX.Y.Z` tag.
3. `release-please.yml` dispatches `publish.yml` and `docs-full.yml` on that tag.
   `publish.yml` builds the package and publishes it to PyPI with Trusted Publishing.
   `docs-full.yml` rebuilds the docs site from the tag.
4. Manual recovery: `gh workflow run publish.yml --ref vX.Y.Z`.
   Use `docs-full.yml` in that command to rebuild the release docs.

`publish.yml` and `docs-full.yml` still trigger on a `v*` tag push.
release-please creates the tag with `GITHUB_TOKEN`.
GitHub skips `on: push` for that tag, so the dispatch is the only run.
A tag pushed with a PAT starts those workflows from the tag trigger alone.
If the release-please step uses a PAT, delete the dispatch step first.

Versions 0.2.1 through 0.3.0 are absent from PyPI.
Backfill of those versions is a manual owner decision.

**Version bump rules (pre-1.0).** Both `bump-minor-pre-major` and `bump-patch-for-minor-pre-major` are true.
These bumps apply while the version is below 1.0.0.
At 1.0.0 and above, a breaking change bumps the major, and `feat` bumps the minor.

A hidden changelog type opens no Release PR, unless that commit carries a breaking-change marker.

| Commit type | Release | Bump |
|---|---|---|
| `feat:` | yes | patch (`0.2.0 → 0.2.1`) |
| `fix:` | yes | patch (`0.2.0 → 0.2.1`) |
| `perf:`, `refactor:`, `docs:` | yes | patch (`0.2.0 → 0.2.1`) |
| `type!:` or a `BREAKING CHANGE:` footer | yes | minor (`0.2.0 → 0.3.0`) |
| `test:`, `chore:`, `ci:`, `build:`, `style:` | no | — |

Observed on this repo: v0.2.2 was a `feat` patch, v0.2.4 was a `docs` patch, and v0.3.0 was a breaking minor.

**Do not** push `vX.Y.Z` tags manually — release-please manages them.

## Gotchas

- **`GITHUB_TOKEN` tags skip `on: push`.** `release-please.yml` dispatches `publish.yml` and `docs-full.yml`. Recover a release with `gh workflow run publish.yml --ref vX.Y.Z`.
- **Don't commit generated files** (`.pyc`, `__pycache__/`, `.ipynb_checkpoints/`).
- **Don't commit secrets** (`.env`, credentials, API keys).
- **One logical change per commit.** Don't mix a bug fix with a refactor.
- **Docs changes get their own commits** with `docs(scope):` prefix.
