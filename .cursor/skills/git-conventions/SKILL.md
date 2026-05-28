---
name: git-conventions
description: >-
  Git commit message conventions and pre-commit checklist for normix.
  Use when committing changes, creating PRs, or when the user asks about
  commit format, branch naming, or git workflow.
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
test(gamma): add edge case tests for alpha near zero
refactor(solver): extract eta-rescaling into shared helper
perf(em): switch E-step to CPU backend for large N
```

## Pre-Commit Checklist

Before committing, verify:

- [ ] `uv run pytest tests/` passes
- [ ] No debug `print` statements left in code
- [ ] Type hints on all public methods
- [ ] If changes advance a migration phase → update `dev-notes/plans/migration_plan.md`
- [ ] If new modules were added → update `dev-notes/ARCHITECTURE.md`
- [ ] If new distribution was added → update `.cursor/rules/project-overview.mdc`
- [ ] If design decisions were made → update `dev-notes/design/design.md`

## Release workflow (release-please)

Versioning and PyPI releases are automated via `googleapis/release-please-action`.
**Never edit `version` in `pyproject.toml` or `__version__` in `normix/__init__.py` by hand.**

How it works:

1. Commits merged to `master` with `feat:` or `fix:` types cause release-please to open
   (or update) a Release PR titled `chore: release X.Y.Z`.
2. Merging the Release PR bumps `pyproject.toml` and `normix/__init__.py`, updates
   `CHANGELOG.md`, and pushes a `vX.Y.Z` tag.
3. The `vX.Y.Z` tag triggers `.github/workflows/publish.yml`, which builds and
   publishes the package to PyPI via Trusted Publishing.

**Version bump rules (pre-1.0):**

| Commit type | Bump |
|---|---|
| `fix:` | patch (`0.2.0 → 0.2.1`) |
| `feat:` | minor (`0.2.0 → 0.3.0`) |
| `feat!:` / `BREAKING CHANGE:` footer | minor (not major until 1.0) |
| `docs:`, `test:`, `refactor:`, `chore:`, `perf:` | no release |

**Do not** push `vX.Y.Z` tags manually — release-please manages them.

## Gotchas

- **Don't commit generated files** (`.pyc`, `__pycache__/`, `.ipynb_checkpoints/`).
- **Don't commit secrets** (`.env`, credentials, API keys).
- **One logical change per commit.** Don't mix a bug fix with a refactor.
- **Docs changes get their own commits** with `docs(scope):` prefix.
