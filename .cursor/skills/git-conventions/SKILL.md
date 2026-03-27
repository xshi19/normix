---
name: git-conventions
description: >-
  Git commit message conventions and pre-commit checklist for normix.
  Use when committing changes, creating PRs, or when the user asks about
  commit format, branch naming, or git workflow.
---

# Git Conventions

> High-level principles: `docs/design/agent_instructions_design.md`

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
| `design` | docs/design/ updates |
| `tech` | docs/tech_notes/ updates |
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
- [ ] If changes advance a migration phase → update `docs/plans/migration_plan.md`
- [ ] If new modules were added → update `docs/ARCHITECTURE.md`
- [ ] If new distribution was added → update `.cursor/rules/project-overview.mdc`
- [ ] If design decisions were made → update `docs/design/design.md`

## Gotchas

- **Don't commit generated files** (`.pyc`, `__pycache__/`, `.ipynb_checkpoints/`).
- **Don't commit secrets** (`.env`, credentials, API keys).
- **One logical change per commit.** Don't mix a bug fix with a refactor.
- **Docs changes get their own commits** with `docs(scope):` prefix.
