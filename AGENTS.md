# normix

JAX package for Generalized Hyperbolic distributions as exponential families.
Built on Equinox. Deps: `jax`, `equinox`, `jaxopt`, `tfp` (Bessel only), `optax` (optional).

## Commands

- Install: `uv sync`
- Tests: `uv run pytest tests/`
- Single test: `uv run pytest tests/test_gamma.py -v`
- Notebooks: `uv run jupyter lab`
- Add dependency: `uv add <package>` / `uv add --dev <package>`
- This project uses **uv**, NOT conda.

## Commits

Conventional commits: `feat|fix|docs|test|refactor(scope): description`

Before committing:
- `uv run pytest tests/` passes
- No debug print statements
- Type hints on all public methods
- Update `docs/plans/migration_plan.md` if changes advance a phase
- Update `docs/ARCHITECTURE.md` if new modules were added

## Core Design

- **Exponential family**: log base measure, sufficient statistics, log partition as single source of truth
- **Three parametrizations**: classical ↔ natural θ ↔ expectation η, all JIT-compatible
- **Autodiff-first**: derivatives via jax.grad/jax.hessian on log partition
- **EM algorithm**: E-step computes conditional expectations, M-step returns new immutable model
- **Immutable**: eqx.Module pytrees, no mutation — updates return new instances
- **Unbatched core**: single-observation methods, batch via jax.vmap

## Design Philosophy

- **Simplicity**: weigh complexity cost against improvement magnitude; simplification wins are great outcomes
- **Numerical efficiency**: leverage linear algebra and optimization knowledge without adding code complexity
- **Numerical robustness**: handle edge cases carefully; work in log-space to prevent overflow

## Context Map

Architecture and module hierarchy → `docs/ARCHITECTURE.md`

| Area | Rule / Doc |
|---|---|
| Distribution code (`normix/`) | `.cursor/rules/coding-conventions.mdc` |
| Tests (`tests/`) | `.cursor/rules/testing-guidelines.mdc` |
| Notebooks (`notebooks/`) | `.cursor/rules/notebook-guidelines.mdc` |
| Rule authoring | `.cursor/rules/rule-authoring.mdc` |
| Design decisions | `docs/design/jax_design.md` |
| Migration status | `docs/plans/migration_plan.md` |
| Bessel functions | `docs/ARCHITECTURE.md` § Bessel Functions |
| GIG η→θ optimization | `docs/ARCHITECTURE.md` § GIG η→θ |
| Package survey | `docs/references/distribution_packages.md` |
