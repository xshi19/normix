# normix

JAX package for Generalized Hyperbolic distributions and related distributions,
implemented as exponential families. Built on Equinox for immutable pytree modules.

Minimalist (like nanoGPT), elegant (inspired by FlowJAX), mathematically robust,
and numerically efficient (Cholesky, log-space, float64).

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

- **Exponential family structure**: log base measure $h(x)$, sufficient statistics $t(x)$, log partition $\psi(\theta)$
- **Three parametrizations**: classical $\leftrightarrow$ natural $\theta$ $\leftrightarrow$ expectation $\eta = \nabla\psi(\theta)$, all JIT-compatible
- **Autodiff-first**: `expectation_params` and `fisher_information` derived from `jax.grad`/`jax.hessian` on `_log_partition_from_theta`
- **EM algorithm**: E-step computes conditional expectations $E[t(Y)|X]$, M-step converts $\eta \to \theta$ via `from_expectation`
- **Immutable**: all distributions are `eqx.Module` pytrees; M-step returns a new model
- **Unbatched core**: `_log_prob` and `_sample` operate on single observations; batch via `jax.vmap`

### Dependencies

`jax`, `equinox`, `jaxopt`, `tensorflow_probability.substrates.jax` (Bessel only), `optax` (optional).

## Context Map

Architecture and module hierarchy → `docs/ARCHITECTURE.md`

When editing specific areas, read the relevant rule:

| Area | Rule / Doc |
|---|---|
| Distribution code (`normix/`) | `.cursor/rules/coding-conventions.mdc` |
| Tests (`tests/`) | `.cursor/rules/testing-guidelines.mdc` |
| Notebooks (`notebooks/`) | `.cursor/rules/notebook-guidelines.mdc` |
| Design decisions | `docs/design/jax_design.md` |
| Migration status | `docs/plans/migration_plan.md` |
| Bessel functions | `docs/ARCHITECTURE.md` § Bessel Functions |
| GIG η→θ optimization | `docs/ARCHITECTURE.md` § GIG η→θ |
| Package survey (TFP, FlowJAX, efax) | `docs/references/distribution_packages.md` |
