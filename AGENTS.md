# normix

JAX package for Generalized Hyperbolic distributions and related distributions,
implemented as exponential families. Built on Equinox for immutable pytree modules.

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
- **Log-partition triad**: every distribution provides three pairs of classmethods — (log-partition, gradient, Hessian) × (JAX JIT-able, CPU numpy/scipy). Defaults use `jax.grad`/`jax.hessian`; subclasses override for analytical or Bessel-heavy implementations.
- **Solver separation**: `grad_fn` and `hess_fn` operate in θ-space only; the solver applies the φ↔θ chain rule internally. Distributions never need to know about the solver's reparametrization.
- **EM algorithm**: E-step computes conditional expectations $E[t(Y)|X]$, M-step converts $\eta \to \theta$ via `from_expectation`
- **Immutable**: all distributions are `eqx.Module` pytrees; M-step returns a new model
- **Unbatched core**: `log_prob`, `pdf`, `cdf` operate on single observations; batch via `jax.vmap`
- **Distribution API**: Every distribution provides `pdf`, `cdf` (where analytical), `mean`, `var`, `std`, `rvs`; joint/marginal mixtures provide `rvs`, `mean`, `cov`

### Dependencies

`jax`, `equinox`, `jaxopt`, `tensorflow_probability.substrates.jax` (Bessel only), `matplotlib` (notebooks), `optax` (optional).

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
| Technical notes | `docs/tech_notes/` |

## Design Philosophy
- **Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. 
Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. 
When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.
- **Numerical efficiency**: Use your knowledge in numerical linear algebra, optimization and statistics to improve the efficiency without adding code complexity. 
For example, use assume_a in scipy.linalg.inv for triangular and positive definite matrices. 
- **Numerical Robustness**: Code need to be robust under extreme scenarios. Handle the edge cases carefully.
For example, you can introduce additional code and logic to prevent the overflow of large z when computing logkv(v,z).
