# normix

JAX package for Generalized Hyperbolic distributions as exponential families.
Built on Equinox. Deps: `jax`, `equinox`, `jaxopt`, `tfp` (Bessel only), `optax` (optional).

## Commands

- Install: `uv sync`
- Tests: `uv run pytest tests/` (fast default: excludes `slow`, `stress`, `integration`, `gpu`)
- Full validation: `uv run pytest tests/ -m "slow or stress or integration"`
- Single test: `uv run pytest tests/test_gamma.py -v`
- Benchmarks: `uv run python benchmarks/run_all.py` (or individual: `uv run python benchmarks/bench_em_mixture.py`)
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

## Design Philosophy

`jax`, `equinox`, `jaxopt`, `tensorflow_probability.substrates.jax` (Bessel only), `matplotlib` (notebooks), `optax` (optional).

## Context Map

Architecture and module hierarchy → `docs/ARCHITECTURE.md`

| Area | Rule / Doc |
|---|---|
| Distribution code (`normix/`) | `.cursor/rules/coding-conventions.mdc` |
| Tests (`tests/`) | `.cursor/rules/testing-guidelines.mdc` |
| Notebooks (`notebooks/`) | `.cursor/rules/notebook-guidelines.mdc` |
| Design decisions | `docs/design/design.md` |
| Fitters & eta rules (D1, done) | `docs/design/design.md` § EM Framework |
| Penalised EM / shrinkage | `docs/design/penalised_em.md`, `docs/theory/shrinkage.rst` |
| Agent instructions design | `docs/design/agent_instructions_design.md` |
| Mathematical theory & derivations | `docs/theory/` (`.rst` format, based on [Shi2016]) |
| Migration status | `docs/plans/migration_plan.md` |
| Bessel functions | `docs/ARCHITECTURE.md` § Bessel Functions |
| GIG η→θ optimization | `docs/ARCHITECTURE.md` § GIG η→θ |
| RVS generation (PINV, TDR) | `docs/ARCHITECTURE.md` § Random Variate Generation |
| Package survey (TFP, FlowJAX, efax) | `docs/references/distribution_packages.md` |
| Technical notes | `docs/tech_notes/` |
| Benchmarks (`benchmarks/`) | `benchmarks/` — EM, Bessel, GIG solvers; `run_all.py` orchestrator; `compare.py` diff tool |
| Git conventions | `.cursor/skills/git-conventions/` |
| Doc/rule/skill maintenance | `.cursor/skills/agent-maintenance/` |
| Docs website build/publish | `.cursor/skills/docs-publish/` |

## Design Philosophy

Priority: Elegancy > Numerical Efficiency & Robustness > Mathematical Clarity > Simplicity.

- **Elegancy**: Reading and using normix should be enjoyable. This applies to code, mathematics, documentation, and agent instructions alike. Think in high-level abstractions — modules, base classes, object hierarchies — and optimise for long-term maintainability. When a new feature feels like it needs a quick-fix function, ask the user whether the underlying design should be refactored instead of patching around it. Prefer redesign over accretion.

- **Numerical Efficiency & Robustness**: normix targets the same standard as professional scientific-computing libraries. Use knowledge of numerical linear algebra, optimisation, and statistics to choose efficient algorithms (e.g. `assume_a` flags for triangular/positive-definite solves, exploiting Cholesky structure). Code must be robust under extreme parameters — handle edge cases, guard against overflow/underflow (e.g. large-$z$ asymptotics in `log_kv`), and prefer log-space arithmetic where magnitudes vary widely.

- **Mathematical Clarity**: Math is the backbone of this package. We start with math theory, and end with practical engineering solutions. Mathematical notation in docstrings, `.rst` theory docs, and code comments must be clean and internally consistent. Maintain a clear correspondence between math symbols and code variable names (e.g. $\theta$ ↔ `theta`, $\eta$ ↔ `eta`, $\psi$ ↔ `log_partition`). Avoid ad-hoc or ambiguous notation.

- **Simplicity**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it; removing something and getting equal or better results is a simplification win. However, simplicity must not sacrifice any of the three higher-priority concerns. Delegating to a general-case implementation when a closed-form specialisation exists is not "simpler" — it is mathematically more complex and numerically wasteful. Special-case distributions (NIG, VG, NInvG) must use their own analytical formulas rather than routing through GeneralizedHyperbolic.
