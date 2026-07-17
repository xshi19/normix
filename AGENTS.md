# normix

JAX package for Generalized Hyperbolic distributions as exponential families.
Built on Equinox. Deps: `jax`, `equinox`, `jaxopt`, `tfp` (Bessel only), `optax` (optional).

## Commands

- Install: `uv sync`
- Tests: `uv run pytest tests/` (fast default: excludes `slow`, `stress`, `integration`, `gpu`)
- Full validation: `uv run pytest tests/ -m "slow or stress or integration"`
- Single test: `uv run pytest tests/test_gamma.py -v`
- Benchmarks: `uv run python benchmarks/run_all.py` (or individual: `uv run python benchmarks/bench_em_mixture.py`)
- Docs (cached build): `uv run make -C docs html`
- Docs (full re-execute): `uv run make -C docs html-strict`
- Docs clean: `uv run make -C docs clean` / `uv run make -C docs clean-cache`
- Notebooks: `uv run jupyter lab`
- Add dependency: `uv add <package>` / `uv add --dev <package>`
- This project uses **uv**, NOT conda.

## Commits

Conventional commits: `feat|fix|docs|test|refactor(scope): description`

Before committing:
- `uv run pytest tests/` passes
- No debug print statements
- Type hints on all public methods
- Update the relevant plan in `dev-notes/plans/` if changes advance a phase
- Update `dev-notes/ARCHITECTURE.md` if new modules were added

## Core Design

- **Exponential family structure**: log base measure $h(x)$, sufficient statistics $t(x)$, log partition $\psi(\theta)$
- **Three parametrizations**: classical $\leftrightarrow$ natural $\theta$ $\leftrightarrow$ expectation $\eta = \nabla\psi(\theta)$, all JIT-compatible
- **Log-partition triad**: every distribution provides three pairs of classmethods — (log-partition, gradient, Hessian) × (JAX JIT-able, CPU numpy/scipy). Defaults use `jax.grad`/`jax.hessian`; subclasses override for analytical or Bessel-heavy implementations.
- **Solver separation**: `grad_fn` and `hess_fn` operate in θ-space only; the solver applies the φ↔θ chain rule internally. Distributions never need to know about the solver's reparametrization.
- **EM algorithm**: E-step computes conditional expectations $E[t(Y)|X]$, M-step converts $\eta \to \theta$ via `from_expectation`
- **Immutable**: all distributions are `eqx.Module` pytrees; M-step returns a new model
- **Unbatched core**: `log_prob`, `pdf`, `cdf` operate on single observations; batch via `jax.vmap`
- **Distribution API**: Every distribution provides `pdf`, `cdf` (where analytical), `mean`, `var`, `std`, `rvs`; joint/marginal mixtures provide `rvs`, `mean`, `cov`
- **Information measures**: exponential families provide `entropy`, `varentropy`, `renyi(alpha)`, `log_density_power(alpha)` (autodiff of $R(\alpha)=\log\int p^\alpha$); marginal mixtures expose `joint_entropy/varentropy/renyi` (marginal-of-X versions are intractable)

## Core Dependencies

`jax`, `equinox`, `jaxopt`, `tensorflow_probability.substrates.jax` (Bessel only), `matplotlib` (notebooks), `optax` (optional).

## Context Map

Published website source → `docs/` (built to https://xshi19.github.io/normix/)

Agent/dev internal knowledge → `dev-notes/` (not published; index at `dev-notes/README.md`)

Architecture and module hierarchy → `dev-notes/ARCHITECTURE.md`

| Area | Rule / Doc |
|---|---|
| Distribution code (`normix/`) | `.cursor/rules/coding-conventions.mdc` |
| Tests (`tests/`) | `.cursor/rules/testing-guidelines.mdc` |
| Notebooks (`notebooks/`) | `.cursor/rules/notebook-guidelines.mdc` |
| Published design rationale | `docs/design/` (exponential_family, mixtures, em_framework, solvers_and_bessel) |
| Design decisions table (internal) | `dev-notes/design/design.md` |
| Agent instructions design | `dev-notes/design/agent_instructions_design.md` |
| Full design doc index (internal) | `dev-notes/design/index.md` |
| Mathematical theory & derivations | `docs/theory/` (MyST `.md`, based on [Shi2016]) |
| Distribution conversions (`to_<name>`) | `dev-notes/tech_notes/distribution_conversions.md` |
| Active plans | `dev-notes/plans/` (`docs_refactor.md`, `finance_architecture.md`, `loops_and_orchestration.md`, `review_roadmap_2026-07-12.md`) |
| 2026-07-12 review roadmap (44 items, not started) | `dev-notes/plans/review_roadmap_2026-07-12.md` |
| `normix.finance` roadmap (Phase D + Phase E done; Phase F diversification proposed) | `dev-notes/plans/finance_architecture.md` |
| Docs refactor plan (Phases 1–7 done; Phase 8 polish in progress) | `dev-notes/plans/docs_refactor.md` |
| Completed/archived plans (JAX migration, review roadmap) | `dev-notes/archive/plans/` |
| Archived design proposals (already implemented) | `dev-notes/archive/design/` |
| Package survey (TFP, FlowJAX, efax) | `dev-notes/references/distribution_packages.md` |
| Technical notes | `dev-notes/tech_notes/` |
| Benchmarks (`benchmarks/`) | `benchmarks/` — EM, Bessel, GIG solvers, JIT solvers, incremental EM, GH path comparison (`bench_gh_paths.py`); `run_all.py` orchestrator; `compare.py` diff tool |
| Git conventions | `.cursor/skills/git-conventions/` |
| Doc/rule/skill maintenance + post-task reflection | `.cursor/skills/agent-maintenance/` |
| Docs website build/publish | `.cursor/skills/docs-publish/` |
| Cross-link conventions | `.cursor/rules/docs-cross-links.mdc` |
| One-way-door design workflow (→ design.md row) | `.cursor/skills/architect/` (calls `arena`, the strong model panel) |
| Adversarial diff review (Bessel/solver/EM/`custom_jvp`) | `.cursor/skills/interrogate/` |
| Walkthroughs & design history | `.cursor/skills/how/`, `.cursor/skills/why/` |
| TDD bug fixes / novel-work playbooks | `.cursor/skills/tdd/`, `.cursor/skills/figure-it-out/` |
| Engineering principles library | `.cursor/skills/principles/` |
| Prose style + writing registers | `.cursor/skills/unslop/` |
| pstack skill survey (archived; implemented) | `dev-notes/archive/references/pstack_skills_review.md` |

## Design Philosophy

Priority: Elegancy > Numerical Efficiency & Robustness > Mathematical Clarity > Simplicity.

- **Elegancy**: Reading and using normix should be enjoyable. This applies to code, mathematics, documentation, and agent instructions alike. Think in high-level abstractions — modules, base classes, object hierarchies — and optimise for long-term maintainability. When a new feature feels like it needs a quick-fix function, ask the user whether the underlying design should be refactored instead of patching around it. Prefer redesign over accretion.

- **Numerical Efficiency & Robustness**: normix targets the same standard as professional scientific-computing libraries. Use knowledge of numerical linear algebra, optimisation, and statistics to choose efficient algorithms (e.g. `assume_a` flags for triangular/positive-definite solves, exploiting Cholesky structure). Code must be robust under extreme parameters — handle edge cases, guard against overflow/underflow (e.g. large-$z$ asymptotics in `log_kv`), and prefer log-space arithmetic where magnitudes vary widely.

- **Mathematical Clarity**: Math is the backbone of this package. We start with math theory, and end with practical engineering solutions. Mathematical notation in docstrings, MyST theory docs, and code comments must be clean and internally consistent. Maintain a clear correspondence between math symbols and code variable names (e.g. $\theta$ ↔ `theta`, $\eta$ ↔ `eta`, $\psi$ ↔ `log_partition`). Avoid ad-hoc or ambiguous notation.

- **Simplicity**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it; removing something and getting equal or better results is a simplification win. However, simplicity must not sacrifice any of the three higher-priority concerns. Delegating to a general-case implementation when a closed-form specialisation exists is not "simpler" — it is mathematically more complex and numerically wasteful. Special-case distributions (NIG, VG, NInvG) must use their own analytical formulas rather than routing through GeneralizedHyperbolic.
