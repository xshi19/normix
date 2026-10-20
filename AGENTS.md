# normix

JAX package for Generalized Hyperbolic distributions as exponential families.
Built on Equinox. Deps: `jax`, `equinox`, `jaxopt`, `numpy`, `scipy`; `matplotlib` (optional).

## Commands

- Install: `uv sync`
- Tests: `uv run pytest tests/` (fast default: excludes `slow`, `stress`, `integration`, `gpu`)
- Slow/stress/integration suite: `uv run pytest tests/ -m "slow or stress or integration"`
- Single test file: `uv run pytest tests/test_bessel_contract.py -v`
- Benchmarks: `uv run python benchmarks/run_all.py` (or individual: `uv run python benchmarks/bench_em_mixture.py`)
- ASV smoke: `cd asv_bench && uv run asv run --python=same --quick`
- ASV tripwire (local): `cd asv_bench && uv run asv continuous --factor 1.5 -b 'Bessel|GIGFromExpectation' master HEAD`
- ASV dashboard: `bash scripts/publish_asv_html.sh` (from committed `asv_bench/results/`)
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
- **Distribution API**: Every distribution provides `pdf`, `cdf` (where analytical), `mean`, `var`, `std`, `rvs`; joint/marginal mixtures provide `rvs`, `mean`, `cov`, `skewness`, `kurtosis` (excess)
- **Information measures**: exponential families provide `entropy`, `varentropy`, `renyi(alpha)`, `log_density_power(alpha)` (autodiff of $R(\alpha)=\log\int p^\alpha$); marginal mixtures expose `joint_entropy/varentropy/renyi` (marginal-of-X versions are intractable)

## Context Map

Published website source → `docs/` (built to https://xshi19.github.io/normix/)

Agent/dev internal knowledge → `dev-notes/` (not published; index at `dev-notes/README.md`)

Architecture and module hierarchy → `dev-notes/ARCHITECTURE.md`

| Area | Doc |
|---|---|
| Published design rationale | `docs/design/` (exponential_family, mixtures, em_framework, solvers_and_bessel, why_not_gradient_descent) |
| Internal design | `dev-notes/design/design.md`, `dev-notes/design/index.md`, `dev-notes/design/agent_instructions_design.md` |
| Theory and research | `docs/theory/` (MyST, [Shi2016]); `docs/research/`; working notes in `dev-notes/research/` |
| Bibliography | `docs/references.md` |
| Conversions and tech notes | `dev-notes/tech_notes/distribution_conversions.md`; `dev-notes/tech_notes/` |
| Plans | Active: `dev-notes/plans/` (`docs_refactor.md` Phase 8 polish, `loops_and_orchestration.md`). Done: `dev-notes/archive/plans/` (`review_roadmap_2026-07-12.md`, `finance_architecture.md`, `asv_benchmarking.md`) |
| Archived proposals | `dev-notes/archive/design/` |
| Surveys | `dev-notes/references/distribution_packages.md`, `dev-notes/references/gpjax_review.md`, `dev-notes/references/humanlayer_skills_review.md`; archived `dev-notes/archive/references/pstack_skills_review.md` |
| Benchmarks | `benchmarks/` (`run_all.py`, `compare.py`); `asv_bench/`; `scripts/publish_asv_html.sh`; `.github/workflows/asv.yml` |

## Reading rules and workflows

Read the matching files below before work in that area. Apply every matching row.
Resolve repository paths from the root. Resolve skill-local references from the skill directory.
Read `.cursor/rules/project-overview.mdc` once for the project catalog. For instruction edits, also read `.cursor/rules/rule-authoring.mdc`.

| When | Read |
|---|---|
| Python code, including tests and scripts | `.cursor/rules/coding-conventions.mdc` |
| Comment Python, or review comments | `.cursor/skills/no-comments/SKILL.md` |
| Add or change tests | `.cursor/rules/testing-guidelines.mdc` |
| Run, select, or profile tests | `.cursor/skills/running-tests/SKILL.md` |
| Add a distribution, module, constant, or design decision | `.cursor/skills/agent-maintenance/SKILL.md` |
| Fix a reproducible bug | `.cursor/skills/tdd/SKILL.md` |
| Change Bessel, GIG/GH solvers, EM, or custom JVPs | `.cursor/skills/interrogate/SKILL.md` |
| Change class boundaries, abstractions, backend selection, device placement, or transformation guarantees | `.cursor/skills/architect/SKILL.md` |
| Explain behavior or trace design history | `.cursor/skills/how/SKILL.md` or `.cursor/skills/why/SKILL.md` |
| Make design, verification, or refactoring decisions | `.cursor/skills/principles/SKILL.md` |
| Edit prose, docstrings, rules, skills, or PR descriptions | `.cursor/skills/unslop/SKILL.md` |
| Edit public docs, internal docs, README, or docstring links | `.cursor/rules/docs-cross-links.mdc` |
| Edit tutorials; build or publish docs | `.cursor/skills/docs-publish/SKILL.md` |
| Work in notebooks/ | `.cursor/rules/notebook-guidelines.mdc` |
| Commit, prepare a PR, or release | `.cursor/skills/git-conventions/SKILL.md` |
| Edit AGENTS.md or ARCHITECTURE.md | `.cursor/rules/maintain-agents-md.mdc` or `.cursor/rules/maintain-architecture-md.mdc`, respectively |
| Edit theory or design/technical/reference notes | `.cursor/rules/maintain-theory-docs.mdc` or `.cursor/rules/maintain-design-docs.mdc`, as applicable |
| Edit rules or skills | `.cursor/rules/maintain-cursor-rules.mdc` or `.cursor/rules/maintain-skills.mdc`, respectively |
| Request a visual explanation | `.cursor/skills/show-me/SKILL.md` |
| Explicitly request a custom playbook | `.cursor/skills/figure-it-out/SKILL.md` |
| Prepare an external design/review package | `.cursor/skills/arena/SKILL.md` (hand-off reference; runner retired) |

## Codex notes

Cursor frontmatter does not load these files in Codex. Read the selected bodies explicitly.
Skills remain canonical under `.cursor/skills/`.
`.agents/skills/` may symlink those directories, and `.codex/agents/` holds project subagents.
Use available host tools.
Cursor tool arguments and model names are not Codex configuration.

Preserve required external hand-offs.
Report a prepared package separately from a completed external review.
Follow the user's task scope.
A workflow pointer does not authorize publishing or releasing.
Host execution and hand-offs: `dev-notes/design/agent_instructions_design.md` § Host support.

## Design Philosophy

Priority: Elegancy > Numerical Efficiency & Robustness > Mathematical Clarity > Simplicity.

- **Elegancy**: Reading and using normix should be enjoyable. This applies to code, mathematics, documentation, and agent instructions alike. Think in high-level abstractions — modules, base classes, object hierarchies — and optimise for long-term maintainability. When a new feature feels like it needs a quick-fix function, ask the user whether the underlying design should be refactored instead of patching around it. Prefer redesign over accretion.

- **Numerical Efficiency & Robustness**: normix targets the same standard as professional scientific-computing libraries. Use knowledge of numerical linear algebra, optimisation, and statistics to choose efficient algorithms (e.g. `assume_a` flags for triangular/positive-definite solves, exploiting Cholesky structure). Code must be robust under extreme parameters — handle edge cases, guard against overflow/underflow (e.g. large-$z$ asymptotics in `log_kv`), and prefer log-space arithmetic where magnitudes vary widely.

- **Mathematical Clarity**: Math is the backbone of this package. We start with math theory, and end with practical engineering solutions. Mathematical notation in docstrings, MyST theory docs, and code comments must be clean and internally consistent. Maintain a clear correspondence between math symbols and code variable names (e.g. $\theta$ ↔ `theta`, $\eta$ ↔ `eta`, $\psi$ ↔ `log_partition`). Avoid ad-hoc or ambiguous notation.

- **Simplicity**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it; removing something and getting equal or better results is a simplification win. However, simplicity must not sacrifice any of the three higher-priority concerns. Delegating to a general-case implementation when a closed-form specialisation exists is not "simpler" — it is mathematically more complex and numerically wasteful. Special-case distributions (NIG, VG, NInvG) must use their own analytical formulas rather than routing through GeneralizedHyperbolic.
