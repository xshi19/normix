# normix

`normix/` implements Generalized Hyperbolic distributions as exponential families in JAX and Equinox.
`tests/` contains the package tests. Sphinx builds `docs/` into `docs/_build/`.
`dev-notes/` holds internal design records and working notes.

## Checks

Run from the repository root with uv.

- Lint the package and tests with `uv run ruff check normix/ tests/`.
- Run the default suite with `uv run pytest tests/`. It excludes `slow`, `stress`, `integration`, and `gpu`.
- Build the docs with `uv run make -C docs html`.

See the [lint rules and numerical checks](docs/development/lint-rules.md) for the contracts each check covers.

## Design philosophy

The priorities, in order, are elegance, numerical efficiency and robustness, mathematical clarity, and simplicity.
Use the analytical formulas for NIG, VG, and NInvG.
The [design decisions](dev-notes/design/design.md) cover immutable distributions, single-observation methods,
parameter conversions, the log-partition triad, and solver derivatives in natural coordinates.

## Context map

Read the matching files before working in each area.
Resolve paths from the repository root and skill-local references from the skill directory.

| When | Read |
|---|---|
| Distribution catalog and public API | `.cursor/rules/project-overview.mdc` |
| Module layout, design rationale, and plans | `dev-notes/ARCHITECTURE.md`, `dev-notes/design/index.md`, `dev-notes/README.md` |
| Published theory, design, or references | `docs/theory/`, `docs/design/`, `docs/references.md` |
| ASV benchmarks and dashboard | `asv_bench/README.md` |
| Edit instructions | `.cursor/rules/rule-authoring.mdc` |
| Python code, including tests and scripts | `.cursor/rules/coding-conventions.mdc` |
| Comment Python, or review comments | `.agents/skills/no-comments/SKILL.md` and, for large diffs, `.codex/agents/comment-sicko.toml` |
| Add or change tests | `.cursor/rules/testing-guidelines.mdc` |
| Run, select, or profile tests | `.agents/skills/running-tests/SKILL.md` |
| Add a distribution, module, constant, or design decision | `.agents/skills/agent-maintenance/SKILL.md` |
| Fix a reproducible bug | `.agents/skills/tdd/SKILL.md` |
| Change Bessel, GIG/GH solvers, EM, or custom JVPs | `.agents/skills/interrogate/SKILL.md` |
| Change class boundaries, abstractions, backend selection, device placement, or transformation guarantees | `.agents/skills/architect/SKILL.md` |
| Explain behavior or trace design history | `.agents/skills/how/SKILL.md` or `.agents/skills/why/SKILL.md` |
| Design, verification, refactoring, benchmarks, or shared engineering skills | `dev-notes/skill-overlays/principles.md` and its matching shared skill |
| Audit numerical tests with mutation testing | `.agents/skills/mutation-testing/SKILL.md` and `dev-notes/skill-overlays/mutation-testing.md` |
| Prove a Lean claim, translate with jax-lean, or edit formal coverage | `.agents/skills/normix-formal-verify/SKILL.md` |
| Edit prose, docstrings, rules, skills, or PR descriptions | `.agents/skills/unslop/SKILL.md` |
| Edit public docs, internal docs, README, or docstring links | `.cursor/rules/docs-cross-links.mdc` |
| Edit tutorials, build docs, or publish docs | `.agents/skills/docs-publish/SKILL.md` |
| Work in `notebooks/` | `.cursor/rules/notebook-guidelines.mdc` |
| Commit, prepare a PR, or release | `.agents/skills/git-conventions/SKILL.md` |
| Edit AGENTS.md or ARCHITECTURE.md | `.cursor/rules/maintain-agents-md.mdc` or `.cursor/rules/maintain-architecture-md.mdc`, respectively |
| Edit theory or design/technical/reference notes | `.cursor/rules/maintain-theory-docs.mdc` or `.cursor/rules/maintain-design-docs.mdc`, as applicable |
| Edit rules or skills | `.cursor/rules/maintain-cursor-rules.mdc` or `.cursor/rules/maintain-skills.mdc`, respectively |
| Request a visual explanation | `.agents/skills/show-me/SKILL.md` |
| Explicitly request a custom playbook | `.agents/skills/figure-it-out/SKILL.md` (`allow_implicit_invocation: false`) |
| Prepare an external design or review package | `.agents/skills/arena/SKILL.md` and the [external review requirements](dev-notes/design/agent_instructions_design.md#external-review) |

### Skills and overlays

`.agents/skills/README.md` lists project skills and pinned shared skills under `.agents/skills/`.
Change a shared skill upstream in `xshi19/xshi-skills`; never edit a pinned copy. Read the matching plain Markdown attachment in `dev-notes/skill-overlays/` alongside its shared skill.
Invocation settings live with each skill. See [host support](dev-notes/design/agent_instructions_design.md#host-support) for discovery and reviewer configuration.
A workflow pointer does not authorize publishing or releasing.
