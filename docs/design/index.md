# Design Documentation Index

> Living design docs for normix. Implemented proposals and historical
> migration plans are in `docs/archive/design/` (read-only). Plans for
> work that hasn't started yet are in `docs/plans/`.

## How this folder is organised

| File | What it covers |
|---|---|
| [`design.md`](design.md) | Design philosophy + canonical decision table (the "why" lookup) |
| [`exponential_family.md`](exponential_family.md) | EF base class, log-partition triad, Bregman solver interface, MVN promotion (D3), `jaxopt` migration (D4) |
| [`mixtures.md`](mixtures.md) | Joint vs Marginal split (D2), `MarginalMixture` ABC, parameter facade, `from_expectation` η→model map, factor-analysis sibling family |
| [`em_framework.md`](em_framework.md) | Model/Fitter separation, η-update rule layers, `Shrinkage` combinator + penalised EM theory, the four covariance regularisations (`none` / `det_sigma_one` / `det_sigma_x` / `a_eq_b`) |
| [`solvers_and_bessel.md`](solvers_and_bessel.md) | Bregman solver internals, GIG η-rescaling, Bessel regimes and CPU/GPU hybrid, RVS generation |
| [`agent_instructions_design.md`](agent_instructions_design.md) | How AGENTS.md, rules, skills, and design docs work together to give coding agents the right context |

## Quick lookups

| Question | Where to read |
|---|---|
| Why two classes for each mixture distribution? | `mixtures.md` § 1 |
| Why is the joint a public exponential family? | `mixtures.md` § 2 |
| Why three classmethod tiers for the log-partition? | `exponential_family.md` § 2 |
| What does `'det_sigma_x'` regularisation do? | `em_framework.md` § 5 |
| Why does `'a_eq_b'` matter beyond `'det_sigma_one'`? | `em_framework.md` § 5.2 |
| Why is `Shrinkage` a combinator, not subclasses? | `em_framework.md` § 4.3 |
| Why hand-rolled Newton instead of `optimistix`? | `solvers_and_bessel.md` § 1.2 |
| Why CPU backend for Bessel and GIG solve? | `solvers_and_bessel.md` § 4 |
| Where do rules vs skills vs docs belong? | `agent_instructions_design.md` |

## See also

- `docs/ARCHITECTURE.md` — module hierarchy, distribution storage table,
  numerical constants table.
- `docs/theory/` — mathematical derivations (`.rst`).
- `docs/tech_notes/` — deep dives (Bessel survey, GIG benchmarks,
  EM profiling, JAX overhead).
- `docs/plans/` — work not yet done. `finance_architecture.md` lives
  there until implemented.
- `docs/archive/design/` — implemented proposals retained for historical
  context (`em_covariance_extensions.md`, `penalised_em.md`,
  `log_partition_triad.md`, `solver_redesign.md`).
