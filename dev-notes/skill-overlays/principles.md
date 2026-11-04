# normix engineering attachments

Read this file for normix design, verification, and refactoring work.
Read the relevant shared skills below alongside their project additions here.
Paths in prose resolve from the repository root; Markdown links resolve from this file.
This file is an attachment, not a discoverable skill.
The priorities and analytical-specialization requirement in `AGENTS.md` govern these additions.

| Work | Shared skill | Project additions |
|---|---|---|
| Parameter and statistic layouts | [principle-code-mirrors-the-math](../../.agents/skills/principle-code-mirrors-the-math/SKILL.md) | [Data structures](#data-structures) |
| Mathematical values and fitting state | [principle-immutable-values-for-math-objects](../../.agents/skills/principle-immutable-values-for-math-objects/SKILL.md) | [Data structures](#data-structures) |
| Constructors, clamps, and failed solves | [principle-fail-loud-on-invalid-domain](../../.agents/skills/principle-fail-loud-on-invalid-domain/SKILL.md) | [Validation and root causes](#validation-and-root-causes) |
| Stable numerical formulas | [principle-choose-the-numerically-honest-formulation](../../.agents/skills/principle-choose-the-numerically-honest-formulation/SKILL.md) | [Validation and root causes](#validation-and-root-causes) |
| Specialized formulas | [principle-closed-form-over-general-case](../../.agents/skills/principle-closed-form-over-general-case/SKILL.md) | `AGENTS.md` design priorities |
| Related values and derivatives | [principle-one-source-of-truth-derive-the-rest](../../.agents/skills/principle-one-source-of-truth-derive-the-rest/SKILL.md) | [Verification and refactoring](#verification-and-refactoring) |
| Backend and transformation contracts | [principle-explicit-backend-no-silent-placement](../../.agents/skills/principle-explicit-backend-no-silent-placement/SKILL.md) | [Benchmark evidence](#benchmark-evidence) |
| Numerical measurements | [benchmarking](../../.agents/skills/benchmarking/SKILL.md) | [Benchmark evidence](#benchmark-evidence) |
| Numerical review | [numerical-review](../../.agents/skills/numerical-review/SKILL.md) | [Validation](#validation-and-root-causes), [verification](#verification-and-refactoring) |
| Design and migration decisions | [decision-records](../../.agents/skills/decision-records/SKILL.md) | [Project workflows](#project-workflows) |

## Data structures

Set `eqx.Module` fields, sufficient-statistic pytrees, and natural/expectation parameter layouts ($\theta$/$\eta$) before writing logic.
Use [design.md](../design/design.md) row E9's theory-order pytrees as the project example.
Keep `eqx.Module` values immutable: its frozen dataclass fields declare shape intent and form a pytree.
Mark non-traced configuration with `eqx.field(static=True)`.
Do not add mutable model state, `_fitted` flags, or Python control flow on traced values.

## Validation and root causes

Validate at `from_classical` and `from_natural`, following row F6's specified clamps and rejections.
Trust those parameters inside pure math kernels; do not repeat validation deep in call chains.
An output clamp can hide a wrong Bessel formula.
Keep constructor clamps required by F6 distinct from such workarounds.
For GIG, F6 also requires rejecting positive natural rate coordinates with `eqx.error_if` (EF-06).

For solver hangs, distinguish compilation, dispatch, and solver progress before blaming iteration limits.
Keep the failure open when a workaround only avoids its execution path.

## Benchmark evidence

Compare numerical candidates within [design.md](../design/design.md) rows S5, S8, and S9.
Those rows govern GIG inversion, Bessel evaluation, and backend placement.
Use the shared benchmarking skill's [JAX timing guidance](../../.agents/skills/benchmarking/references/async-jit-backends.md) for warm-up, synchronization, and separate compilation measurements.

For EM, solver, or benchmark changes, provide a relevant `normix.utils.plotting` figure and/or a `benchmarks/compare.py` delta.
Support convergence claims with a convergence plot.
Extend `benchmarks/utils.py::save_result` (JSON plus git hash) and `benchmarks/compare.py` instead of collecting results by hand.
Run the relevant suite and inspect actual measurements; a subagent summary does not verify the result.
For benchmark resumption and docs publishing, distinguish completed artifacts from stale or incomplete results.
Check repeated runs and restarts after partial completion; the library itself remains pure-functional.

## Verification and refactoring

Treat numerical values, parameter round-trips, and promised JIT or autodiff behavior as observable normix contracts.
Before refactoring, pin verified behavior with golden values or Hypothesis property tests.
Rerun those checks after changing structure, using justified numerical tolerances.
Keep contract and EM-convergence tests green between every refactor or migration phase.
Planned intermediate breakage does not waive this requirement.
For GIG or Bessel assertion audits, read [the mutation-testing attachment](mutation-testing.md) with its shared skill.
For bug fixes, follow [TDD](../../.cursor/skills/tdd/SKILL.md).

Row E4's `BatchEMFitter` and `IncrementalEMFitter` replaced two obsolete fitters; remove obsolete paths as callers migrate.
Migrate internal API callers and delete legacy APIs in the same wave.
Delete temporary compatibility shims through the plan that introduced them.
For public APIs shipped on PyPI, retain one `DeprecationWarning` cycle before deletion in a following minor release.

## Project workflows

Route structural and execution-contract changes through [architect](../../.cursor/skills/architect/SKILL.md).
Propagate accepted design changes through types, docs, and decision rows.
Use [agent-maintenance](../../.cursor/skills/agent-maintenance/SKILL.md#agent-makes-a-recurring-mistake) for recurring corrections.
In Cursor, `project-overview.mdc` loads in every session; keep it concise.
For authorized multi-agent or worktree runs, give each writer its own output path.
Preserve the [external hand-off contract](../design/agent_instructions_design.md#normix-engineer-hand-off) for external comparisons.
