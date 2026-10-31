---
name: principles
description: >-
  The normix engineering principles library (adapted from pstack): design,
  verification, execution, and code-structure principles with normix
  grounding. Use when making design decisions, reviewing or refactoring
  code, verifying numerical work, planning multi-step changes, or when
  another skill cites a principle by name (prove-it-works,
  subtract-before-you-add, build-the-lever, ...).
---

# Principles

The four priorities — Elegance > Numerical efficiency & robustness >
Mathematical clarity > Simplicity — live in `AGENTS.md` (always in context)
and `dev-notes/design/design.md` (with rationale). This skill does not
restate them; it adds the operational principles for executing under them.
Workflow skills (architect, arena, interrogate, figure-it-out, tdd) cite
sections here by name.

## Design

**Foundational thinking.** Data structures first: get the `eqx.Module`
fields, sufficient-statistic pytrees, and θ/η layouts right before writing
logic — the right shape makes downstream code obvious (row E9's
theory-order pytrees are the house example). Scaffold before features when
every later phase benefits. A late data-structure change is a rewrite; an
early one is a one-line diff.

**Exhaust the design space.** When a structural decision has no precedent,
sketch 2–3 competing shapes before committing — the architect skill
makes this concrete. Skip alternative structural sketches for routine fixes.
Benchmark numerical choices within the governing contracts (rows S5, S8, S9).

**Redesign from first principles.** Integrating a new requirement means
redesigning as if it had been known on day one, not bolting on. AGENTS.md
already demands "redesign over accretion"; the method: read the affected
design holistically, ask "what would we have built", then propagate the
change through every reference — types, docs, design rows.

**Attack the premise.** Repeated failures can be evidence against the assumption shared by the attempted fixes.
After two fixes fail the same check, write that shared premise in one sentence.
Test it with a rerunnable observation that would differ if the premise were false.
If the evidence contradicts it, investigate what the failures share that the premise did not predict.
If the evidence supports it, trace the cause further before another fix.
For example, solver stalls attributed to iteration limits may instead come from compilation or dispatch.

**Subtract before you add.** Remove dead weight first and build on the
simpler base (row E4: two fitters replaced two obsolete ones). No
speculative guards, validators, or configuration beyond what the math
demands.

**Laziness protocol and reader load.** Smallest diff that solves the
problem; prefer deletion; one decision in one place. If tracing an answer
takes more than 3 files or layers, flatten. The reader-side test: can
someone answer "where does X come from and what can change X" in 30
seconds? `eqx` immutability answers the second for free — keep it that
way.

## Verification

**Benchmark within the contract.** List the user goal and governing decision IDs before comparing candidates.
Reject candidates that violate those constraints unless explicit user authorization covers the specific change.
Value agreement, transformation support, and device placement are separate claims.
Require evidence for each claim affected by the change.
A timeout rejects the measured run, not every implementation of that approach.

Before measuring, write down the decision, expected number, noise floor, and result that would change the decision.
For JAX execution timing, make an untimed warm-up call for each input signature to exclude JIT compilation.
Wait for that call with `jax.block_until_ready` before starting the timer.
Wait for each timed result with `jax.block_until_ready` before stopping the timer.
JAX dispatch is asynchronous; timing dispatch alone does not measure completed computation.
Measure compilation separately when it matters to the decision.

Explain every measured number before trusting, reporting, or acting on it.
Identify the limiting resource or code path from a separate profile or counters collected during execution.
Check that the intended work ran and produced correct outputs.
Keep the units, run count, spread, and limiter evidence with the number.
Treat an effect smaller than the run-to-run spread as inconclusive.

**Prove it works.** Hand back an inspectable artifact, not a claim. The
normix convention: any EM/solver/benchmark change ships with the relevant
figure (`normix.utils.plotting`) and/or the `benchmarks/compare.py` delta —
"it converged" is not evidence; the convergence plot is. Bug fixes carry a
failing-then-passing test (the tdd skill). Verify the real thing: run the
suite and read the actual numbers; a subagent's summary is not
verification.

For a refactor, first pin verified behavior with golden values or Hypothesis property tests.
Then change the structure.
Finally, rerun those checks to prove that observable behavior stays unchanged within justified numerical tolerances.

**Test behavior, not implementation.** Tests describe what callers can observe through the interface they use.
A useful test fails when that behavior breaks and survives a change to internal structure.
Judge each test against both imagined edits: a wrong observable result and a refactor that preserves the result.
Derive expectations from the mathematical contract or an independent reference, rather than the implementation under test.
For normix, numerical values, parameter round-trips, and promised JIT or autodiff behavior are observable contracts.
Private helper choices and internal call order usually are not.

**Mutation testing.** Use deliberate code changes to check whether numerical tests detect broken behavior.
Read [Mutation testing](references/mutation-testing.md) when auditing GIG or Bessel tests, especially when coverage hides weak assertions.

**Sequence verifiable units.** Small units, each ending in a check; verify
before advancing, never batch checks to the end. Between refactor phases
the contract and EM-convergence tests stay green — that is the normix
carve-out to "planned intermediate breakage". Deliver in the order that
proves the work: failing test first, fix second; baseline captured before
treatment.

**Build the lever.** For any non-trivial sweep, edit run, or analysis,
build the rerunnable tool — script, codemod, benchmark harness — instead
of hand-doing it. normix already lives this: `benchmarks/utils.py`
(`save_result`: JSON + git hash) and `benchmarks/compare.py` (before/after
deltas) are the lever; extend them rather than hand-collecting numbers.
If you cited this principle and the diff contains no script, you didn't
apply it.

## Execution

**Never block on the human.** Proceed with routine reversible work inside the established contract.
Reuse existing user authorization.
Route structural or execution-contract changes through architect.
Request agreement only for unresolved contract changes or irreversible actions outside that authorization.

**Fix root causes.** A symptom shows where a wrong state becomes visible.
Reproduce it.
Trace back to the first wrong decision or failure to handle a legitimate state.
If the triggering state is legitimate, handling it there can be the fix.
If the state violates the contract, a guard there hides its source.

For example, an output clamp can hide a wrong Bessel formula.
Constructor clamps remain part of the contract (row F6).
Check other call sites for the same cause.
For a stall, distinguish compilation, dispatch, and solver progress before claiming a cause.
Keep the failure open when a workaround only avoids its execution path.

**Outcome-oriented execution.** In planned migrations, converge on the
target architecture rather than preserving every intermediate state — but
tests stay green between phases (see Sequence verifiable units), and any
temporary compatibility shim is deleted by the same plan that created it.

**Migrate callers, then delete legacy APIs.** Internal APIs: migrate and
delete in one wave; no append-only dual paths. Public API (normix ships on
PyPI): a `DeprecationWarning` cycle first, deletion in a following minor
release.

**Guard the context window.** Route bulk output to subagents and keep
summaries in the main thread; read selectively. In Cursor,
`project-overview.mdc` is always applied — every line there costs every
Cursor conversation.

**Encode lessons in structure.** A recurring correction should change what the repository permits.
Follow [Agent makes a recurring mistake](../agent-maintenance/SKILL.md#agent-makes-a-recurring-mistake) for the enforcement order.
A reminder alone leaves the same mistake available to the next contributor.

## Code structure

**Boundary discipline.** Validate and clamp at constructors
(`from_classical`, `from_natural` — row F6); trust parameters
unconditionally inside. Math kernels are pure functions; no re-validation
deep in call chains.

**Type and structure discipline.** `eqx.Module` (frozen dataclass, pytree)
is the type system here: fields declare shape intent,
`eqx.field(static=True)` marks non-traced config, and immutability makes
illegal state transitions unrepresentable. Don't smuggle in mutable state,
`_fitted` flags, or Python control flow on traced values.

## Rarely needed (know they exist)

- **Make operations idempotent** — repeated runs should converge to the target state after interruption.
  Derive each step from the state that exists.
  Apply only the remaining change; a repeat then finds nothing left to do.
  Partial benchmark or docs artifacts need evidence that distinguishes completed work from stale or incomplete results.
  Check both a repeated run and a restart after partial completion.
  This matters for benchmark resumption and docs publishing; the library itself is pure-functional.
- **Separate before serializing shared state** — relevant to
  multi-agent/worktree runs (best-of-n or an external bake-off); give each
  writer its own path.

## Gotchas

- These principles rank below the four priorities. When "smallest diff"
  collides with "special cases use their own analytical formulas" (the
  Simplicity carve-out in AGENTS.md), the priorities win.
- Don't cite a principle as decoration; cite it when it changed what you
  did.
