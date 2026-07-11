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
sketch 2–3 competing shapes before committing — the architect and arena
skills make this concrete. Not for mechanical work, bug fixes, or
numerical-method choices; benchmarks decide those (rows S5, S8, S9).

**Redesign from first principles.** Integrating a new requirement means
redesigning as if it had been known on day one, not bolting on. AGENTS.md
already demands "redesign over accretion"; the method: read the affected
design holistically, ask "what would we have built", then propagate the
change through every reference — types, docs, design rows.

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

**Prove it works.** Hand back an inspectable artifact, not a claim. The
normix convention: any EM/solver/benchmark change ships with the relevant
figure (`normix.utils.plotting`) and/or the `benchmarks/compare.py` delta —
"it converged" is not evidence; the convergence plot is. Bug fixes carry a
failing-then-passing test (the tdd skill). Verify the real thing: run the
suite and read the actual numbers; a subagent's summary is not
verification.

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

**Never block on the human.** normix changes are reversible (PR + tests +
release-please): proceed and present, don't ask permission. Reserve
checkpoints for genuine one-way doors (architect Phase C) and irreversible
actions.

**Fix root causes.** Reproduce, then ask why until you hit bottom: a NaN
from `log_kv` at large $z$ is a regime-boundary bug, not a missing output
clamp. Distinguish structural clamps (row F6: constructors clamp inputs —
by design) from band-aid clamps that hide a wrong formula. Grep for the
pattern, not just the instance.

**Outcome-oriented execution.** In planned migrations, converge on the
target architecture rather than preserving every intermediate state — but
tests stay green between phases (see Sequence verifiable units), and any
temporary compatibility shim is deleted by the same plan that created it.

**Migrate callers, then delete legacy APIs.** Internal APIs: migrate and
delete in one wave; no append-only dual paths. Public API (normix ships on
PyPI): a `DeprecationWarning` cycle first, deletion in a following minor
release.

**Guard the context window.** Route bulk output to subagents and keep
summaries in the main thread; read selectively. `project-overview.mdc` is
always loaded — every line there costs every conversation.

**Encode lessons in structure.** A recurring correction becomes a constant
(`normix/utils/constants.py`), a test, a lint rule, or a rule-file gotcha —
not a prose reminder. Route via the agent-maintenance triggers; pick the
strongest mechanism available.

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

- **Make operations idempotent** — relevant to `benchmarks/run_all.py`
  resumption and docs publishing; the library itself is pure-functional.
- **Separate before serializing shared state** — relevant to
  multi-agent/worktree runs (arena, best-of-n); give each writer its own
  path.

## Gotchas

- These principles rank below the four priorities. When "smallest diff"
  collides with "special cases use their own analytical formulas" (the
  Simplicity carve-out in AGENTS.md), the priorities win.
- Don't cite a principle as decoration; cite it when it changed what you
  did.
