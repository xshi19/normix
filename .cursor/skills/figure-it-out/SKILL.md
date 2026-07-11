---
name: figure-it-out
description: >-
  Design a custom auditable playbook when no existing skill fits: a new
  family of distributions, a fundamentally different solver, a multi-week
  refactor, or long autonomous work reviewed after the fact. Use for
  /figure-it-out or genuinely novel cross-cutting work — not for routine
  additions (the agent-maintenance skill covers those).
disable-model-invocation: true
---

# Figure it out

When the task matches no playbook, the first deliverable is the playbook
itself: phases, a falsifiable definition of done, and a decision trail the
maintainer can audit after stepping away.

Routine work routes elsewhere: a new distribution → the agent-maintenance
trigger list; a clean bug fix → the tdd skill; a one-way-door design
decision → the architect skill.

## Phase A: Frame

Don't start the run until you can state:

- **Done, as a checkable predicate.** A test that passes, a benchmark
  threshold ("≥ X% faster and the `slow or stress` suite green"), a
  numerical property (likelihood-per-batch monotone within tolerance).
  A vague goal makes a loop spin.
- **Scope, quantified.** Rough units of work and the blockers grounding
  surfaced — raise them now, not fifty commits in.
- **Rigor level, biased high.** One-way doors and wide blast radius get
  gates and artifacts; reversible low-stakes steps get less. Rigor means
  checks, not "try harder".

A multi-hour run earns one checkpoint here: present the framing and
tradeoffs before committing.

## Phase B: Design the workflow

- Decompose into independently-landable units; sequence
  riskiest-unknown-first so option value stays high.
- Build the verification harness **before** the work, with the baseline
  captured from the pre-change state so every check reads "old vs new".
  For benchmarks, `benchmarks/utils.py::save_result` and
  `benchmarks/compare.py` are the existing lever — extend them, don't
  hand-collect numbers.
- One-way-door design decisions inside the plan → run the architect skill.
  Mechanical work whose shape is already concrete → don't; a second arena
  over a settled design is over-engineering.
- Write the phase list into a plan file under `dev-notes/plans/` (or
  `dev-notes/investigations/` for investigation-shaped work). That file is
  what the human reviews.

## Phase C: Run the loop

Each unit is an experiment: hypothesis → smallest change → measure against
the predicate on the real artifact → keep if it advanced, revert if it
didn't. Verify each unit before starting the next (principles skill
§ Sequence verifiable units); never batch checks to the end. A verdict is
VERIFIED, NOT VERIFIED, or INCONCLUSIVE — inconclusive is not a pass.

## Phase D: Decision trail

Keep a decision log in the plan file as you go, one row per decision:
`when | decision | why | evidence`. Evidence is artifact links — test
output, `compare.py` delta, plot path — not prose claims. Commit the log
with the work so the reviewer can audit instead of replaying the session.

## Phase E: Verify and hand back

Check the whole against the Phase A predicate on the real target (full
test suite, real benchmark), not just the harness. Encode any recurring
correction as structure — a test, a constant, a rule gotcha — via the
agent-maintenance triggers. Reply with: the playbook, what's verified
against the predicate, the decision-log path, and what's still open.

## Gotchas

- A regression beyond tolerance stops the line and reports. Never relax
  the predicate or weaken a test to make a unit pass.
- Don't over-fan: parallel subagents only across genuine seams (isolated
  worktrees), which normix rarely has.
