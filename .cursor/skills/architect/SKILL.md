---
name: architect
description: >-
  Design before implementing: sketch eqx.Module fields, classmethod
  signatures, and module boundaries before code, fan candidates through the
  arena skill, and land the decision as a dev-notes/design/design.md row.
  Use for one-way-door structural changes — a new ABC layer, a new mixture
  family shape, an η-update abstraction — or when the user asks to
  design/architect something non-trivial. Not for routine work (a new
  analytical Hessian, a pdf bug fix, a doc edit).
---

# Architect

Design first for one-way-door structural changes. The trigger test: **would
this change add or amend a row in `dev-notes/design/design.md`?** If not,
skip this skill and just implement.

Open a todo per phase: Ground, Sketch, Agree, Implement, Scrap.

## Phase A: Ground

Read before designing — the constraints live in the docs, not in guesses:

- `dev-notes/design/design.md` — philosophy + decision table (any new shape
  must honor or explicitly amend these rows)
- `dev-notes/design/index.md` quick lookups → the relevant topical doc
  (`exponential_family.md`, `mixtures.md`, `em_framework.md`,
  `solvers_and_bessel.md`)
- `dev-notes/ARCHITECTURE.md` — current module hierarchy
- The code the design touches (Grep + Read; normix is small, direct reading
  beats fan-out — reach for the how skill only for a subsystem you
  genuinely don't know)

If the design redefines ownership or layering, run the why skill on the
existing shape first: superseded rationale in `dev-notes/archive/design/`
is a constraint, not trivia.

## Phase B: Sketch (arena)

Run `.cursor/skills/arena/SKILL.md` with the design task and the Phase A
grounding. Pass each runner `references/runner-prompt.md`; each returns a
design package shaped per `references/rationale-template.md` — caller's
usage first, then the field sketch, classmethod signatures with
`NotImplementedError` bodies, module map, and rationale. Arena returns one
synthesized package with its synthesis decision filled in.

## Phase C: Agree (checkpoint by default)

Present the synthesized package and a draft `design.md` row to the user
before implementing. Architect only fires on one-way doors, which is
exactly where human attention belongs — reversible work never reaches this
skill (principles skill § Never block on the human covers the rest). Skip
the checkpoint only when the user asked for design-and-implement in one go.

For extra adversarial pressure on a high-stakes sketch, run the interrogate
skill on the design package before implementing.

## Phase D: Implement against the sketch

The sketch is the contract; replace `NotImplementedError` bodies with code.
A deviation — a signature needing a parameter the sketch missed, an extra
field — is signal worth surfacing, not friction to absorb silently. Ask
whether the sketch was wrong, a requirement was missed, or the
implementation is overreaching.

## Phase E: Scrap when the architecture is wrong

Repeated friction of the same shape means the sketch is wrong: the same
workaround recurring across unrelated code, special-case branches
multiplying, callers needing the abstraction's internal rules. A few hard
edge cases are not a scrap signal — GH is legitimately complex, and
complexity in the math is not complexity in the design.

When scrapping: re-ground with the implementation lessons as inputs,
subtract before adding (the new sketch should be smaller before it grows),
and re-run Phase B.

## Output — landing the decision

Follow the agent-maintenance skill's "Design Decision Made" trigger:

1. New row in the `dev-notes/design/design.md` decision table, with a
   Why/Detail link
2. Rationale in the relevant topical design doc — or a proposal under
   `dev-notes/plans/` for subsystem-scale work (archived to
   `dev-notes/archive/design/` once implemented)
3. `docs/design/` update if the rationale is user-facing

The rejected alternatives from arena go into the rationale doc. That nuance
is a large part of why architect was run at all.

## Gotchas

- Numerical-method choices inside a design (solver, backend, tolerance) are
  settled by benchmarks, not by the arena — leave them as open questions in
  the package with a benchmark plan.
- Don't let the sketch drift into implementation: bodies stay
  `NotImplementedError` until Phase D, or the "design" is just code review
  of premature code.
