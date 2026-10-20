---
name: architect
description: >-
  Design before implementing: sketch eqx.Module fields, classmethod
  signatures, and module boundaries before code, hand the sketch to
  Normix Engineer, and land the decision as a dev-notes/design/design.md row.
  Use for structural changes or changes to backend selection, device
  placement, or transformation guarantees. Also use for an explicit
  design request. Skip routine implementations that preserve those contracts.
---

# Architect

Design before structural changes or changes to established execution contracts.
Check whether the change revises a decision in `dev-notes/design/design.md`.
A stable signature does not imply stable backend or device semantics.
Skip routine implementation work that preserves the decisions.

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

For ownership, layering, or execution-contract changes, run the why skill before selecting a candidate.
Record the user goal, governing decision IDs, and rejected alternatives with their evidence.
Check related decisions even when the diff edits only one row.
Separate performance targets from requirements that every candidate must preserve.
If a candidate needs a contract change, identify the user instruction authorizing that change.
Keep that candidate a proposal when the authority is missing.

Completion criterion: each candidate preserves the cited constraints or names an unresolved contract change.

## Phase B: Sketch (hand-off)

Prepare the package in `dev-notes/design/agent_instructions_design.md` § Host support.
Do not spawn candidates in this session.
The user relays the package. This session does not call Normix Engineer.

Stop when the package is ready.
With no returned synthesis, the state is "ready for external review".
Do not start Phase C until that synthesis is back.

Completion criterion: the package is ready, this session spawned no candidate, and the state is ready for external review.

## Phase C: Agree (checkpoint by default)

Start this phase only after the synthesized package is back.
Present the synthesized package and proposed decision changes before implementing.
Use existing user authorization when it covers those changes.
Otherwise, request agreement on the specific unresolved contract change.
Do not ask again for an already authorized change.

For extra adversarial pressure on a high-stakes sketch, run the interrogate
skill on the design package before implementing.

Completion criterion: user authorization covers the package and its proposed contract changes.

## Phase D: Implement against the sketch

The sketch is the contract; replace `NotImplementedError` bodies with code.
A deviation — a signature needing a parameter the sketch missed, an extra
field — is signal worth surfacing, not friction to absorb silently. Ask
whether the sketch was wrong, a requirement was missed, or the
implementation is overreaching.

Completion criterion: the implementation matches the sketch; every
deviation was surfaced, not absorbed.

## Phase E: Scrap when the architecture is wrong

Repeated friction of the same shape means the sketch is wrong: the same
workaround recurring across unrelated code, special-case branches
multiplying, callers needing the abstraction's internal rules. A few hard
edge cases are not a scrap signal — GH is legitimately complex, and
complexity in the math is not complexity in the design.

When scrapping: re-ground with the implementation lessons as inputs,
subtract before adding (the new sketch should be smaller before it grows),
and re-run Phase B.

Completion criterion: the new sketch is smaller than the one that
failed, and the implementation lessons are inputs to Phase B, not
footnotes.

## Output — landing the decision

Follow the agent-maintenance skill's "Design Decision Made" trigger:

1. New row in the `dev-notes/design/design.md` decision table, with a
   Why/Detail link
2. Rationale in the relevant topical design doc — or a proposal under
   `dev-notes/plans/` for subsystem-scale work (archived to
   `dev-notes/archive/design/` once implemented)
3. `docs/design/` update if the rationale is user-facing

The rejected alternatives from the synthesized package go into the rationale
doc. That nuance is a large part of why architect was run at all.

Completion criterion: a new or amended `design.md` row exists, with
rationale in the topical doc (or a plan file for subsystem-scale work).

## Gotchas

- Benchmark numerical choices only after the candidates satisfy the governing constraints.
  Backend semantics are design decisions; faster timings do not authorize changing them.
  Attach a benchmark plan for choices that remain open.
- Don't let the sketch drift into implementation: bodies stay
  `NotImplementedError` until Phase D, or the "design" is just code review
  of premature code.
