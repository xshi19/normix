---
name: agent-maintenance
description: >-
  Maintain agent-facing knowledge: rules, docs, AGENTS.md, ARCHITECTURE.md,
  skills. Use when adding new modules, distributions, constants, or design
  decisions. Use when docs may be stale or rules may contradict. Use when
  the user asks to update or audit documentation, rules, or skills. Also
  use to reflect after a long or corrective session ("reflect") and route
  lessons into rules and skills.
---

# Agent Maintenance

> High-level principles: `dev-notes/design/agent_instructions_design.md`

This skill codifies **when and how** to update the agent-facing knowledge
system. Each trigger below tells you what to update and where.

## Triggers

### New Module or File Added Under `normix/`

1. `dev-notes/ARCHITECTURE.md` — update module hierarchy tree
2. If it's a distribution → also follow "New Distribution Added" below
3. If it adds a new area of work → add a row to the context map in `AGENTS.md`

### New Distribution Added

1. `.cursor/rules/project-overview.mdc` — add to distribution catalog table
2. `dev-notes/ARCHITECTURE.md` — add to module hierarchy and distribution storage table
3. `dev-notes/design/design.md` — if new design decisions were made, add to decision table
4. Add tests under `tests/` (see `.cursor/rules/testing-guidelines.mdc`)
5. Add notebook under `notebooks/` (see `.cursor/rules/notebook-guidelines.mdc`)
6. Add theory derivation under `docs/theory/` if mathematical background is needed

### New Numerical Constant

1. Define in `normix/utils/constants.py` (canonical location)
2. `dev-notes/ARCHITECTURE.md` — update numerical constants table
3. `.cursor/rules/coding-conventions.mdc` — update constants table if commonly used

### Design Decision Made

1. `dev-notes/design/design.md` — add row to the decision table
2. If architecturally significant → update `dev-notes/ARCHITECTURE.md`
3. If user-facing rationale → also update the relevant file under `docs/design/`
4. If it changes how code should be written → update relevant `.cursor/rules/` file

### Agent Makes a Recurring Mistake

1. Add to the "Things to Avoid" / "Gotchas" section of the relevant rule
2. If no relevant rule exists, consider whether the mistake warrants a new rule

### Advancing a Plan Phase

1. Update the relevant plan under `dev-notes/plans/` (e.g. `docs_refactor.md`,
   `finance_architecture.md`). Completed plans live in `dev-notes/archive/plans/`.
2. If modules were restructured → update `dev-notes/ARCHITECTURE.md`

### New Multi-Step Workflow Identified

1. Check existing skills — does one already cover this?
2. If repeated >3 times or complex enough → create a skill in `.cursor/skills/`
3. Follow `.cursor/rules/maintain-skills.mdc` for skill structure

### Rule Growing Beyond ~150 Lines

1. Split into sub-rules or move detail to `dev-notes/`
2. Update `.cursor/rules/maintain-cursor-rules.mdc` current rules table

### Rules May Be Contradicting

1. Read all rules in `.cursor/rules/`
2. Identify contradictions
3. Resolve by keeping the more specific/recent intent
4. Remove duplicated content — each fact lives in one place

### After a Long or Corrective Session (Reflect)

Run a reflection pass when the user says "reflect", when a complex task
just landed and the recipe is worth keeping, when the user corrected your
approach mid-task, or when you hit dead ends before finding the working
path. Skip for trivial sessions — one-offs are not lessons.

1. Review the session for three kinds of signal: recurring corrections,
   the dead-end → working-path pattern, and rules/skills that were
   missing, stale, or failed to trigger when they should have.
2. Structural check first: if a lesson can be a test, a constant, a lint,
   or a glob fix, encode that instead of prose (principles skill § Encode
   lessons in structure). The instruction is often the symptom.
3. Route each surviving lesson through the triggers above (recurring
   mistake → rule gotcha; new multi-step workflow → skill; design insight
   → `design.md` row).
4. Present the proposed edits as an Accepted / Rejected list and wait for
   approval before applying — rule and skill edits affect every future
   session. Note dropped findings with a one-line reason.

## Single Source of Truth

Before updating anything, check that you're updating the **canonical location**:

| Fact | Canonical Location |
|---|---|
| Module hierarchy, distribution storage | `dev-notes/ARCHITECTURE.md` |
| Distribution catalog | `.cursor/rules/project-overview.mdc` |
| Coding conventions | `.cursor/rules/coding-conventions.mdc` |
| Numerical constants (values) | `normix/utils/constants.py` |
| Design rationale, decision table | `dev-notes/design/design.md` |
| Published design rationale | `docs/design/` |
| Commands, context map | `AGENTS.md` |
| Mathematical derivations | `docs/theory/` |

If the fact already exists elsewhere, **add a pointer**, don't duplicate.

## Pre-Update Checklist

Before making any documentation update:

- [ ] Read the target file first — is the content already there?
- [ ] Check related files — will this addition contradict something?
- [ ] Verify existing entries are still accurate — fix stale entries while you're there
- [ ] Follow the appropriate `maintain-*.mdc` rule for the file you're editing

## Gotchas

- **Don't update AGENTS.md for every change.** Only when the map needs a new entry.
- **Don't add implementation details to rules.** Rules say *what* and *what not*.
- **`project-overview.mdc` is always loaded.** Every line costs context in every conversation.
- **Check before creating.** The most common maintenance mistake is adding something that already exists in a different file.
- **Never link `dev-notes/` from `docs/` or normix docstrings** — promote content to `docs/` first.
