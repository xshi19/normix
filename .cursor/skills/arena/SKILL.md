---
name: arena
description: >-
  Retired inside this repo. Do not run a multi-model bake-off in a Cursor
  or Codex session. Normix Engineer runs that bake-off outside the repo.
  Read this only to prepare the hand-off package: task, grounding,
  artifact shape, and a 3–6 item rubric. Not for numerical-method choices.
disable-model-invocation: true
---

# Arena

Do not run this skill in a Cursor session or a Codex session.
Normix Engineer runs multi-model bake-offs outside this repo.
It runs one Codex GPT candidate, one Cursor Grok candidate, and its own
candidate. It compares the three and grafts a synthesis.

## Hand-off contract

Prepare this package for Normix Engineer. Then stop.

- **Task.** State the design question in one place.
- **Grounding.** State the constraints the candidates must honor. For an
  architect hand-off, use the Phase A notes.
- **Artifact shape.** State what each candidate returns. For a design
  sketch, include
  `.cursor/skills/architect/references/runner-prompt.md` and
  `.cursor/skills/architect/references/rationale-template.md`.
- **Rubric.** Write 3–6 concrete gradeable criteria. Candidates do not
  see the rubric. Write "the sketch keeps `grad_fn` in θ-space per row
  S2". Do not write "the code is correct".

Normix Engineer returns one synthesized package. That note names the
base, the grafts, and the rejections. Record it. Do not treat the
bake-off as a pass (principles skill § Prove it works).

## Gotchas

- Do not use a bake-off to settle a numerical choice (solver, backend,
  tolerance). Benchmarks decide those. A bake-off may widen the candidate
  list. The benchmark still decides.
