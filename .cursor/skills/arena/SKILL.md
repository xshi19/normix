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

## Hand-off

Do not run a bake-off in this repo.
The package, the relay, and the return are in `dev-notes/design/agent_instructions_design.md` § Host support.
Prepare that package. Then stop.
With no returned synthesis, the state is "ready for external review".
Do not treat the bake-off as a pass (principles skill § Prove it works).

## Gotchas

- Do not use a bake-off to settle a numerical choice (solver, backend,
  tolerance). Benchmarks decide those. A bake-off may widen the candidate
  list. The benchmark still decides.
