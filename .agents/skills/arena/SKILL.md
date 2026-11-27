---
name: arena
description: >-
  Prepare the review package for an independent comparison of design candidates.
  Include the task, grounding, artifact shape, and a 3–6 item rubric.
  Use benchmarks for numerical-method choices.
disable-model-invocation: true
---

# Arena

Prepare the review package described in
[External review](../../../dev-notes/design/agent_instructions_design.md#external-review).
Report “ready for external review” until the independent written synthesis is available.
Record which candidates contributed and any missing or substituted reviews.

Read [benchmark evidence](../../../dev-notes/skill-overlays/principles.md#benchmark-evidence).

## Gotchas

- Do not use a bake-off to settle a numerical choice (solver, backend,
  tolerance). Benchmarks decide those. A bake-off may widen the candidate
  list. The benchmark still decides.
