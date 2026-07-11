---
name: arena
description: >-
  Fan out parallel design candidates across the strong model panel,
  cross-judge, pick a base, graft the best ideas from the losers. Called by
  the architect skill for one-way-door designs; invoke directly only for an
  explicit "arena this" / multi-model bake-off. Not for numerical-method
  choices — benchmarks decide those.
disable-model-invocation: true
---

# Arena

Fan out N parallel attempts at the same task, cross-judge, pick the strongest
as base, graft the best ideas from the rest. Expensive by design: use for
one-way-door structural decisions, roughly 1–2 invocations per quarter, not
routine work.

## Model panel (Cursor only)

One candidate per model. This table is the canonical strong panel — the
interrogate skill's escalation tier points here.

| Role | Model slug |
|---|---|
| Runner 1 | `claude-fable-5-thinking-max` |
| Runner 2 | `gpt-5.6-sol-max` |
| Runner 3 | `grok-4.5-fast-xhigh` |
| Cross-judge | one model from the cheap trio (`composer-2.5-fast`, `claude-sonnet-5-thinking-high`, `gpt-5.6-terra-medium`), family different from the parent's |

If a slug is rejected when spawning, pick the closest same-family slug from
the Task tool's error message, proceed, and update this table afterwards.
Other harnesses (Claude Code, Codex) cannot spawn cross-family subagents;
there, run N candidates on the models available and note the reduced
diversity in the synthesis note.

## Phases

Open a todo per phase: Frame, Fan out, Cross-judge, Pick, Graft, Verify.

1. **Frame.** All candidates get the same prompt, so the prompt is the
   contract. State the artifact each candidate produces, then derive a
   rubric of 3–6 concrete gradeable criteria ("the sketch keeps `grad_fn`
   in θ-space per row S2", not "code is correct"). Candidates see only the
   task; the rubric is the judge's and picker's tool.
2. **Fan out.** Spawn all runners in a single message,
   `subagent_type: generalPurpose`, `run_in_background: true`. For design
   sketches (the architect case): `readonly: true`, package returned in the
   response body. For code candidates: `best-of-n-runner` subagents instead
   (each gets its own worktree and branch). A rationale naming the
   alternatives each candidate considered and rejected is mandatory —
   without it the parent cannot tell principled structure from accident.
   A dropout means proceed with N−1 and note it.
3. **Cross-judge.** After all candidates complete (never while they are
   still writing), spawn one readonly judge with the rubric and the
   candidates by label. It scores each criterion and recommends a base.
4. **Pick.** Read every candidate end to end — skimming surfaces only the
   most familiar-looking one. Score criterion by criterion, compare with the
   judge. Agreement confirms the base; disagreement means bias or an
   ambiguous rubric — re-read both rationales before deciding. Tie-break
   toward the smaller surface a future maintainer can extend without
   breaking invariants.
5. **Graft.** Walk each loser once more; the yield is usually one or two
   ideas per candidate. Fold grafts in by hand so the result stays coherent
   under one mental model. Record grafts, and record rejections with
   reasons — the rejection notes are the highest-signal part of the record.
   All candidates converging on one shape = strong consensus, ship it;
   wild divergence = Phase 1 was under-specified, reframe and re-run.
6. **Verify.** The synthesized artifact gets normal scrutiny (principles
   skill § Prove it works). The arena does not earn a pass.

## Output

One synthesized artifact plus a synthesis note: base, grafts (with source
candidate), rejections (with reasons), dropouts, verification result. When
called from architect, the note fills the design package's "Synthesis
decision" section and feeds the `dev-notes/design/design.md` row.

## Gotchas

- Don't use arena to settle numerical questions (solver choice, backend,
  tolerances): rows S1–S9 were decided by benchmarks. Arena may widen the
  candidate list; the benchmark still decides.
- Don't spawn the judge in parallel with the runners — it reads partial
  output and reports it as dropouts.
- N candidates writing files into one shared directory is shared mutable
  state; give each its own path or worktree.
