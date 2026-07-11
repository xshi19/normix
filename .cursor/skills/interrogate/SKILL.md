---
name: interrogate
description: >-
  Multi-model adversarial review of a diff. Use for changes touching
  normix/utils/bessel.py, the GIG/GH solvers, the EM fitter, or any
  @jax.custom_jvp function, and on request ("interrogate", "adversarial
  review", "stress test this", "tear this apart"). Cheap cross-family panel
  by default, escalated to the strong arena panel for deep-math diffs. Skip
  for docs-only or typing-only changes.
---

# Interrogate

One reviewer per panel model, all given the same prompt and rubric. The
adversarial signal comes from model diversity — models differ in blind
spots and priors, so agreement across models is high-confidence signal and
lone-model findings are lower confidence. The deliverable is a synthesized
verdict. Do NOT auto-apply changes.

## Panels (Cursor only)

| Tier | Reviewers | When |
|---|---|---|
| Default | `composer-2.5-fast`, `claude-sonnet-5-thinking-high`, `gpt-5.6-terra-medium` | any requested review |
| Strong | the arena panel — `.cursor/skills/arena/SKILL.md` | diff touches `normix/utils/bessel.py`, a `@jax.custom_jvp` function, solver convergence logic, or posterior-conjugacy / M-step math |

The strong tier exists because a sign error in an asymptotic series or a
wrong Bessel-recurrence step is precisely what cheap reviewers miss; for
everything else, the cheap trio already covers three model families and
the consensus mechanism does the work. If a slug is rejected when
spawning, pick the closest same-family slug from the Task tool's error
message, proceed, and update this table afterwards. In other harnesses
(Claude Code, Codex) run the reviewers on available same-family models and
note the reduced diversity.

## Steps

1. **Scope.** The diff: files the user points at, or
   `git diff master...HEAD` on a feature branch. Package it with the
   surrounding context reviewers need — the touched class and its base,
   and for numerical work the relevant `docs/theory/` or
   `dev-notes/tech_notes/` page (without it every model re-derives the
   math, badly).
2. **Intent.** One paragraph derived from the user's message, commits, and
   the code: what is this change trying to accomplish? Reviewers challenge
   the execution, never the intent. If the intent is unclear, ask before
   spawning.
3. **Spawn.** One reviewer per panel model in a single message,
   `subagent_type: generalPurpose`, `readonly: true`. Fill
   `references/reviewer-prompt.md` with the intent, the diff plus context,
   and pass it identically to every reviewer.
4. **Synthesize.** Deduplicate findings described differently by different
   models; mark consensus findings (2+ models); note explicit
   disagreements — they are useful context for the verdict.
5. **Lead judgment.** You are a pragmatic lead, not an aggregator.
   Reviewers saw a slice; you have `design.md`, the conversation, and the
   call sites. Filter: hypotheticals whose input path can't occur (trace
   the call site), premature-abstraction advice, "I'd have done it
   differently". When dismissing a finding that a decision row settles,
   cite the row. Be slow to dismiss numerical-correctness findings — when
   a reviewer claims a formula error, check against the theory page or the
   cited reference before ruling.

## Output

- **Intent** — the paragraph from step 2
- **Reviewers** — one line each: model, N findings
- **Act on** — real correctness/maintainability issues; description, which
  models raised it, why it matters (more than ~5 items usually means
  under-filtering)
- **Consider** — legitimate but with unclear cost/benefit; the tradeoff
- **Noted** — valid, low priority
- **Dismissed** — with one-line reasons; this is the trust mechanism that
  lets the user override your judgment
- **Agreement map** — where models agreed and diverged, and what the
  pattern says

## Gotchas

- A finding that contradicts a `design.md` row isn't automatically wrong —
  but dismissing it requires citing the row, not vibes.
- The verdict is not a merge gate; the user decides. Act-on items they
  decline become Noted, not silently dropped.
- Don't run interrogate on the sketch *and* the diff of the same change
  with the strong panel twice — once at design time (architect Phase C) or
  once at review time is enough unless the user asks.
