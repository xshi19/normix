---
name: interrogate
description: >-
  Adversarial review of a diff. Use for changes touching
  normix/utils/bessel.py, the GIG/GH solvers, the EM fitter, or any
  @jax.custom_jvp function, and on request ("interrogate", "adversarial
  review", "stress test this", "tear this apart"). Default tier: Cursor
  Grok 4.7 reviewers with distinct lenses. Escalation: hand off to Normix
  Engineer. Skip for docs-only or typing-only changes.
---

# Interrogate

Reviewers share one diff and one intent. On the default tier, every
reviewer is Cursor Grok 4.7 (`grok-4.7-xhigh`). Diversity comes from
distinct lens prompts. Agreement across lenses is the high-confidence
signal. A finding from one lens is lower confidence.

The deliverable is a synthesized verdict. Do not auto-apply changes.

## Tiers

| Tier | Reviewers | When |
|---|---|---|
| Default | Three reviewers on Cursor Grok 4.7 (`grok-4.7-xhigh`), each with a distinct lens prompt | any requested review that is not an escalation |
| Escalation | Hand off to Normix Engineer. It runs Codex GPT and Cursor Grok reviewers outside this repo. | the diff touches `normix/utils/bessel.py`, a GIG or GH solver, the EM fitter, or a `@jax.custom_jvp` function; or the user asks for a cross-family review |

Escalate for a sign error or a wrong Bessel step. A second model family
catches that class of bug. On that tier, do not spawn reviewers here.
Prepare the scope and intent package, then stop.

## Steps

1. **Scope.** The diff: files the user points at, or
   `git diff master...HEAD` on a feature branch. Package it with the
   surrounding context reviewers need. Include the touched class and its
   base. For numerical work, include the relevant `docs/theory/` or
   `dev-notes/tech_notes/` page. Without that page, every reviewer
   re-derives the math badly.

   Completion criterion: reviewers receive the diff plus the class/base
   and the theory or tech-note page the change depends on.
2. **Intent.** One paragraph derived from the user's message, commits, and
   the code: what is this change trying to accomplish? Reviewers challenge
   the execution, never the intent. If the intent is unclear, ask before
   you spawn or hand off.

   Completion criterion: one intent paragraph exists; reviewers will
   challenge execution against it, not the goal.
3. **Spawn.** On the default tier, spawn three readonly reviewers in one
   message. Use `subagent_type: generalPurpose`, `readonly: true`, and model
   `grok-4.7-xhigh`.
   Fill `references/reviewer-prompt.md` with the intent and the diff. Give
   each reviewer a different lens from that file. On the escalation tier,
   do not spawn. Hand the same package to Normix Engineer and stop.

   Completion criterion: three `grok-4.7-xhigh` reviewers run on distinct
   lenses, or the escalation hand-off is ready.
4. **Synthesize.** Deduplicate findings that reviewers describe in
   different words. Mark a consensus finding when 2 or more reviewers
   raise it. List explicit disagreements.

   Completion criterion: consensus findings are marked, and disagreements
   are listed.
5. **Lead judgment.** You are a pragmatic lead, not an aggregator.
   Reviewers saw a slice; you have `design.md`, the conversation, and the
   call sites. Filter: hypotheticals whose input path can't occur (trace
   the call site), premature-abstraction advice, "I'd have done it
   differently". When dismissing a finding that a decision row settles,
   cite the row. Be slow to dismiss numerical-correctness findings — when
   a reviewer claims a formula error, check against the theory page or the
   cited reference before ruling.

   Completion criterion: every finding is in Act on / Consider / Noted /
   Dismissed; dismissals that a `design.md` row settles cite the row.

## Output

- **Intent** — the paragraph from step 2
- **Reviewers** — one line each: lens, model, and finding count
- **Act on** — real correctness or maintainability issues. Name the
  reviewers who raised each one. Say why it matters. More than five items
  usually means the filter is too weak.
- **Consider** — legitimate but with unclear cost/benefit; the tradeoff
- **Noted** — valid, low priority
- **Dismissed** — with one-line reasons; this is the trust mechanism that
  lets the user override your judgment
- **Agreement map** — where reviewers agreed and diverged, and what the
  pattern says

Completion criterion: the user can act on the Act-on list without
re-reading the reviewer transcripts.

## Gotchas

- A finding that contradicts a `design.md` row isn't automatically wrong —
  but dismissing it requires citing the row, not vibes.
- The verdict is not a merge gate; the user decides. Act-on items they
  decline become Noted, not silently dropped.
- Do not run the escalation tier on both the sketch and the diff.
  Design time means architect Phase C. Review time is the other chance.
  Once is enough, unless the user asks for both.
