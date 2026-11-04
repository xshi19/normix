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

Reviewers share the diff, user goal, and governing constraints.
On the Cursor default tier, every reviewer uses Cursor Grok 4.7 (`grok-4.7-xhigh`).
Diversity comes from
distinct lens prompts. Agreement helps prioritize checks; it does not settle correctness.
Verify a cited contract violation even when only one lens reports it.

That Cursor tier is not available in Codex.
A sequential lens pass there is a preliminary check.
It is not independent cross-family review.

The deliverable is a synthesized verdict. Do not auto-apply changes.

Every finding must carry a reproducible run against the reviewed revision.
Record the exact command, observed output, and expected behavior under the governing contract.
Include any repro script and inputs needed to repeat the command.
For a design or plan, run a prototype or probe the current tree.
Omit claims that no run demonstrates.
Include these requirements in the external review package.

## Tiers

| Tier | Where | Reviewers | When |
|---|---|---|---|
| Default | Cursor | Three reviewers, model `grok-4.7-xhigh`, distinct lenses | a requested review that is not an escalation |
| Default | Codex | One sequential lens pass with the host model. Label it preliminary. | same, when the host is Codex |
| Escalation | either | Do not spawn reviewers. Prepare the § Host support package and stop. | the diff touches `normix/utils/bessel.py`, a GIG or GH solver, the EM fitter, or a `@jax.custom_jvp` function; or the user asks for a cross-family review |

Escalate for a sign error or a wrong Bessel step. A second model family
catches that class of bug. On that tier, do not spawn reviewers here.
Prepare the package in `dev-notes/design/agent_instructions_design.md` § Host support.
The user relays it. Stop.
The state is "ready for external review".

## Steps

1. **Scope.** The diff: files the user points at, or
   `git diff master...HEAD` on a feature branch. Package it with the
   surrounding context reviewers need. Include the touched class and its
   base. For numerical work, include the relevant `docs/theory/` or
   `dev-notes/tech_notes/` page. Without that page, every reviewer
   re-derives the math badly.

   Include governing decision rows from the base revision and their topical rationale.
   Include related unchanged rows and relevant rejected alternatives.
   Mark proposed decision edits separately from the constraints they would replace.

   Completion criterion: reviewers receive the code, numerical context, and the pre-change contract with citations.
2. **Goal and constraints.** State the user's requested outcome and its source.
   List the governing constraints with decision IDs.
   Describe the author's mechanism separately.
   Cite any user authorization to change a constraint; otherwise mark that change unresolved.
   Reviewers must challenge the mechanism and any conflict with those constraints.

   Completion criterion: the packet distinguishes the user goal, constraints, proposed mechanism, and authority for changes.
3. **Review.**
   **Cursor only.** Spawn three readonly reviewers in one message.
   This spawn is a Cursor `Task` call.
   Use `subagent_type: generalPurpose`, `readonly: true`, and model `grok-4.7-xhigh`.
   That `subagent_type`, `readonly`, and model recipe is Cursor-only.
   Fill `references/reviewer-prompt.md` with the diff and context.
   Replace its entire Intent section with step 2's goal, constraints, mechanism, and authority.
   Include this instruction: "Challenge the mechanism and any conflict with the constraints."
   Give every reviewer that contract check, then assign a distinct lens from the template.
   **Codex.** Use the tools this host has.
   You may apply the lenses in sequence with the host model.
   Label that pass preliminary.
   That pass is not independent cross-family review.
   **Escalation, either host.** Do not spawn reviewers.
   Prepare the package in `dev-notes/design/agent_instructions_design.md` § Host support.
   The user relays it. Stop.
   The state is "ready for external review".

   Completion criterion: Cursor ran three `grok-4.7-xhigh` reviewers on distinct lenses, or Codex recorded a preliminary pass.
   Otherwise the escalation package is ready for external review.
4. **Synthesize.**
   **Cursor.** Deduplicate findings that reviewers describe in different words.
   Mark a consensus finding when 2 or more reviewers raise it.
   List explicit disagreements.
   **Codex.** You used one model.
   Do not mark consensus.
   Do not write an agreement map.
   This pass is not cross-family review.

   Completion criterion: Cursor marks consensus and lists disagreements.
   Codex lists lenses checked and does not claim reviewer agreement.
5. **Lead judgment.** You are a pragmatic lead, not an aggregator.
   Reviewers saw a slice; you have `design.md`, the conversation, and the
   call sites. Filter: hypotheticals whose input path can't occur (trace
   the call site), premature-abstraction advice, "I'd have done it
   differently". When dismissing a finding that a decision row settles,
   cite the row. Be slow to dismiss numerical-correctness findings — when
   a reviewer claims a formula error, check against the theory page or the
   cited reference before ruling.

   Check the proposal against related decisions, including rows outside the diff.
   A rewritten decision row cannot authorize itself.

   Rerun every finding's command against the reviewed revision before retaining it.
   Keep only findings whose rerun confirms a failure of the goal or governing contract.
   Drop findings that fail to reproduce or cannot be rerun, with the reason recorded under Dismissed.

   Completion criterion: every retained finding has a confirming lead rerun and contract evidence; every dismissal has a reason.

## Output

Report the reviewed revision and review state under `dev-notes/design/agent_instructions_design.md` § Host support.
Separate test results from preliminary checks and returned external review.
For each retained finding, include the command, observed lead-rerun output, and expected behavior.

**Cursor.** The bullets below are the three-reviewer tier.

- **Goal and constraints** — the packet from step 2
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

**Codex preliminary output.** Start with `Preliminary. One model. Not cross-family review.`
One host model checked the lenses.
Do not name independent reviewers.
Do not write an agreement map.

- **Goal and constraints** — the packet from step 2
- **Lenses checked** — one line each: the lens, and whether it raised a finding
- **Act on** — correctness or maintainability issues from those lenses
- **Consider** — legitimate, but the cost is unclear
- **Noted** — valid, and low priority
- **Dismissed** — one-line reasons

Completion criterion: the user can act on the Act-on list without
re-reading the lens notes or the reviewer transcripts.

## Gotchas

- A finding that contradicts a `design.md` row isn't automatically wrong —
  but dismissing it requires citing the row, not vibes.
- The verdict is not a merge gate; the user decides. Act-on items they
  decline become Noted, not silently dropped.
- A sequential lens pass is not cross-family review.
  With no returned synthesis, the state is "ready for external review".
- Do not run the escalation tier on both the sketch and the diff.
  Design time means architect Phase C. Review time is the other chance.
  Once is enough, unless the user asks for both.
