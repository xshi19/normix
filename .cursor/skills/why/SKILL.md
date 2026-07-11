---
name: why
description: >-
  Design history and rationale: "why does X work this way", "why did we
  pick Y", what alternatives were rejected, where a threshold came from.
  Searches normix's design docs, archived proposals, tech notes, and git
  history — no MCP fan-out. Use the how skill for current behavior.
---

# Why

Answer motivation questions from the written record, with citations and
calibrated confidence. The code tells you what it does, rarely why it
exists — don't infer intent from code shape.

## Source order

Work down; stop when answered, and note which sources you didn't reach.

1. `dev-notes/design/design.md` — the decision table. Most "why X"
   questions are a row here; follow its Why/Detail link.
2. `dev-notes/design/index.md` quick lookups → topical doc
   (`exponential_family.md`, `mixtures.md`, `em_framework.md`,
   `solvers_and_bessel.md`)
3. `dev-notes/archive/design/` and `dev-notes/archive/plans/` — superseded
   proposals: what was tried and rejected, and the shape of earlier
   thinking
4. `dev-notes/tech_notes/` — numerical deep dives with benchmark evidence
   (where thresholds and constants come from)
5. Git: `git log --follow -p -- <file>`, `git blame -L <start>,<end>
   <file>`, commit bodies; `gh pr view <n> --json title,body,comments,
   reviews` when authenticated
6. `docs/design/` — the published, user-facing voice of the same decisions

This is a reading task; do it inline by default. For an archaeology sweep
across many files, one readonly subagent keeps the parent context clean.

## Epistemics (the product)

- **Cite everything**: design.md row number, doc § heading, commit hash,
  PR number. An uncited claim is inference and must be labeled as such.
- **Hedge deliberately**: "appears to", "likely" for indirect evidence;
  confident language only for direct statements. Don't strip hedges to
  sound authoritative — the calibration is the value.
- **Surface contradictions**: an archived proposal conflicting with the
  current design usually means the design evolved. Check dates; present
  both.
- **Name gaps honestly**: "not recorded in the design docs or git history"
  beats a plausible story.
- If the user proposes a reason ("for performance, I assume?"), treat it
  as a hypothesis to check against the record, not to confirm.

## Output

- **Question** and **the code/decision in question** — one line each
- **What the record says** — cited findings
- **What's inferred** — with the inference chain ("given A and B, likely C")
- **Gaps** — what wasn't found and where you looked
- **Sources consulted** — one line per source, including the empty ones

## Gotchas

- Recency bias: the current shape is an accretion of earlier decisions;
  trace back past the most recent commit.
- design.md rows are dense — follow the Why/Detail link before declaring a
  gap.
- Commit messages can lie and proposals change between draft and
  implementation; prefer the merged diff and the archived final version.
