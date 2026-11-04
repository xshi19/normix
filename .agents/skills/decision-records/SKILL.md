---
name: decision-records
description: "Use when recording consequential choices or inconclusive alternatives, or recovering prior authority before proposing design changes."
---

# Decision Records (ADR)

- Search canonical decisions and topical rationale using subsystem names and synonyms; follow supersession links, scope, and reopening conditions. Recency does not confer authority; obsolete history is not a veto.
- If pstack is installed, run `why` first when asked why a formula or design was chosen to find the existing record.
- Reuse the repository's format and index. If none exists, use `docs/decisions/NNNN-short-title.md`, a small index, and [references/template.md](references/template.md); otherwise skip the template.
- Preserve significant rejected and inconclusive alternatives even when nothing lands. Routine edits without enduring rationale need no record.
- Keep scope, reason, authority by role/source, and reopening conditions explicit. Separate recording, decision, and evidence dates; unknowns stay unknown. Measurements need revision, conditions, and a reproducible artifact.
- Accept, reject, retire, or supersede only with explicit responsible-authority evidence. A PR, merge, passing checks, or edited record alone supplies none; explicit approval within a PR can.
- Reuse authorization already given. For a current conversation decision, record role/date and `session-reported, no durable source`. A report of an earlier decision stays unconfirmed until authority is established; the drafter is not automatically the decision-maker.
- Distinguish `chosen`, `rejected`, `not selected`, `inconclusive`, and `superseded` alternatives. Record why, dated evidence, and reconsideration conditions. A timeout does not refute an approach; an empirical result does not authorize a decision.
- An accepted constraint remains active while its replacement is proposed. After explicit acceptance, link both records, mark the old one superseded with date/source, state any unreplaced scope, and update the index.
- Correct typos and links in place; date later evidence as addenda. Preserve original rationale and make current status, scope, alternatives, and reopening conditions discoverable.
