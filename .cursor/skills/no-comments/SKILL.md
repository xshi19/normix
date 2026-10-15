---
name: no-comments
description: >-
  Review inline comments in normix. Report keep, delete, or move-to-doc
  findings. Edit only when the user asks to apply. Use when asked to
  comment a diff, strip comments, or check comment policy. Keep every
  docstring. In Cursor, use Grok 4.7. In Codex, use the host model.
---

# No comments

Comment policy for normix. The default is a report.
Edit only when the user asks to apply.

## Keep

- Keep every docstring. Do not delete a docstring. Do not thin a docstring.
- Keep a comment only for a real trap.
  A trap is a Bessel regime boundary, a numerical threshold, or a log-space trick.
- A one-line comment may point at the doc that explains the trap.

## Move

Put design rationale, a decision, or background in a doc or a commit message.
Do not leave that prose in an inline comment.

| Kind of note | Put it in |
|---|---|
| Decision or rationale | `dev-notes/design/design.md` |
| Derivation | `docs/theory/` |
| Published rationale | `docs/design/` |
| Superseded discussion | `dev-notes/archive/` |
| Why this change | the commit message |

## Delete

- Delete a comment when the name is self-explanatory.
- Delete a comment when the logic is simple.
- Delete a stale comment.
- Do not add a comment that narrates the next line.

## Steps

1. **Read.** Read the diff or the files the user names.

   Completion criterion: the comment sites are listed.
2. **Report.** For each comment, write `file:line`, then `keep`, `delete`, or `move-to-doc`.
   For `move-to-doc`, name the target file.

   Completion criterion: every comment has one of those three labels.
3. **Apply.** Edit only after the user asks to apply the report.

   Completion criterion: the diff matches the approved labels, and every docstring is intact.

## Model

- In Cursor, use Grok 4.7.
- In Codex, use the host model.
- Do not use Claude, Gemini, or Composer.

## Gotchas

- A docstring is not a comment. The delete list does not apply to it.
- Do not move a public formula out of a docstring into `dev-notes/`.
