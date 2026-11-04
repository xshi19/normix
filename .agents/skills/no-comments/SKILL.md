---
name: no-comments
description: >-
  Review inline comments in normix. Report keep, delete, or move-to-doc
  findings. Edit only when the user asks to apply. Use when asked to
  comment a diff, strip comments, or check comment policy. Keep every
  docstring. Protect functional comments. A large diff uses one
  read-only Comment Sicko reviewer.
---

# No comments

Use this skill on library code, tests, scripts, and notebook Python.
The default is a report.
Edit only when the user asks to apply.

## Protect

These comments are never delete candidates.

- A release marker such as `# x-release-please-version` is protected.
- A tooling directive is protected.
  Examples: `noqa`, `type: ignore`, `pragma`, `fmt: off`, and `fmt: on`.
- A pytest marker written in a comment is protected.
- A JAX marker written in a comment is protected.
- A jupytext header is a structure marker.
- A `# %%` cell marker is a structure marker.
- A license header is protected.

## Keep

- Keep every docstring.
  Do not delete a docstring.
  Do not thin a docstring.
- Keep a comment that states a non-obvious correctness constraint.
  Numerical examples: quadrature cancellation, a threshold, a log-space trick.
  Other examples: ordering, a pytree or immutability invariant, a JAX tracing or static-arg gotcha.
- Do not keep a comment when the name is clear.
- Do not keep a comment when the logic is clear.
- A one-line comment may point at the doc for the trap.
  In `normix/`, name a public doc.
  Do not write a `dev-notes/` path in `normix/`.
  `scripts/check_doc_links.sh` rejects that path.

## Move

Put design rationale in a doc or in the commit message.
Do not leave that prose in an inline comment.

| Kind of note | Put it in |
|---|---|
| Decision or rationale | `dev-notes/design/design.md` |
| Derivation | `docs/theory/` |
| Numerical or implementation note | `dev-notes/tech_notes/` |
| Published rationale | `docs/design/` |
| Superseded discussion | `dev-notes/archive/` |
| Why this change | the commit message |

## Delete

- Delete a comment when the name is self-explanatory.
- Delete a comment when the logic is simple.
- Delete a stale comment.
- Do not add a comment that narrates the next line.
- Do not delete a protected comment.

## Steps

1. **Read.** Read the diff or the files the user names.

   Completion criterion: the comment sites are listed.
2. **Report.** For each comment, write `file:line`, then `keep`, `delete`, or `move-to-doc`.
   For `move-to-doc`, name the target file.
   Label a protected comment `keep`.

   Completion criterion: every comment has one of those three labels.
3. **Apply.** Edit only after the user asks to apply the report.

   Completion criterion: the diff matches the approved labels, and every docstring is intact.

## Who reviews

Delegate when the diff touches more than three files.
Also delegate when you cannot list every comment in one pass.
Otherwise, the main agent reviews the diff.

1. **Small diff.** The main agent does the read and the report.

   Completion criterion: the report covers every comment in the diff.
2. **Large diff.** Delegate one read-only reviewer named Comment Sicko.
   Give the reviewer this skill and the diff.
   The reviewer returns `file:line` with `keep`, `delete`, or `move-to-doc`.
   The reviewer does not edit files.
   Spawn that reviewer in Cursor with a `Task` call.
   Use `subagent_type: generalPurpose`, `readonly: true`, and model `grok-4.7-xhigh`.
   That `Task` recipe is Cursor-only.
   Do not use Claude, Gemini, or Composer.
   On Codex, ask in plain language to spawn the `comment_sicko` agent.
   If that agent is absent, ask for the built-in `explorer` agent.
   Wait for the report.
   Do not ask Codex to set a model on `comment_sicko`.
   Codex subagents inherit the parent sandbox.
   A real read-only run needs a read-only parent sandbox.
   The agent toml does not enforce that sandbox.
   Custom agents: https://learn.chatgpt.com/docs/agent-configuration/subagents

   Completion criterion: one reviewer report is back, and no files were edited.
3. **Check.** The main agent checks the report before presenting it.
   Confirm every docstring is `keep`.
   Confirm every protected comment is `keep`.
   Reject a `delete` that drops a trap.

   Completion criterion: the presented report matches this skill.

## Gotchas

- A docstring is not a comment. The delete list does not apply to it.
- Do not move a public formula out of a docstring into `dev-notes/`.
- Do not treat a tooling directive as prose.
