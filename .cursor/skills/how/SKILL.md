---
name: how
description: >-
  Codebase walkthroughs: "how does X work", subsystem architecture,
  placement and layering questions ("where should this live", "is this the
  right layer"), and optional architecture critique. One agent, no
  fan-out. Use the why skill for history and motivation.
---

# How

Answer "how does X work?" at onboarding grade: enough for a working mental
model, not annotated source code. Two modes — **Explain** (default) and
**Critique** (when asked for architectural issues or improvements).

## Ground in docs first

normix documents itself; read before exploring code:

- `dev-notes/design/index.md` quick-lookups table → the topical design doc
- `dev-notes/ARCHITECTURE.md` — module hierarchy and storage tables
- `docs/design/` — published rationale

Then read the actual code (Grep + Read). Don't guess from file names.

## One agent, two depths

- **Small question** (one module, one function, one flow): answer inline —
  normix is small enough that Grep + Read beats spawning anything.
- **Subsystem question** (the Joint/Marginal mixture machinery, the φ↔θ
  solver chain rule, the EM flow end to end): spawn ONE readonly subagent
  (default model `claude-sonnet-5-thinking-high`) that explores and writes
  the explanation, keeping the bulk file reads out of the parent context.
  No parallel explorers.

If the question's scope is ambiguous, state your best-guess interpretation
and proceed; let the user redirect.

## Explain output

Adapt — not every section is needed for every question:

- **Overview** — what it is and why it exists, 1–2 paragraphs
- **Key concepts** — only the types/abstractions needed for the rest
- **How it works** — the flow in prose, citing specific files and
  functions; use the θ/η/ψ notation where it clarifies
- **Where things live** — the files needed to start working in this area
- **Gotchas** — non-obvious traps (both-branch `jnp.where` evaluation, the
  CPU/JAX backend split, static vs traced fields, historical residue)

## Critique mode

Explain first — you cannot critique what you don't understand. Then the
same agent (or the parent, for inline answers) reviews the architecture
against: abstraction fit (each layer earning its keep), data-model fit
(pytree shapes vs actual access patterns), boundary discipline (validation
at constructors only), evolution readiness (how much changes when the next
likely distribution or feature lands), complexity vs value, and consistency
with sibling subsystems.

Present the explanation first, then the critique in lead-judgment buckets:
Act on / Consider / Noted / Dismissed. Check `dev-notes/design/design.md`
before flagging something a decision row already settles — cite the row
either way.

## Gotchas

- The explanation must stand on its own; a reader who only wants
  understanding shouldn't wade through critique.
- Cite design.md rows rather than re-deriving rationale — motivation
  questions belong to the why skill.
