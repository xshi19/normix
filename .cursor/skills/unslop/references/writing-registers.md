# normix writing registers

Four surfaces, four voices. Identify the register before writing or
editing. The unslop patterns and carve-outs apply to all four.

## `dev-notes/` — engineer's notebook (internal, agent-facing)

- Decision-oriented: tables and short bullets over prose; sentence
  fragments are fine inside tables.
- Terse — every line costs agent context. Link, don't inline (per-subdir
  rules in `.cursor/rules/maintain-design-docs.mdc`).
- Em dashes, decision-row shorthand (F6, S5), and TODO markers all fine.
- No audience-pleasing: no warm-up intros, no restating what the reader
  just read.

## Docstrings (`normix/**/*.py` → autodoc → public API reference)

- NumPy style per `.cursor/rules/coding-conventions.mdc`: one-line
  imperative summary, blank line, detail; Parameters / Returns / Notes.
- Every formula in `:math:` or `.. math::`; parameters state their domains
  (`alpha : float` — shape, $\alpha > 0$).
- No derivations — link the theory page with `:doc:`. No first person, no
  project history, no marketing adjectives.
- Voice: reference manual. The caller's contract, nothing else.

## `docs/tutorials/` — MyST executable narrative

- "We" voice; short prose between cells; each cell and each sentence earns
  its place.
- Key formulas inline (```{math}`), full derivations linked to
  `docs/theory/` — selective enrichment, not duplication (per
  `.cursor/rules/maintain-theory-docs.mdc`).
- Plots are the evidence: say what the figure shows and why it matters,
  never "beautiful plot!".
- Friendly but not chatty: no "Let's dive in!", no exclamation-mark
  enthusiasm.

## `docs/theory/` — formal derivations (MyST, paper-appendix voice)

- Notation defined before use; assumptions and parameter domains stated up
  front.
- "We" active voice ("we show", "we obtain"); full sentences; derivations
  in prose and display math, not bullet chains.
- ```{math}` blocks; `:label:` + `{eq}` only for equations that are
  referenced; cite [Shi2016] via the citation defined in `index.md`.
- Pure math: implementation details go to `dev-notes/tech_notes/`, per
  `.cursor/rules/maintain-theory-docs.mdc`.

## Quick test

Read the paragraph aloud: would a careful human author *of this register*
have written that sentence? A docstring sentence in tutorial voice ("we can
now explore…") or a theory sentence in marketing voice ("the powerful GH
family") fails the test.
