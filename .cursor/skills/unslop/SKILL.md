---
name: unslop
description: >-
  Strip AI-writing patterns from prose while preserving normix's
  mathematical voice. Use when writing or editing README, docs/ pages,
  docstrings, dev-notes, PR descriptions, or when the user says unslop,
  de-AI, or "this reads like AI". Includes the normix carve-outs and the
  four writing registers (dev-notes, docstrings, tutorials, theory).
---

# Unslop

Cut AI tells; preserve meaning, tone, and the math. Process: identify the
register → scan for patterns → rewrite → self-audit ("what still reads as
AI-generated?").

Pick the register first — what's right in dev-notes is wrong in a
docstring: `references/writing-registers.md`.

## normix carve-outs (override the patterns below)

1. **Em dashes are allowed** — normal in mathematical asides and
   decision-table rationale. Cut only when several pile up in a paragraph.
2. **Mid-sentence colons are allowed** where they introduce an equation,
   definition, or list: "define the log-partition: $\psi(\theta) = \dots$".
3. **Technical senses of flagged words are fine**: *vector* in
   $\mathbb{R}^d$, *multimodal(ity)* for densities, *substrate* in
   `tfp.substrates.jax`, *primitive* for JAX primitives, *surface* as "API
   surface" in dev-notes. Banned only as empty metaphors ("a new vector for
   growth").
4. **"Note that" is fine** — the mathematical convention for flagging a
   subtlety. "It is important to note that" is filler; delete it.
5. **Active voice means "we" in math prose**: "we show", "we integrate by
   parts" — standard mathematical style, not chattiness.
6. **Calibrated hedging is intentional** in design-history answers (the why
   skill's epistemics); don't strip it there.
7. **Heading case**: match the page's existing convention; prefer sentence
   case for new long-form pages; don't churn existing headings.

## Patterns to detect and fix

**Content.** Significance inflation ("pivotal", "testament to", "evolving
landscape") → state what happened. Superficial -ing tails ("…highlighting
the importance of robustness") → delete or say the concrete thing.
Promotional adjectives ("powerful", "seamless", "blazing", "cutting-edge")
→ neutral description or the number. Vague attributions ("it is well
known", "experts note") → cite [Shi2016] or the specific reference, or
delete. Formulaic challenge framing ("despite challenges… continues to") →
specific facts.

**Language.** AI vocabulary (delve, crucial, intricate, interplay,
tapestry, showcase, underscore, foster, leverage, robustify) → plain words:
"leverages the intricate interplay of Bessel asymptotics" → "uses the
Hankel-regime asymptotic". Copula avoidance ("serves as", "boasts",
"features") → "is" / "has". Negative parallelism ("it's not just X, it's
Y") → state the point. Forced groups of three → the natural number.
Synonym cycling → one term per concept (see math rules below). False
ranges ("from Gamma to GH") → list them.

**Style.** Don't bold every noun. Inline-header bullets that restate the
line ("**Performance:** performance improved…") → prose; a bold lead-in
followed by genuinely new detail is fine. No decorative emojis. Straight
quotes.

**Communication artifacts.** No chatbot phrases ("I hope this helps",
"Let's dive in!"), no sycophancy, no knowledge-cutoff disclaimers.

**Filler.** "In order to" → "to"; "due to the fact that" → "because";
"It is important to note that" → delete; stacked hedges ("could
potentially possibly") → "may"; generic conclusions ("the future looks
bright") → specific facts or nothing.

**Plain speech.** Say the mechanism or the number, not the vibe:
"significantly faster" → "15× faster E-step on the SP500 GH benchmark".
One idea per sentence; split anything the reader must re-read. Active
voice: "the solver applies the chain rule", not "the chain rule is
applied". Cut adverbs propping up weak verbs. "utilize" → "use",
"facilitate" → "help", "numerous" → "many".

## Math writing (normix additions)

- **Every symbol is defined at first use** on the page, or points to where
  the notation is established. A formula with an unexplained symbol is a
  broken link.
- **LaTeX everywhere**, per the format table in
  `.cursor/rules/coding-conventions.mdc` (`:math:` in docstrings, `$...$`
  in MyST/markdown). Plain-text math ("alpha > 0") is not acceptable.
- **One symbol per object, one term per concept, across pages**: $\Sigma$
  stays $\Sigma$; "natural parameters", never cycled with "canonical";
  the symbol↔code correspondence holds ($\theta$ ↔ `theta`, $\eta$ ↔
  `eta`, $\psi$ ↔ `log_partition`).
- **Formulas carry their domain**: "for $a, b > 0$". An unstated domain is
  a future bug report.
- **Label equations only when referenced** (```{math}` with `:label:`,
  cross-referenced via `{eq}`); unnumbered display math otherwise.
- **Numerical claims are numbers with sources** (benchmark file, tolerance,
  machine), not adverbs.

## Gotchas

- Docstrings feed Sphinx autodoc: rewriting them edits the public website.
  Keep the NumPy-style section structure intact.
- Don't "fix" quoted output, error messages, or code identifiers.
- Unslop is meaning-preserving: if a rewrite changed a mathematical
  statement, revert it.
