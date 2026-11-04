---
name: notation-economy
description: Use when finalizing mathematical exposition (notes, papers, docs, proofs) for readers — audit the final draft and remove auxiliary symbols and aliases that do not earn their place, while preserving every mathematical fact. Not for scratch work or in-progress derivations.
---

# Notation economy

## When

Apply to the **final draft** of math exposition, after claims and proofs have settled. Scratch work and intermediate derivations may use any helpful notation. Audit only what the reader will see.

## When a symbol earns its place

- Prefer the project's notation canon, then established notation from a relevant textbook or paper; record the source and mapping when conventions differ. Standard usage elsewhere does not by itself justify an alias here.
- Introduce an auxiliary symbol only if it names a meaningful object used substantively later, or substantially shortens or clarifies otherwise unwieldy formulas. Recurrence alone is not enough: a letter standing for a short, readable expression still costs the reader a lookup.
- Judge the rendered formula plus the cost of remembering the definition, not LaTeX characters saved. Prefer an explicit indicator or survival probability when its event is the point.
- Keep necessary primary variables, parameters, bound indices, and named objects. This is an audit of avoidable notation, not a symbol quota. A proof-local abbreviation may stay in the proof without entering the statement.
- Define retained symbols near first substantive use, with meaning and scope. No alias chains or decorative subscripts.

## Final-draft audit

1. Inventory introduced symbols. Read the whole page (and related pages) before calling anything redundant; separate substantive uses from definitions, restatements, and unrelated reuse of a letter.
2. For each auxiliary symbol: keep it for a stated reader benefit, inline it, or move its definition to where it is needed. Scrutinize symbols used once or twice; a count is a signal, never a deletion rule. Never add uses to meet a threshold.
3. Compare revision with original claim by claim. Preserve assumptions, quantifiers, domains, units, parameter dependence, conditional vs unconditional quantities, strict vs non-strict events, endpoint atoms, positivity/finiteness conditions, and possibly infinite expectations. Do not create undefined products, quotients, or ∞ − ∞.
4. Lose no fact, caveat, attribution, or proof dependency along with its shorthand. Update all affected uses, captions, labels, and cross-references. If meaning or scope is unclear, settle it before simplifying.
5. Reread in order and check the rendered formulas. The result should reduce lookup effort without creating long unreadable expansions. Record non-obvious decisions in the review, not as a notation table on the page.

## Example

Before:

> Let $A = \{X > u\}$, $p = P(A)$, and $m = E[(X-u)_+]$. Then $E[X 1_A] = u p + m$.

After:

> $E[X 1_{\{X>u\}}] = u \bar F(u) + E[(X-u)_+]$, where $\bar F(u) = P(X>u)$.

$A$, $p$, $m$ are each used once and hide the event that is the point; the survival function $\bar F$ is standard and stays. The strict inequality $X > u$ is preserved.

## Optional audit script

`scripts/notation_audit.py` (Python 3, stdlib only) lists symbols introduced by cues like "Let/Define/denote" or `:=` and counts their later in-page occurrences. It is report-only: it never edits and always exits 0 on findings. It counts in-page uses only, misses prose-only uses, macros, and cross-page use, and **cannot catch frequently used aliases** — an alias used ten times can still be needless. Treat its output as candidates for human judgment.

```
python3 scripts/notation_audit.py path/to/page.md   # files or directories (*.md)
```

## Basis

- Halmos, [*How to Write Mathematics*](https://sites.math.washington.edu/~lind/Resources/Halmos.pdf), §15 "Resist symbols" and §5.
- Knuth, Larrabee, Roberts, [*Mathematical Writing*](https://jmlr.csail.mit.edu/reviewing-papers/knuth_mathematical_writing.pdf), §§1 and 5.
- Geyer, [notes on mathematical writing](http://users.stat.umn.edu/~geyer/math.pdf).
