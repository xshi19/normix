# STE-style pass (from ASD-STE100)

ASD-STE100 Simplified Technical English (STE) is a controlled language for
maintenance manuals. It has two parts: writing rules and a dictionary of
about 900 approved words. normix uses the rules only. Call the result
"STE-style". Never call it "STE" or "STE-compliant": without the dictionary,
the vocabulary is a guess.

## When

- Run this pass last. Settle the content first: facts, math, checks. Then
  rewrite the settled text.
- Do not put "answer in STE" in a prompt for research, review, or derivation
  work (arena, interrogate, figure-it-out, why). Simple output written
  before the thinking loses sources and quality.
- If the user asks for a simple or STE explanation, write the full answer
  first. Then rewrite it. Keep every caveat that changes a decision.

## Scope

| Text | Pass | Why |
|---|---|---|
| `.cursor/rules/`, `.cursor/skills/`, `AGENTS.md` | Full | Instructions for an agent; 20% imperative sentences, 2% with math |
| Procedures anywhere: install, commands, how-to steps, PR "how verified" | Full | One action per step is the STE core |
| Chat explanations the user asked to be simple; show-me captions | Full | The use case in Karpathy's tip |
| `docs/design/`, `dev-notes/design/`, `docs/user_guide/` prose, changelog, PR summary | Descriptive rules only | Rationale, not steps; about 1 in 3 published sentences has math |
| Docstrings | Descriptive rules only | Summary line is already imperative; keep NumPy sections |
| `docs/tutorials/` prose, `dev-notes/tech_notes/` problem and method parts | Descriptive rules only | Keep the "we" voice and the analysis |
| `docs/theory/`, derivations in `docs/research/` and `dev-notes/research/` | None | 64–70% of sentences have math; STE has no rules for proofs |
| why-skill answers, quoted output, code, bibliography, non-English text | None | Hedges, quotes, and code are exact; STE is English-only |

"Descriptive rules only" means: sentence length, paragraph length, same
term, and articles. It does not mean verb-form changes in an argument.

## Rules (in addition to the unslop patterns)

Label each sentence first: procedure (an instruction) or description.

1. Procedure: imperative mood, one instruction, 20 words maximum. Put the
   condition first: "If the build fails, read the log."
2. Description: 25 words maximum. A paragraph has one topic and 6 sentences
   maximum.
3. Verbs: use the simple tenses, the imperative, and the infinitive. Do not
   use progressive forms ("is running") or perfect forms ("has been").
   Do not use the passive in a procedure.
4. Noun clusters: 3 words maximum. A code identifier or a normix term (for
   example "expectation parameters") counts as one word.
5. Keep the articles ("the", "a", "this") in sentences. Table cells can
   stay fragments.
6. Use the same word for the same thing. Do not change normix terms or
   code identifiers: they are STE "technical names".
7. Use a vertical list for steps and for 3 or more parallel items.
8. Put a warning before the step it applies to (force push, publish).
9. Do not change code blocks, inline code, paths, identifiers, math, or
   quoted output.

## Conflicts with the unslop carve-outs

The carve-outs and the math rules stay as they are. Resolution:

- Em dashes and colons: allowed. In a procedure, an aside that adds a
  second instruction becomes a separate sentence.
- "We" voice and "Note that": these belong to theory and tutorials, which
  are out of scope or descriptive-only. Do not change them.
- Calibrated hedging: STE never removes "likely" or "appears to" from a
  claim with indirect evidence.
- Math: symbols are technical names. "One symbol per object" still applies.
- Dev-notes shorthand (decision rows, fragments in tables) stays.

## Non-English text

STE has no Chinese version. Do not apply it to Chinese text, and do not
translate text to apply it. Three rules transfer to any language: one
instruction per sentence, condition first, same term for the same thing.
In bilingual text, apply the pass to the English side and keep the terms
aligned between the two sides.

## Check

```bash
python3 .cursor/skills/unslop/scripts/ste_check.py <files or dirs>
```

The checker skips code, math, tables, and headings. It prints
`path:line: RULE detail` for long sentences (`STE-LEN-P` / `STE-LEN-D`),
long paragraphs (`STE-PARA`), wordy words (`STE-WORD`), verb forms
(`STE-VERB`), and normix term drift (`STE-TERM`). It ignores text in double
quotes. It is advisory: it finds a procedure by its first verb only. Fix
the real findings. Report the remaining findings with a reason for each.
