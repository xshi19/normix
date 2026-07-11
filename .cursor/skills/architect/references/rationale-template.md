# Design package template

One page of prose plus the sketch. Replace the italic notes with content.

## Problem

*One paragraph: what we're building, and which existing constraints
(design.md rows, existing types, callers that can't break) make the shape
non-obvious.*

## Usage (caller's view)

*Written first. The quickstart a consumer would read, plus 2–3 realistic
call sites. What they import, what they call, what comes back. The sketch
below is derived from this.*

## Shape

*The recommended architecture: `eqx.Module` field sketch, classmethod
signatures, module map. Name the load-bearing decisions and which design.md
rows each honors or amends. State what the design deliberately does not
do.*

## Synthesis decision

*Filled in by arena: which candidate became the base and why, what was
grafted from each other candidate, what was rejected and why.*

## Tradeoffs accepted

*One bullet per tradeoff, in the form "we accept X in exchange for Y".
Include anything a future reader might mistake for an oversight.*

## Alternatives considered

*At least one concrete alternative shape and one line on why it lost —
design alternatives this candidate weighed, distinct from the other
runners' candidates covered in the synthesis decision.*

## Open questions and risks

*Questions for the maintainer, phrased so an answer resolves them. Include
any numerical-method choice deferred to a benchmark, with the benchmark
plan.*

## Proposed design.md row

*Draft the decision-table row: `# | Decision | Choice | Why / Detail
link`.*
