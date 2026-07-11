---
name: tdd
description: >-
  Failing-test-first bug fixing. Use when fixing a bug with a clear, cheap
  test path, or when the user asks for TDD, a failing test, or a regression
  test. Skip when the test would need heavy fixtures, GPU, or vague
  reproduction steps — say so and use the closest executable check instead.
---

# TDD bug fix

Make the broken behavior executable before touching production code: a
focused regression test that fails before the fix and passes after.

## Workflow

1. **Understand the bug.** Intended behavior, current behavior, and the
   smallest observable reproduction — distribution, parameter values, the
   offending call.
2. **Pick the narrowest check.** Usually a new parametrized case in the
   existing `tests/test_<distribution>.py`. Tolerances come from
   `.cursor/rules/testing-guidelines.mdc`; markers per the running-tests
   skill (a numerical regression test is typically `contract`; don't mark
   it `slow` unless it is).
3. **Write the failing test first.** Encode the intended behavior, not the
   current implementation. For extreme-parameter bugs, pin the actual
   failing parameter values from the report.
4. **Run it before fixing:** `uv run pytest tests/test_x.py -k <case> -v` —
   confirm it fails *for the intended reason*. If it passes or fails for an
   unrelated reason, fix the test or the reproduction first.
5. **Fix the bug.** Smallest change that satisfies the behavior, at the
   root cause — a NaN traced to a wrong regime boundary is fixed at the
   boundary, not by flooring the output (principles skill § Fix root
   causes).
6. **Rerun** the regression test, then nearby validation: the
   distribution's full test file, and `uv run pytest tests/` before
   committing.

## When a failing test is impractical

Don't silently skip the step. Explain why, then use the closest executable
check: a reproduction script, a property check over a parameter grid, a
benchmark comparison. Prefer no new test over a bad test — bad means it
tests implementation details, needs `slow`-scale sampling for a small fix,
or depends on seeds so tightly any refactor breaks it.

## Guardrails

- Never weaken an existing assertion or tolerance to make a fix pass. A
  genuine behavior change gets its tolerance change justified in the commit
  message.
- The property/contract suites (parameter roundtrip, gradient consistency,
  sample statistics vs `expectation_params`) may already almost catch the
  bug — extend their parametrization before writing a new test from
  scratch.
- Flaky numerical test → fix the seed and state the tolerance rationale.

## Final report

Report evidence, not just outcome: the failing-before test and its failure,
the passing-after run, and the nearby validation performed. If
failing-before evidence couldn't be demonstrated, what was used instead and
why.
