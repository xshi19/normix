---
name: mutation-testing
description: "Use when passing numerical tests leave doubt that plausible regressions would be detected."
---

# Mutation Testing

- Bound the audit to a public operation. Keep mutations isolated and exploratory tooling out of the project lockfile; establish a passing baseline.
- Exercise uncovered behavior through supported callers before interpreting survivors. Use a fresh process or build when needed so the changed code actually executes.
- For behavioral survivors, add an independent contract-derived assertion; verify original-pass/mutant-fail. Never tighten tolerances solely to improve a mutation score.
- Separate assertion-detected changes, equivalent mutants, surviving behavioral changes, uncovered paths, invalid mutants, timeouts, and runner failures. Infrastructure failures are not meaningful kills.
- Equivalence needs evidence that supported observations cannot distinguish the mutant; failure to find a counterexample is insufficient.
- Derive tolerances from required accuracy and oracle error. Retain failing sampled inputs for replay.
- Delete code only with evidence of no supported callers or reachable contract behavior; missing coverage is insufficient.
- Restore unmutated code, rerun relevant tests, and report commands, baseline, outcomes, and unresolved behavioral gaps.
