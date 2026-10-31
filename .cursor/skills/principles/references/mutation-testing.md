# Mutation testing

Coverage shows which code ran; mutation testing asks whether tests notice when that code behaves incorrectly.
Use this principle when numerical tests pass but their ability to catch regressions remains unclear.
Start with a small GIG or Bessel numerical path using `mutmut`.
Use a disposable environment without adding `mutmut` to `pyproject.toml` or `uv.lock`.

- **Cover the uncovered.** Exercise reachable, untested behavior through its callers' interface before judging the mutation results.
  A mutant outside the executed path says nothing about assertion quality.
- **Kill survivors.** Inspect each surviving mutant for a change to behavior promised by the mathematical contract.
  Add a test that exposes that change.
  Confirm that it passes on the original code and fails on the mutant.
  Record equivalent mutants with a reason; do not distort the contract to kill them.
- **Delete dead code.** Remove code only after confirming it has no supported caller or reachable contract behavior.
  Missing coverage alone does not establish that code is dead.

Floating-point tolerances can let meaningful numerical mutants survive.
Pair mutation testing with Hypothesis property tests over valid parameters, including boundary and extreme regimes.
Useful properties include mathematical identities, parameter round-trips, and derivative consistency against an independent reference.
Choose tolerances from the numerical contract and reference accuracy.
Do not tighten them merely to improve the mutation score.

Keep the scope, commands, observed results, and explanations for remaining survivors with the review.
Completion criterion: tests catch each confirmed behavioral mutant, equivalent survivors have explanations, and any deletion preserves supported behavior.
