# Mutation testing in normix

Read [the shared mutation-testing skill](../../.agents/skills/mutation-testing/SKILL.md) before using these project additions.
This file is an attachment, not a discoverable skill.

Start with a small GIG or Bessel numerical path using `mutmut`.
Use a disposable environment without adding `mutmut` to `pyproject.toml` or `uv.lock`.

Floating-point tolerances can let meaningful numerical mutants survive.
Pair mutation testing with Hypothesis property tests over valid parameters, including boundary and extreme regimes.
Useful properties include mathematical identities, parameter round-trips, and derivative consistency against an independent reference.
The shared skill governs tolerance justification, survivor classification, and original-pass/mutant-fail evidence.
