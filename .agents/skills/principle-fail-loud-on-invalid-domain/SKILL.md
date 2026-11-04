---
name: principle-fail-loud-on-invalid-domain
description: "Use when handling invalid parameters, failed solves, infinities, clamps, or numerical thresholds."
disable-model-invocation: true
---

# Fail Loud on Invalid Domain

Reject invalid requests and make numerical failure visible to callers. Preserve the distinction between a valid limiting answer and an algorithm that did not produce an answer.

**Why:** A clamp or substituted finite value can silently change a likelihood or feasible set, letting an optimizer fit a different model while reporting success. Suppressed solver failure also contaminates downstream derivatives and uncertainty estimates, where the original failure is harder to locate.

**Pattern:**

- Distinguish invalid requests, algorithm failure, and valid limiting values. Propagate failure status; a finite last iterate is not convergence.
- Infinity requires a mathematically defined extended-real value; an undefined quantity is not automatically positive infinity.
- Separate roundoff handling from projection or regularization that changes the requested problem; make the latter an explicit contract.
- Give thresholds a purpose, scale, domain, and justification. Share constants by meaning, not numeric equality or a mandatory global registry.
- Check valid boundaries, specified rejections, and exhausted iteration budgets; verify that a plausible finite result cannot hide failure status.
- If validation runs under tracing or JIT compilation, read [references/traced-validation.md](references/traced-validation.md); otherwise skip it.
