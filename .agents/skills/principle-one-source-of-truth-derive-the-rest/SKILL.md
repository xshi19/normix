---
name: principle-one-source-of-truth-derive-the-rest
description: "Use when implementing related values, derivatives, or inverse maps that could drift apart."
disable-model-invocation: true
---

# One Source of Truth, Derive the Rest

Derive related production quantities from one authoritative primitive wherever their numerical contracts agree. Keep justified exceptions explicit and verification independent.

**Why:** Separately maintained values and derivatives drift when an approximation or branch changes in only one implementation. An optimizer can then follow a gradient that no longer belongs to its objective, while tests that reuse the production primitive reproduce the same error and conceal the mismatch.

**Pattern:**

- Derive related production quantities from a shared primitive when their numerical contracts agree; keep test oracles independent.
- Autodiff differentiates the implemented approximation, including its errors. Check derivative accuracy separately.
- Stable analytic or custom derivatives need evidence over their stated domain, including boundary limits; optimizer progress alone is insufficient.
- Implicit derivatives require a converged solution and locally invertible Jacobian. Define behavior for failed solves, singular points, and active constraints.
- Finite-difference diagnostics need smooth interior inputs, scaling, and a step-size sweep.
- When changing an approximation or branch threshold, review dependent values, derivatives, and inverse maps together for contract drift.
