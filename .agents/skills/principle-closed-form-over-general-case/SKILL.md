---
name: principle-closed-form-over-general-case
description: "Use when choosing a specialized formula or considering deduplication through a general solver."
disable-model-invocation: true
---

# Closed Form Over General Case

Prefer a specialized formula when it satisfies the required domain and accuracy contract. Replace it with general machinery only when the numerical evidence justifies the extra dependencies and failure modes.

**Why:** Deduplicating a direct formula through a solver imports initialization, stopping rules, and convergence failures into callers that previously needed none. An exact boundary value can become a failed solve, while a shorter implementation becomes harder to differentiate and verify.

**Pattern:**

- Compare domains, boundary behavior, derivative accuracy, and cost before replacing a special case with general machinery.
- Check cancellation and singular limits; a closed form can lose accuracy where an iterative method remains reliable.
- At transitions between specialized and general paths, check values and any promised derivatives on both sides.
- Count imported convergence criteria and failure modes, not just lines removed.
- Verify difficult inputs with an independent oracle; agreement between paths sharing the same unstable primitive is weak evidence.
- Keep separate formulas when domains, limits, or independent verification roles differ; record that reason where future deduplication would erase it.
