---
name: numerical-review
description: "Use when reviewing numerical formulas, derivative rules, iterative solvers, or execution-contract changes."
---

# Numerical Review

- Read the governing derivation and accepted decisions. Check assumptions at actual callers; matching mistakes can make round-trips pass.
- Probe cancellation, degeneracy, and transitions between formulas. Test defined boundaries, rejection of excluded inputs, and supported one-sided limits before accepting floors.
- Check derivatives against independent identities or references. Finite differences need smooth interior points, scaling, and a step-size sweep; implicit derivatives need convergence, residual, rank, and active-constraint checks.
- Trace failure status to callers. Distinguish invalid inputs, valid infinities, iteration limits, and numerical breakdown; a finite fallback proves none of them resolved.
- Check oracle accuracy and domain; sharing the implementation's primitive can hide its error. Include absolute error near zero and justify tolerances.
- Separate a mathematical theorem, program-translation certificate, approximation-error bound, and floating-point guarantee. State each one's scope; proof failure alone does not demonstrate a program bug.
- Report demonstrated findings by consequence with the violated contract, smallest public reproduction, command, actual output, tolerance rationale, and environment. Keep unexecuted concerns and unavailable checks as evidence gaps.
- If the code uses traced, JIT, or GPU execution (JAX, PyTorch CUDA), read [references/transformed-execution.md](references/transformed-execution.md); otherwise skip it.
