---
name: principle-choose-the-numerically-honest-formulation
description: "Use when choosing formulations, representations, and precision for numerical code."
disable-model-invocation: true
---

# Choose the Numerically Honest Formulation

Choose the formulation, representation, and precision so useful digits survive within the error budget. Fix an unstable formulation before tightening tolerances or increasing precision.

**Why:** Products of tiny densities, explicit inverses of ill-conditioned matrices, cancellation, and formulas evaluated at extreme arguments can destroy useful digits even when the final quantity is well behaved. More iterations or smaller tolerances cannot recover information already lost in intermediates.

**Pattern:**

- Keep densities and likelihoods in log space; use `logsumexp` for sums and `log1p`/`expm1` to avoid cancellation. Investigate cancellation and conditioning before adding output floors or extra precision.
- Solve via factorization instead of forming an explicit inverse; check assumptions, including definiteness and rank, at actual callers.
- Switch special functions such as Bessel functions to asymptotic or series forms at extreme arguments, checking their error against the budget.
- Choose dtype from the error budget and verify the precision actually used by intermediates and outputs.
- Treat rank truncation as a changed problem. Justify the cutoff and inspect the residual against the original equations.
- Scale coordinates and stopping criteria so unit changes do not accidentally decide convergence.
- Check errors near zero, at supported boundaries, and across extreme scales against an independent reference; solver success alone is insufficient.
- If pstack is installed, use `principle-explain-the-number` before reporting an error, residual, or speedup from a solver or precision change, to account for where that number comes from.
