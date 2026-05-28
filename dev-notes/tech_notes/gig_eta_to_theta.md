# GIG η→θ Optimization

## Problem

Given expectation parameters η = (E[log Y], E[1/Y], E[Y]), find natural parameters θ
such that ∇ψ(θ) = η, where ψ is the GIG log-partition function.

## η-rescaling

The Fisher matrix condition number can reach 10^30 for extreme a/b ratios.
Rescaling symmetrizes the problem:

```
s = √(η₂/η₃)
η̃ = (η₁ + ½ log(η₂/η₃),  √(η₂η₃),  √(η₂η₃))
```

After solving η̃ → θ̃ (symmetric GIG with ã = b̃), unscale:
θ₂ = θ̃₂/s, θ₃ = s·θ̃₃.

## Multi-start Strategy

Initial guesses from special-case limits:
1. **Gamma limit** (b→0): match η₁ = E[log X], η₃ = E[X] via `Gamma.from_expectation`
2. **InverseGamma limit** (a→0): match η₁, η₂ via `InverseGamma.from_expectation`
3. **InverseGaussian limit** (p=-½): match η₂, η₃ via `InverseGaussian.from_expectation`

Perturbed copies at scales 0.1, 0.5, 2.0, 10.0 are added for robustness.

## Solver

`scipy.optimize.minimize` with `method='L-BFGS-B'` and bounds θ₂ ≤ 0, θ₃ ≤ 0.
Not JIT-able due to multi-start control flow and scipy dependency.

## Joint Distribution Expectation Parameters

The joint distribution `JointNormalMixture` computes `expectation_params()` via `jax.grad`
on the joint log-partition, which parameterizes the subordinator through the GIG limit
(e.g., b → ε for VG). This can yield numerically inaccurate values for E[Y] when the
subordinator is at a degenerate limit. For theoretical moment computation (E[X], Cov[X]),
use `subordinator().mean()` and `subordinator().var()` directly instead.
