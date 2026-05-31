# Divergences

normix provides two information divergences between distributions as
first-class, closed-form operations: the **squared Hellinger distance** and the
**Kullback–Leibler divergence**. For exponential families both reduce to
evaluations of the log-partition $\psi$, so they are exact and differentiable —
no Monte Carlo.

## Distribution-level API

```python
from normix import squared_hellinger, kl_divergence

squared_hellinger(p, q)   # symmetric distance in [0, 1]
kl_divergence(p, q)       # asymmetric divergence in [0, ∞)
```

Both take two distributions of the **same family** (two `Gamma`s, two
`NormalInverseGaussian`s, …). $H^2$ is a bounded, symmetric, metric-like
quantity — prefer it when you want a comparison; KL is the asymmetric
information divergence used in likelihood theory.

```{warning}
These compare members of the *same* family. A Hellinger distance between, say, a
Variance Gamma and an NIG is not meaningful — their log-partitions differ. To
compare *different* families on data, use out-of-sample log-likelihood instead
(see {doc}`../tutorials/finance/01_univariate_index`).
```

## The functional `*_from_psi` core

The distribution-level functions are a thin layer over a functional core that
operates directly on the log-partition and natural-parameter vectors. This is
what the optimization and EM code calls internally, and it is fully
differentiable:

```python
from normix import squared_hellinger_from_psi, kl_divergence_from_psi

squared_hellinger_from_psi(psi, theta_p, theta_q)
kl_divergence_from_psi(psi, grad_psi, theta_p, theta_q)
```

Here `psi` is a callable $\theta \mapsto \psi(\theta)$ (for instance
`Gamma._log_partition_from_theta`), `grad_psi` is $\nabla\psi$, and `theta_p`,
`theta_q` are natural-parameter vectors. Feeding a distribution's own $\psi$
reproduces the convenience result exactly.

## Typical uses

- **Estimation error.** $H^2(\text{true}, \hat p_n)$ measures how far a fitted
  model is from a reference; it shrinks at the parametric $1/n$ rate as the
  sample grows.
- **Stability.** $H^2$ between a model fit on two periods (e.g. train vs test)
  quantifies distributional drift.
- **Gradients of distance.** Because the core takes a plain callable, you can
  `jax.grad` through $H^2$ with respect to natural parameters — useful for
  moment-projection and model-distillation tasks.

Worked examples are in {doc}`../tutorials/stats/01_divergences`, and per-sample
goodness-of-fit diagnostics (QQ plots, CDF overlays, KS tests) are in
{doc}`../tutorials/stats/02_goodness_of_fit`.
