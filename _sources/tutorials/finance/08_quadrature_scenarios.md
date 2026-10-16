---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 600
---

# Quadrature scenarios for portfolio CVaR

We compare deterministic quadrature with conditional Monte Carlo for one NIG
portfolio. Both methods integrate the conditional normal exactly and differ
only in their measure of the mixing variable $Y$. The value is the
{ref}`Rockafellar–Uryasev <rockafellaruryasev2000>` objective; its derivation
and quadrature error analysis are in {doc}`../../theory/quadrature_scenarios`.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norminvgauss
from normix import NormalInverseGaussian
from normix.finance import CVaR, DiscreteMeasure, MeanRiskProblem

model = NormalInverseGaussian.from_classical(
    mu=jnp.array([0.01, 0.03, -0.01]),
    gamma=jnp.array([-0.2, 0.1, 0.05]),
    sigma=jnp.array([[0.7, 0.1, 0.0], [0.1, 0.4, 0.05], [0.0, 0.05, 0.2]]),
    mu_ig=1.0, lam=1.0,
)
w = jnp.array([0.4, 0.3, 0.3])
proj = model.project(w)
risk = CVaR(0.05)
Q = model.subordinator().quadrature()
print(f"nodes: {Q.Y.size}, probability sum: {float(Q.p.sum()):.15f}")
print(f"CVaR: {float(risk.value(proj)):.12f}")
```

## An independent reference

We use [SciPy](https://docs.scipy.org/doc/scipy/) to integrate the NIG return
density. For this subordinator, $\mu_Y=\lambda=1$, so the NIG shape parameters
are $a=\sqrt{1+(\tilde\gamma/\tilde\sigma)^2}$ and
$b=\tilde\gamma/\tilde\sigma$, with scale $\tilde\sigma$.

```{code-cell} python
mu, gamma, sigma = map(float, (proj._mu_scalar, proj._gamma_scalar, proj._sigma_scalar))
b = gamma / sigma
reference_law = norminvgauss(np.hypot(1.0, b), b, loc=mu, scale=sigma)
x_alpha = reference_law.ppf(risk.alpha)
reference = -quad(lambda x: x * reference_law.pdf(x), -np.inf, x_alpha,
                  epsabs=1e-12, epsrel=1e-12)[0] / risk.alpha
rows = []
for n in (64, 128, 256, 512):
    q = proj.subordinator().quadrature(n)
    value = float(risk.value(proj, q))
    rows.append({"method": "quadrature", "nodes": n, "CVaR": value,
                 "relative error": abs(value / reference - 1)})
Y = proj.subordinator().rvs(20_000, seed=0)
mc = float(risk.value(proj, Y))
rows.append({"method": "conditional MC", "nodes": Y.size, "CVaR": mc,
             "relative error": abs(mc / reference - 1)})
print(f"independent reference: {reference:.12f}")
print(pd.DataFrame(rows).to_string(index=False))
np.testing.assert_allclose(risk.value(proj, Q), reference, rtol=1e-11)
```

The MC error in this table belongs to one seed. The repository benchmark
reports RMSE over multiple seeds. An array still supplies equal-weight draws;
weighted observations use `DiscreteMeasure.from_weights(Y, weights)`.

## The same measure throughout optimization

`MeanRiskProblem` retains one quadrature measure for the surface and frontier.
Expected returns use the original model's subordinator mean. We compare the
optimized weights with the explicit MC path at the same target returns.

```{code-cell} python
problem = MeanRiskProblem(model, risk)
targets = jnp.linspace(0.01, 0.05, 5)
frontier = problem.efficient_frontier(targets, gamma_bounds=(-0.5, 0.5))
mc_frontier = problem.efficient_frontier(targets, Y, gamma_bounds=(-0.5, 0.5))
comparison = pd.DataFrame({
    "target": np.asarray(targets),
    "quadrature CVaR": np.asarray(frontier.risk),
    "MC CVaR": np.asarray(mc_frontier.risk),
    "max weight difference": np.max(np.abs(frontier.weights - mc_frontier.weights), axis=1),
})
print(comparison.to_string(index=False))
np.testing.assert_allclose(frontier.weights.sum(axis=1), 1.0, atol=1e-12)
```

The objective and its derivatives share the same measure. First derivatives
of the value agree with the analytic weight gradient:

```{code-cell} python
gradient = jax.grad(lambda weights: risk.value_w(model, weights, Q))(w)
np.testing.assert_allclose(gradient, risk.gradient_w(model, w, Q), rtol=1e-11, atol=1e-12)
print("gradient:", np.asarray(gradient))
print("step resolution:", float(risk.step_resolution(proj, Q)))
```

Use `risk.hessian_w` for the Hessian. Applying autodiff twice to the value
would omit quantile motion because the root is frozen for the envelope derivative.

## A narrow transition needs more nodes

Reducing conditional variance makes the CDF transition narrower in $\log Y$.
The diagnostic compares its width with local node spacing; it is not an error
bound. This example deliberately enters that regime.

```{code-cell} python
sharp = proj.replace(gamma=jnp.array([-0.5]), L_Sigma=jnp.array([[0.003]]))
for n in (512, 2048, 8192):
    q = sharp.subordinator().quadrature(n)
    print(f"nodes={n:5d}  R={float(risk.step_resolution(sharp, q)):.3f}  "
          f"CVaR={float(risk.value(sharp, q)):.10f}  "
          f"H_gamma_gamma={float(risk.hessian_scalar(sharp, q)[1, 1]):.8f}")
```

The Hessian can change appreciably after the value appears stable. Inspect
resolution and convergence before using such curvature in a transaction-cost
QP or generalized ENB. Moment existence is a separate condition: an
inverse-gamma law can have finite CVaR and infinite skew curvature at zero skew.
See {doc}`../../design/quadrature_scenarios` for the supported defaults and limits.
