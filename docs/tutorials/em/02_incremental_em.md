---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 600
---

# Incremental (mini-batch) EM

When data arrives in a stream, or is too large to sweep each iteration,
`IncrementalEMFitter` updates the model from **mini-batches**. Each step
computes the batch expectation parameters $\hat\eta$, then blends them into a
running estimate $\eta_t$ through an **$\eta$-update rule** before the M-step.
The choice of rule controls the bias/variance and the forgetting behaviour of
the online estimate.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from normix import (
    NormalInverseGaussian,
    IdentityUpdate, RobbinsMonroUpdate, SampleWeightedUpdate,
    EWMAUpdate, AffineUpdate, Shrinkage, eta0_from_model,
)
from normix.fitting.em import IncrementalEMFitter
from normix.utils.plotting import set_theme

set_theme()
np.set_printoptions(precision=4, suppress=True)
```

## Setup

```{code-cell} python
true = NormalInverseGaussian.from_classical(
    mu=jnp.array([0.0, 0.0]),
    gamma=jnp.array([0.4, -0.3]),
    sigma=jnp.array([[1.0, 0.3], [0.3, 1.0]]),
    mu_ig=1.0, lam=1.5)
X = true.rvs(20_000, seed=0)
init = NormalInverseGaussian.default_init(X)
key = jax.random.PRNGKey(0)
```

## The six $\eta$-update rules

A rule maps the previous estimate $\eta_{t-1}$ and the batch estimate
$\hat\eta$ to the new $\eta_t$. normix ships six:

| Rule | Update $\eta_t$ |
|---|---|
| `IdentityUpdate` | $\hat\eta$ (no memory) |
| `RobbinsMonroUpdate(tau0)` | step-size $\propto 1/(t + \tau_0)$ |
| `SampleWeightedUpdate` | weight by cumulative sample count |
| `EWMAUpdate(w)` | exponential moving average, weight $w$ |
| `AffineUpdate(a, b, c)` | $a + b\,\eta_{t-1} + c\,\hat\eta$ |
| `Shrinkage(base, eta0, tau)` | wrap a base rule, shrink toward $\eta_0$ |

```{code-cell} python
rules = {
    "Identity": IdentityUpdate(),
    "RobbinsMonro": RobbinsMonroUpdate(tau0=10.0),
    "SampleWeighted": SampleWeightedUpdate(),
    "EWMA(0.1)": EWMAUpdate(w=0.1),
    "Affine(½,½)": AffineUpdate(b=0.5, c=0.5),
    "Shrinkage": Shrinkage(IdentityUpdate(), eta0_from_model(init), tau=0.3),
}

target = float(true.marginal_log_likelihood(X))
print(f"target mean log-likelihood (true model): {target:.4f}\n")

finals = {}
for name, rule in rules.items():
    fitter = IncrementalEMFitter(
        batch_size=512, max_steps=60, eta_update=rule,
        e_step_backend="cpu", m_step_backend="cpu")
    res = fitter.fit(init, X, key=key)
    finals[name] = float(res.model.marginal_log_likelihood(X))
    print(f"{name:16s} final mean log-lik = {finals[name]:.4f}")
```

```{code-cell} python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
names = list(finals)
ax.barh(names, [finals[n] for n in names], color="#2D5A8A")
ax.axvline(target, color="0.4", ls="--", lw=1.2, label="true-model LL")
ax.set_xlabel("final mean log-likelihood")
ax.set_xlim(min(finals.values()) - 0.01, target + 0.005)
ax.set_title("Mini-batch EM: $\\eta$-update rules after 60 steps")
ax.legend()
plt.show()
```

All rules climb to within a fraction of a nat of the true-model likelihood after
just 60 mini-batches. They differ mainly in *how* they get there — the averaging
rules (`SampleWeighted`, `EWMA`, `Affine`) damp the per-step noise, while
`Identity` and the decaying `RobbinsMonro` step are more volatile.

## Following a single trajectory

With `verbose=1` the fitter records the log-likelihood at diagnostic
checkpoints, which we can plot to see the mini-batch ascent of two contrasting
rules:

```{code-cell} python
fig, ax = plt.subplots()
for name, rule in [("Identity", IdentityUpdate()),
                   ("EWMA(0.1)", EWMAUpdate(w=0.1))]:
    res = IncrementalEMFitter(
        batch_size=512, max_steps=60, eta_update=rule, verbose=1,
        e_step_backend="cpu", m_step_backend="cpu").fit(init, X, key=key)
    ll = np.asarray(res.log_likelihoods)
    ax.plot(np.linspace(0, 60, len(ll)), ll, marker="o", ms=3, label=name)
ax.axhline(target, color="0.4", ls="--", lw=1.2, label="true-model LL")
ax.set_xlabel("mini-batch step"); ax.set_ylabel("mean log-likelihood")
ax.set_title("Incremental EM trajectories")
ax.legend()
plt.show()
```

The `Identity` rule is noisier because it discards all history; `EWMA`
smooths the estimate across batches.

## Shrinkage toward a target

`Shrinkage` wraps any base rule and pulls the estimate toward a fixed
$\eta_0$ — a regularizer for small batches or noisy streams. The targets module
builds sensible $\eta_0$ values:

- `eta0_from_model(model)` — the model's own current expectation parameters.
- `eta0_isotropic(model, sigma2)` — isotropic covariance target.
- `eta0_diagonal(model, diag)` — diagonal covariance target.
- `eta0_with_sigma(model, Sigma0)` — explicit covariance target.

```{code-cell} python
from normix import eta0_isotropic

rule = Shrinkage(RobbinsMonroUpdate(tau0=10.0), eta0_isotropic(init, 1.0), tau=0.5)
res = IncrementalEMFitter(
    batch_size=256, max_steps=60, eta_update=rule,
    e_step_backend="cpu", m_step_backend="cpu").fit(init, X, key=key)
print("shrinkage-to-isotropic final mean log-lik:",
      float(res.model.marginal_log_likelihood(X)))
```

## Takeaways

- `IncrementalEMFitter` updates from mini-batches; `fit(model, X, key=...)`
  needs a PRNG key for batch sampling.
- Six $\eta$-update rules trade off memory vs responsiveness; averaging rules
  reduce variance, `Identity` reacts fastest.
- `Shrinkage` + the `eta0_*` targets regularize the online estimate toward a
  chosen structure.

Next: {doc}`03_initialization_and_multistart` looks at where the EM loop starts
and how to make fits robust to local optima.
