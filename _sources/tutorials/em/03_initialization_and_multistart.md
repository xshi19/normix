---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 600
---

# Initialization and multi-start

EM converges to a *local* optimum of the likelihood, so where it starts matters.
normix gives you three tools: `default_init` for a data-driven starting model,
warm-starting through `theta0`, and `jax.vmap` to run many starts in parallel.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from normix import NormalInverseGaussian, Gamma
from normix.utils.plotting import set_theme

set_theme()
np.set_printoptions(precision=4, suppress=True)
```

## `default_init`: a data-driven start

`default_init(X)` matches the empirical mean and covariance and picks reasonable
subordinator parameters, giving EM a sensible place to begin. Starting there, the
fit converges quickly:

```{code-cell} python
true = NormalInverseGaussian.from_classical(
    mu=jnp.array([0.0, 0.0]),
    gamma=jnp.array([0.5, -0.4]),
    sigma=jnp.array([[1.0, 0.4], [0.4, 1.0]]),
    mu_ig=1.0, lam=1.2)
X = true.rvs(3000, seed=0)

init = NormalInverseGaussian.default_init(X)
ll0 = float(init.marginal_log_likelihood(X))
res = init.fit(X, max_iter=100, tol=1e-4, e_step_backend="cpu")
ll1 = float(res.model.marginal_log_likelihood(X))
print(f"mean log-lik at default_init : {ll0:.4f}")
print(f"mean log-lik after EM        : {ll1:.4f}  ({res.n_iter} iters)")
```

## Multi-start for robustness

A single arbitrary start can land in a worse optimum. Running EM from several
random initializations and keeping the best fit guards against this. Here we
compare random starts against `default_init`:

```{code-cell} python
emp_cov = jnp.asarray(np.cov(np.asarray(X), rowvar=False))
rng = np.random.default_rng(1)

def random_init(seed):
    g = jnp.asarray(rng.normal(scale=0.6, size=2))
    return NormalInverseGaussian.from_classical(
        mu=jnp.asarray(X.mean(axis=0)), gamma=g, sigma=emp_cov,
        mu_ig=float(rng.uniform(0.5, 2.0)), lam=float(rng.uniform(0.5, 2.0)))

scores = []
for s in range(5):
    r = random_init(s).fit(X, max_iter=100, tol=1e-4, e_step_backend="cpu")
    scores.append(float(r.model.marginal_log_likelihood(X)))

print("random-start mean log-liks:", np.round(scores, 4))
print(f"best random start : {max(scores):.4f}")
print(f"default_init start : {ll1:.4f}")
```

The best random start matches `default_init`, while the worst trails it — the
takeaway is to *use `default_init`* (it is already a strong start) and to *keep
the best of several starts* when the likelihood surface is rugged.

## Vectorized starts with `jax.vmap`

For exponential-family distributions, the $\eta \mapsto \theta$ solve in
`from_expectation` is pure JAX, so it `vmap`s. That lets us fit many datasets —
or many bootstrap resamples — in a single vectorized call instead of a Python
loop:

```{code-cell} python
g_true = Gamma(alpha=jnp.array(2.0), beta=jnp.array(1.5))
datasets = jnp.stack([g_true.rvs(2000, seed=s) for s in range(8)])   # (8, 2000)

# Each dataset's expectation parameters, then a single batched solve.
etas = jax.vmap(lambda d: jax.vmap(Gamma.sufficient_statistics)(d).mean(0))(datasets)
fits = jax.vmap(Gamma.from_expectation)(etas)        # batched Gamma pytree

print("fitted alphas:", np.asarray(fits.alpha))
print("fitted betas :", np.asarray(fits.beta))
print("alpha mean ± sd: %.3f ± %.3f" % (float(fits.alpha.mean()), float(fits.alpha.std())))
```

The result `fits` is a single `Gamma` pytree whose leaves carry a leading batch
axis — exactly the shape you want for bootstrap confidence intervals.

## Warm-starting with `theta0`

`fit_mle` and `from_expectation` accept a `theta0` to seed the solver. A warm
start near the solution converges in fewer iterations and lands at the same
optimum:

```{code-cell} python
X1 = g_true.rvs(5000, seed=0)
cold = Gamma.fit_mle(X1)
warm = Gamma.fit_mle(X1, theta0=g_true.natural_params())
print("cold start (alpha, beta):", float(cold.alpha), float(cold.beta))
print("warm start (alpha, beta):", float(warm.alpha), float(warm.beta))
```

## Takeaways

- `default_init(X)` is a moment-matched starting model; EM converges quickly
  from it.
- For rugged likelihoods, run several starts and keep the highest-likelihood
  fit.
- `jax.vmap` over `from_expectation` fits many datasets/resamples at once;
  `theta0` warm-starts the solver.

Next, the {doc}`../stats/01_divergences` tutorial measures how close two fitted
models are.
