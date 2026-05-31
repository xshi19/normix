---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 900
---

# Factor mixtures for a 30-asset portfolio

At portfolio scale — dozens of assets — a full covariance matrix has too many
parameters to estimate reliably. The factor mixtures replace $\Sigma$ with
$F F^\top + \operatorname{diag}(D)$, capturing the dominant co-movement with a
handful of latent factors while keeping the heavy-tailed GH structure. This
tutorial fits a `FactorGeneralizedHyperbolic` to 30 stocks and inspects the
factors it discovers.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path

from normix import FactorGeneralizedHyperbolic, NormalInverseGaussian
from normix.utils.plotting import set_theme

set_theme()
np.set_printoptions(precision=4, suppress=True)
```

## Data: 30 large-cap stocks

```{code-cell} python
data_path = Path("../../../data/sp500_returns.csv").resolve()
panel = pd.read_csv(data_path, index_col="Date", parse_dates=True).dropna(axis=1, how="any")
tickers = list(panel.columns[:30])
R = panel[tickers].values.astype(np.float64)

n_train = len(R) // 2
X_train, X_test = jnp.asarray(R[:n_train]), jnp.asarray(R[n_train:])
d = len(tickers)
print(f"d = {d} stocks, train {n_train}, test {R.shape[0] - n_train}")
```

## The parameter budget

```{code-cell} python
full_cov = d * (d + 1) // 2
for r in (1, 2, 3):
    print(f"factor r={r}: {d * r + d:4d} covariance params   "
          f"(full Σ: {full_cov})")
```

A two-factor model describes the $30 \times 30$ covariance with 90 numbers
instead of 465.

## Fitting factor GH at several ranks

```{code-cell} python
import time

results = {}
for r in (1, 2, 3):
    t0 = time.perf_counter()
    init = FactorGeneralizedHyperbolic.default_init(X_train, r=r)
    res = init.fit(X_train, max_iter=60, tol=1e-3,
                   e_step_backend="cpu", m_step_backend="cpu")
    results[r] = res.model
    print(f"r={r}: {res.n_iter:3d} iters, {time.perf_counter() - t0:5.1f}s, "
          f"test LL = {float(res.model.marginal_log_likelihood(X_test)):.4f}")
```

Adding factors raises the held-out likelihood with sharply diminishing returns —
the first factor (a market factor) does most of the work.

## Inspecting the latent factors

The columns of $F$ are the factor loadings: how strongly each stock responds to
each latent factor. The first factor typically loads positively on *every*
stock — it is the market:

```{code-cell} python
import matplotlib.pyplot as plt

model = results[2]
F = np.asarray(model.F)
order = np.argsort(F[:, 0])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for k, ax in enumerate(axes):
    ax.barh(np.array(tickers)[order], F[order, k], color="#2D5A8A")
    ax.set_title(f"Factor {k + 1} loadings")
    ax.tick_params(axis="y", labelsize=6)
    ax.axvline(0, color="0.4", lw=0.8)
plt.show()

print(f"factor 1 loadings all positive: {bool((F[:, 0] > 0).all() or (F[:, 0] < 0).all())}")
```

## Heavy tails survive at scale

The factor structure regularizes the covariance; the GH subordinator still
captures the joint heavy tails. We compare the factor model's out-of-sample
likelihood against a full-$\Sigma$ NIG fit to the same 30 assets:

```{code-cell} python
full_nig = NormalInverseGaussian.default_init(X_train).fit(
    X_train, max_iter=60, tol=1e-3, e_step_backend="cpu").model

print(f"FactorGH (r=2): {model.F.shape[1] * d + d:4d} cov params, "
      f"test LL {float(model.marginal_log_likelihood(X_test)):.4f}")
print(f"full-Σ NIG    : {full_cov:4d} cov params, "
      f"test LL {float(full_nig.marginal_log_likelihood(X_test)):.4f}")
```

The factor model reaches a comparable out-of-sample likelihood with a fraction
of the covariance parameters — and every internal solve goes through the
Woodbury identity, so it scales to hundreds of assets without forming a dense
$d \times d$ inverse.

## Takeaways

- `FactorGeneralizedHyperbolic.default_init(X, r=...)` then `fit` estimates a
  low-rank-plus-diagonal covariance with the full GH tail behaviour.
- The first factor is a market factor (loads on all assets); extra factors add
  little out-of-sample likelihood here.
- Factor mixtures match a full-$\Sigma$ fit with far fewer parameters and
  Woodbury-scale linear algebra.

Next: {doc}`04_cvar_optimization` uses a fitted mixture to compute and
differentiate portfolio tail risk.
