---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
mystnb:
  execution_timeout: 900
---

# EM vs MCECM Algorithm Comparison

This notebook replicates Table 4 from [Shi2016], comparing the EM and MCECM
algorithms for fitting Generalized Hyperbolic (GH) distributions.

**Procedure** (following the thesis):
1. Fit a 2D GH distribution to real stock return data via EM → base model
   with parameters $(\mu, \gamma, \Sigma, p_0, a, b)$.
2. For each $p \in \{-10, -9, \ldots, 10\}$, construct a "true" model by
   replacing $p_0$ with $p$ in the base model (all other parameters unchanged).
3. Generate 5000 i.i.d. multivariate GH samples from each "true" model.
4. Re-fit each sample (all parameters free) using **both** EM and MCECM.
5. Compare log-likelihoods and squared Hellinger distances $H^2$.

**Expected result**: Both algorithms converge to essentially the same MLE.

```{code-cell} python
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)
from normix import GeneralizedHyperbolic, squared_hellinger

np.set_printoptions(precision=6, suppress=False)
```

```{code-cell} python
# myst-nb executes with cwd = docs/tutorials/em/
data_path = Path("../../../data/sp500_returns.csv").resolve()
returns = pd.read_csv(data_path, index_col=0)
cols = ['AAPL', 'MSFT']
X_real = jnp.array(returns[cols].dropna().values, dtype=jnp.float64)
print(f'Data: {X_real.shape[0]} observations, {X_real.shape[1]} stocks')
```

## Step 1: Fit base model to real data

```{code-cell} python
model_base = GeneralizedHyperbolic.default_init(X_real)
result_base = model_base.fit(
    X_real, max_iter=200, tol=1e-2,
    regularization='det_sigma_one', verbose=1)
model_base = result_base.model

j = model_base.joint
print(f'\nBase model parameters:')
print(f'  p = {float(j.p):.4f},  a = {float(j.a):.4f},  b = {float(j.b):.4f}')
print(f'  mu = {np.array(j.mu)}')
print(f'  gamma = {np.array(j.gamma)}')
```

## Step 2–5: Sweep $p$, simulate, re-fit via EM and MCECM

```{code-cell} python
p_values = list(range(-10, 11))
results = []

t_em_total = 0.0
t_mcecm_total = 0.0

for p_val in p_values:
    print(f'\np = {p_val:+3d} ', end='', flush=True)

    joint_true = eqx.tree_at(lambda j: j.p, model_base.joint, jnp.float64(p_val))
    model_true = GeneralizedHyperbolic(joint_true)

    X_synth = model_true.rvs(5000, seed=p_val + 100)

    try:
        t0 = time.perf_counter()
        result_em = model_true.fit(
            X_synth, algorithm='em', max_iter=200, tol=1e-2,
            regularization='det_sigma_one', verbose=0)
        t_em = time.perf_counter() - t0
        t_em_total += t_em
        model_em = result_em.model
        ll_em = float(model_em.marginal_log_likelihood(X_synth))
        h2_em = float(squared_hellinger(model_true.joint, model_em.joint))
        print(f'[EM: {result_em.n_iter} iters, {t_em:.1f}s] ', end='', flush=True)
    except Exception as e:
        print(f'[EM FAILED: {e}] ', end='', flush=True)
        ll_em, h2_em = np.nan, np.nan

    try:
        t0 = time.perf_counter()
        result_mcecm = model_true.fit(
            X_synth, algorithm='mcecm', max_iter=200, tol=1e-4,
            regularization='det_sigma_one', verbose=0)
        t_mcecm = time.perf_counter() - t0
        t_mcecm_total += t_mcecm
        model_mcecm = result_mcecm.model
        ll_mcecm = float(model_mcecm.marginal_log_likelihood(X_synth))
        h2_mcecm = float(squared_hellinger(model_true.joint, model_mcecm.joint))
        print(f'[MCECM: {result_mcecm.n_iter} iters, {t_mcecm:.1f}s]', end='', flush=True)
    except Exception as e:
        print(f'[MCECM FAILED: {e}]', end='', flush=True)
        ll_mcecm, h2_mcecm = np.nan, np.nan

    results.append({
        'p': p_val,
        'LL_EM': ll_em, 'H2_EM': h2_em,
        'LL_MCECM': ll_mcecm, 'H2_MCECM': h2_mcecm,
    })

print(f'\n\nTotal EM time:    {t_em_total:.1f}s')
print(f'Total MCECM time: {t_mcecm_total:.1f}s')
```

## Table 4: Comparison between EM and MCECM

```{code-cell} python
df = pd.DataFrame(results)

print('Table 4: Comparison between the EM algorithm and the MCECM algorithm')
print('=' * 75)
print(f'{"p":>4s}  |  {"EM algorithm":^25s}  |  {"MCECM algorithm":^25s}')
print(f'{"":>4s}  |  {"Log-likelihood":>14s}  {"H^2":>8s}  |  {"Log-likelihood":>14s}  {"H^2":>8s}')
print('-' * 75)
for _, row in df.iterrows():
    p = int(row['p'])
    ll_em = f"{row['LL_EM']:.4f}" if not np.isnan(row['LL_EM']) else '    N/A'
    h2_em = f"{row['H2_EM']:.4f}" if not np.isnan(row['H2_EM']) else '    N/A'
    ll_mc = f"{row['LL_MCECM']:.4f}" if not np.isnan(row['LL_MCECM']) else '    N/A'
    h2_mc = f"{row['H2_MCECM']:.4f}" if not np.isnan(row['H2_MCECM']) else '    N/A'
    print(f'{p:+4d}  |  {ll_em:>14s}  {h2_em:>8s}  |  {ll_mc:>14s}  {h2_mc:>8s}')
print('=' * 75)
```
