---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 600
---

# Transaction costs and local-quadratic rebalancing

The mean-risk reduction of {doc}`05_mean_risk_optimization` collapses portfolio
choice to two coordinates $(\tilde\mu, \tilde\gamma)$. An $\ell_1$ turnover
penalty breaks that reduction: $\|w - w_0\|_1$ is not a function of the reduced
coordinates alone. When transaction costs keep the solution near the current
portfolio $w_0$, a second-order Taylor expansion of the coherent risk $r$
recovers a **quadratic program** in buy/sell variables. This tutorial walks
through that construction on the same Dow basket used in the mean-risk page.

Formal derivation: {doc}`../../theory/transaction_costs`.

This page is intentionally separate from
{doc}`05_mean_risk_optimization`: that tutorial is about the frictionless
dimension reduction; here the turnover term *breaks* that reduction and the
API is a different object (`TransactionCostProblem`).

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path

from normix import GeneralizedHyperbolic
from normix.finance import CVaR, MeanRiskProblem, TransactionCostProblem
from normix.utils.plotting import set_theme, COLORS

set_theme()
np.set_printoptions(precision=5, suppress=True)
```

## Fit once, rebalance often

```{code-cell} python
basket = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "WMT", "CAT"]
data_path = Path("../../../data/sp500_returns.csv").resolve()
R = pd.read_csv(data_path, index_col="Date", parse_dates=True)[basket].dropna()
X = jnp.asarray(R.values, dtype=jnp.float64)

model = (GeneralizedHyperbolic.default_init(X)
         .fit(X, max_iter=100, tol=1e-4, e_step_backend="cpu").model
         .regularize_a_eq_b())
cvar = CVaR(0.05)
Y = model.joint.subordinator().rvs(15_000, seed=0)
print(f"mean log-likelihood {float(model.marginal_log_likelihood(X)):.4f}")
```

Take the minimum-CVaR frontier portfolio as the *current* holding $w_0$ — a
manager who already solved the frictionless problem and now faces a cost of
trading.

```{code-cell} python
prob = MeanRiskProblem(model, cvar)
mv_mu, mv_gamma = prob.min_variance_point()
mv_ret = float(prob.expected_return(mv_mu, mv_gamma))
targets = jnp.linspace(mv_ret, mv_ret + 4e-4, 25)
ga = float(mv_gamma)
span = 1.5 * max(float(np.asarray(model.gamma).max()
                      - np.asarray(model.gamma).min()), 4e-4)
frontier = prob.efficient_frontier(
    targets, Y, gamma_bounds=(ga - span, ga + span), n_iter=40)
k = int(np.argmin(np.asarray(frontier.risk)))
w0 = frontier.weights[k]
print(f"anchor return {float(frontier.expected_return[k]):.3e}  "
      f"CVaR {float(frontier.risk[k]):.4f}")
pd.Series(np.asarray(w0), index=basket).round(3)
```

## Local quadratic program

`TransactionCostProblem` reuses the same `CVaR` object — only its weight-space
gradient and Hessian at $w_0$ enter the QP. The objective is

$$
\max_w \; w^\top m - c_1\, r(w) - c_2 \|w - w_0\|_1
\quad\text{s.t.}\quad w^\top e = 1,\; w \ge 0,
$$

approximated by replacing $r$ with its Taylor model at $w_0$ and introducing
buy/sell variables $v = (v^+; v^-)$.

```{code-cell} python
tc = TransactionCostProblem(model, cvar, c1=5.0, c2=5e-2)
A = -jnp.eye(len(basket), dtype=jnp.float64)   # long-only: -w ≤ 0
b = jnp.zeros(len(basket), dtype=jnp.float64)
result = tc.solve(w0, Y, A=A, b=b)

print(f"improved over hold?  {bool(result.improved)}")
print(f"turnover ‖Δw‖₁      {float(result.turnover):.4f}")
print(f"approx obj  hold → * {float(result.hold_objective):.6e} → "
      f"{float(result.approx_objective):.6e}")
print(f"exact  obj  hold → * {float(tc.true_objective_at(w0, w0, Y)):.6e} → "
      f"{float(tc.true_objective_at(result.weights, w0, Y)):.6e}")
```

The matrices themselves are available for an external QP solver if needed:

```{code-cell} python
qp = result.qp
print(f"m̃ shape {qp.m_tilde.shape},  H̃ shape {qp.H_tilde.shape}")
print(f"budget residual ẽᵀv = {float(qp.e_tilde @ result.v):.2e}")
```

## How costs reshape the trade

Sweeping $c_2$ shows the continuum from frictionless rebalancing to a pure hold.

```{code-cell} python
import matplotlib.pyplot as plt

c2_grid = np.geomspace(1e-3, 2e-1, 12)
rows = []
for c2 in c2_grid:
    r = TransactionCostProblem(model, cvar, c1=5.0, c2=float(c2)).solve(
        w0, Y, A=A, b=b)
    rows.append({
        "c2": c2,
        "turnover": float(r.turnover),
        "approx_gain": float(r.approx_objective - r.hold_objective),
        "exact_gain": float(
            tc.true_objective_at(r.weights, w0, Y)
            - tc.true_objective_at(w0, w0, Y)
        ) if bool(r.improved) else 0.0,
    })
sweep = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.plot(sweep["c2"], sweep["turnover"], "o-", color=COLORS["brick"], lw=2)
ax.set_xscale("log")
ax.set_xlabel(r"turnover penalty $c_2$")
ax.set_ylabel(r"turnover $\|w^\star - w_0\|_1$")
ax.set_title("Local QP: turnover vs transaction-cost weight")
plt.show()
sweep.round(5)
```

At large $c_2$ the solver correctly returns the hold ($w^\star = w_0$). At small
$c_2$ the quadratic model can recommend a large move — outside the regime where
the Taylor expansion is trustworthy — so in practice one verifies that the
**exact** Monte Carlo objective also improves, exactly as the theory note warns.

```{code-cell} python
cmp = pd.DataFrame(
    {"w0": np.asarray(w0), "w*": np.asarray(result.weights)},
    index=basket,
)
cmp["Δ"] = cmp["w*"] - cmp["w0"]
cmp.round(3)
```

## Takeaways

- Transaction costs live in an **optimization layer**, not inside the risk
  measure: the same `CVaR` object feeds both the efficient surface and the
  local QP.
- `TransactionCostProblem.build_qp` exposes the theory matrices
  $(\tilde m, \tilde H, \tilde e, \tilde A, \tilde b)$; `.solve` runs a
  SciPy SLSQP default without adding a QP dependency.
- Always check `improved` (and ideally the exact objective): if the approximate
  gain is non-positive, hold $w_0$.

See {doc}`../../theory/transaction_costs` for the QP derivation and
{doc}`05_mean_risk_optimization` for the frictionless surface this rebalancing
starts from.
