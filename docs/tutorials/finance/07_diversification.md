---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 600
---

# Diversification: variance ENB vs CVaR ENB

Meucci's **effective number of bets** (ENB) asks how many uncorrelated risk
factors a portfolio is truly exposed to. The answer depends on which risk you
diagonalize. Variance ENB uses $\mathrm{Cov}[X]$; generalized ENB uses the
Hessian of squared coherent risk $\rho^2$ — here CVaR — and can see
tail concentrations that a second-moment measure misses.

Formal derivations: {doc}`../../theory/enb`,
{doc}`../../theory/generalized_enb`.

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from normix import VarianceGamma
from normix.finance import CVaR, VarianceENB, GeneralizedENB, MinimumTorsion
from normix.utils.plotting import set_theme, COLORS, COLOR_CYCLE

set_theme()
np.set_printoptions(precision=4, suppress=True)
```

## Synthetic case: $\mathrm{Cov}[X] \propto I$, one skewed name

Construct a four-asset VG portfolio whose *return covariance* is a multiple of
the identity, but asset 0 carries all the skewness. Variance ENB must then
report $N = 4$ for equal weights; CVaR ENB should load more risk on asset 0.

```{code-cell} python
E_Y, Var_Y = 1.0, 0.2
g = -1.2
beta = E_Y / Var_Y
alpha = E_Y * beta
s2 = jnp.ones(4).at[0].set(1.0 - Var_Y * g * g / E_Y)

model = VarianceGamma.from_classical(
    mu=jnp.zeros(4),
    gamma=jnp.zeros(4).at[0].set(g),
    sigma=jnp.diag(s2),
    alpha=alpha,
    beta=beta,
)
cov = np.asarray(model.cov())
print("Cov[X] diagonals:", np.diag(cov))
print("off-diagonal max |entry|:", np.max(np.abs(cov - np.diag(np.diag(cov)))))

w = jnp.full(4, 0.25)
Y = model.joint.subordinator().rvs(40_000, seed=0)

res_var = VarianceENB().evaluate(model, w)
res_cvar = GeneralizedENB(CVaR(0.01)).evaluate(model, w, Y)

print(f"variance ENB = {float(res_var.enb):.3f}   vol = {float(res_var.risk):.4f}")
print(f"CVaR ENB     = {float(res_cvar.enb):.3f}   CVaR = {float(res_cvar.risk):.4f}")
print("p_var :", np.asarray(res_var.p))
print("p_cvar:", np.asarray(res_cvar.p))
```

```{code-cell} python
fig, ax = plt.subplots()
idx = np.arange(4)
ax.bar(idx - 0.2, np.asarray(res_var.p), 0.4, label="variance", color=COLOR_CYCLE[0])
ax.bar(idx + 0.2, np.asarray(res_cvar.p), 0.4, label="CVaR", color=COLORS["brick"])
ax.set_xticks(idx)
ax.set_xticklabels([f"asset {i}" for i in idx])
ax.set_ylabel("normalized risk contribution $p_k$")
ax.set_title("Where the risk lives")
ax.legend()
fig.tight_layout()
```

Variance treats the four names as exchangeable. CVaR assigns a larger share of
tail risk to the skewed asset — the gap that justifies a second ENB.

## Minimum torsion factors

The default diagonalization is constrained minimum torsion
(`MinimumTorsion`). The transform $T$ itself is reusable for Meucci-style
factor construction:

```{code-cell} python
dec = MinimumTorsion().decompose(model.cov())
print("T Cov T' ≈ I?", np.allclose(dec.T @ model.cov() @ dec.T.T, np.eye(4), atol=1e-10))
print("diag(D) =", np.asarray(dec.d))
```

## API sketch

```python
from normix.finance import VarianceENB, GeneralizedENB, CVaR

Y = model.joint.subordinator().rvs(n, seed=0)   # once; common random numbers
res_var  = VarianceENB().evaluate(model, w)                 # no Y
res_cvar = GeneralizedENB(CVaR(0.05)).evaluate(model, w, Y) # one CMC VaR solve
```

Both return the same `ENBResult` pytree (`enb`, `p`, `risk`, `v`, `d`, `T`,
`eigenvalues`). Swap the torsion with `VarianceENB(torsion=PCATorsion())` when
you want a principal-component baseline.
