---
file_format: mystnb
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  execution_mode: cache
  execution_timeout: 180
---

# Bessel functions and `log_kv`

The densities of the GIG and Generalized Hyperbolic distributions are written
in terms of the **modified Bessel function of the second kind**, $K_\nu(z)$
({ref}`DLMF <dlmf>` §10). Evaluating it naively overflows and underflows badly,
and the standard library versions are neither JIT-able nor differentiable.
normix provides `log_kv`, a log-space implementation of $K_\nu$ as a
192-point quadrature (two array backends):

$$
\texttt{log\_kv}(\nu, z) = \log K_\nu(z).
$$

`jax.grad` differentiates that sum; `log_kv_moments` writes the same
derivatives as expectations (shown below).

```{code-cell} python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from normix import log_kv, log_kv_moments
from normix.utils.plotting import set_theme

set_theme()
np.set_printoptions(precision=6, suppress=False)
```

## Two backends, one function

`log_kv` has a JIT-able [JAX](https://docs.jax.dev/en/latest/) backend (the
default) and a [NumPy](https://numpy.org/doc/stable/) CPU backend. Both agree,
and both match
[`scipy.special.kve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kve.html)
(the exponentially-scaled Bessel function, $K_\nu(z)\,e^{z}$, via AMOS
{ref}`Amos1986 <amos1986>` — used here as a reference, not a runtime path):

```{code-cell} python
from scipy.special import kve

v, z = 1.7, 4.0
jax_val = float(log_kv(v, z))               # JAX backend
cpu_val = float(log_kv(v, z, backend="cpu"))  # numpy/scipy backend
ref = float(np.log(kve(v, z)) - z)           # log K_v = log kve - z

print(f"log_kv  (jax) = {jax_val:.12f}")
print(f"log_kv  (cpu) = {cpu_val:.12f}")
print(f"scipy   ref   = {ref:.12f}")
```

## Symmetry and vectorization

$K_\nu = K_{-\nu}$, and `log_kv` broadcasts over both arguments like any JAX
ufunc, so you can `vmap` or evaluate on grids directly:

```{code-cell} python
print("K_v == K_-v :", bool(jnp.allclose(log_kv(0.5, 2.0), log_kv(-0.5, 2.0))))

vs = jnp.array([0.0, 0.5, 1.0, 2.0])
zs = jnp.linspace(0.5, 5.0, 4)
grid = log_kv(vs[:, None], zs[None, :])   # (4, 4) via broadcasting
print("grid shape:", grid.shape)
```

## Numerical stability in the tails

For large $z$, $K_\nu(z)$ decays like $e^{-z}$ and underflows to exactly zero in
double precision — so `log(scipy.special.kv(...))` returns $-\infty$. Because
`log_kv` works in log space throughout, it stays finite and accurate:

```{code-cell} python
from scipy.special import kv

for z_big in [50.0, 200.0, 700.0]:
    with np.errstate(divide="ignore"):
        naive = np.log(kv(0.5, z_big))        # underflows to -inf for large z
    stable = float(log_kv(0.5, z_big))
    print(f"z = {z_big:6.1f}   log(kv) = {naive:>10}   log_kv = {stable:.4f}")
```

```{code-cell} python
import matplotlib.pyplot as plt

zgrid = jnp.linspace(0.05, 20.0, 400)
fig, ax = plt.subplots()
for nu in [0.0, 1.0, 5.0, 20.0]:
    ax.plot(np.asarray(zgrid), np.asarray(log_kv(nu, zgrid)), label=f"$\\nu={nu:g}$")
ax.set_xlabel("z"); ax.set_ylabel(r"$\log K_\nu(z)$")
ax.set_title("log_kv across orders")
ax.legend()
plt.show()
```

## How `log_kv` is computed

$K_\nu(z)$ has the whole-line integral representation
({ref}`DLMF <dlmf>` [10.32.9](https://dlmf.nist.gov/10.32.E9))

$$
2K_\nu(z)=\int_{\mathbb R}e^{\nu u-z\cosh u}\,du, \qquad z>0.
$$

normix does not call a library Bessel routine. It replaces that integral by a
**weighted sum of 192 function values** (Gauss–Legendre quadrature, two panels).
That sum is the 192-node kernel. Both `backend="jax"` and `backend="cpu"`
evaluate the same sum; there is no regime dispatch.

The integrand of $u \mapsto e^{\nu u-z\cosh u}$ peaks at

$$
u_0=\operatorname{asinh}(\nu/z).
$$

The code shifts $x=u-u_0$ so the mass sits at $x=0$, then places 96 nodes on
each side of the peak. Node locations are treated as constants
(`stop_gradient`); only the weights still depend on $(\nu,z)$.

## Derivatives: autodiff and `log_kv_moments`

**Autodiff** (automatic differentiation) is `jax.grad`: it differentiates the
JAX expression for `log_kv`. It is not a finite-difference stencil. Because the
nodes are frozen, that derivative is exactly an expectation under the 192-point
discrete measure:

$$
\partial_z\log K_\nu = -E[\cosh u], \qquad
\partial_\nu\log K_\nu = E[u].
$$

`log_kv_moments(v, z)` evaluates those expectations from the same weights and
stores them as `d_arg` and `d_order`. So

```python
jax.grad(lambda z: log_kv(v, z))(z)   # autodiff of the sum
log_kv_moments(v, z).d_arg            # the same expectation, written out
```

are the same calculation, two ways to read one quadrature:

```{code-cell} python
v0, z0 = 1.3, 2.5
z_arr = jnp.array(z0)
ad_z = float(jax.grad(lambda z: log_kv(v0, z))(z_arr))
ad_v = float(jax.grad(lambda v: log_kv(v, z0))(jnp.array(v0)))
m = log_kv_moments(v0, z0)

print(f"jax.grad d/dz  = {ad_z:.12f}")
print(f"d_arg          = {float(m.d_arg):.12f}")
print(f"jax.grad d/dv  = {ad_v:.12f}")
print(f"d_order        = {float(m.d_order):.12f}")
```

GIG’s $\eta=\nabla\psi$ and Fisher $H$ use `log_kv_moments` so the mean and the
$3\times 3$ covariance come from one pass. Ordinary `log_prob` code can keep
using `jax.grad(log_kv)`.

The $z$-derivative also matches the classical recurrence
$K_\nu'(z)=-\tfrac12\big(K_{\nu-1}(z)+K_{\nu+1}(z)\big)$:

```{code-cell} python
recur = -0.5 * (
    float(jnp.exp(log_kv(v0 - 1, z0) - log_kv(v0, z0)))
    + float(jnp.exp(log_kv(v0 + 1, z0) - log_kv(v0, z0)))
)
print(f"recurrence d/dz = {recur:.12f}")
```

## Which backend should I use?

- **`backend="jax"`** (default) — use inside anything that is JIT-compiled,
  differentiated with `jax.grad`, or vectorized with `jax.vmap`, and on GPU.
  This is what distribution `log_prob` methods call.
- **`backend="cpu"`** — the same sums in NumPy. Use it from Python EM
  loops (`e_step_backend="cpu"`) that should not dispatch through JAX.

The two are numerically interchangeable; the choice is purely about
performance and the surrounding execution context.

## Takeaways

- `log_kv(v, z)` is a 192-point quadrature for $\log K_\nu(z)$, not a call
  to `scipy.special.kv`.
- $u_0=\operatorname{asinh}(\nu/z)$ is the mode of the integrand; nodes sit
  around that peak.
- `jax.grad(log_kv)` is autodiff of that sum. `log_kv_moments(v, z).d_arg`
  is the same $\partial_z$ as an explicit average; `d_order` is
  $\partial_\nu$. Neither is a finite difference.
- Pick `backend="jax"` for JIT/grad/vmap/GPU; `backend="cpu"` for a NumPy
  EM loop that should not enter JAX.

Next: {doc}`04_random_sampling` uses these densities to draw and validate
samples from every distribution.
