# Solvers and Bessel Functions

> **Scope.** Why the Bregman solver is decoupled from `ExponentialFamily`,
> why Bessel evaluation is one moment-quadrature kernel with two array
> backends, and why the EM hot path still splits quad forms (JAX) from
> the GIG solve (CPU).
>
> **Where things live.** The `backend × method` matrix is in
> {doc}`exponential_family` § 3. This file owns the deeper rationale.

---

## 1. Bregman Solver (`fitting/solvers.py`)

The η→θ inversion is

$$
\theta_* = \arg\min_\theta\,[\,\psi(\theta) - \theta\cdot\eta\,].
$$

This problem is convex in $\theta$ for any convex $\psi$. The solver
takes $\psi$ as a generic `f` callable, not a log-partition method:

```python
solve_bregman(f, eta, theta0, *, backend, method, bounds,
              grad_fn, hess_fn, max_steps, tol, verbose) -> BregmanResult
```

| Decision | Choice | Rationale |
|---|---|---|
| Generic `f` (vs `log_partition_fn`) | generic | Bregman works for any convex function; the solver shouldn't know about EFs |
| `grad_fn` + `hess_fn` separate | separate, both θ-space | Solver applies $\theta \leftrightarrow \phi$ chain rule via `jax.jacobian(to_theta)`; distributions never touch reparametrization |
| Result type | `BregmanResult` | Survives `lax.scan` (loose-typed `Any` scalars where needed) |
| Multi-start | orthogonal `solve_bregman_multistart` | Not baked into solver names; `vmap` for JAX, Python `for` for CPU |

### 1.1 Bounds: reparam vs native

| Bound | Transform $\theta \to \phi$ | Inverse $\phi \to \theta$ |
|---|---|---|
| $(-\infty, 0)$ | $\phi = \log(-\theta)$ | $\theta = -\exp(\phi)$ |
| $(0, +\infty)$ | $\phi = \log(\theta)$ | $\theta = \exp(\phi)$ |
| $(\ell, h)$ | $\phi = \mathrm{logit}((\theta-\ell)/(h-\ell))$ | $\theta = \ell + (h-\ell)\sigma(\phi)$ |
| $(-\infty, +\infty)$ | $\phi = \theta$ | $\theta = \phi$ |

`backend='cpu'` passes `bounds` directly to
[`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
(native L-BFGS-B box constraints).
[`jaxopt.LBFGSB`](https://jaxopt.github.io/stable/_api/jaxopt.html#jaxopt.LBFGSB)
also supports bounds natively. Other JAX backends reparameterise.

### 1.2 Newton: hand-rolled, JIT-cached

No JAX library provides a Newton **minimizer** that accepts a user-supplied
Hessian:

| Library | Newton | Custom Hessian | Box constraints |
|---|---|---|---|
| [Optimistix](https://docs.kidger.site/optimistix/) | root-finding only | no | no |
| [JAXopt](https://jaxopt.github.io/stable/) | none | n/a | yes (LBFGSB only) |
| [Optax](https://optax.readthedocs.io/en/latest/) | none | n/a | n/a |

So we ship a hand-rolled Newton via `lax.while_loop` (true early
stopping). For repeated warm-started solves on the same shape (the GIG
EM hot path), `make_jit_newton_solver(f, grad_fn, hess_fn, bounds)`
builds a `@jax.jit`-decorated specialised solve whose XLA cache survives
across calls — critical, otherwise per-call retracing dominated GH
EM time.

### 1.3 `BregmanResult` and `lax.scan`

```python
@dataclass(frozen=True)
class BregmanResult:
    theta: jax.Array
    fun:        Any   # may be JAX scalar (under scan) or Python float
    grad_norm:  Any
    num_steps:  int
    converged:  Any   # bool / 0-d JAX bool
    elapsed_time: float = 0.0
```

Loose `Any` typing is deliberate: forcing Python `float`/`bool` would
raise `ConcretizationTypeError` when the result flows through
`lax.scan`. `verbose` is threaded into the solver for printed
diagnostics.

---

## 2. GIG η→θ

The GIG Fisher information can be ill-conditioned (condition number up
to $10^{30}$) when $a \ll b$ or $a \gg b$. Vanilla L-BFGS-B fails
without rescaling.

### 2.1 η-rescaling

Before optimization:

$$
s = \sqrt{\eta_2/\eta_3},\qquad
\tilde\eta = \bigl(\eta_1 + \tfrac12\log(\eta_2/\eta_3),\,\sqrt{\eta_2\eta_3},\,\sqrt{\eta_2\eta_3}\bigr).
$$

The rescaled GIG has $\tilde a = \tilde b = \sqrt{ab}$ and a
symmetric Fisher matrix. After solving for $\tilde\theta$:

$$
\theta = (\tilde\theta_1,\;\tilde\theta_2/s,\;s\cdot\tilde\theta_3).
$$

### 2.2 Solver choice in EM

Default: `backend='cpu', method='lbfgs'` — `scipy.optimize.minimize`
with the numpy Bessel kernel. This avoids GPU kernel dispatch overhead on a
3-D scalar problem.

For the warm-started Newton path (`backend='jax', method='newton'`),
the cached `_gig_jax_newton_jit` keeps a single XLA executable across
all warm-started solves.

When `theta0` is **not** provided, `GeneralizedInverseGaussian.from_expectation`
runs `solve_bregman_multistart` on the η-rescaled problem, with seeds
from the Gamma / InverseGamma / InverseGaussian special cases.

---

## 3. Bessel Functions

{py:func}`normix.utils.bessel.log_kv` is the unified entry point.

One centered whole-line Gauss–Legendre kernel implements both `log_kv`
and {py:func}`normix.utils.bessel.log_kv_moments`. Geometry is frozen
under `stop_gradient`; autodiff of `log_kv` yields cumulants of that
discrete measure. No regimes, no `lax.cond`, no `custom_jvp`, no finite
differences.

### 3.1 The kernel

$$
2K_\nu(z)=\int_{\mathbb R}e^{\nu u-z\cosh u}\,du, \qquad z>0
$$

({ref}`DLMF <dlmf>` [10.32.9](https://dlmf.nist.gov/10.32.E9)). Production
`log_kv` is this integral as a **192-point Gauss–Legendre sum** (96 nodes
on each side of the peak). That is the kernel: a weighted sum, not a
library Bessel call and not a finite-difference stencil.

The integrand peaks at $u_0=\operatorname{asinh}(\nu/z)$. The code shifts
$x=u-u_0$ so the mass sits at the origin. Node locations are frozen
(`stop_gradient`); only the weights depend on $(\nu,z)$.

**Autodiff** here means `jax.grad` / `jax.hessian` of that sum.
Because the nodes are constants,

$$
\partial_z\log K_\nu=-E[\cosh u],\qquad
\partial_\nu\log K_\nu=E[u].
$$

`log_kv_moments(v, z).d_arg` is the first identity written out;
`jax.grad(lambda z: log_kv(v, z))(z)` is the same identity by differentiating
the log-sum-exp. They agree to $\sim 10^{-13}$
(`tests/test_bessel_contract.py::test_ad_equals_bundle`). GIG $\eta$ and
$H=D\,\mathrm{cov}\,D$ use the moment bundle so both come from one pass.

`backend='jax'` and `backend='cpu'` are the same sums in JAX and NumPy.
The exponent uses $\mathrm{expm1}$ with $\kappa-\nu=z^2/(\kappa+\nu)$.

`log_kv_moments` returns `BesselMoments(log_k, u0, mean, cov)` of
$(x, e^{-x}-1, e^{x}-1)$. [`scipy.special.kve`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kve.html)
({ref}`Amos1986 <amos1986>`) is a test oracle, not a runtime path.

Hankel / Olver / small-$z$ formulae ( {ref}`DLMF <dlmf>`
[10.40.2](https://dlmf.nist.gov/10.40.E2),
[10.41.3–4](https://dlmf.nist.gov/10.41),
[10.30.2](https://dlmf.nist.gov/10.30.E2) ) are identities in the
contract tests, not production branches. The old four-regime `lax.cond`
plus FD $\partial_\nu$ produced $L_{\nu\nu}<0$ at GIG$(25,1,1)$.

### 3.2 Why `backend` is a Python-level string

Resolved before JAX tracing begins. `backend='jax'` keeps the code
traceable; `backend='cpu'` runs eagerly — appropriate because EM loops
are already Python `for` loops at the CPU end.

### 3.3 CPU triad for Bessel-dependent distributions

**Design rule:** any distribution that calls `log_kv` must override the
Tier 3 CPU classmethods so the CPU solver path
(`solve_bregman(backend='cpu')`) avoids JAX dispatch entirely. The
three classmethods are `_log_partition_cpu`, `_grad_log_partition_cpu`,
`_hessian_log_partition_cpu` — all numpy in / numpy out.

Distributions that don't call `log_kv` (Gamma, InverseGamma,
InverseGaussian) inherit the default wrappers. They pay nothing.

---

## 4. CPU/GPU Hybrid Backend

EM timing on 468 stocks, 2552 observations (GH; pre-S10, `kve` on CPU):

| Phase | JAX (GPU) | CPU hybrid | Speedup |
|---|---|---|---|
| E-step | ~1.1 s | ~0.07 s | ~15× |
| M-step (GIG solve) | ~5–7 s | ~0.01 s | ~500× |

After S10 the CPU E-step is the same 192-node kernel in NumPy, not
`kve`; it is slower than AMOS on $N=2552$. The split (quad forms in JAX,
GIG solve on CPU) is unchanged. `_fit_defaults` is a separate decision.

**Hybrid strategy:**

- Quad forms ($L_\Sigma^{-1}(x-\mu)$ etc.) stay in JAX (d-dimensional,
  GPU-friendly).
- `log_kv` calls and GIG optimization move to CPU
  (`backend='cpu'`).

`NormalMixture.e_step(X, backend='cpu')` is the hybrid path:

- Quad forms (`L⁻¹(x−μ)`, `‖z‖²`, `‖w‖²`) stay in JAX `vmap`
  (GPU-friendly).
- Bessel calls go to CPU via `GIG.expectation_params_batch(backend='cpu')`
  (same quadrature as JAX, NumPy backend).
- `_posterior_gig_params(z2, w2)` lives on each
  `JointNormalMixture` subclass.

Default fitter settings reflect the hot path:
`e_step_backend='jax'`, `m_step_backend='cpu'`, `m_step_method='newton'`.

---

## 5. Random Variate Generation

PINV (Polynomial-Interpolation-based Numerical Inversion;
{ref}`HormannLeydold2011 <hormannleydold2011>`) in `utils/rvs.py` is pure JAX
and works for any univariate log-kernel — no normalising constant needed:

- `build_pinv_table(log_kernel, mode, *, x_of_w, n_grid, tail_eps)`
  builds a quantile table in JAX. Tail bisection via `lax.fori_loop`,
  trapezoidal CDF via `jnp.cumsum`.
- `rvs_pinv(key, u_grid, x_grid, n)` samples via `jnp.interp`
  (GPU-friendly, vectorised).

Distributions on $(0,\infty)$ supply
`log_kernel(w) = log_prob(exp(w)) + w` and seed the table at
`jnp.log(self.mode())`. Closed-form `mode()` lives on the distribution
itself (`Gamma`, `InverseGamma`, `InverseGaussian`, `GIG`).
`InverseGaussian.ppf` and both `GIG.cdf` / `GIG.ppf` inline a single
`build_pinv_table` call — `log_prob` is the only kernel.

GIG-specific sampling is inlined in
`distributions/generalized_inverse_gaussian.py`:

- `_gig_rvs_devroye(key, p, a, b, n)` — TDR on $w = \log x$
  ({ref}`Devroye2014 <devroye2014>`), batch-parallel (no `while_loop`).
- `GIG.rvs(method='pinv')` — `quantile_table().rvs` via
  `build_pinv_table` / `rvs_pinv` in `utils/rvs.py`.

Neither method evaluates the Bessel normalising constant.

### Quantile Functions (`cdf`, `ppf`)

- `Gamma.ppf` and `InverseGamma.ppf` invert the regularised incomplete
  gamma via `normix.utils.gammaincinv` — a pure-JAX Newton iteration on
  `jax.scipy.special.gammainc` with a Wilson–Hilferty seed
  ({ref}`WilsonHilferty1931 <wilsonhilferty1931>`). This is the JAX analogue of
  `scipy.special.gammaincinv`.
- `InverseGaussian.ppf`, `GIG.cdf`, `GIG.ppf` build a PINV table from
  `log_prob` (above).
- Univariate `Normal`-mixture marginals (`UnivariateVarianceGamma`,
  `UnivariateNormalInverseGamma`, `UnivariateNormalInverseGaussian`,
  `UnivariateGeneralizedHyperbolic`) use the same generic PINV machinery
  with `log_kernel(w) = self.log_prob(jnp.atleast_1d(w))`, seeded at
  `self.mean()` (no closed-form mode for Bessel mixtures).

---

## 6. Cross-References

- Triad design: {doc}`exponential_family`.
- Why EM / `fit_mle` rather than NLL gradient descent: {doc}`why_not_gradient_descent`.
- Theory: {doc}`GIG distribution <../theory/gig>`, {doc}`EM algorithm <../theory/em_algorithm>`.
