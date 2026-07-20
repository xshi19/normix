# Solvers and Bessel Functions

> **Scope.** Why the Bregman solver is decoupled from `ExponentialFamily`,
> why Bessel evaluation has two backends and four numerical regimes,
> and why the EM hot path runs on a CPU/GPU hybrid.
>
> **Where things live.** The `backend × method` matrix is in
> `exponential_family.md` § 3. ARCHITECTURE.md has the
> compact runtime view. This file owns the deeper rationale.

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

`backend='cpu'` passes `bounds` directly to `scipy.optimize.minimize`
(native L-BFGS-B box constraints). `jaxopt.LBFGSB` also supports bounds
natively. Other JAX backends reparameterise.

### 1.2 Newton: hand-rolled, JIT-cached

No JAX library provides a Newton **minimizer** that accepts a user-supplied
Hessian:

| Library | Newton | Custom Hessian | Box constraints |
|---|---|---|---|
| Optimistix | root-finding only | no | no |
| JAXopt | none | n/a | yes (LBFGSB only) |
| Optax | none | n/a | n/a |

So we ship a hand-rolled Newton via `lax.while_loop` (true early
stopping). For repeated warm-started solves on the same shape (the GIG
EM hot path), `make_jit_newton_solver(f, grad_fn, hess_fn, bounds)`
builds a `@jax.jit`-decorated specialised solve whose XLA cache survives
across calls — critical, otherwise per-call retracing dominated GH
EM time (see `../tech_notes/jax_overhead_diagnosis.md` § Resolution).

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
with `scipy.special.kve`. This avoids GPU kernel dispatch overhead on a
3-D scalar problem.

For the warm-started Newton path (`backend='jax', method='newton'`),
the cached `_gig_jax_newton_jit` keeps a single XLA executable across
all warm-started solves.

When `theta0` is **not** provided, `GeneralizedInverseGaussian.from_expectation`
runs `solve_bregman_multistart` on the η-rescaled problem, with seeds
from the Gamma / InverseGamma / InverseGaussian special cases.

See `../tech_notes/gig_eta_to_theta.md` for the derivation, the
seven-Bessel analytical Hessian, and benchmark comparisons.

---

## 3. Bessel Functions

`log_kv(v, z, backend='jax'|'cpu')` is the unified entry point in
`normix/utils/bessel.py`.

### 3.1 Pure-JAX backend (default)

Four-regime dispatch via `lax.cond` (only the selected branch executes
at runtime):

| Regime | Trigger | Method |
|---|---|---|
| Hankel | $z > \max(25, v^2/4)$ | DLMF 10.40.2 asymptotic |
| Olver | $\|v\| > 25$, not Hankel | DLMF 10.41.3-4 uniform expansion |
| Small-$z$ | $z < 10^{-6}$, $\|v\| > 0.5$ | leading asymptotic |
| Quadrature | otherwise | 64-point Gauss–Legendre (Takekawa 2022) |

Custom JVP via `@jax.custom_jvp`:

- $\partial/\partial z$: exact recurrence
  $K'_\nu = -(K_{\nu-1} + K_{\nu+1})/2$.
- $\partial/\partial v$: central FD with $\varepsilon = 10^{-5}$.

### 3.2 CPU backend (EM hot path)

`scipy.special.kve`, fully vectorised NumPy. Not JIT-able. For large
$N$ a single `kve` C-call per element beats vmapping JAX's
`lax.cond`-dispatched implementation, which causes separate kernel
launches per condition check.

### 3.3 Why `backend` is a Python-level string

Resolved before JAX tracing begins. `backend='jax'` keeps the code
traceable; `backend='cpu'` runs eagerly — appropriate because EM loops
are already Python `for` loops at the CPU end.

### 3.4 CPU triad for Bessel-dependent distributions

**Design rule:** any distribution that calls `log_kv` must override the
Tier 3 CPU classmethods so the CPU solver path
(`solve_bregman(backend='cpu')`) avoids JAX dispatch entirely. The
three classmethods are `_log_partition_cpu`, `_grad_log_partition_cpu`,
`_hessian_log_partition_cpu` — all numpy in / numpy out.

Distributions that don't call `log_kv` (Gamma, InverseGamma,
InverseGaussian) inherit the default wrappers. They pay nothing.

See `../tech_notes/bessel_implementations_survey.md` for benchmarks.

---

## 4. CPU/GPU Hybrid Backend

EM timing on 468 stocks, 2552 observations (GH distribution):

| Phase | JAX (GPU) | CPU hybrid | Speedup |
|---|---|---|---|
| E-step | ~1.1 s | ~0.07 s | ~15× |
| M-step (GIG solve) | ~5–7 s | ~0.01 s | ~500× |

**Hybrid strategy:**

- Quad forms ($L_\Sigma^{-1}(x-\mu)$ etc.) stay in JAX (d-dimensional,
  GPU-friendly).
- `log_kv` calls and GIG optimization move to CPU
  (`backend='cpu'`).

`NormalMixture.e_step(X, backend='cpu')` is the hybrid path:

- Quad forms (`L⁻¹(x−μ)`, `‖z‖²`, `‖w‖²`) stay in JAX `vmap`
  (GPU-friendly).
- Bessel calls go to CPU via `GIG.expectation_params_batch(backend='cpu')`.
- `_posterior_gig_params(z2, w2)` lives on each
  `JointNormalMixture` subclass.

Default fitter settings reflect the hot path:
`e_step_backend='jax'`, `m_step_backend='cpu'`, `m_step_method='newton'`.

See `../tech_notes/em_gpu_profiling.md`.

---

## 5. Random Variate Generation

PINV (Polynomial-Interpolation-based Numerical Inversion) in
`utils/rvs.py` is pure JAX and works for any univariate log-kernel — no
normalising constant needed:

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

- `_gig_rvs_devroye(key, p, a, b, n)` — TDR on $w = \log x$,
  batch-parallel (no `while_loop`).
- `_gig_rvs_pinv(key, u_grid, x_grid, n)` — alias of `rvs_pinv` used by
  `GIG.rvs(method='pinv')`.

Neither method evaluates the Bessel normalising constant. See
`../tech_notes/gig_rvs.md`.

### Quantile Functions (`cdf`, `ppf`)

- `Gamma.ppf` and `InverseGamma.ppf` invert the regularised incomplete
  gamma via `normix.utils.gammaincinv` — a pure-JAX Newton iteration on
  `jax.scipy.special.gammainc` with a Wilson–Hilferty seed. This is the
  JAX analogue of `scipy.special.gammaincinv`.
- `InverseGaussian.ppf`, `GIG.cdf`, `GIG.ppf` build a PINV table from
  `log_prob` (above).
- Univariate `Normal`-mixture marginals (`UnivariateVarianceGamma`,
  `UnivariateNormalInverseGamma`, `UnivariateNormalInverseGaussian`,
  `UnivariateGeneralizedHyperbolic`) use the same generic PINV machinery
  with `log_kernel(w) = self.log_prob(jnp.atleast_1d(w))`, seeded at
  `self.mean()` (no closed-form mode for Bessel mixtures).

### 5.1 Quantile-table reuse: `QuantileTable` (DEC-5)

> Decision row: `design.md` § *2026-07 review Phase 0*, DEC-5. Decided
> 2026-07-20; implementation lands with roadmap item E3 (Phase 5).

Every `cdf`/`ppf` call above rebuilds the 4000-point table — 4000
Bessel-heavy `log_prob` evaluations per call. Distributions are
immutable (`eqx.Module`, row F1), so self-caching is out; DEC-5 gives
repeated quantile workloads an explicit handle instead:

```python
class QuantileTable(eqx.Module):
    """Frozen PINV quantile table; a pytree, so jit/vmap/scan-safe."""
    u_grid: jax.Array   # (n_grid,) trapezoidal CDF values
    x_grid: jax.Array   # (n_grid,) observation-axis values

    def cdf(self, x): ...   # jnp.interp(x, x_grid, u_grid, left=0.0, right=1.0)
    def ppf(self, q): ...   # jnp.interp(q, u_grid, x_grid)
    def rvs(self, n, seed=42): ...   # inverse-CDF sampling (rvs_pinv)
```

- Lives in `utils/rvs.py` next to `build_pinv_table`, which stays the
  raw functional core (the table object holds its output; interp
  bookkeeping like `left=0.0, right=1.0` moves off the call sites).
- PINV-backed distributions gain `quantile_table() -> QuantileTable`
  (the `_UnivariateNormalMixtureMixin`, `GIG`, `InverseGaussian`);
  their `cdf`/`ppf`/`rvs(method='pinv')` re-express through it with
  per-call semantics unchanged — holding the table is the caller's
  opt-in amortisation.
- Interaction: GIG's degenerate-regime `cdf`/`ppf` delegation to
  Gamma/InverseGamma closed forms (roadmap B4) bypasses the table —
  `quantile_table()` is only meaningful in the PINV regime; E3's
  implementation must not regress B4's resolution.

Alternative rejected: documenting `u_grid, x_grid = dist._pinv_grids()`
reuse. The helper exists only on the univariate-mixin today (no uniform
accessor across GIG / InverseGaussian), raw tuples push interp
bookkeeping onto every caller, and a bare tuple communicates nothing
about grid semantics. The table object is the same two arrays with the
three obvious methods attached.

---

## 6. Cross-References

- Architecture surface: `../ARCHITECTURE.md` § *Bessel Functions*,
  § *GIG η→θ Optimization*.
- Triad design: `exponential_family.md`.
- Tech notes: `../tech_notes/bessel_implementations_survey.md`,
  `../tech_notes/gig_eta_to_theta.md`,
  `../tech_notes/em_gpu_profiling.md`,
  `../tech_notes/jax_overhead_diagnosis.md`,
  `../tech_notes/gig_rvs.md`.
- Historical / archived: `../archive/design/solver_redesign.md`,
  `../archive/design/log_partition_triad.md`.
