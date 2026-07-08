# Mixture Architecture

> **Scope.** Why normal-mixture distributions are split into a Joint and
> a Marginal class, why the joint is itself an `ExponentialFamily`,
> and how the factor-analysis family slots in as a sibling without
> forcing the joint hierarchy to lie about its EF signature.
>
> **Where things live.** Class diagram and storage table are in
> `../ARCHITECTURE.md` § *Mixture Structure*. EM details are in
> `em_framework.md`.

---

## 1. Two Classes, Not One

The Generalized Hyperbolic family is a normal variance–mean mixture:

$$X \mid Y \sim \mathcal{N}(\mu + \gamma y,\; \Sigma y),\qquad Y \sim \text{subordinator}.$$

```
JointNormalMixture(ExponentialFamily)        f(x, y) — IS an exponential family
    ↑                                         (closed-form natural / sufficient
JointVarianceGamma, JointNIG, …               statistics; exact M-step)

MarginalMixture(eqx.Module)                  abstract; fitter contract
    ↑                                         (Bessel-required marginal density)
NormalMixture, FactorNormalMixture           f(x) = ∫ f(x,y) dy — NOT an EF
    ↑                                         on its own
VarianceGamma, GH, FactorGH, …
```

The joint is exponential — we exploit that for closed-form M-steps. The
marginal needs numerical integration (Bessel functions) — it cannot be an
EF. Mixing the two roles into one class breaks both stories.

---

## 2. D2 — Joint Classes Are Public

`JointVarianceGamma`, `JointNormalInverseGamma`,
`JointNormalInverseGaussian`, `JointGeneralizedHyperbolic`, and the
abstract `JointNormalMixture` are first-class public `ExponentialFamily`
objects: exported from `normix`, documented alongside marginals, intended
for direct use where the joint law $f(x,y)$ matters (simulation,
complete-data MLE, divergences, custom EM variants).

### 2.1 Observation vector convention

`sufficient_statistics`, `log_base_measure`, and inherited `log_prob` /
`pdf` take a **single flat array** `xy = jnp.concatenate([x, y])`, with
$x$ of shape `(d,)` and scalar $y > 0$ last. This matches the
sufficient-statistic block $[\log y,\,1/y,\,y,\,\ldots]$ used internally.
For readability, `log_prob_joint(x, y)` and `rvs(n, seed) -> (X, Y)`
remain the preferred entry points when $x$ and $y$ are already separate.

### 2.2 GIG sign convention

For GIG-based joints, natural parameters $\theta_2,\theta_3$ must align
with the **GIG** convention
$\theta_{\mathrm{GIG}} = [p-1,\,-b/2,\,-a/2]$ on $[\log y,\,1/y,\,y]$:
scalar coefficients $-(b/2 + \cdots)$ and $-(a/2 + \cdots)$ on $1/y$ and
$y$, **not** $-b$ and $-a$. Gamma and InverseGamma joints already match
this limit; GH and NIG joints follow the same pattern.

### 2.3 `from_natural` for joints

`JointGeneralizedHyperbolic` inverts the full joint family directly.
`JointVarianceGamma`, `JointNormalInverseGamma`, and
`JointNormalInverseGaussian` validate that `theta` lies on the
constrained subfamily before reconstructing classical parameters.

---

## 3. Marginal Mixture API

`MarginalMixture` (in `normix/mixtures/marginal.py`) is the abstract
contract that fitters and downstream code depend on:

```python
class MarginalMixture(eqx.Module):
    # distribution surface
    def log_prob(self, x: jax.Array) -> jax.Array: ...
    def pdf(self, x: jax.Array) -> jax.Array: ...
    def mean(self) -> jax.Array: ...
    def cov(self) -> jax.Array: ...
    def rvs(self, n: int, seed: int = 42) -> jax.Array: ...
    def marginal_log_likelihood(self, X: jax.Array) -> jax.Array: ...

    # EM hooks (stats type chosen by subclass)
    def e_step(self, X, *, backend='jax'): ...
    def m_step(self, eta, **kw) -> "MarginalMixture": ...
    def m_step_normal(self, eta) -> "MarginalMixture": ...
    def m_step_subordinator(self, eta, **kw) -> "MarginalMixture": ...
    def compute_eta_from_model(self): ...
    def em_convergence_params(self): ...

    # convenience
    def fit(self, X, **kw) -> "EMResult": ...
```

`NormalMixture` (full Σ) and `FactorNormalMixture` are the two
implementations. Fitters depend only on this ABC.

---

## 4. Parameter Facade on `NormalMixture`

The marginal density is parameterised by the same classical tuple
$(\mu,\gamma,\Sigma,\text{subordinator})$ as the joint. Rather than
exposing this through `model._joint.*`, the marginal forwards them:

| Forwarded | Read | Write (via `replace`) |
|---|---|---|
| `mu`, `gamma`, `L_Sigma` | property | `replace(mu=...)`, `replace(L_Sigma=...)` |
| `sigma()` | method | `replace(sigma=...)` (Cholesky internally) |
| `log_det_sigma()` | method | — |
| Subordinator fields | per-subclass property | `replace(<key>=...)` (per `_subordinator_keys`) |

`replace(**updates)` validates keys against
`_NORMAL_KEYS ∪ _subordinator_keys()`, treats `sigma` as a write-only
alias for `L_Sigma`, and uses `eqx.tree_at` for an immutable update. The
storage stays in `_joint`; the facade is forwarders + immutable update —
no duplicated state.

```python
vg2 = vg.replace(mu=new_mu)
vg3 = vg.replace(alpha=2.5, beta=0.5)
vg4 = vg.replace(sigma=sigma2 * jnp.eye(d))
gh2 = gh.replace(p=1.0, a=3.0, b=4.0)
```

The marginal `mean()` and `cov()` (=$E[Y]\Sigma + \mathrm{Var}[Y]\gamma\gamma^\top$)
remain distinct from `mu` and `sigma()`: those are the **conditional**
Gaussian's parameters, not the marginal's moments.

---

## 5. `from_expectation` as the Canonical η→Model Map

The closed-form M-step is exactly the conjugate-dual map η → θ on the
joint exponential family. Both layers expose it:

```python
JointNormalMixture.from_expectation(eta: NormalMixtureEta, **kw) -> JointNormalMixture
NormalMixture.from_expectation(eta: NormalMixtureEta, **kw)      -> NormalMixture
```

The joint method dispatches on `isinstance(eta, NormalMixtureEta)`:

- **pytree path** — closed form (`_mstep_normal_params` for $\mu,\gamma,\Sigma$;
  `_subordinator_from_eta` for the subordinator).
- **flat `jax.Array`** — falls back to the inherited Bregman solver.
  Kept for API symmetry; not the recommended route for high-dimensional
  joint EFs.

Per-subclass plumbing collapses to two hooks:

| Method | Purpose |
|---|---|
| `_subordinator_from_eta(cls, eta, *, theta0=None, **kw)` | Fit the subordinator from $(E[\log Y], E[1/Y], E[Y])$. `theta0` warm-starts GIG; closed-form subordinators ignore it. |
| `_from_normal_and_subordinator(cls, μ, γ, L, sub)` | One `cls(...)` call per joint subclass to bridge differing field names (`α/β`, `μ_ig/λ`, `p/a/b`). |

The instance `m_step` is a thin wrapper. Only `GeneralizedHyperbolic`
overrides `m_step` (and keeps a sanity-check `m_step_subordinator`):
the GIG solver needs a warm-start `theta0` and a fall-back when it
wanders out of the sane region. VG / NInvG / NIG drop their per-subclass
`m_step_subordinator` overrides.

Inverting any prior or shrinkage target back to a model is a one-liner:

```python
sigma_recovered = VarianceGamma.from_expectation(eta0_iso).sigma()
joint_recovered = JointVarianceGamma.from_expectation(eta0_iso)
```

---

## 6. Factor Analysis Family

`FactorNormalMixture` is a **sibling** of `NormalMixture`, **not** a
subclass: it stores `(μ, γ, F, D, subordinator)` directly with
$\Sigma = F F^\top + \mathrm{diag}(D)$. `FactorVarianceGamma`,
`FactorNormalInverseGamma`, `FactorNormalInverseGaussian`, and
`FactorGeneralizedHyperbolic` follow.

### 6.1 Why no shared `JointNormalMixture`

The factor-analysis complete-data structure is over $(X, Y, Z)$ with
**ten** sufficient statistics (`FactorMixtureStats` in `fitting/eta.py`),
not the six of `NormalMixtureEta`. Reusing `JointNormalMixture` would
force the joint hierarchy to lie about its EF signature. A separate
sibling family keeps each story clean.

### 6.2 Woodbury everywhere

All Σ-related linear algebra goes through Woodbury at $O(d r^2 + r^3)$,
never forming a dense $d \times d$ solve:

```python
def _M(self):           return I_r + Fᵀ D⁻¹ F                    # (r, r)
def _solve(self, x):    return D⁻¹ x − D⁻¹ F · M⁻¹ · Fᵀ D⁻¹ x    # Σ⁻¹ x
def _quad_form(self,x): return x · _solve(x)
def _log_det_sigma(self): return Σ log D + slogdet(M)
def _beta(self):        # β = Fᵀ Σ⁻¹, used per E-step pass
```

`_beta` is computed once per E-step pass, not per observation.

### 6.3 `D` positivity and `F` gauge

- `D = jnp.maximum(D, D_FLOOR)` after each M-step
  (`D_FLOOR = 1e-8` in `utils/constants.py`).
- `F` is identifiable only up to a right $r \times r$ orthogonal
  rotation, so `(μ, γ, F, D)` would never converge in norm. The
  convergence hook `em_convergence_params` returns
  `(μ, γ, Σ = F F^\top + \mathrm{diag}(D))` — invariant to the
  rotation.

### 6.4 `default_init` for FactorGH

Mirrors the standard `GeneralizedHyperbolic.default_init`: short EM
on `FactorNIG`, `FactorVG`, `FactorNInvG` (each converted into the
GIG embedding), plus a moment-based fallback `(p=1, a=1, b=1)`. Picks
the candidate with the highest marginal log-likelihood.
JAX-native: no `try/except`, no Python branching on data values.

### 6.5 Why not a `DispersionModel` ABC yet

A previous draft proposed `DispersionModel` with `FullDispersion(L_Σ)`
and `FactorDispersion(F, D)` implementations. We defer this:

- Only two storage variants exist today.
- An interface designed for two implementations adds indirection without
  observable benefit.
- The third variant's needs (SVD? banded? low-rank-plus-diagonal?) are
  unknown — committing to an interface now is the worst time.

When at least three variants are needed, the starting sketch is:

```python
class DispersionModel(eqx.Module):
    def solve(self, x): ...
    def solve_matrix(self, X): ...
    def quad_form(self, x): ...
    def log_det(self): ...
    def sigma(self): ...
    def sample_noise(self, key, shape=()): ...
```

Until then, `JointNormalMixture._quad_forms` and
`FactorNormalMixture._solve` carry the linear algebra inline.

---

## 7. Sufficient Statistics in Theory Order

Both stats classes use **descriptive field names** in **theory order**
(matching `../../docs/theory/shrinkage.md` and `../../docs/theory/factor_analysis.md`):

```python
class NormalMixtureEta(eqx.Module):
    E_inv_Y:     # s_1 = E[Y⁻¹],    scalar
    E_Y:         # s_2 = E[Y],       scalar
    E_log_Y:     # s_3 = E[log Y],   scalar
    E_X:         # s_4 = E[X],       (d,)
    E_X_inv_Y:   # s_5 = E[X / Y],   (d,)
    E_XXT_inv_Y: # s_6 = E[X X^T/Y], (d, d)

class FactorMixtureStats(eqx.Module):
    # fields s_1 … s_6 identical to NormalMixtureEta
    E_XZT_inv_sqrtY: # s_7  = E[X Z^T Y^{-1/2}], (d, r)
    E_Z_inv_sqrtY:   # s_8  = E[Z Y^{-1/2}],     (r,)
    E_Z_sqrtY:       # s_9  = E[Z Y^{1/2}],      (r,)
    E_ZZT:           # s_10 = E[Z Z^T],          (r, r)
```

Sharing the first six fields means shrinkage targets, weights, and
tests written for the standard family port directly to the factor
family.

---

## 8. Cross-References

- Architecture: `../ARCHITECTURE.md` § *Mixture Structure*.
- EM machinery: `em_framework.md`.
- Theory: `../../docs/theory/gh.md`, `../../docs/theory/factor_analysis.md`,
  `../../docs/theory/shrinkage.md`.
- Historical / archived rationale: `../archive/design/em_covariance_extensions.md`.
