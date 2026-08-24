# Why not gradient descent

> **Scope.** Why the supported fitters are exponential-family MLE and EM,
> not Adam or L-BFGS on the observed-data negative log-likelihood (NLL).
> Why `log_prob` is still differentiable, and how to plug in an external
> optimiser if a surrounding model requires it.
>
> **Where things live.** `fit_mle` on exponential families and
> {py:class}`~normix.fitting.em.BatchEMFitter` for mixtures. The η→θ
> solve is in {doc}`solvers_and_bessel`. EM structure is in
> {doc}`em_framework`.

---

## 1. The supported fitters

For a regular exponential family the MLE is the mean sufficient statistic
followed by a constrained Bregman inversion:

$$
\hat\eta = n^{-1}\sum_{i=1}^n t(x_i),\qquad
\hat\theta = \arg\min_\theta\bigl[\psi(\theta)-\theta\cdot\hat\eta\bigr].
$$

{py:class}`~normix.distributions.generalized_inverse_gaussian.GIG`
implements that as `fit_mle`. The inversion is a 3-D convex problem after
one $O(n)$ reduction; it does not re-evaluate $K_\nu$ on every observation
at every step. The GIG Fisher information can have condition number
$10^{30}$ at extreme $(a,b)$; the solve η-rescales before walking in
$\theta$ ({doc}`solvers_and_bessel` § 2).

Normal variance-mean mixtures
({py:class}`~normix.distributions.generalized_hyperbolic.GeneralizedHyperbolic`
and special cases) are not exponential families in $x$ alone. Fitting is
EM ({ref}`Dempster1977 <dempster1977>`): the E-step returns
$\mathbb{E}[t(Y)\mid X]$, the M-step is the same `from_expectation` map.
Complete-data likelihood is monotone; the normal M-step is closed form.

That is the supported path. The rest of this page is why a
[GPJax](https://github.com/thomaspinder/GPJax) /
[FlowJAX](https://github.com/danielward27/flowjax)-style loop — store
unconstrained parameters, apply `softplus`, run Adam or L-BFGS on the
marginal NLL — is not a second public fitter.

---

## 2. What was compared

On GIG and Gamma, NLL methods store $\phi=(p,\,\varphi_a,\,\varphi_b)$
with $a=\mathrm{softplus}(\varphi_a)$, $b=\mathrm{softplus}(\varphi_b)$,
and minimise $-\frac1n\sum_i\log p(x_i\mid p,a,b)$. GH uses an analogous
softplus on $(a,b)$ and a Cholesky factor of $\Sigma$.

| Method | What it optimises |
|---|---|
| `fit_mle` / EM | $\hat\eta\to\theta$ (GIG: η-rescaled Bregman; GH: EM) |
| L-BFGS + softplus | observed NLL in unconstrained $\phi$ ([JAXopt](https://jaxopt.github.io/stable/) L-BFGS) |
| Adam + softplus | same NLL, first-order (hand-rolled; [Optax](https://optax.readthedocs.io/en/latest/) is not a dependency) |
| L-BFGS-B box NLL | GIG only: [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) `L-BFGS-B` bounds $a,b>0$, no softplus |

Reproduction: `uv run python benchmarks/bench_gradient_fitting.py`.
CPU, [JAX](https://docs.jax.dev/en/latest/) 0.9.1, $n=2000$ (Gamma, GIG),
$n=600$ (GH, $d=2$).

---

## 3. What the comparison shows

**Gamma** $(\alpha,\beta)=(2,1.5)$ — no Bessel, well-conditioned control.
L-BFGS + softplus matches `fit_mle` in NLL and in $(\hat\alpha,\hat\beta)$.
Adam reaches the same NLL in 1500 steps. The optimiser harness works; Gamma
does not need it (`fit_mle` is a 2-D convex solve after one mean of
$t(x)=(\log x,\,x)$).

**Interior GIG** $(p,a,b)=(1,2,1)$. L-BFGS + softplus reproduces `fit_mle`
to working precision ($\Delta$NLL $= -4.7\times 10^{-4}$ for both;
parameters agree to printed digits). Wall-clock: `fit_mle` $0.19\,\mathrm{s}$,
L-BFGS $2.57\,\mathrm{s}$ ($\sim 14\times$). Adam after 1500 steps is still
$0.275$ in hybrid parameter error and slightly *worse* NLL than the MLE.

**Degenerate GIG.** At large $\sqrt{ab}$ or $a/b$ extremes, the NLL ridge
is nearly flat in $(p,a,b)$ and the MLE in classical coordinates is
weakly identified. η-rescaled Bregman stays in a reasonable neighbourhood.
L-BFGS + softplus matches that neighbourhood when the point is only mildly
asymmetric, and drives $b$ to underflow ($10^{-37}$) at
$(p,a,b)=(1,10^4,10^{-3})$. Box-constrained NLL collapses $b$ onto the
bound. Adam can flip the sign of $p$. Softplus cannot represent the
Variance-Gamma boundary $b=0$ *exactly*; EM's degenerate-GIG branches can.

**GH, $d=2$.** EM and L-BFGS reach the same NLL to $\sim 10^{-4}$
($\Delta$NLL $\approx -0.013$ vs the truth on the interior draw). EM:
$0.45$–$1.0\,\mathrm{s}$ (25 iterations). L-BFGS: $11$–$13\,\mathrm{s}$
(40 iterations), $\sim 10$–$25\times$ slower. Adam at 200 steps (already
$4$–$8\times$ EM wall-clock) is $0.09$–$0.19$ nats *worse* than the truth
and has not left the initialisation neighbourhood. Subordinator
$(p,a,b)$ differs across EM and L-BFGS at matched NLL (GH scale gauge).

The ML recipe is a valid MLE on well-conditioned interiors. It is the
wrong default here: cost, first-order lag, GIG-boundary collapse, and
the exact $b=0$ boundary that VG occupies.

---

## 4. What this does not show

A historical reason for avoiding gradient descent was
$\partial_\nu\log K_\nu$: older JAX Bessel wrappers (TensorFlow
Probability's `log_besselk`) returned a **zero** $\nu$-tangent, so Adam
could not move $p$. That bug is gone.
`log_kv` uses `@jax.custom_jvp` (exact
$\partial_z$ recurrence; central difference $\partial_\nu$ at
$10^{-5}$). On the same GIG grid, `jax.grad` of the observed NLL with
respect to $(p,a,b)$ matches a CPU finite difference that never touches
the custom JVP, relative error $\sim 10^{-9}$ (`tests/test_gig_properties.py`).

Quasi-Newton on the GIG NLL *can* recover the interior MLE. We still do
not lead with it, for the reasons in § 3, not because autodiff through
$K_\nu$ is broken.

---

## 5. Bring your own optimiser

`log_prob` is JIT-able and differentiable. If a surrounding model must
co-optimise GIG parameters with non-normix parameters, the density is
the primitive — not a `method="adam"` fitter. Interior $(p,a,b)$ only;
the caveats in § 3 still apply.

```python
import jax
import jax.numpy as jnp
from jax.nn import softplus
from normix import GIG

def unpack(phi):
    return phi[0], softplus(phi[1]), softplus(phi[2])

def nll(phi, X):
    p, a, b = unpack(phi)
    dist = GIG(p=p, a=a, b=b)
    return -jnp.mean(jax.vmap(dist.log_prob)(X))

# jax.grad(nll)(phi, X)  — plug into JAXopt L-BFGS, Optax, or a custom loop
```

Adam and L-BFGS remain unsupported as public fitters: there is no grid
point where they beat `fit_mle` / EM, and shipping them means owning
$b\to 0$ collapse, the GH gauge, and step-count / learning-rate knobs
whose only documented advice would be "prefer `fit_mle`".

Reopen the decision if a concrete caller needs one of:

- **streaming / minibatch** where a full E-step over $n$ is infeasible;
- **joint NLL** with parameters that are not a normix exponential family.

The first abstraction in either case is a documented `nll(params, X)`
helper, not an in-tree Adam.

---

## 6. Cross-references

- {doc}`exponential_family` — $\psi$, $\eta=\nabla\psi$, clamp vs bijections.
- {doc}`em_framework` — model / fitter split, M-step as `from_expectation`.
- {doc}`solvers_and_bessel` — η-rescaling, `log_kv` JVP, CPU/GPU hybrid.
- {doc}`EM algorithm <../theory/em_algorithm>`,
  {doc}`GIG distribution <../theory/gig>`.
- {doc}`Fitting with EM <../user_guide/em_fitting>`.
