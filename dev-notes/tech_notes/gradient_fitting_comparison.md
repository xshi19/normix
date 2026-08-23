# Gradient descent vs exponential-family / EM fitting

**Date:** 2026-08-23
**Status:** measured. Reproduction:
`uv run python benchmarks/bench_gradient_fitting.py --save`
**Results JSON:** `benchmarks/results/2026-08-23_48cc052_gradient_fitting.json`
(CPU, JAX 0.9.1, git `48cc052`). `--quick` is a smoke path (fewer samples / steps).

Companion: the record audit
[`../reviews/gradient_descent_decision_audit_2026-08-23.md`](../reviews/gradient_descent_decision_audit_2026-08-23.md)
(what was written down historically). This note is the missing head-to-head.

---

## 1. Problem

GPJax, FlowJAX, and efax fit constrained parameters by storing an
*unconstrained* value and applying a bijection (`softplus`, `exp`) inside
the loss, then running Adam / L-BFGS in unconstrained space. normix does
not: exponential-family MLE is $\hat\eta = n^{-1}\sum t(x_i)$ followed by
a constrained Bregman solve $\eta\to\theta$; GH-family models use EM.

The question this study answers: **if we implement the ML recipe on
normix's own `log_prob` (including the `log_kv` custom JVP), does it
recover parameters as well as the current fitters, and if not, why?**

Three hypotheses from the audit, tested separately:

| # | Hypothesis | How we test it |
|---|---|---|
| H1 | $\partial_\nu\log K_\nu$ from the custom JVP is too noisy for GD | `jax.grad` of the NLL wrt $p$ vs a CPU finite difference that *does not* go through the JVP |
| H2 | First-order Adam + softplus fails where quasi-Newton on the same NLL succeeds | Adam vs jaxopt L-BFGS, identical reparametrisation and init |
| H3 | Walking the NLL is the wrong problem (ill-conditioned / non-identified $(p,a,b)$; slower than moment matching) | NLL methods vs `fit_mle` / EM; include the 2026-03 degenerate GIG grid |

A **Gamma control** (no Bessel) checks that the Adam/L-BFGS harness itself
is correct: if Gamma works and GIG does not, the gap is the GIG geometry /
Bessel, not a coding bug.

---

## 2. Notation

### 2.1 Distributions

**Gamma** (control). $X\sim\mathrm{Gamma}(\alpha,\beta)$, $\alpha>0$,
$\beta>0$. Density $\beta^\alpha x^{\alpha-1}e^{-\beta x}/\Gamma(\alpha)$.
Sufficient statistic $t(x)=(\log x,\, x)$. Closed-form $\eta\to\theta$
(digamma).

**GIG.** $Y\sim\mathrm{GIG}(p,a,b)$ for $a>0$, $b>0$, $p\in\mathbb{R}$:

$$
f(y\mid p,a,b)
= \frac{(a/b)^{p/2}}{2 K_p(\sqrt{ab})}
  \, y^{p-1}
  \exp\bigl(-(ay + b/y)/2\bigr),
  \qquad y>0.
$$

Natural parameters $\theta=(p-1,\,-b/2,\,-a/2)$ with $\theta_2\le 0$,
$\theta_3\le 0$. Sufficient statistic
$t(y)=(\log y,\, 1/y,\, y)$.
Expectation parameters $\eta=\nabla\psi(\theta)=(E[\log Y],\,E[1/Y],\,E[Y])$.
Log-partition $\psi(\theta)$ contains $\log K_p(\sqrt{ab})$.

Scale / concentration (Barndorff-Nielsen):
$\delta=\sqrt{b/a}$ (scale), $\omega=\sqrt{ab}$ (concentration).
Large $\omega$ $\Rightarrow$ $Y$ concentrates about $\delta$;
$(p,a,b)$ become poorly identified from $(E[\log Y], E[1/Y], E[Y])$.
See `gig_eta_to_theta.md` and solvers_and_bessel.md § 2.

**GH, $d=2$.** $X\mid Y\sim\mathcal N(\mu+\gamma Y,\, \Sigma Y)$,
$Y\sim\mathrm{GIG}(p,a,b)$. Marginal density is closed form (two
$K_\nu$ evaluations). Not an exponential family in $x$ alone. Fitted
by EM on the joint complete-data $\eta$.

$\Sigma=L_\Sigma L_\Sigma^\top$ (Cholesky). GH has a scale gauge:
$(\Sigma, a, b)\mapsto (c\Sigma,\, a/c,\, cb)$ leaves the law of $X$
unchanged up to a compensating shift in the GIG. EM defaults to
`regularization='det_sigma_one'` ($\lvert\Sigma\rvert=1$). NLL methods
in this study do **not** impose that gauge, so $(p,a,b)$ are not
comparable across EM vs Adam; **NLL is the comparable metric**.

### 2.2 Softplus reparametrisation

For a coordinate $u>0$ the ML recipe stores $\varphi\in\mathbb{R}$ with
$u=\mathrm{softplus}(\varphi)=\log(1+e^\varphi)$. Inverse (stable):

```python
def inv_softplus(x):
    x = jnp.maximum(x, 1e-12)
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(x)))
```

| Family | Unconstrained $\varphi$ | Reconstruction |
|---|---|---|
| Gamma | $(\varphi_\alpha, \varphi_\beta)$ | $\alpha=\mathrm{sp}(\varphi_\alpha)$, $\beta=\mathrm{sp}(\varphi_\beta)$ |
| GIG | $(\varphi_p, \varphi_a, \varphi_b)$ | $p=\varphi_p$ (already $\mathbb{R}$), $a=\mathrm{sp}(\varphi_a)$, $b=\mathrm{sp}(\varphi_b)$ |
| GH $d=2$ | $\mu,\gamma\in\mathbb{R}^2$; $\mathrm{diag}(L_\Sigma)$ via softplus; strict lower triangle free; $p$ free; $a,b$ via softplus | 10 scalars |

`sp` = `jax.nn.softplus`. No `paramax` dependency — same bijection, 8 lines.

### 2.3 Objectives

Mean negative log-likelihood on a sample $X_{1:n}$:

$$
\mathrm{NLL}(\phi)
= -\frac1n\sum_{i=1}^n \log f(X_i\mid \mathrm{unpack}(\phi)).
$$

$\Delta\mathrm{NLL} = \mathrm{NLL}(\hat\phi) - \mathrm{NLL}(\phi_\mathrm{true})$.
Negative is allowed: the finite-sample MLE can beat the true parameter's NLL.

Exponential-family MLE (normix `fit_mle`):

$$
\hat\eta = \frac1n\sum_{i=1}^n t(X_i),
\qquad
\hat\theta = \arg\min_\theta\bigl[\psi(\theta)-\theta\cdot\hat\eta\bigr].
$$

For GIG the second step is $\eta$-rescaled, multi-start CPU L-BFGS-B
(`from_expectation` with `theta0=None`). It never differentiates the
sample NLL.

EM (GH): E-step $E[t(X,Y)\mid X]$ (CPU Bessel), M-step closed-form
$(\mu,\gamma,\Sigma)$ plus the same GIG $\eta\to\theta$ on the
subordinator block; `det_sigma_one` after each M-step.

---

## 3. Methods compared

Every iterative NLL method starts from the **same perturbed init**, not
from the truth.

### 3.1 GIG / Gamma

| Tag | What it optimises | Constraints | Engine |
|---|---|---|---|
| `fit_mle` | Bregman $\psi(\theta)-\theta\cdot\hat\eta$ | $\theta_2,\theta_3\le 0$ (GIG); closed form (Gamma) | current normix |
| `adam+softplus` | sample NLL | softplus on positive coords | hand-rolled Adam, `lax.scan`, JIT |
| `lbfgs+softplus` | sample NLL | same softplus | `jaxopt.LBFGS` |
| `lbfgsb-box NLL` (GIG only) | sample NLL | scipy box $a\ge 10^{-12}$, $b\ge 10^{-12}$ | `scipy.optimize.minimize(L-BFGS-B)` + `jax.grad` |

Adam hyperparameters: $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$.
GIG / Gamma: 1500 steps, $\mathrm{lr}=10^{-2}$ (GIG) / $5\cdot 10^{-3}$ (Gamma).
L-BFGS: `maxiter=150`.

GIG init: $p\leftarrow p+0.7$, $a\leftarrow\max(2.5a, 0.05)$,
$b\leftarrow\max(0.4b, 0.05)$.

### 3.2 GH ($d=2$)

Shared naive init: `GH._from_init_params` (sample mean, $\gamma=0$,
sample covariance, $p=a=b=1$). Not `default_init` (that already runs EM
on NIG/VG/NInvG — would credit EM twice).

| Tag | Engine | Budget |
|---|---|---|
| `EM` | `BatchEMFitter(e_step_backend='cpu', regularization='det_sigma_one')` | `max_iter=25`, `tol=1e-5` |
| `adam+softplus` | same Adam as GIG | 200 steps, $\mathrm{lr}=10^{-3}$ |
| `lbfgs+softplus` | jaxopt L-BFGS | `maxiter=40` |

200 Adam steps is *more* wall-clock than 25 EM iterations. If Adam has
not caught up by then, first-order NLL is not competitive on time.

### 3.3 What the methods isolate

```
         same NLL, different optimiser          same optimiser class, different problem
                  │                                         │
   Adam+softplus ─┴─ L-BFGS+softplus ──── L-BFGS-B box ──── fit_mle / EM
   (first-order)     (quasi-Newton,           (quasi-Newton,    (moment match
                      reparam)                 native bounds)    + Bregman / EM)
```

- Adam vs L-BFGS+softplus $\Rightarrow$ H2 (first-order vs quasi-Newton).
- L-BFGS+softplus vs L-BFGS-B box $\Rightarrow$ reparametrisation vs bounds,
  same NLL.
- L-BFGS+softplus vs `fit_mle` $\Rightarrow$ H3 (NLL vs exponential-family
  MLE).

---

## 4. Metrics

| Metric | Definition | When it is meaningful |
|---|---|---|
| $\mathrm{NLL}$ | mean $-\log f(X_i\mid\hat\phi)$ | always (primary) |
| $\Delta\mathrm{NLL}$ | NLL minus NLL at the true parameter | always |
| `param_err` (GIG/Gamma) | $\max_k \lvert\hat u_k-u_k\rvert/(1+\lvert u_k\rvert)$ over classical coords | identifiable regimes only |
| wall-clock | seconds, including JIT compile of that call | reported; first GIG Adam/L-BFGS call pays compile |
| `finite` | all of $(\hat\phi, \mathrm{NLL})$ finite | crash / overflow flag |
| $n_\mathrm{iter}$ | L-BFGS / EM iterations | — |

For GH, `param_err` on $(p,a,b)$ is **not** reported as a ranking
criterion (gauge). NLL only.

Sample sizes: $n=2000$ (Gamma, GIG, gradient diagnostic), $n=600$ (GH).
Seeds fixed (`seed=0` for fits, `seed=1` for the gradient grid).

### 4.1 Gradient diagnostic (H1)

At the *true* GIG, with the same sample:

- $\partial\mathrm{NLL}/\partial p$ via `jax.grad` of the JAX `log_prob`
  (flows through `@jax.custom_jvp` on `log_kv`, $\varepsilon_\nu=$
  `BESSEL_EPS_V` $=10^{-5}$).
- Central difference of a **CPU** NLL that uses
  `GIG._log_partition_cpu` / `scipy.special.kve`, step $10^{-6}$.
  This path does not use the custom JVP.

Relative error $\lvert g_\mathrm{jax}-g_\mathrm{cpu}\rvert/(1+\lvert g_\mathrm{cpu}\rvert)$.
The same comparison is made for $\partial_\nu\log K_\nu(z)$ itself
(`jax.grad(log_kv)` vs FD of $\log\mathrm{kve}-\,z$).

---

## 5. Sample code (what is actually being run)

Harness: `benchmarks/bench_gradient_fitting.py`. Snippets below are the
mathematical core, not the logging.

### 5.1 GIG NLL in unconstrained space

```python
def gig_unpack(phi):
    p, a, b = phi[0], jax.nn.softplus(phi[1]), jax.nn.softplus(phi[2])
    return p, a, b

def nll_phi(phi):
    p, a, b = gig_unpack(phi)
    dist = GIG(p=p, a=a, b=b)
    return -jnp.mean(jax.vmap(dist.log_prob)(X))   # unbatched core, batched here
```

`log_prob` is the exponential-family formula
$\log h(y)+\theta^\top t(y)-\psi(\theta)$, and $\psi$ calls `log_kv`.

### 5.2 Adam (`lax.scan`, JIT)

```python
b1, b2, eps = 0.9, 0.999, 1e-8
g = jax.grad(nll_phi)(phi)
m = b1 * m + (1 - b1) * g
v = b2 * v + (1 - b2) * g**2
phi = phi - lr * (m / (1 - b1**t)) / (jnp.sqrt(v / (1 - b2**t)) + eps)
```

No `optax` — it is not a core dependency. This is the FlowJAX/GPJax
update with the same hyperparameters.

### 5.3 Current fitter (the baseline)

```python
# GIG: one mean of t(X), then constrained η→θ (η-rescaled, CPU multi-start)
model = GIG.fit_mle(X)

# GH: E-step (CPU Bessel) + M-step (closed-form normal + GIG η→θ)
result = GH._from_init_params(mu, gamma, sigma).fit(
    X, max_iter=25, e_step_backend="cpu", regularization="det_sigma_one",
)
```

### 5.4 Independent CPU gradient (H1)

```python
theta = np.asarray(GIG(p, a, b).natural_params())
t = np.stack([np.log(X), 1 / X, X], axis=1)
psi = float(GIG._log_partition_cpu(theta))          # scipy.kve, no custom JVP
nll_cpu = psi - np.mean(t @ theta)
# then (nll_cpu(p+ε) - nll_cpu(p-ε)) / (2ε)
```

---

## 6. Test grid

**Gamma.** $(\alpha,\beta)=(2, 1.5)$. One well-conditioned control.

**GIG** (includes the 2026-03 notebook regimes):

| Label | $(p, a, b)$ | Why |
|---|---|---|
| interior | $(1, 2, 1)$ | well-conditioned |
| invgauss $p=-1/2$ | $(-0.5, 2, 1)$ | Inverse-Gaussian special case |
| asymmetric $a\gg b$ | $(0.5, 10, 0.1)$ | Fisher ill-conditioning |
| asymmetric $a\ll b$ | $(-1, 0.1, 10)$ | the other asymmetry |
| near-Gamma $b=10^{-4}$ | $(2, 2, 10^{-4})$ | VG-like boundary; $b=0$ unreachable under softplus |
| large $\sqrt{ab}=100$ | $(1, 100, 100)$ | concentrated; $(p,a,b)$ non-identified |
| large $a=10^4$, $b=10^{-3}$ | $(1, 10^4, 10^{-3})$ | 2026-03 failure case |

**GH $d=2$:**

| Label | $(\mu,\gamma,\Sigma,p,a,b)$ |
|---|---|
| interior | $\mu=0$, $\gamma=(0.3,-0.1)$, $\Sigma=\bigl(\begin{smallmatrix}1&0.3\\0.3&0.8\end{smallmatrix}\bigr)$, $p=1$, $a=b=2$ |
| near-VG | $\mu=0$, $\gamma=(0.2,0.2)$, $\Sigma=I$, $p=2$, $a=2$, $b=10^{-3}$ |
| asymmetric $a\gg b$ | $\mu=(0.1,-0.1)$, $\gamma=(0.4,0)$, $\Sigma=I$, $p=0.5$, $a=10$, $b=0.1$ |

---

## 7. Results

### 7.1 H1 — Bessel $\partial_\nu$ is *not* the failure mode

Relative error of `jax.grad(NLL)` wrt $p$ against the CPU finite
difference, $n=2000$:

| $(p,a,b)$ | $\partial\mathrm{NLL}/\partial p$ jax | CPU FD | rel. err. | $\partial_\nu\log K$ rel. |
|---|---:|---:|---:|---:|
| $(1,2,1)$ | $-9.78\times 10^{-3}$ | same | $1.6\times 10^{-9}$ | $1.0\times 10^{-10}$ |
| $(5,1,1)$ | $-3.0\times 10^{-3}$ | same | $5\times 10^{-11}$ | $4\times 10^{-11}$ |
| $(1,100,100)$ | $7.8\times 10^{-5}$ | same | $5.6\times 10^{-9}$ | $0$ (to printed precision) |
| $(2,2,10^{-4})$ | $-5.7\times 10^{-3}$ | same | $3\times 10^{-11}$ | $1\times 10^{-11}$ |
| $(1,10^4,10^{-3})$ | $5.5\times 10^{-3}$ | same | $6\times 10^{-11}$ | $3\times 10^{-11}$ |
| $(-0.5,2,1)$ | $1.90\times 10^{-2}$ | same | $8\times 10^{-9}$ | $7\times 10^{-10}$ |

H1 is **rejected**. Today's `log_kv` custom JVP (exact $\partial_z$,
central FD $\partial_\nu$ at $\varepsilon=10^{-5}$) matches an independent
`kve` finite difference to $\sim 10^{-9}$ on this grid, including the
regimes that broke unrescaled L-BFGS-B in March 2026.

That March failure (TFP / logbesselk returning **exactly 0** for
$\partial_\nu$) was real — it is why the custom JVP exists. It is no
longer the reason to avoid gradient descent. Any user-facing "why not
GD" page should not lead with a stale Bessel-gradient story.

### 7.2 Gamma control — the harness works

$n=2000$, true $(\alpha,\beta)=(2, 1.5)$, NLL$^*=1.1384$.

| Method | $\hat\alpha$ | $\hat\beta$ | $\Delta$NLL | time |
|---|---:|---:|---:|---:|
| `fit_mle` | 2.111 | 1.613 | $-1.16\times 10^{-3}$ | 0.15 s |
| `adam+softplus` (1500 steps) | 2.116 | 1.617 | $-1.16\times 10^{-3}$ | 0.12 s |
| `lbfgs+softplus` (12 iters) | 2.111 | 1.613 | $-1.16\times 10^{-3}$ | 0.91 s |

Adam and L-BFGS reach the same NLL as the closed-form MLE. L-BFGS matches
parameters to all printed digits. The optimisation stack is not broken.

### 7.3 GIG — well-conditioned cases

$n=2000$. `param_err` is hybrid-scale max relative error on $(p,a,b)$.
Times include JIT on the first Adam/L-BFGS call of that process.

**Interior** $(1,2,1)$, NLL$^*=1.2448$:

| Method | $(\hat p,\hat a,\hat b)$ | param_err | $\Delta$NLL | time |
|---|---|---:|---:|---:|
| `fit_mle` | $(1.055,\, 2.084,\, 0.937)$ | 0.032 | $-4.7\times 10^{-4}$ | 0.19 s |
| `lbfgs+softplus` (20 it) | same as `fit_mle` | 0.032 | $-4.7\times 10^{-4}$ | 2.57 s |
| `lbfgsb-box NLL` (15 it) | same | 0.032 | $-4.7\times 10^{-4}$ | 0.48 s |
| `adam+softplus` (1500) | $(1.549,\, 2.476,\, 0.561)$ | 0.275 | $+1.3\times 10^{-3}$ | 0.57 s |

Quasi-Newton on the NLL **matches** exponential-family MLE to working
precision. Adam has not arrived after 1500 steps (NLL close, parameters
not). `fit_mle` is $\sim 5$–$14\times$ faster than NLL L-BFGS because it
averages $t(X)$ once and solves a 3-D convex problem; it does not
re-evaluate Bessel on every datum every iteration.

**InvGauss** and **$a\ll b$**: same pattern — `fit_mle` $=$
`lbfgs+softplus` $=$ box L-BFGS-B; Adam lags on `param_err`.

### 7.4 GIG — degenerate / ill-conditioned cases

**$a\gg b$** $(0.5, 10, 0.1)$:

| Method | $(\hat p,\hat a,\hat b)$ | param_err | $\Delta$NLL |
|---|---|---:|---:|
| `fit_mle` / `lbfgs+softplus` | $(0.605,\, 10.85,\, 0.089)$ | 0.077 | $-6.0\times 10^{-4}$ |
| `adam+softplus` | $(1.38,\, 16.5,\, 0.035)$ | 0.59 | $+0.010$ |
| `lbfgsb-box NLL` | $(1.67,\, 17.1,\, 10^{-12})$ | 0.78 | $+0.024$ |

Box constraints on the NLL **collapsed $b$ onto the bound**. Softplus
L-BFGS matched `fit_mle`. Native bounds on the NLL are not a substitute
for $\eta$-rescaling + moment matching.

**Near-Gamma** $b=10^{-4}$: `fit_mle` recovers $b\approx 8\times 10^{-3}$
(the sample cannot resolve $b$ that small — NLL is flat). Box L-BFGS-B
drove $b$ to $10^{-12}$ (`success=False` after 7 iters). Softplus stayed
with `fit_mle`. Softplus *cannot represent $b=0$* (VG); it can only
approach $0$ as $\varphi_b\to-\infty$.

**Large $\sqrt{ab}=100$** $(1, 100, 100)$, NLL$^*=-0.894$:

| Method | param_err | $\Delta$NLL | note |
|---|---:|---:|---|
| `fit_mle` | 23.8 | $-0.0011$ | $(p,a,b)$ wildly off; NLL matched |
| `lbfgs+softplus` | 23.8 | $-0.0011$ | same basin as `fit_mle` |
| `lbfgsb-box NLL` | 34.7 | $-0.0010$ | different $(p,a,b)$, similar NLL |
| `adam+softplus` | 7.3 | **$+22.6$** | first-order diverged on a flat ridge |

This is the 2026-03 non-identifiability: many $(p,a,b)$ share the same
$\eta$ when $\omega$ is large. Quasi-Newton methods find *a* NLL
minimiser; the classical parameters are meaningless. Adam's NLL is
catastrophically worse — H2 holds here.

**Large $a=10^4$, $b=10^{-3}$:**

| Method | $(\hat p,\hat a,\hat b)$ | param_err | $\Delta$NLL |
|---|---|---:|---:|
| `fit_mle` | $(1.51,\, 1.15\times 10^4,\, 8.5\times 10^{-4})$ | 0.26 | $-8.5\times 10^{-4}$ |
| `adam+softplus` | $(-0.47,\, 2.5\times 10^4,\, 4.2\times 10^{-3})$ | 1.50 | $+0.44$ |
| `lbfgs+softplus` | $(4.06,\, 1.74\times 10^4,\, 7\times 10^{-37})$ | 1.53 | $+0.008$ |
| `lbfgsb-box NLL` | $(5.63,\, 2.5\times 10^4,\, 10^{-12})$ | 2.32 | $+0.044$ |

Only `fit_mle` (η-rescaled Bregman) stays in a reasonable neighbourhood.
NLL quasi-Newton drives $b$ to underflow; Adam flips the sign of $p$.
**This is H3, not H1.**

### 7.5 GH $d=2$

Shared moment init. NLL$^*$ = NLL at the true parameter. $\Delta$NLL
negative $\Rightarrow$ fitted model beats the truth on this sample
(expected).

| Case | NLL$^*$ | NLL init | EM $\Delta$NLL (time) | L-BFGS $\Delta$NLL (time) | Adam $\Delta$NLL (time) |
|---|---:|---:|---|---|---|
| interior | 3.206 | 3.468 | **$-0.0135$ (1.0 s, 25 it)** | $-0.0134$ (13.2 s, 40 it) | $+0.089$ (4.3 s, 200 st) |
| near-VG | 3.466 | 3.726 | **$-0.0100$ (0.45 s)** | $-0.0105$ (11.3 s) | $+0.093$ (4.7 s) |
| $a\gg b$ | 1.084 | 1.542 | **$-0.0132$ (0.60 s)** | $-0.0131$ (11.9 s) | $+0.192$ (4.7 s) |

EM did not trip `converged=True` at 25 iterations (`tol=1e-5` on
$(\mu,\gamma,L_\Sigma)$); it nevertheless matches L-BFGS NLL to
$\sim 10^{-4}$. L-BFGS still had `opt_error` $\sim 10^{-2}$ at
`maxiter=40` — the NLL surface is flatter in the GH gauge than EM's
constrained M-steps.

Adam at 200 steps ($\sim 4$–$8\times$ EM wall-clock) is still $0.09$–$0.19$
NATS worse than the truth, i.e. has not left the init neighbourhood in
likelihood. First-order NLL is not a substitute for EM here.

$(p,a,b)$ differ across EM and L-BFGS (gauge + `det_sigma_one`). Example,
interior: EM $(1.90,\, 2.64,\, 0.053)$ vs L-BFGS $(1.94,\, 0.78,\, 0.23)$
at essentially identical NLL.

---

## 8. What this does and does not show

**Holds:**

1. The ML recipe is a *correct* MLE procedure on well-conditioned GIG and
   on Gamma: L-BFGS + softplus reproduces `fit_mle` to working precision.
2. It is the **wrong default** for this package:
   - **Cost.** GIG `fit_mle` is one 3-D convex solve after a single
     $O(n)$ mean; NLL L-BFGS re-evaluates `log_kv` on every observation,
     $\sim 5$–$40\times$ slower here. GH EM is $\sim 10$–$25\times$
     faster than NLL L-BFGS at the same NLL.
   - **First-order.** Adam lags on GIG and does not arrive on GH in a
     budget already larger than EM. GPJax can use Adam because a GP
     marginal likelihood is a smooth, moderately conditioned Cholesky
     problem, not a 3-parameter Bessel family with $\kappa(I(\theta))$ up
     to $10^{30}$.
   - **Degeneracy.** At large $\omega$ or $a/b$ extremes, NLL methods
     collapse $b\to 0$ or wander on a ridge. η-rescaling + bounds in
     $\theta$-space is the geometry the GIG MLE actually has.
   - **Boundaries.** VG is $b=0$ *exactly*. Softplus never lands there.
     EM + degenerate GIG branches do.
3. Monotonicity: EM's NLL is monotone in the complete-data likelihood
   (standard EM). Adam on the marginal is not. Not measured as a curve
   here; still a design reason to prefer EM for GH.

**Does not hold (update the folklore):**

- "Gradient descent fails because $\partial_\nu\log K_\nu$ is a bad
  finite difference." Not on this grid, not with the current `log_kv`.
- "Quasi-Newton on the GIG NLL cannot recover the MLE." It can, when
  $(p,a,b)$ are identified. The reason we still do not *lead* with it
  is cost + the degenerate cases + EM's closed-form normal M-step.

**Not tested (deliberate):**

- VG unbounded-likelihood region $\alpha\le d/2$ (both EM and NLL need
  `alpha_min`; that is a different note).
- Mini-batch SGD, learning-rate schedules, Optax, paramax wrappers.
- GPU. This run is CPU JAX; GH EM's advantage on GPU is already in
  `em_gpu_profiling.md`.
- Warm-start from `GH.default_init` (would help every method).

---

## 9. Recommendation

Keep **EM + constrained Bregman $\eta\to\theta$** as the supported fitter.
Record the decision as: *correct and cheaper on the problem we actually
solve; the ML recipe is a valid MLE on the interior and a poor default
at the GIG boundary / GH gauge.*

User-facing `docs/design/` page (next, not this note): lead with the
exponential-family identity and the timing table in § 7.3–7.5; mention
Bessel $\partial_\nu$ only as historical context (TFP returned 0; we
fixed it; it is no longer the blocker).

Optional later: a `GradientFitter` was reserved in the 2026-03-07 fit
design (`git show 5c679e8:docs/jax_normix_fit_design.md` § 5.3 item 6)
and never built. This study does not motivate building it for GIG/GH.
Gamma-like interiors would not benefit either (`fit_mle` is already
closed form).

---

## 10. Reproduction

```bash
uv run python benchmarks/bench_gradient_fitting.py --save
uv run python benchmarks/bench_gradient_fitting.py --quick          # smoke
uv run python benchmarks/bench_gradient_fitting.py --section grad  # H1 only
```

`--save` writes `benchmarks/results/<date>_<git>_gradient_fitting.json`.

Adam / L-BFGS step counts are constants at the bottom of `main()`
(`n_adam=1500`, `n_lbfgs=150`, GH `n_em=25`, `n_adam_gh=200`,
`n_lbfgs_gh=40`). Raise GH Adam steps there if a longer first-order
run is needed; 200 steps already exceeded EM wall-clock.

---

## References

- This repo: `../reviews/gradient_descent_decision_audit_2026-08-23.md`,
  `gig_eta_to_theta.md`, `../design/solvers_and_bessel.md` § 2,
  `../design/exponential_family.md` § 4.3 (clamp vs paramax),
  `../references/gpjax_review.md` § 2.2 / § 5,
  `../references/distribution_packages.md` (FlowJAX `fit_to_data`, efax
  `ExpToNat`).
- Git-only founding docs: `5c679e8:docs/jax_normix_fit_design.md`,
  `485282e:docs/jax_fitting_design_analysis.md`.
- `normix/utils/bessel.py` — `@jax.custom_jvp`, `BESSEL_EPS_V`.
