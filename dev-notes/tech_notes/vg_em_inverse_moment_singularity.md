# The Inverse-Moment Singularity in the VG EM (and the `b_post` floor)

> Companion to the bug report [`investigations/variance_gamma_em_nan.md`](../investigations/variance_gamma_em_nan.md).
> Theory: [`docs/theory/gh.rst`](../../docs/theory/gh.rst), [`docs/theory/em_algorithm.rst`](../../docs/theory/em_algorithm.rst).
> Code: `normix/mixtures/{joint,marginal,factor}.py`,
> `normix/distributions/{variance_gamma,generalized_hyperbolic,generalized_inverse_gaussian,gamma}.py`,
> `normix/utils/constants.py` (`B_POST_FLOOR`).

## Problem

Fitting `VarianceGamma` by EM to a strongly heavy-tailed series drives the Gamma
subordinator shape $\alpha$ downward until the covariance M-step emits `nan` —
even though the marginal log-likelihood is still *improving*.
`NormalInverseGaussian` and `GeneralizedHyperbolic` fit the same data without
trouble.

This note derives the E-step and M-step for VG and GH side by side, isolates
**exactly** which conditional expectation diverges and at what threshold,
explains why VG is the *only* family in `normix` that fails this way, and records
the fix we adopted: a floor on the posterior scale $b_{\text{post}}$, applied
uniformly in the shared E-step.

**Notation.** We follow `docs/theory`:

- $Y \sim \mathrm{GIG}(p, a, b)$, $\;Z \sim \mathcal N(0, \Sigma)$,
  $\;X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y}\,Z$, with $\mu,\gamma\in\mathbb R^d$.
- $q(x) = (x-\mu)^\top\Sigma^{-1}(x-\mu)$ — squared **Mahalanobis distance** of $x$ from $\mu$.
- $\tilde q = \gamma^\top\Sigma^{-1}\gamma$.
- For VG the subordinator is $Y\sim\mathrm{Gamma}(\alpha,\beta)$, i.e. the GIG limit
  $p=\alpha,\;a=2\beta,\;b\to 0$. **The Gamma shape $\alpha$ is the VG analogue of the GH index $p$.**

(In Nitithumbundit & Chan (2015) the same quantity $q(x)$ is written $\delta_i^2$;
we use $q(x)$ throughout to match `normix`'s theory docs.)

---

## 1. Setup: the normal variance–mean mixture

The complete-data density factorises (`docs/theory/gh.rst` eq. gh-joint;
`JointNormalMixture.log_prob_joint`):

$$
\log f(x, y) = \underbrace{-\tfrac{d}{2}\log(2\pi) - \tfrac12\log|\Sigma|
- \tfrac{d}{2}\log y - \tfrac{1}{2y}\,q(x) + \gamma^\top\Sigma^{-1}(x-\mu)
- \tfrac{y}{2}\,\tilde q}_{\log f(x\mid y)} \;+\; \log f_Y(y).
$$

The Gaussian block carries the sufficient statistics

$$
t(x,y) = \big[\,\log y,\;\; y^{-1},\;\; y,\;\; x,\;\; x\,y^{-1},\;\; x x^\top y^{-1}\,\big].
$$

The $y^{-1}$ multiplying the Gaussian quadratic form is intrinsic — it *is* the
inverse-variance weight of a scale mixture. **No EM/ECM/MCECM formulation of this
model can avoid $E[Y^{-1}\mid x]$**: it drives the M-step for $\mu$, $\gamma$,
*and* $\Sigma$ (§4).

---

## 2. VG as the $b\to 0$ limit of GH

| Family | Subordinator | GIG $(p,a,b)$ | Index | $b$ |
|---|---|---|---|---|
| **GH** | $\mathrm{GIG}(p,a,b)$ | $(p,\,a,\,b)$, $a,b>0$ | $p$ | $b>0$ |
| **NIG** | InverseGaussian | $(-\tfrac12,\; \lambda/\mu_{IG}^2,\; \lambda)$ | $-\tfrac12$ | $b=\lambda>0$ |
| **NInvG** | InverseGamma | $(-\alpha,\; 0,\; 2\beta)$ | $-\alpha$ | $b=2\beta>0$ |
| **VG** | $\mathrm{Gamma}(\alpha,\beta)$ | $(\alpha,\; 2\beta,\; b\to 0)$ | $\alpha$ | **$b=0$** |

VG is the unique `normix` family with $b=0$ — the root of everything below.

---

## 3. E-step: the posterior is GIG (shared); the decisive role of $b$

The posterior of the mixing variable is conjugate
(`docs/theory/em_algorithm.rst`):

$$
Y\mid X=x \;\sim\; \mathrm{GIG}\big(\underbrace{p - \tfrac{d}{2}}_{p_{\text{post}}},\;
\underbrace{a + \tilde q}_{a_{\text{post}}},\;
\underbrace{b + q(x)}_{b_{\text{post}}}\big).
$$

| | $p_{\text{post}}$ | $a_{\text{post}}$ | $b_{\text{post}}$ |
|---|---|---|---|
| **GH** | $p-\tfrac d2$ | $a+\tilde q$ | $\;b+q(x)$ |
| **VG** | $\alpha-\tfrac d2$ | $2\beta+\tilde q$ | $\;q(x)$ |

(`JointGeneralizedHyperbolic._posterior_gig_params`, `JointVarianceGamma._posterior_gig_params`.)

**The decisive difference is the last column.** GH carries $b_{\text{post}} = b+q(x) \ge b > 0$,
bounded away from zero. VG has $b_{\text{post}} = q(x)$, which $\to 0$ for any
observation approaching the fitted location $\mu$.

### Conditional moments (Bessel ratios)

Let $\omega := \sqrt{a_{\text{post}}\,b_{\text{post}}}$ be the **posterior Bessel
argument**. The GIG moment formula gives
(`GeneralizedInverseGaussian._expectation_params_batch_cpu`):

$$
E[Y\mid x] = \sqrt{\tfrac{b_{\text{post}}}{a_{\text{post}}}}\,
\frac{K_{p_{\text{post}}+1}(\omega)}{K_{p_{\text{post}}}(\omega)},\qquad
E[Y^{-1}\mid x] = \sqrt{\tfrac{a_{\text{post}}}{b_{\text{post}}}}\,
\frac{K_{p_{\text{post}}-1}(\omega)}{K_{p_{\text{post}}}(\omega)},
$$
$$
E[\log Y\mid x] = \tfrac12\log\tfrac{b_{\text{post}}}{a_{\text{post}}}
+ \frac{\partial_\nu K_\nu(\omega)\big|_{\nu=p_{\text{post}}}}{K_{p_{\text{post}}}(\omega)}.
$$

The batch E-step (`NormalMixture._aggregate_eta`) averages these into the six
expectation parameters $\hat\eta = \tfrac1n\sum_i E[t(x_i,Y)\mid x_i]$, in the
order of `docs/theory/gh.rst`:

$$
\eta_1 = E[\log Y],\;\; \eta_2 = E[Y^{-1}],\;\; \eta_3 = E[Y],\;\;
\eta_4 = E[X],\;\; \eta_5 = E[X Y^{-1}],\;\; \eta_6 = E[X X^\top Y^{-1}].
$$

---

## 4. M-step: identical closed form for VG and GH

Maximising $\sum_i E[\log f(x_i, Y)\mid x_i]$ over the Gaussian block gives the
**same** closed form for every family (`docs/theory/gh.rst` eq. gh-m-step;
`JointNormalMixture._mstep_normal_params`):

$$
\mu = \frac{\eta_4 - \eta_3\,\eta_5}{1 - \eta_2\,\eta_3},\qquad
\gamma = \frac{\eta_5 - \eta_2\,\eta_4}{1 - \eta_2\,\eta_3},
$$
$$
\Sigma = \eta_6 - \eta_5\,\mu^\top - \mu\,\eta_5^\top
+ \eta_2\,\mu\mu^\top - \eta_3\,\gamma\gamma^\top.
$$

**$E[Y^{-1}\mid x]$ enters all three updates**: through $\eta_2 = \widehat{E[Y^{-1}]}$
(the denominator $1-\eta_2\eta_3$), through $\eta_5 = \tfrac1n\sum_i x_i\,E[Y^{-1}\mid x_i]$,
and through $\eta_6 = \tfrac1n\sum_i x_i x_i^\top\,E[Y^{-1}\mid x_i]$. The families
differ **only** in the subordinator update:

| | Subordinator update | code |
|---|---|---|
| **VG** | digamma inversion $\psi(\alpha)-\log\alpha = \eta_1-\log\eta_3$, then $\beta=\alpha/\eta_3$ | `Gamma.from_expectation` |
| **GH** | GIG $(\eta_1,\eta_2,\eta_3)\to(p,a,b)$ Bregman/Newton solve | `GeneralizedHyperbolic.m_step_subordinator` |

---

## 5. The singularity: posterior moments as $q(x)\to 0$

Everything hinges on the small-argument behaviour of $K_\nu$. As $z\to0^+$,

$$
K_\nu(z) \sim \tfrac12\Gamma(|\nu|)\big(\tfrac{z}{2}\big)^{-|\nu|}\ (\nu\neq0),
\qquad K_0(z)\sim -\log z .
$$

For VG, $b_{\text{post}} = q(x)$, $\omega = \sqrt{a_{\text{post}}\,q(x)}$, and write
$\nu_\star := p_{\text{post}} = \alpha - \tfrac d2$. Substituting into
$E[Y^{-1}\mid x] = \sqrt{a_{\text{post}}/q(x)}\;K_{\nu_\star-1}(\omega)/K_{\nu_\star}(\omega)$:

**Case $\nu_\star>1$** ($\alpha > \tfrac d2+1$): $K_{\nu_\star-1}/K_{\nu_\star}\sim \tfrac{1}{\nu_\star-1}\tfrac{\omega}{2}$, so

$$
E[Y^{-1}\mid x]\;\longrightarrow\;\frac{a_{\text{post}}}{2(\nu_\star-1)}=\frac{a_{\text{post}}}{2\alpha-d-2}\quad(\text{finite}).
$$

**Case $0<\nu_\star<1$** ($\tfrac d2<\alpha<\tfrac d2+1$): using $K_{\nu_\star-1}=K_{1-\nu_\star}$,

$$
E[Y^{-1}\mid x]\;\sim\; C\,q(x)^{\,\nu_\star-1}=C\,q(x)^{\,\alpha-\frac d2-1}\;\longrightarrow\;\infty .
$$

**Case $\nu_\star<0$** ($\alpha<\tfrac d2$): $E[Y^{-1}\mid x]\sim (d-2\alpha)/q(x)\to\infty$
(at $\alpha=\tfrac d2$ the borderline is $-1/(q(x)\log\sqrt{q(x)})$).

Collecting the thresholds for divergence **as $q(x)\to 0$**:

| Quantity | diverges when | $d=1$ |
|---|---|---|
| $E[Y^{-1}\mid x]$ — drives $\mu,\gamma,\Sigma$ | $\alpha\le \tfrac d2+1$ | $\alpha\le 1.5$ |
| $E[\log Y\mid x]$ — drives $\alpha$ | $\alpha\le \tfrac d2$ | $\alpha\le 0.5$ |
| marginal pdf $f(x)$ at $x=\mu$ | $\alpha\le \tfrac d2$ | $\alpha\le 0.5$ |

**Why VG alone fails.** $E[Y^{-1}\mid x]$ is the *first* to blow up, at
$\alpha\le \tfrac d2+1$ — well before the density itself becomes singular at
$\alpha\le \tfrac d2$. For GH/NIG/NInvG, $b_{\text{post}} = b+q(x)\ge b>0$, so
$\omega\ge\sqrt{a_{\text{post}}\,b}$ never enters the small-$z$ regime and all
three moments stay finite for every $x$, **whatever the index $p$**. Only VG
($b=0$) lets $b_{\text{post}} = q(x)\to 0$.

**Connection to the unbounded likelihood.** This is *not merely numerical*. For
$\alpha < \tfrac d2$ the VG marginal density is genuinely unbounded at $x=\mu$
($\propto q(x)^{\alpha - d/2}\to\infty$). As $\mu\to$ a data point and $\alpha$
shrinks, the likelihood diverges — the same degeneracy as the unbounded
likelihood of Gaussian *mixtures* (Day 1969). The EM is correctly climbing toward
an unbounded spike; the remedy must *regularize*, not "fix arithmetic".

**Match to the observed failure** (`investigations/variance_gamma_em_nan.md`):
the run blew up at $\alpha\approx 0.69$ ($d=1$), which is $<1.5$ (deep in the
$E[Y^{-1}\mid x]$-unbounded regime, so $\Sigma$ overflows) yet $>0.5$ (so the
density and log-likelihood are still finite and improving). Numerically, with
$\alpha=0.7,\,\beta=1$ the repro reported $E[Y^{-1}\mid x]\approx 4.4\times10^{23}$
at the exact mode; flooring $b_{\text{post}}$ at $10^{-6}$ brings it to
$\approx 3.0\times10^{4}$.

> **Footnote (the $4.4\times10^{23}$ is a clamp artifact).** At exactly
> $q(x)=0$ the true conditional moment is $+\infty$ ($\nu_\star=0.2<1$), so no
> finite value can follow from $q=0$ literally. The finite number measures an
> internal clamp: the JAX expectation path floors the GIG scale at
> `LOG_EPS` $=10^{-30}$ before the Bessel calls, and
> $E[Y^{-1}\mid x]\big|_{b=10^{-30}} = C\,(10^{-30})^{\nu_\star-1} \approx 4.4\times10^{23}$
> reproduces the anecdote exactly (the CPU path's `TINY` $=10^{-300}$ would
> give $4.4\times10^{239}$). The number characterizes the clamp, not the model.

---

## 6. The fix we adopted: floor $b_{\text{post}}$

We keep $E[Y^{-1}\mid x]$ and $E[\log Y\mid x]$ but prevent $b_{\text{post}}$ from
reaching zero. In the shared E-step, after the family-specific
`_posterior_gig_params` returns $(p_{\text{post}}, a_{\text{post}}, b_{\text{post}})$:

$$
\boxed{\;b_{\text{post}} \;\leftarrow\; \max\!\big(b_{\text{post}},\; b_{\min}\big),\qquad
b_{\min} = \mathtt{B\_POST\_FLOOR} = 10^{-6}.\;}
$$

This bounds the prefactor $\sqrt{a_{\text{post}}/b_{\text{post}}}\le\sqrt{a_{\text{post}}/b_{\min}}$
and the argument $\omega\ge\sqrt{a_{\text{post}}\,b_{\min}}$ together, so all three
conditional moments stay finite for every observation, including $x=\mu$.

### Why floor $b$ rather than $\omega$

Nitithumbundit & Chan (2015) floor the **argument** $\omega$ at a constant
$\Delta$ (their "delta region": when $\omega = \sqrt{a_{\text{post}}\,q(x)}<\Delta$,
replace $q(x)$ by $\Delta^2/a_{\text{post}}$). Because
$\omega = \sqrt{a_{\text{post}}\,b_{\text{post}}}$ couples $a$ and $b$, the two
choices are *not* the same bound — they translate into each other only through
$a_{\text{post}}$:

$$
\text{floor }b_{\text{post}}\ge b_{\min}\;\Longrightarrow\;\omega\ge\sqrt{a_{\text{post}}\,b_{\min}}
\quad(\text{$\omega$-floor drifts with }\sqrt{a_{\text{post}}}),
$$
$$
\text{floor }\omega\ge\Delta\;\Longleftrightarrow\;b_{\text{post}}\ge\Delta^2/a_{\text{post}}
\quad(\text{$b$-floor drifts with }1/a_{\text{post}}).
$$

**Decision: floor $b_{\text{post}}$ at a constant $b_{\min}$, uniformly for all
families.** Rationale:

- *Simplicity.* A constant `jnp.maximum(b_post, B_POST_FLOOR)` needs no
  per-observation $a_{\text{post}}$ coupling and reads identically in every E-step
  path.
- *Uniformity of the cap.* In the singular regime the $b$-floor caps
  $E[Y^{-1}\mid x]$ uniformly in $a_{\text{post}}$ and only linearly in $d$;
  the NC $\omega$-floor's cap grows linearly in $a_{\text{post}}$ (next
  subsection).
- *It coincides with GH's existing guard.* GH already keeps its prior $b$ away
  from zero (the GIG solve reverts $a,b$ below `GIG_CLAMP_LO = 1e-6`), so flooring
  $b_{\text{post}}$ at the same $10^{-6}$ extends one consistent rule —
  "the GIG scale never sits below $10^{-6}$" — from the prior to the posterior.
- *Calibration.* For $a_{\text{post}}\sim\mathcal O(1)$ the induced argument floor
  is $\omega\ge\sqrt{1\cdot10^{-6}} = 10^{-3}$, squarely inside NC's recommended
  $\Delta\in[10^{-5},10^{-3}]$.

### How large can the capped moment get?

With $b_{\text{post}}$ pinned at $b_{\min}$ and
$\nu := p_{\text{post}} = \alpha - \tfrac d2$, the small-$z$ Bessel
asymptotics of §5 give, as $\omega = \sqrt{a_{\text{post}}\,b_{\min}} \to 0$,

$$
E[Y^{-1}\mid x]\Big|_{b=b_{\min}} \;\sim\;
\begin{cases}
\dfrac{\Gamma(1-\nu)}{\Gamma(\nu)}\, 2^{1-2\nu}\,
a_{\text{post}}^{\,\nu}\, b_{\min}^{\,\nu-1}, & 0<\nu<1,\\[2ex]
\dfrac{2|\nu|}{b_{\min}} \;=\; \dfrac{d-2\alpha}{b_{\min}}, & \nu<0,
\end{cases}
$$

with logarithmic corrections at the borderlines $\nu\in\{0,1\}$. For large
$\omega$ (huge $a_{\text{post}}$; the Bessel ratio $\to 1$) the cap is
$E[Y^{-1}\mid x] \approx \sqrt{a_{\text{post}}/b_{\min}}$. The worst case is
therefore

$$
E[Y^{-1}\mid x] \;\lesssim\; \max\!\Big(\frac{d-2\alpha}{b_{\min}},\;
\sqrt{\frac{a_{\text{post}}}{b_{\min}}}\Big)
$$

— **linear in $d$ and uniform in $a_{\text{post}}$** in the singular regime
$\nu<0$. Even at $d=100,\ \alpha=0.5$ the cap is only
$99/b_{\min} \approx 10^8$, comfortably inside float64 range. (An earlier
revision of this note extrapolated the $0<\nu<1$ formula to $\nu<0$, predicted
caps exploding with $|\nu|$, and recommended switching to the NC $\omega$-floor
for high-dimensional VG; the case split above corrects both claims.)

By contrast, the $\omega$-floor's induced bound
$b_{\text{post}}\ge\Delta^2/a_{\text{post}}$ caps the moment at
$2|\nu|\,a_{\text{post}}/\Delta^2$ in the same regime — **growing linearly in
$a_{\text{post}}$** (large $\beta$, strong skewness $\tilde q$). The constant
$b$-floor is the *more* uniform of the two regularizers; no high-$d$ escape
hatch is needed.

**Numerical verification** (scipy `kve`, $b_{\min}=10^{-6}$, $\Delta=10^{-3}$):

| Case | Asymptotic above | Exact |
|---|---|---|
| $\nu=0.2,\ a_{\text{post}}=2$ ($d=1,\ \alpha=0.7$) | $2.79\times10^4$ | $2.99\times10^4$ |
| $\nu=-0.5,\ a_{\text{post}}=2$ ($d=1,\ \alpha\to0$) | $2\lvert\nu\rvert/b_{\min}=10^6$ | $1.0\times10^6$ |
| $\nu=-49.5,\ a_{\text{post}}=2$ ($d=100,\ \alpha=0.5$) | $99/b_{\min}=9.9\times10^7$ | $9.9\times10^7$ |
| $a$-scaling, $\nu=-0.5$, $a_{\text{post}}\in[0.1,100]$ | — | $b$-floor cap $\approx10^6$ (flat); $\omega$-floor cap $10^5\to10^8$ |

Repro:

```python
import numpy as np
from scipy.special import kve

E_inv_Y = lambda p, a, b: np.sqrt(a/b) * kve(p-1, np.sqrt(a*b)) / kve(p, np.sqrt(a*b))
E_inv_Y(0.2, 2.0, 1e-6)    # b-floor cap, 0<nu<1     -> 2.99e4
E_inv_Y(-0.5, 2.0, 1e-6)   # b-floor cap, nu<0       -> 1.0e6
E_inv_Y(-0.5, 100.0, 1e-6) # uniform in a_post       -> 1.0e6
E_inv_Y(-0.5, 100.0, 1e-6/100.0)  # omega-floor cap  -> 1.0e8
```

### Gauge dependence of $b_{\min}$

$b_{\text{post}} = q(x) = (x-\mu)^\top\Sigma^{-1}(x-\mu)$ is not invariant
under the model's scale gauge $Y \to sY$, $\gamma \to \gamma/s$,
$\Sigma \to \Sigma/s$ (which leaves the marginal law of $X$ unchanged): under
it $q(x) \to s\,q(x)$, so a fixed $b_{\min} = 10^{-6}$ binds with different
strength depending on how the gauge is pinned. In particular,
`regularization='det_sigma_one'` ($|\Sigma|=1$ after each M-step) and
`'none'` ($\Sigma$ at the empirical scale) place the same data at different
distances from the floor. The floor is a numerical guard, not a
gauge-invariant model statement; fits that actually sit near the floor should
be compared under a common gauge.

---

## 7. Applicability across families & where it lives

The floor is applied in the **shared** E-step chokepoints, so it is uniform by
construction and not a VG special-case:

| Path | Method | File |
|---|---|---|
| Joint, JAX (and direct callers) | `JointNormalMixture._compute_posterior_expectations` | `mixtures/joint.py` |
| Joint, CPU | `NormalMixture._e_step_subordinator_cpu` | `mixtures/marginal.py` |
| Factor, JAX | `FactorNormalMixture._conditional_expectations` | `mixtures/factor.py` |
| Factor, CPU | `FactorNormalMixture._e_step_subordinator_cpu` | `mixtures/factor.py` |

`_posterior_gig_params` itself stays **pure** (returns the raw posterior
$(p,a,b)$); the floor is applied by the caller, so the posterior-parameter unit
tests are unaffected.

Effect per family:

- **VG**: $b_{\text{post}} = q(x)$, so the floor is the active regularizer for
  near-mode observations.
- **GH / NIG / NInvG**: $b_{\text{post}} = b_{\text{prior}} + q(x)\ge b_{\text{prior}}$,
  and $b_{\text{prior}} \ge 10^{-6}$ in normal operation, so the floor is a
  **dormant safety net** — it never binds except in the extreme near-VG corner
  ($b_{\text{prior}}\to 0$), where it harmlessly tightens an already-safe update.

The duplicated per-subclass `_compute_posterior_expectations` overrides were
removed; the centralized base method computes the floored GIG moments from each
family's `_posterior_gig_params`.

---

## 8. Why we do *not* add NC's extra E-step

NC bundle a *second* adjustment: an extra E-step that recomputes $E[Y^{-1}\mid x]$
at the updated $(\mu,\gamma)$ **before** forming $\Sigma$. It targets a different,
second-order effect: in the closed-form M-step, $\Sigma$ uses weights
$E[Y^{-1}\mid x_i]$ evaluated at $\theta^{(t)}$ but residuals $(x_i-\mu^{(t+1)})$
at the new $\mu$. If an observation sat at the *old* mode, a stale-large weight
paired with a moved residual inflates $\Sigma$.

We keep `normix`'s VG EM as **classical EM (one E-step, one M-step)** and do *not*
add this, because:

1. **The single-E/single-M update is the exact joint maximizer of the
   EM $Q$-function.** $(\mu,\gamma)$ solve a linear system independent of $\Sigma$,
   then $\Sigma$ is the weighted residual covariance — there is no inner fixed
   point to iterate.
2. **Once $b_{\text{post}}$ is floored, every $E[Y^{-1}\mid x_i]$ is finite**, so
   that exact M-step is finite and well-posed — no `nan`. The extra E-step only
   shaves a transient $\Sigma$ inflation that needs *both* a near-old-mode
   observation *and* a large $\mu$ move in one step; with a sane floor and the
   small per-iteration $\mu$ moves of `normix`'s regime, it does not materialize.
3. The extra E-step is, by definition, **not classical EM** — it is a multi-cycle
   ECM refinement. `normix` already exposes `algorithm='mcecm'`
   (E → M$_{\text{normal}}$ → E → M$_{\text{sub}}$) as an escape hatch if extra
   robustness is ever wanted.

---

## References

- T. Nitithumbundit, J. S. K. Chan (2015). *An ECM algorithm for Skewed
  Multivariate Variance Gamma Distribution in Normal Mean-Variance
  Representation.* arXiv:1504.01239 — §2 (unbounded density), §3.3.1 (moment
  asymptotics, the $\Delta$ delta-region bound, and the extra E-step).
- A. J. McNeil, R. Frey, P. Embrechts (2005). *Quantitative Risk Management*,
  §3.2 — the canonical MCECM algorithm for GH/VG.
- D. Lüthi, W. Breymann. `ghyp` R package — reference MCECM implementation;
  documents the $x=\mu$ VG singularity and the fix-$\lambda$ option.
- M. Bee, M. M. Dickson, F. Santi (2018). *Likelihood-based risk estimation for
  variance-gamma models.* Statistical Methods & Applications 27(1).
- N. E. Day (1969). *Estimating the components of a mixture of normal
  distributions.* Biometrika 56(3) — origin of the unbounded-likelihood degeneracy.
- `normix` theory: [`docs/theory/gh.rst`](../../docs/theory/gh.rst),
  [`docs/theory/em_algorithm.rst`](../../docs/theory/em_algorithm.rst).
