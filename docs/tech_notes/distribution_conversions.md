# Distribution Conversions: KL Projection and Exact Embedding

> Code: `to_<name>` methods on each distribution.
> Tests: `tests/test_distribution_conversions.py`.

## Problem

normix exposes a hierarchy of nested exponential families:

```
GIG  ⊃  Gamma, InverseGamma, InverseGaussian      (univariate subordinators)
GH   ⊃  VG,    NInvG,        NIG                  (joint normal mixtures)
```

Each special case is a sub-manifold of the more general family. We want
two operations between any (general $\mathcal{P}$, special $\mathcal{Q}$)
pair, both exposed as `source.to_<name>()`:

1. **Special → General** (e.g. `Gamma.to_gig`). The special case lies on
   a sub-manifold; the conversion is an exact algebraic embedding.
2. **General → Special** (e.g. `GIG.to_gamma`). The conversion is the KL
   projection
   $$q^* = \arg\min_{q \in \mathcal{Q}} D_{\mathrm{KL}}(p \,\|\, q).$$

When the source $p$ already lies in $\mathcal{Q}$ both operations agree
and the round-trip `source.to_<general>().to_<source>()` is a numerical
identity.

## 1. KL Projection via Moment Matching

For an exponential-family target with sufficient statistics $t_q(x)$,
log base measure $\log h_q(x)$ and log partition $\psi_q(\theta)$,

$$
\log q_\theta(x) = \log h_q(x) + \theta^\top t_q(x) - \psi_q(\theta).
$$

Then for any source $p$ (not necessarily in the same family),

$$
D_{\mathrm{KL}}(p \,\|\, q_\theta)
= -H(p) - E_p[\log h_q(X)] - \theta^\top E_p[t_q(X)] + \psi_q(\theta).
$$

Differentiating with respect to $\theta$:

$$
\nabla_\theta D_{\mathrm{KL}}(p \,\|\, q_\theta)
= -E_p[t_q(X)] + \nabla\psi_q(\theta)
= -E_p[t_q(X)] + \eta_q(\theta).
$$

The first-order condition for the projection is the moment-matching
identity

$$
\boxed{\eta_q(\theta^*) \;=\; E_p[t_q(X)]}.
$$

Convexity of $\psi_q$ makes $\theta^*$ unique on the interior of the
domain. Concretely: compute the expected target sufficient statistics
under the source, then call `q.from_expectation(...)`.

The base measure $h_q$ does not enter the gradient, so cross-family
projections (e.g. GIG → InverseGaussian, where $h_q(x) = (2\pi)^{-1/2}
x^{-3/2}$ but $h_p(x) = 1$) are no harder than projections within the
same parametric family.

## 2. The Subordinator Projections (GIG → Gamma / InvGamma / InvGauss)

GIG sufficient statistics are $t_p(x) = [\log x,\; 1/x,\; x]$, so
$\eta_p = (E[\log X],\; E[1/X],\; E[X])$ is what `expectation_params()`
returns.

| Target | $t_q(x)$ | $E_p[t_q(X)]$ in terms of $\eta_p$ |
|---|---|---|
| `Gamma`           | $[\log x,\; x]$     | $[\eta_1,\;\; \eta_3]$ |
| `InverseGamma`    | $[-1/x,\; \log x]$  | $[-\eta_2,\; \eta_1]$ |
| `InverseGaussian` | $[x,\; 1/x]$        | $[\eta_3,\;\; \eta_2]$ |

Each target has a closed-form (or near-closed-form via Newton-on-digamma)
`from_expectation`, so the projection is a single small computation —
not a Bregman optimisation.

### Sanity Conditions

- **Gamma**: always well-posed; `Gamma.from_expectation` solves
  $\psi(\alpha) - \log\alpha = \eta_1 - \log\eta_3$ for $\alpha > 0$,
  then $\beta = \alpha/\eta_3$.
- **InverseGamma**: same shape of equation; always well-posed.
- **InverseGaussian**: requires $\eta_2 > 1/\eta_3$, which is **Jensen's
  inequality** applied to the strictly convex function $\varphi(x) = 1/x$
  on $x > 0$:

  $$
  E_p[1/X] \;\ge\; \frac{1}{E_p[X]}, \qquad
  \text{equality iff } X \text{ is constant a.s.}
  $$

  The InverseGaussian closed form derives from
  $E_q[X] = \mu,\; E_q[1/X] = 1/\mu + 1/\lambda$, so

  $$
  \mu = \eta_3, \qquad
  \lambda = \frac{1}{\eta_2 - 1/\eta_3}.
  $$

  Thus $\lambda > 0$ is exactly Jensen's strict inequality. For any
  non-degenerate GIG (positive variance), the inequality is strict in
  exact arithmetic, so the projection is well-posed.

  **When the gap is small.** The denominator $\eta_2 - 1/\eta_3$ shrinks
  near the Gamma limit (where the projected $\lambda \to \infty$). For a
  true Gamma($\alpha,\beta$) source ($\alpha > 1$):

  $$
  \eta_2 - \frac{1}{\eta_3}
  = \frac{\beta}{\alpha-1} - \frac{\beta}{\alpha}
  = \frac{\beta}{\alpha(\alpha-1)}
  \;\xrightarrow[\alpha \to \infty]{}\; 0,
  $$

  with the projected $\lambda = \alpha(\alpha-1)/\beta \to \infty$
  (the InverseGaussian degenerates to a point mass at $\mu = \alpha/\beta$).

  **Floating-point failure mode.** When the source GIG is lifted from a
  Gamma via `Gamma.to_gig` (i.e. stored with $b = 0$), GIG's
  `_grad_log_partition` clamps $b$ to `LOG_EPS = 1e-30` and computes
  $\eta_2,\, \eta_3$ via small-$z$ Bessel asymptotics. Both ratios are of
  order $|\beta/\alpha|$, but their difference is of order
  $|\beta/(\alpha^2)|$ — a roughly $\alpha$-fold relative cancellation.
  For very large $\alpha$ (or for an $\alpha \le 1$ Gamma where $E[1/X]$
  is genuinely infinite), roundoff can drive
  $\eta_2 - 1/\eta_3 \le 0$ even though the mathematical value is
  positive. `InverseGaussian.from_expectation` guards against this with
  `lam = jnp.maximum(1/(eta_2 - 1/eta_3), LOG_EPS)`, returning the
  numerically degenerate but valid InverseGaussian
  $(\mu = \eta_3,\; \lambda = 10^{30})$. The KL divergence to such a
  projection is large (as it should be: a sharp Gamma is a poor
  InverseGaussian fit), so consumers who care about projection quality
  should check `kl_divergence` rather than rely on the projection alone.

  In practice this matters mostly when projecting a *true* Gamma to
  InverseGaussian; if that's the goal, route through GIG explicitly via
  `Gamma(...).to_gig().to_inverse_gaussian()` so the assumption is
  visible at the call site.

## 3. The Exact Embeddings (Special → GIG)

GIG with classical parameters $(p, a, b)$ has density
$f(x) \propto x^{p-1} \exp(-(ax + b/x)/2)$.

| Source | GIG image | Boundary |
|---|---|---|
| `Gamma(α, β)`           | $(p,a,b) = (\alpha,\; 2\beta,\; 0)$           | $b = 0$ |
| `InverseGamma(α, β)`    | $(p,a,b) = (-\alpha,\; 0,\; 2\beta)$          | $a = 0$ |
| `InverseGaussian(μ, λ)` | $(p,a,b) = (-1/2,\; \lambda/\mu^2,\; \lambda)$ | strict interior |

Verification for Gamma: substituting $a = 2\beta$, $b = 0$ into the GIG
kernel removes the $b/x$ term and recovers $x^{\alpha-1} e^{-\beta x}$
(up to normalisation). The other two are analogous.

### Why `boundary_eps`?

Storing $b = 0$ (or $a = 0$) is mathematically pure, and GIG's
`_log_partition_from_theta` recognises this regime via its degenerate
branch (`use_gamma`/`use_invg` flags driven by `GIG_DEGEN_THRESHOLD`).
However:

- `_grad_log_partition` clamps both bounds to `LOG_EPS = 1e-30` to keep
  the Bessel ratios finite. The Bessel asymptotics on
  $\sqrt{ab} \approx 0$ still recover the correct Gamma / InverseGamma
  expectations for $E[\log X]$ and the dominant statistic, but
  $E[1/X]$ (Gamma case) or $E[X]$ (InverseGamma case) is computed using
  small-$z$ asymptotics whose constant depends on which Bessel ratio is
  taken; consumers reading the *unused* GIG sufficient statistic from a
  freshly lifted GIG should not assume sharp asymptotic behaviour.

The KL projection back to the special case ignores that unused
component (its target sufficient statistics don't include it), so the
round-trip is exact regardless. Pass `boundary_eps > 0` only when
downstream code wants to inspect the lifted GIG as a strict-interior
distribution.

## 4. Joint Mixtures: GH ↔ VG / NInvG / NIG

Each `JointXxx` shares the same Normal block
$f(x \mid y) = \mathcal{N}(\mu + \gamma y,\; \Sigma y)$ and differs only
in the subordinator $f_Y(y)$. The joint KL therefore decomposes:

$$
D_{\mathrm{KL}}(\text{JointGH} \,\|\, \text{JointVG})
= E_{Y \sim \text{GIG}}\!\left[
  D_{\mathrm{KL}}(p_{X|Y}\,\|\,q_{X|Y})
  + \log\frac{f_Y(Y)}{g_Y(Y)}
\right].
$$

Because $p_{X|Y} = q_{X|Y}$ (identical Gaussians given $Y$), the first
term vanishes and

$$
D_{\mathrm{KL}}(\text{JointGH} \,\|\, \text{JointVG})
= D_{\mathrm{KL}}(\text{GIG} \,\|\, \text{Gamma}).
$$

So projecting the joint reduces to projecting the subordinator. The
implementation does exactly that:

```python
def to_joint_variance_gamma(self):
    gamma = self.subordinator().to_gamma()         # GIG -> Gamma KL projection
    return JointVarianceGamma(
        mu=self.mu, gamma=self.gamma, L_Sigma=self.L_Sigma,
        alpha=gamma.alpha, beta=gamma.beta,
    )
```

The marginal layer (`GeneralizedHyperbolic.to_variance_gamma`, etc.)
delegates to the joint conversion and re-wraps.

## 5. Round-Trip Identity

For any special-case source $s \in \mathcal{Q}$:

$$
s \;\xrightarrow{\text{lift}}\; s' \in \mathcal{P}
\;\xrightarrow{\text{project}}\; s^{**} \in \mathcal{Q}.
$$

By construction $s' \in \mathcal{Q}$ (it lies on the sub-manifold), and
the KL projection of any point in $\mathcal{Q}$ onto $\mathcal{Q}$ is
the point itself. So $s^{**} = s$ up to numerical error from the
`from_expectation` solver (~1e-9 for Gamma/InvGamma's digamma Newton,
exact for InvGauss).

`tests/test_distribution_conversions.py` checks this in three forms:

1. parameter equality after the round-trip;
2. `kl_divergence(original, round_trip)` is below 1e-10 at the joint
   level for VG / NInvG and below 1e-12 for NIG (no boundary);
3. for a generic GH that is *not* on a sub-manifold, the projected VG
   sits at a Monte-Carlo-estimated local minimum of cross-family KL.

## 6. Why Same-Name `to_<x>()` for Both Directions

The mathematical operations differ (lossless embedding vs. KL-minimising
approximation), but from the user's perspective the question is the
same: *"what is the closest distribution in family $\mathcal{Q}$ to
this one?"* When the source is on the manifold, the answer is the
source itself (both operations agree); when it is not, the answer is
the unique KL projection. A single verb keeps the call sites uncluttered
and lets the mathematical distinction live in the docstrings, where it
belongs.
