# Mathematical formalization checklist

2026-10-24 · Tracking document

This file tracks the mathematical inventory from [the Lean feasibility report](01_lean_math_feasibility.md). Work on one identified item at a time. Whoever works on an item updates its fields, dependencies, status, and Notes/PR. Keep IDs stable. Split an item if its statement cannot be reviewed or proved independently.

Use these statuses: `not-started`, `statement-drafted`, `statement-reviewed`, `proof-in-progress`, `proved`, and `blocked`. All items initially have status `not-started`. Before proof work, write an English proposition card with a prose statement, Lean-style signature, assumptions, intended-spec rationale, covered normix function, and risks of weakening the statement. Use [the report's template](01_lean_math_feasibility.md#64-proposition-cards-and-specification-review). For `blocked`, record the missing mathematical fact, library support, or implementation correspondence. For `proved`, record the reviewed statement, checked commit, proof location, and axiom audit. A conditional theorem does not complete its distribution instance or code bridge.

The reading baseline is `51e053c57237a29c71b56d3a2a86988277477bd9`. No Lean proof, test, benchmark, or build was run for this inventory. **local check** means source or documentation evidence from the original static investigation. **derivation** means the calculation or recalled mathematics described here. **to verify** marks uncompiled statements, unchecked library support, and literature details. Source labels N1–N8 and J1–J5 refer to [the report's sources](01_lean_math_feasibility.md#sources).

Lean module paths below are planned, not existing proof artifacts. Files, directories, modules, and namespaces use UpperCamelCase; definitions use lowerCamelCase; theorems use snake_case. The lowercase `formal/lean/` is the package directory. Hand-written Lean models and the jaxpr bridge proceed in parallel. The bridge aims for broad normix coverage; difficult translations receive explicit hand-written models and separate correspondence obligations.

## Difficulty, value, and effort

Difficulty A means finite algebra. B combines existing analysis tools. C requires continuous probability or substantial integration infrastructure. D requires special-function or error-analysis development. Values reflect relevance to normix. Grades and effort estimates are planning judgments, not measured results. A person-week is five working days. Estimates assume an experienced Lean proof engineer and mathematical review. They are incremental after relevant foundations exist, so they cannot simply be added.

| Target group | Difficulty / value | Incremental estimate | Proof work | Review focus |
|---|---|---|---|---|
| GIG/Gamma parameter roundtrips and statistic order | A / High | 2–5 person-days | Definitions, algebra, mappings | Parameter conventions and clamping boundaries |
| Finite weighted-sum gradients, covariance, positivity | A–B / Very high | 1–3 person-weeks | Finite sums and calculus | Fixed-node meaning and strict positivity |
| Continuous EF convexity, gradient, Fisher information | C / Very high | 4–8 person-weeks | Integrability and derivative lemmas | Regularity, support, nonminimal representations |
| Legendre duality and local inverse | B–C / High | 2–5 person-weeks | Convex analysis and chain rule | Invertible domain and global-surjectivity claims |
| Gamma/InverseGamma normalization and moments | B–C / High | 2–4 person-weeks | Substitution and Gamma integrals | Rate/scale conventions and moment domains |
| IG normalization, moments, closed CDF | C / High | 2–5 person-weeks | Gaussian integrals and derivatives | Parameterization, endpoints, tails |
| GIG normalization and arbitrary real moments | C / Very high | 2–4 person-weeks, plus Bessel foundations | Log substitution and normalization ratios | Interior versus exact Gamma boundaries |
| Mixture representation, mean, covariance | C / Very high | 2–4 person-weeks | Product measures and measurable maps | Independence and required moments |
| GH closed density and conditional GIG | C / Very high | 2–5 person-weeks, using preceding results | Complete the square, substitute, integrate | Constants, matrix domain, special points |
| Specialized NIG/VG/NInvG formulas | B–C / High | 1–3 person-weeks per family | Specialized integrals and boundary proofs | Exact boundaries cannot be replaced by numerical limits |
| General CDF properties and ideal inverse sampling | C / High | 2–4 person-weeks | Monotonicity and pushforward laws | Discontinuities, endpoints, quantization scope |
| Ideal IG/GIG rejection algorithms | C–D / Medium–high | 4–10 person-weeks each | Proposal laws, acceptance, termination | Valid envelopes and failure events |
| One-step EM monotonicity and penalized EM | C / Very high | 3–6 person-weeks after density foundations | KL/Jensen arguments and instance conditions | Exact E/M-steps and objective |
| Normal-block and IG M-steps | A–C / Very high | 1–3 person-weeks | Algebra and objective comparison | Nonsingularity and optimality, not just stationarity |
| Stationarity of EM accumulation points | C–D / Medium | 6–12 person-weeks | Compactness and limits | Applicability to actual GH fitting |
| Bessel symmetry, recurrence, parameter derivatives | C–D / Very high | 4–8 person-weeks for foundations; 2–4 more for identities | Integral bounds, differentiation, integration by parts | Dominating functions and boundary terms |
| Bessel asymptotics, quadrature and derivative error | D / High, costly | 2–6 person-months per restricted region | Auxiliary bounds and formal proofs | Error metric, parameter uniformity, analysis plan |

Finite exponential families support discrete Bessel moments. Continuous EF analysis and Bessel integrals support GIG. Mixture densities and conditional GIG laws then support EM instances.

## 1. Exponential families

Let $\rho$ be a fixed reference measure, $h(x)\ge0$ a base-density factor, and $t(x)\in\mathbb R^k$ the sufficient statistic. The statistic collects the observation functions paired with parameters in the exponent. Define

$$
Z(\theta)=\int h(x)e^{\langle\theta,t(x)\rangle}\,d\rho(x),\qquad
\psi(\theta)=\log Z(\theta),\qquad
p_\theta(x)=h(x)e^{\langle\theta,t(x)\rangle-\psi(\theta)}.
$$

Start on the domain $0<Z(\theta)<\infty$. Derivative results use its open interior. `log_base_measure` returns $\log h$, not $h$. [N2; local check]

| ID | Item | normix function/page | Difficulty | Value | Dependencies | Lean module (planned) | Status | Notes/PR |
|---|---|---|---|---|---|---|---|---|
| EF-01 | GIG $b$ roundtrip: $-2(-b/2)=b$ | `normix/distributions/generalized_inverse_gaussian.py`: `natural_params`, `from_natural` | A | High | Real arithmetic | `formal/lean/Normix/Distributions/GigParameters.lean` | statement-drafted | Formula before clamping; local check / derivation; Lean statement checked on branch `formal/w2-m0`; review pending (`formal/lean/StatementCards.md`). |
| EF-02 | GIG $a$ roundtrip | Same GIG conversion functions | A | High | Real arithmetic | `formal/lean/Normix/Distributions/GigParameters.lean` | statement-drafted | Review coordinate order; Lean statement checked on branch `formal/w2-m0`; review pending (`formal/lean/StatementCards.md`). |
| EF-03 | GIG $p$ roundtrip | Same GIG conversion functions | A | High | Real arithmetic | `formal/lean/Normix/Distributions/GigParameters.lean` | statement-drafted | Review statistic convention; Lean statement checked on branch `formal/w2-m0`; review pending (`formal/lean/StatementCards.md`). |
| EF-04 | Gamma classical/natural parameter roundtrip | `normix/distributions/gamma.py`: parameter conversions | A | High | Positive shape and rate | `formal/lean/Normix/Distributions/GammaParameters.lean` | statement-drafted | Separate constructor clamps; Lean statement checked on branch `formal/w2-m0`; review pending (`formal/lean/StatementCards.md`). |
| EF-05 | Named conversion between GIG and mixture statistic order | `normix/fitting/eta.py`: `NormalMixtureEta`; GIG statistics | A | High | Named field definitions | `formal/lean/Normix/Foundations/StatisticCoordinates.lean` | statement-drafted | $(\log Y,1/Y,Y)$ versus $(1/Y,Y,\log Y,\ldots)$; Lean statement checked on branch `formal/w2-m0`; review pending (`formal/lean/StatementCards.md`). Y-part only. |
| EF-06 | Parameter-domain preservation for ideal GIG conversion | GIG `natural_params`, `from_natural` | A | High | EF-01–EF-03 | `formal/lean/Normix/Distributions/GigParameters.lean` | statement-drafted | State $a,b>0$; clamps need their own correspondence; Lean statement checked on branch `formal/w2-m0`; review pending (`formal/lean/StatementCards.md`). |
| EF-07 | Normalize finite positive exponential weights | `normix/utils/bessel.py`: `_moments_from_weights` | A | Very high | Nonempty finite index set, positive weights | `formal/lean/Normix/Numerics/FiniteMoments.lean` | statement-drafted | Positive denominator and total mass $1$; Lean statement checked on branch `formal/w2-m0`; review pending (`formal/lean/StatementCards.md`). |
| EF-08 | Gradient of finite log-partition equals weighted statistic | `_moments_from_weights`, `log_kv_moments` | A–B | Very high | EF-07; finite differentiation | `formal/lean/Normix/Numerics/FiniteMoments.lean` | statement-drafted | Nodes and weights fixed; Lean statement checked on branch `formal/w2-m0`; review pending (`formal/lean/StatementCards.md`). One coordinate only. |
| EF-09 | Centered covariance is a positive-semidefinite Gram matrix | `_moments_from_weights`, `BesselMoments.cov` | A | Very high | EF-07; centered statistics | `formal/lean/Normix/Numerics/FiniteMoments.lean` | statement-drafted | Equality with the modeled matrix is required; Lean statement checked on branch `formal/w2-m0`; review pending (`formal/lean/StatementCards.md`). |
| EF-10 | Hessian of finite log-partition equals covariance | `_moments_from_weights`, `log_kv_moments` | A–B | Very high | EF-08 | `formal/lean/Normix/Numerics/FiniteMoments.lean` | not-started | Strict positivity needs no degenerate direction |
| EF-11 | Convexity of continuous log-partition | `normix/exponential_family.py`; `docs/design/exponential_family.md` | C | Very high | EF definitions, finite partition domain, Hölder | `formal/lean/Normix/ExponentialFamily/Convexity.lean` | not-started | No Hessian existence needed |
| EF-12 | Continuous gradient equals expected sufficient statistic | `ExponentialFamily.expectation_params` | C | Very high | Local integrable domination and differentiation | `formal/lean/Normix/ExponentialFamily/Derivatives.lean` | not-started | Support and regularity explicit |
| EF-13 | Continuous Hessian equals statistic covariance | `ExponentialFamily` log-partition triad | C | Very high | EF-12; second derivative domination | `formal/lean/Normix/ExponentialFamily/Derivatives.lean` | not-started | Moment existence explicit |
| EF-14 | Score-based Fisher information equals Hessian | `ExponentialFamily.fisher_information` | C | Very high | EF-12, EF-13; fixed support | `formal/lean/Normix/ExponentialFamily/Fisher.lean` | not-started | Do not define Fisher as the desired Hessian |
| EF-15 | Legendre equality on the gradient image | `ExponentialFamily`; `docs/design/exponential_family.md` | B–C | High | EF-11, EF-12; convex dual definition | `formal/lean/Normix/ExponentialFamily/Duality.lean` | not-started | No claim of surjectivity onto the full moment interior |
| EF-16 | Local inverse derivative solves $H\,d\theta=d\eta$ | `ExponentialFamily._implicit_natural_params` | B–C | High | EF-13; interior exact solution, invertible Hessian | `formal/lean/Normix/ExponentialFamily/LocalInverse.lean` | not-started | Excludes residual and failure branches |
| EF-17 | Identify redundant symmetric-matrix coordinates | `normix/distributions/normal.py`; `normix/mixtures/joint.py` | A–B | High | Symmetric matrix/flat-array correspondence | `formal/lean/Normix/Foundations/SymmetricCoordinates.lean` | not-started | Full flattened Fisher need not be positive definite |
| EF-18 | Ambient moment contract for a restricted subfamily | `JointNormalMixture._ambient_moment_contract` | B–C | High | EF-13; explicit embedding and moment domain | `formal/lean/Normix/ExponentialFamily/RestrictedFamilies.lean` | not-started | Joint VG/NInvG/NIG share GH statistics |
| EF-19 | Intrinsic Fisher is $J^\top\operatorname{Cov}(t)J$ | `_ambient_moment_contract`; restricted parameter maps | B–C | High | EF-14, EF-18; smooth interior map | `formal/lean/Normix/ExponentialFamily/RestrictedFamilies.lean` | not-started | Boundary moments require separate checks |
| EF-20 | `bregman_objective` differs from a Bregman divergence by a constant | `normix/fitting/solvers.py`: `bregman_objective` | B | High | EF-11, EF-12; $\eta=\nabla\psi(\theta_0)$ | `formal/lean/Normix/ExponentialFamily/Duality.lean` | not-started | Objective itself need not be nonnegative |
| EF-21 | Error of approximate expectation inversion on a specified domain | `_implicit_natural_params`; `normix/fitting/solvers.py` | C–D | High | EF-16; residual and conditioning bounds | `formal/lean/Normix/Numerics/InverseError.lean` | not-started | Domain and error metric to verify |
| EF-22 | Derivative error of approximate expectation inversion | `_implicit_natural_params` | C–D | High | EF-16, EF-21; Hessian perturbation control | `formal/lean/Normix/Numerics/InverseError.lean` | not-started | Separate from value error |

The intended continuous results are:

- Convexity of $\psi$ along segments in the natural-parameter domain follows from Hölder's inequality without first constructing a Hessian.
- Under local conditions allowing differentiation under the integral, $\nabla\psi=E_\theta[t]$ and $\nabla^2\psi=\operatorname{Cov}_\theta(t)$.
- Define the score by $s_\theta(x)=\nabla_\theta\log p_\theta(x)$. With fixed support and suitable regularity, $I(\theta)=E[s_\theta s_\theta^\top]=\nabla^2\psi(\theta)$. Defining Fisher as the Hessian would avoid this proof obligation.
- Define $\psi^*(\eta)=\sup_{\theta\in\Theta}\{\langle\theta,\eta\rangle-\psi(\theta)\}$. First prove dual equalities and inverse relations on $\eta\in\nabla\psi(\Theta)$. Equality of that image with the whole moment-space interior needs further assumptions.

These are **derivation** targets based on standard theory and recalled *Graphical Models, Exponential Families, and Variational Inference* (2008). The original and a complete Mathlib exponential-family framework were not checked.

Three normix-specific issues affect the difficulty:

**Minimality.** A representation is minimal when its statistics have no nontrivial almost-everywhere affine relation. This usually controls strict convexity and identifiability. `MultivariateNormal` and joint mixtures flatten every entry of $xx^\top$, including duplicate symmetric entries. The Fisher matrix need not be positive definite on the whole flat coordinate space. Use symmetric matrices or an explicit identifiable subspace for inverses. [N2, N4; local check and derivation]

**Restricted subfamilies.** `_ambient_moment_contract` returns moments and covariance of common GH statistics for joint VG/NInvG/NIG. The gradient of a restricted log-partition is not generally that ambient expectation vector. For a smooth interior map $\theta=\theta(u)$ with Jacobian $J$, intrinsic Fisher information is $J^\top\operatorname{Cov}(t)J$. Boundary cases need separate moment-existence checks. [N4, N7; local check and derivation]

**Numerical versus mathematical inversion.** `_implicit_natural_params` uses $H(\theta)d\theta=d\eta$, where $H=\nabla^2\psi$. The local inverse theorem needs an exact solution, an interior point, and invertible Hessian. Actual solves use residual tolerances and may reach boundaries or failure branches. Value and derivative errors are separate targets. A Bregman divergence is the gap between a convex function and its tangent plane. `bregman_objective` computes $\psi(\theta)-\theta^\top\eta$. For $\eta=\nabla\psi(\theta_0)$ it differs from $D_\psi(\theta,\theta_0)$ by a term independent of $\theta$; the objective itself need not be nonnegative. [N2, N6; local check and derivation]

## 2. Densities, moments, and mixtures

Let $V>0$ be the mixing variable, usually named `Y` in source. Let $\mu,\gamma\in\mathbb R^d$, let $\Sigma$ be positive definite, and let $Z\sim N(0,\Sigma)$ be independent of $V$. Define

$$X=\mu+\gamma V+\sqrt V\,Z.$$

| ID | Item | normix function/page | Difficulty | Value | Dependencies | Lean module (planned) | Status | Notes/PR |
|---|---|---|---|---|---|---|---|---|
| DM-01 | Gamma density normalization | `normix/distributions/gamma.py`: `_log_partition_from_theta` | B–C | High | Gamma integral; positive shape/rate | `formal/lean/Normix/Distributions/Gamma.lean` | not-started | Rate convention |
| DM-02 | Gamma real-order moment formula and existence domain | `Gamma.raw_moment` | B–C | High | DM-01; Gamma integral with shifted shape | `formal/lean/Normix/Distributions/GammaMoments.lean` | not-started | $r>-\alpha$ |
| DM-03 | Reciprocal Gamma transformation gives InverseGamma | `normix/distributions/inverse_gamma.py` | B–C | High | DM-01; reciprocal Jacobian | `formal/lean/Normix/Distributions/InverseGamma.lean` | not-started | Positive support |
| DM-04 | InverseGamma density normalization | `InverseGamma` density | B–C | High | DM-03 | `formal/lean/Normix/Distributions/InverseGamma.lean` | not-started | Exact density, not a numerical limit |
| DM-05 | InverseGamma moment formula and existence domain | `InverseGamma.raw_moments`, `mean`, `var` | B–C | High | DM-02, DM-03 | `formal/lean/Normix/Distributions/InverseGammaMoments.lean` | not-started | $r<\alpha$ |
| DM-06 | InverseGaussian density normalization | `normix/distributions/inverse_gaussian.py` | C | High | Gaussian integral or equivalent special-function route | `formal/lean/Normix/Distributions/InverseGaussian.lean` | not-started | Distinct from InverseGamma |
| DM-07 | IG mean equals $m$ | `InverseGaussian.mean` | C | High | DM-06; first-moment integrability | `formal/lean/Normix/Distributions/InverseGaussianMoments.lean` | not-started | $m,\lambda>0$ |
| DM-08 | IG variance equals $m^3/\lambda$ | `InverseGaussian.var` | C | High | DM-07; second moment | `formal/lean/Normix/Distributions/InverseGaussianMoments.lean` | not-started | Finite variance proof |
| DM-09 | IG reciprocal mean equals $1/m+1/\lambda$ | `InverseGaussian.from_expectation` | C | High | DM-06; inverse-moment integral | `formal/lean/Normix/Distributions/InverseGaussianMoments.lean` | not-started | Used by the IG M-step |
| DM-10 | GIG kernel mass equals its Bessel normalizer | `GeneralizedInverseGaussian.natural_params`; `docs/theory/gig.md` | C | Very high | BES-01, BES-02; log substitution | `formal/lean/Normix/Distributions/GIG.lean` | not-started | $p\in\mathbb R$, $a,b>0$ |
| DM-11 | Every real-order moment is integrable in the GIG interior | `GeneralizedInverseGaussian.raw_moment` | C | Very high | BES-01; shifted-order kernel | `formal/lean/Normix/Distributions/GIGMoments.lean` | not-started | Prove before taking normalization ratios |
| DM-12 | GIG real-order moment formula | `raw_moment`, `_eta_H_from_moments` | C | Very high | DM-10, DM-11 | `formal/lean/Normix/Distributions/GIGMoments.lean` | not-started | Boundary distributions separate |
| DM-13 | Normal mixture density integrates to one | `normix/mixtures/joint.py`, `marginal.py` | C | Very high | Normalized mixing law; conditional Gaussian; nonnegative integral exchange | `formal/lean/Normix/Mixtures/Measure.lean` | not-started | General mixing law |
| DM-14 | GH closed density equals the mixture integral | `GeneralizedHyperbolic.log_prob`; `docs/theory/gh.md` | C | Very high | DM-10, DM-13; positive-definite matrix algebra | `formal/lean/Normix/Mixtures/GHDensity.lean` | not-started | Exact formula before numerical floors |
| DM-15 | Conditional mixing law is GIG | `JointNormalMixture._posterior_gig_params` | C | Very high | DM-10, DM-14; positive marginal density | `formal/lean/Normix/Mixtures/Posterior.lean` | not-started | `_floored_posterior_gig_params` changes parameters |
| DM-16 | Mixture mean identity | `NormalMixture.mean` | C | Very high | Independent $V,Z$; required integrability | `formal/lean/Normix/Mixtures/Moments.lean` | not-started | Begin with sufficient second-moment assumptions |
| DM-17 | Mixture covariance identity | `NormalMixture.cov` | C | Very high | DM-16; $E[V^2]<\infty$ | `formal/lean/Normix/Mixtures/Moments.lean` | not-started | Conditional covariance and total covariance |
| DM-18 | Weaker moment assumptions when $\gamma=0$ | Mixture `mean`, `cov` | C | High | DM-16, DM-17; explicit integrability | `formal/lean/Normix/Mixtures/SymmetricMoments.lean` | not-started | Symmetry alone does not establish a mean |
| DM-19 | Specialized NIG density and IG/GIG parameter identity | `NormalInverseGaussian.log_prob` | B–C | High | BES-09; DM-06, DM-14 | `formal/lean/Normix/Mixtures/NIGDensity.lean` | not-started | $p=-1/2$, $a=\lambda/m^2$, $b=\lambda$ |
| DM-20 | Exact Gamma-mixture VG density | `VarianceGamma.log_prob` | B–C | High | DM-01, DM-13; direct mixture integral | `formal/lean/Normix/Mixtures/VGDensity.lean` | not-started | $p=\alpha$, $a=2\beta$, $b=0$ |
| DM-21 | VG behavior at $x=\mu$ | `VarianceGamma.log_prob` | B–C | High | DM-20; parameter-dependent local integrability | `formal/lean/Normix/Mixtures/VGBoundary.lean` | not-started | Separate pointwise singularity from the law |
| DM-22 | Exact InverseGamma-mixture NInvG density | `NormalInverseGamma.log_prob` | B–C | High | DM-04, DM-13; direct mixture integral | `formal/lean/Normix/Mixtures/NInvGDensity.lean` | not-started | $p=-\alpha$, $a=0$, $b=2\beta$ |
| DM-23 | Student-type boundary at $\gamma=0$ | `NormalInverseGamma.log_prob` | B–C | High | DM-22 | `formal/lean/Normix/Mixtures/NInvGBoundary.lean` | not-started | Exact specialization |
| DM-24 | NInvG moment divergence domains | NInvG and mixture moment methods | B–C | High | DM-05, DM-16–DM-18 | `formal/lean/Normix/Mixtures/NInvGMoments.lean` | not-started | API `inf` can mean unavailable, not signed expectation $+\infty$ |
| DM-25 | Ideal rescaling preserves the marginal law of $X$ | `JointNormalMixture._rescale` | B–C | High | Mixture map and transformed mixing law | `formal/lean/Normix/Mixtures/Rescaling.lean` | not-started | Positive scale; clamps inactive |
| DM-26 | Joint-density Jacobian under $V\mapsto sV$ | `_rescale` | B–C | High | DM-25; change of variables | `formal/lean/Normix/Mixtures/Rescaling.lean` | not-started | Joint density is not unchanged in the same coordinates |
| DM-27 | Conditions under which clamped rescaling equals ideal rescaling | GH `_build_rescaled` | A–B | High | DM-25; explicit clamp bounds | `formal/lean/Normix/Bridge/Rescaling.lean` | not-started | Source correspondence to verify |
| DM-28 | A specified GIG-to-boundary weak limit | `docs/theory/gig.md`, `gh.md` | C–D | High | DM-10; selected parameter path and boundary law | `formal/lean/Normix/Distributions/GIGLimits.lean` | not-started | Choose one boundary and path per proposition card |
| DM-29 | Moment convergence along that boundary path | Same theory pages and moment methods | C–D | High | DM-28; additional moment control | `formal/lean/Normix/Distributions/GIGLimits.lean` | not-started | Weak convergence alone is insufficient |

The exact density and moment targets are as follows. All densities below have support $(0,\infty)$. [N3; local check for source correspondence; derivation for formulas]

- Gamma has shape $\alpha>0$ and rate $\beta>0$:
  $f(v)=\beta^\alpha v^{\alpha-1}e^{-\beta v}/\Gamma(\alpha)$ and
  $E[V^r]=\Gamma(\alpha+r)/(\Gamma(\alpha)\beta^r)$ for $r>-\alpha$.
- InverseGamma has $\alpha,\beta>0$:
  $f(v)=\beta^\alpha v^{-\alpha-1}e^{-\beta/v}/\Gamma(\alpha)$ and
  $E[V^r]=\beta^r\Gamma(\alpha-r)/\Gamma(\alpha)$ for $r<\alpha$.
- InverseGaussian, or IG, has mean parameter $m>0$ and shape $\lambda>0$:
  $f(v)=\sqrt{\lambda/(2\pi v^3)}\exp[-\lambda(v-m)^2/(2m^2v)]$.
  Its targets are $E[V]=m$, $\operatorname{Var}(V)=m^3/\lambda$, and $E[1/V]=1/m+1/\lambda$.
- GIG has $p\in\mathbb R$ and $a,b>0$. Prove its normalizer from the Bessel integral, then all real-order moments.
- GH uses $V\sim\mathrm{GIG}(p,a,b)$. NIG uses IG mixing and the exact GIG parameters $p=-1/2$, $a=\lambda/m^2$, $b=\lambda$.
- VG uses Gamma mixing at the exact boundary $p=\alpha$, $a=2\beta$, $b=0$. Its density needs separate treatment at $x=\mu$.
- NInvG uses InverseGamma mixing at $p=-\alpha$, $a=0$, $b=2\beta$. Treat its Student-type $\gamma=0$ boundary and moment divergence explicitly.

IG and InverseGamma are different distributions. Lean can share mathematical lemmas while the runtime retains each specialized formula. Mathematical reuse does not require routing all code through GH.

The core GIG integral is

$$
\int_0^\infty v^{p-1}e^{-(av+b/v)/2}\,dv
=2(b/a)^{p/2}K_p(\sqrt{ab}).
$$

Substitute $v=\sqrt{b/a}\,e^u$. Replacing $p$ by $p+r$ and taking a normalization ratio then gives

$$E[V^r]=(b/a)^{r/2}\frac{K_{p+r}(\sqrt{ab})}{K_p(\sqrt{ab})}.$$

First establish integrability for every required order. The boundaries $a=0$ and $b=0$ are outside this formula's direct domain. Boundary densities, weak convergence, and moment convergence are distinct targets. Weak convergence controls expectations of bounded continuous functions; it does not imply convergence of higher moments.

For GH, set

$$
q(x)=(x-\mu)^\top\Sigma^{-1}(x-\mu),\quad
A=a+\gamma^\top\Sigma^{-1}\gamma,\quad B=b+q(x),\quad r=p-d/2.
$$

Expanding the conditional Gaussian quadratic form and using the same GIG integral gives

$$
f_X(x)=\frac{(a/b)^{p/2}e^{(x-\mu)^\top\Sigma^{-1}\gamma}}
{(2\pi)^{d/2}|\Sigma|^{1/2}K_p(\sqrt{ab})}
(B/A)^{r/2}K_r(\sqrt{AB}).
$$

The same calculation gives $V\mid X=x\sim\mathrm{GIG}(r,A,B)$, matching `_posterior_gig_params`. `_floored_posterior_gig_params` changes those parameters and cannot directly use the exact conjugacy theorem. [N4; local check and derivation]

Normalization can first follow from the mixture definition. Each conditional Gaussian integrates to $1$; exchange nonnegative integrals. Then prove that the closed formula equals that mixture integral. This separates existence of a probability law from correctness of the code's explicit formula.

When the needed moments exist, conditional expectation yields

$$E[X]=\mu+\gamma E[V],\qquad
\operatorname{Cov}(X)=E[V]\Sigma+\operatorname{Var}(V)\gamma\gamma^\top.$$

A first common proof can assume $E[V^2]<\infty$. Weaker conditions, including $\gamma=0$, deserve separate statements. Symmetric heavy tails do not establish a zero mean through cancellation. A Cauchy mean does not exist. An API `inf` indicating an unavailable moment is not a claim that its signed expectation is $+\infty$. [N4; local check and derivation]

## 3. CDFs

The cumulative distribution function is $F(x)=P(X\le x)$. For a real distribution with density $f$, it is $F(x)=\int_{-\infty}^x f(t)\,dt$. Establish its basic probability properties before connecting analytical formulas or numerical tables.

| ID | Item | normix function/page | Difficulty | Value | Dependencies | Lean module (planned) | Status | Notes/PR |
|---|---|---|---|---|---|---|---|---|
| CDF-01 | $0\le F(x)\le1$ | Distribution and univariate-mixture `cdf` specifications | C | High | Probability measure definition | `formal/lean/Normix/Foundations/CDF.lean` | not-started | Exact CDF |
| CDF-02 | CDF monotonicity | Same CDF specifications | C | High | Monotonicity of measure | `formal/lean/Normix/Foundations/CDF.lean` | not-started | All real thresholds |
| CDF-03 | CDF right continuity | Same CDF specifications | C | High | Continuity of finite measures | `formal/lean/Normix/Foundations/CDF.lean` | not-started | Do not assume a continuous density |
| CDF-04 | Limits at negative and positive infinity | Same CDF specifications | C | High | Probability mass and monotone limits | `formal/lean/Normix/Foundations/CDF.lean` | not-started | Limits $0$ and $1$ |
| CDF-05 | $F'=f$ at continuity points of the density | Distribution `cdf`, `pdf` | C | High | Density representation; local continuity | `formal/lean/Normix/Foundations/CDFDerivative.lean` | not-started | Derivative existence explicit |
| CDF-06 | Gamma CDF equals the regularized incomplete Gamma expression | `Gamma.cdf` | B–C | High | DM-01; incomplete Gamma definition | `formal/lean/Normix/Distributions/GammaCDF.lean` | not-started | Pinned Mathlib support to verify |
| CDF-07 | IG CDF equals its Gaussian-CDF formula | `InverseGaussian.cdf` | C | High | DM-06; Gaussian CDF, endpoints and tails | `formal/lean/Normix/Distributions/InverseGaussianCDF.lean` | not-started | `log_ndtr` rounding is separate |
| CDF-08 | Normalized cumulative table masses are monotone | `normix/utils/rvs.py`: `build_pinv_table` | A | High | Increasing grid; nonnegative masses; positive total | `formal/lean/Normix/Numerics/CDFTable.lean` | not-started | Finite data property |
| CDF-09 | Linear CDF interpolation remains in $[0,1]$ | `GIG.cdf`; `_UnivariateNormalMixtureMixin.cdf` | A | High | CDF-08; interpolation interval conditions | `formal/lean/Normix/Numerics/CDFTable.lean` | not-started | Endpoints explicit |
| CDF-10 | Tail probability bound for a selected table window | `build_pinv_table`: `tail_eps` | C–D | High | Distribution-specific integral tail bound | `formal/lean/Normix/Numerics/CDFError.lean` | not-started | A density-height threshold is not a probability bound |
| CDF-11 | Uniform CDF approximation bound | `build_pinv_table`; table-based `cdf` | D | High | CDF-10; trapezoid, normalization, interpolation error | `formal/lean/Normix/Numerics/CDFError.lean` | not-started | Specify $\sup_x\lvert\widehat F(x)-F(x)\rvert\le\varepsilon$ |

The local implementations have three forms. `Gamma.cdf` uses regularized incomplete Gamma. `InverseGaussian.cdf` has an analytical Gaussian-CDF formula; `log_ndtr` is a stable evaluation path, not a rounding proof. `GIG.cdf` and `_UnivariateNormalMixtureMixin.cdf` use numerical tables. `build_pinv_table` truncates to a finite interval, applies trapezoidal integration, normalizes, and linearly interpolates. [N3, N5; local check]

PINV abbreviates polynomial-interpolation numerical inversion. The name alone does not justify using error theorems for a particular higher-order inversion algorithm.

A multivariate CDF is $P(X_1\le x_1,\ldots,X_d\le x_d)$. It is not the CDF of a one-dimensional projection. Initial mixture CDF targets concern `Univariate*` classes. A mathematical definition does not imply that all multivariate classes implement it.

The finite-table proofs need an increasing grid, nonnegative masses, and positive total mass. Accuracy needs the separate bound in CDF-11, where $\widehat F$ is the table CDF and $\varepsilon$ is a derived error bound. `tail_eps` is a density-height threshold. A relative density drop does not imply that the tail probability is below the same number. Bessel `log_drop` has the same distinction. Tail errors need integral inequalities. [derivation]

## 4. Sampling

“Correct random generation” has three different meanings: a correct distributional transformation, a correct ideal sampling algorithm, and a claim about a real pseudorandom number generator. Keep them separate.

| ID | Item | normix function/page | Difficulty | Value | Dependencies | Lean module (planned) | Status | Notes/PR |
|---|---|---|---|---|---|---|---|---|
| SAMP-01 | Measurability of the generalized quantile | Distribution inverse-sampling specification | C | High | CDF-02–CDF-04 | `formal/lean/Normix/Foundations/Quantile.lean` | not-started | Endpoint null events explicit |
| SAMP-02 | Ideal uniform input maps to the target law under the quantile | Quantile-based `rvs` specification | C | High | SAMP-01; generalized inverse property | `formal/lean/Normix/Foundations/Quantile.lean` | not-started | Discontinuous CDFs require care |
| SAMP-03 | Ideal independent-input mixture map has the target law | `JointNormalMixture.rvs` | C | Very high | DM-13; measurable map and product measure | `formal/lean/Normix/Mixtures/Sampling.lean` | not-started | May assume correct $V,Z$ directly |
| SAMP-04 | IG normal-square/uniform transformation has the IG density | `InverseGaussian.rvs` | C–D | Medium–high | DM-06; transformed densities and independent inputs | `formal/lean/Normix/Distributions/InverseGaussianSampling.lean` | not-started | Derive the two-choice transformation |
| SAMP-05 | TDR envelope dominates the GIG target everywhere | `normix/utils/rvs.py`: `_gig_rvs_tdr` | C–D | Medium–high | GIG log-density; proposal construction | `formal/lean/Normix/Numerics/GIGEnvelope.lean` | not-started | Finite-precision envelope is an additional obligation |
| SAMP-06 | An accepted ideal TDR proposal has the target law | `_gig_rvs_tdr` | C–D | Medium–high | SAMP-05; proposal and acceptance probabilities | `formal/lean/Normix/Numerics/GIGRejection.lean` | not-started | Repeated proposals independent |
| SAMP-07 | Infinite ideal TDR retries terminate almost surely | `_gig_rvs_tdr` ideal model | C–D | Medium–high | SAMP-06; positive acceptance probability | `formal/lean/Normix/Numerics/GIGRejection.lean` | not-started | Nontermination has probability zero |
| SAMP-08 | Finite retry failure bound | `_gig_rvs_tdr` maximum rounds | C | Medium–high | Independent proposals; acceptance at least $a_0>0$ | `formal/lean/Normix/Numerics/GIGRejection.lean` | not-started | At most $(1-a_0)^R$; union bound for multiple columns |
| SAMP-09 | Distribution conditional on success with bounded retries | `_gig_rvs_tdr` NaN failure result | C–D | Medium–high | SAMP-06, SAMP-08; explicit failure result | `formal/lean/Normix/Numerics/GIGRejection.lean` | not-started | Do not erase exhausted-round failures |
| SAMP-10 | Identify the law induced by a quantile table | `QuantileTable.rvs` | C | High | Table map; ideal uniform input | `formal/lean/Normix/Numerics/QuantileSampling.lean` | not-started | It is an approximation to the original law |
| SAMP-11 | Quantile-table distribution error under a selected metric | `QuantileTable.rvs` | C–D | High | SAMP-10; CDF-11 or other explicit table bound | `formal/lean/Normix/Numerics/QuantileSampling.lean` | not-started | Specify metric and quantization domain |
| SAMP-12 | State the independence assumptions linking seeds to ideal inputs | `JointNormalMixture.rvs`: `seed`, `seed + 1` | C–D | High | Explicit PRNG or ideal random model | `formal/lean/Normix/Bridge/RandomInputs.lean` | not-started | Distinct seeds or `split` alone do not prove independence |

For the first meaning, view random variables as measurable functions on a sample space. Measurability lets event preimages receive probabilities. The output law obtained by a measurable map is its **pushforward measure**.

Let $U\sim\mathrm{Uniform}(0,1)$ and $G\sim N(0,I_d)$ be independent. Let $Q_V(u)=\inf\{v:F_V(v)\ge u\}$ be a generalized quantile. For $L_\Sigma L_\Sigma^\top=\Sigma$, define

$$T(u,g)=\mu+\gamma Q_V(u)+\sqrt{Q_V(u)}L_\Sigma g.$$

Proving that the product input measure pushes forward to GH or another target law proves the ideal mixture transformation. It needs measurability and endpoint null-event handling. Starting directly with independent $V,Z$ of the correct laws avoids proving a Gaussian sampling algorithm at the outset. [derivation]

For the second meaning, concrete ideal algorithms have further obligations:

- `InverseGaussian.rvs` uses a squared normal, a uniform variable, and a two-choice transformation. Derive that transformation's density.
- `_gig_rvs_tdr` uses transformed density rejection, or TDR. Prove global envelope domination, correct acceptance probabilities, and independence of repeated proposals. Infinite retries can then terminate almost surely.
- Actual `_gig_rvs_tdr` has a maximum round count and returns NaN on exhaustion. Permit failure in the specification and prove the law conditional on success. If each independent proposal accepts with probability at least $a_0>0$, failure after $R$ rounds is at most $(1-a_0)^R$. Multiple columns permit a union bound. The recorded claim $a_0\ge e^{-1}$ is **to verify**, including its relation to finite-precision envelope construction.
- `QuantileTable.rvs` produces the table-induced law. Even ideal uniform inputs do not make it exactly GIG. [N5; local check]

For the third meaning, a **PRNG** deterministically maps finite seeds to bits. Its finite-seed, floating-point output law is discrete and cannot equal a continuous Gaussian law. Implementation behavior, statistical properties, and finite-precision output distributions require different specifications. Total variation distance between discrete output and a continuous target can be $1$; a small total variation bound cannot be assumed. [derivation]

`JointNormalMixture.rvs` samples the mixing variable with `seed` and the Gaussian with `seed + 1`. Distinct seeds do not prove independence. Replacing them with `split` would still require an ideal random model or a PRNG theorem. jax-lean's random model covers finite uniform integer sampling and explicitly excludes proof of real PRNG independence and uniformity. [N4, J4; local check]

The first deliverable should say “ideal sampling transformation proved,” conditional on correct independent input laws. It must not say that all of `rvs` is verified.

## 5. EM

EM is expectation-maximization. For fixed observations $x_1,\ldots,x_n$, let $p_\theta(x,v)$ be a joint density. Define its marginal $m_\theta(x)=\int p_\theta(x,v)\,dv$ and posterior density $q_\theta(v\mid x)=p_\theta(x,v)/m_\theta(x)$. The average log-likelihood and auxiliary objective are

$$
\ell(\theta)=\frac1n\sum_{i=1}^n\log m_\theta(x_i),\qquad
Q(\theta'\mid\theta)=\frac1n\sum_{i=1}^n
\int q_\theta(v\mid x_i)\log p_{\theta'}(x_i,v)\,dv.
$$

Require $n>0$, positive finite relevant marginal densities, normalized posteriors, and existence of the logarithmic integrals. General cases with zero densities also require absolute-continuity conditions: the new posterior must be defined where the old posterior has mass.

| ID | Item | normix function/page | Difficulty | Value | Dependencies | Lean module (planned) | Status | Notes/PR |
|---|---|---|---|---|---|---|---|---|
| EM-01 | Normalize the posterior defined by a density ratio | `docs/theory/em_algorithm.md`; mixture E-step | C | Very high | Joint density, positive finite marginal | `formal/lean/Normix/EM/Posterior.lean` | not-started | Avoid general conditional-measure existence initially |
| EM-02 | Nonnegativity of the required KL divergence | `docs/theory/em_algorithm.md` | C | Very high | Normalized densities, log integrability, Jensen | `formal/lean/Normix/EM/KLDivergence.lean` | not-started | Absolute continuity explicit |
| EM-03 | Likelihood/Q/KL decomposition | `BatchEMFitter._em_step` mathematical specification | C | Very high | EM-01, EM-02; logarithmic integrals | `formal/lean/Normix/EM/Decomposition.lean` | not-started | Derive from joint/posterior/marginal identity |
| EM-04 | A Q-improving step does not decrease likelihood | `_em_step`; `docs/theory/em_algorithm.md` | C | Very high | EM-03 | `formal/lean/Normix/EM/Monotonicity.lean` | not-started | Exact global Q maximization unnecessary |
| EM-05 | Normal-block candidate satisfies the normal equations | `JointNormalMixture._mstep_normal_params` | A–B | Very high | Named statistics; $s_1s_2>1$ | `formal/lean/Normix/EM/NormalMStep.lean` | not-started | Ideal formulas before floors and `SIGMA_REG` |
| EM-06 | Normal-block candidate maximizes Q | `_mstep_normal_params` | B–C | Very high | EM-05; positive-definite candidate covariance | `formal/lean/Normix/EM/NormalMStep.lean` | not-started | Stationary equations alone are insufficient |
| EM-07 | IG closed M-step is optimal on its stated domain | `InverseGaussian.from_expectation` | A–C | Very high | DM-07, DM-09; nondegenerate moments | `formal/lean/Normix/EM/InverseGaussianMStep.lean` | not-started | Positive reciprocal denominator |
| EM-08 | Exact Gamma shape root satisfies the M-step specification | `Gamma.from_expectation`, `_newton_digamma` | B–C | High | DM-01, DM-02; digamma/root analysis | `formal/lean/Normix/EM/GammaMStep.lean` | not-started | Numerical convergence is separate; not closed form |
| EM-09 | Exact InverseGamma shape root satisfies the M-step specification | `InverseGamma.from_expectation` | B–C | High | DM-04, DM-05; shape-root analysis | `formal/lean/Normix/EM/InverseGammaMStep.lean` | not-started | Not closed form |
| EM-10 | Exact GIG expectation inversion satisfies the M-step specification | GIG `from_expectation`; `normix/fitting/solvers.py` | C | Very high | DM-12; EF convexity and invertible domain | `formal/lean/Normix/EM/GIGMStep.lean` | not-started | Three-dimensional inversion; approximate solver separate |
| EM-11 | Bounded monotone likelihood values have a limit | `docs/theory/em_algorithm.md` | B–C | Medium | EM-04; upper bound | `formal/lean/Normix/EM/LikelihoodLimit.lean` | not-started | Does not prove parameter convergence |
| EM-12 | Nonstationary points strictly improve under a specified exact update | Exact M-step/Q derivative relation | C–D | Medium | EM-04; differentiability and exact update assumptions | `formal/lean/Normix/EM/StrictImprovement.lean` | not-started | State constrained versus interior stationarity |
| EM-13 | Accumulation points are stationary for a continuous single-valued update | `docs/theory/em_algorithm.md` | C–D | Medium | EM-11, EM-12; compact invariant subset of the open domain | `formal/lean/Normix/EM/Stationarity.lean` | not-started | Multivalued or discontinuous updates need more theory |
| EM-14 | Penalized EM increases $\ell-P$ | `docs/theory/shrinkage.md`; penalized fitting | C | Very high | EM-03; fixed penalty and penalized Q improvement | `formal/lean/Normix/EM/Penalized.lean` | not-started | Arbitrary field shrinkage need not share one penalty |
| EM-15 | A full MCECM cycle preserves the appropriate improvement property | `BatchEMFitter._mcecm_step` | C | High | EM-04; correct conditional objective at every cycle | `formal/lean/Normix/EM/MCECM.lean` | not-started | Multi-Cycle ECM |
| EM-16 | Bound likelihood loss for a specified approximate E/M-step | `_em_step`; approximate GIG solve and posterior floors | C–D | High | EM-03; derived approximation bounds | `formal/lean/Normix/Numerics/EMError.lean` | not-started | Derive $\varepsilon_k$; a test tolerance is not a bound |
| EM-17 | Map the actual batch step to the exact or bounded-error theorem | `BatchEMFitter`: floors, fallback, rescaling, stopping | C–D | Very high | EM-04 or EM-16; DM-27; branch specifications | `formal/lean/Normix/Bridge/BatchEM.lean` | not-started | Record exclusions; Aitken stopping does not prove stationarity |

The KL divergence is $D_{\mathrm{KL}}(q\|r)=\int q\log(q/r)$. Jensen's inequality, which compares a convex function of an average with its average value, yields nonnegativity under the required hypotheses. Then

$$
\ell(\theta')-\ell(\theta)
=Q(\theta'\mid\theta)-Q(\theta\mid\theta)
+\frac1n\sum_iD_{\mathrm{KL}}\bigl(q_\theta(\cdot\mid x_i)\|q_{\theta'}(\cdot\mid x_i)\bigr).
$$

Thus $Q(\theta'\mid\theta)\ge Q(\theta\mid\theta)$ implies nondecreasing likelihood. The new parameter need not globally maximize Q. This is the basic generalized EM condition. The decomposition follows by substituting $\log p=\log q+\log m$. Its literature background is *Maximum Likelihood from Incomplete Data via the EM Algorithm* (1977), recalled rather than reread. [derivation]

The first Lean model can define posteriors by density ratios. This avoids starting with conditional-distribution existence on arbitrary measure spaces. Assuming one-step likelihood improvement and proving it over a loop would not establish EM theory itself.

For the normal M-step, use `NormalMixtureEta` fields. Write $s_1=E[1/V]$, $s_2=E[V]$, $s_4=E[X]$, $s_5=E[X/V]$, and $s_6=E[XX^\top/V]$. For nondegenerate inputs, $s_1s_2>1$. The candidate is

$$
\mu_* = \frac{s_4-s_2s_5}{1-s_1s_2},\qquad
\gamma_* = \frac{s_5-s_1s_4}{1-s_1s_2},
$$

$$
\Sigma_*=s_6-s_5\mu_*^\top-\mu_*s_5^\top
+s_1\mu_*\mu_*^\top-s_2\gamma_*\gamma_*^\top.
$$

First prove the normal equations. Then prove optimality under conditions including $\Sigma_*\succ0$. A zero derivative does not imply an optimum. Insufficient data can produce singular scatter, leaving no finite maximizer in the open positive-definite domain. [derivation]

`_mstep_normal_params` floors its denominator and adds `SIGMA_REG` to covariance. The ideal closed form and full implementation therefore differ. The IG mixing step has $m_*=E[V]$ and $\lambda_*=(E[1/V]-1/E[V])^{-1}$. Gamma and InverseGamma need shape roots; GIG needs a three-dimensional inversion. [N3, N4, N6; local check]

Convergence has three different levels:

1. A monotone likelihood sequence bounded above has a limit.
2. Stationary accumulation points need further conditions: a compact invariant set inside the open parameter domain, differentiability, suitable continuity of the update, and strict improvement at nonstationary points. Interior unconstrained stationarity means zero gradient. Constraints require their own first-order condition.
3. Convergence of the whole parameter sequence, a local maximum as its limit, and a global MLE are stronger claims. Monotonicity alone proves none of them.

A restricted route proves the compact-set result for a continuous single-valued update first. It then derives strict improvement from exact M-step and Q derivative relations. Multiple optima or discontinuous updates require set-valued theory and more work. *On the Convergence Properties of the EM Algorithm* (1983) is recalled background. Exact assumptions and the applicable version are **to verify**; this checklist does not claim to reproduce its full theorem.

GH does not automatically meet these conditions. Scale nonidentifiability, boundary solutions, and unbounded VG density at $x=\mu$ need attention. `BatchEMFitter` also has posterior floors, approximate GIG solves, fallback, rescaling, and Aitken stopping. Aitken extrapolates a likelihood limit from nearby values; it is not a stationarity proof. [N6; local check]

Batch EM results do not apply automatically to every fitter. Penalized EM must increase $\ell(\theta)-P(\theta)$ for a fixed penalty $P$. Arbitrary per-field shrinkage need not correspond to such a penalty. MCECM means Multi-Cycle ECM; every cycle needs the appropriate conditional objective. `IncrementalEMFitter` uses random minibatches and custom `EtaUpdateRule` updates. It generally lacks per-step monotonicity of the full-data likelihood. Step-size, noise, and stability conditions belong in the separate online-EM items below.

For controlled approximate steps, a target is $\ell(\theta_{k+1})\ge\ell(\theta_k)-\varepsilon_k$. Derive $\varepsilon_k$ from error analysis. A testing tolerance is not a theoretical error bound.

## 6. Bessel functions

For real order $\nu\in\mathbb R$ and argument $z>0$, use the integral corresponding to the current kernel:

$$
K_\nu(z)=\frac12\int_{\mathbb R}e^{\nu u-z\cosh u}\,du
=\int_0^\infty e^{-z\cosh u}\cosh(\nu u)\,du.
$$

Splitting the real line into positive and negative halves gives the second representation. The first directly matches `normix/utils/bessel.py`; the second connects to classical formulas. These are standard integral representations of the modified Bessel function of the second kind, based on local technical material and recalled literature. If pinned Mathlib uses another definition, prove equivalence before using its symbol. [N5; local check and derivation; Mathlib equivalence to verify]

| ID | Item | normix function/page | Difficulty | Value | Dependencies | Lean module (planned) | Status | Notes/PR |
|---|---|---|---|---|---|---|---|---|
| BES-01 | Whole-line Bessel kernel is integrable | `normix/utils/bessel.py`; `docs/theory/gig.md` | C–D | Very high | Tail bounds for $e^{\nu u-z\cosh u}$ | `formal/lean/Normix/SpecialFunctions/BesselIntegral.lean` | not-started | Part of 4–8 person-week foundations |
| BES-02 | $K_\nu(z)>0$ | `log_kv` mathematical target | C | Very high | BES-01; positive integrand on a set of positive measure | `formal/lean/Normix/SpecialFunctions/BesselIntegral.lean` | not-started | Needed for logarithms and ratios |
| BES-03 | Whole-line and half-line representations agree | `bessel.py`; Bessel theory | C | High | BES-01; split and reflect integral | `formal/lean/Normix/SpecialFunctions/BesselIntegral.lean` | not-started | Required if using a different library representation |
| BES-04 | Order symmetry $K_{-\nu}(z)=K_\nu(z)$ | `log_kv`, GIG/NIG formulas | C | Very high | BES-01; $u\mapsto-u$ substitution | `formal/lean/Normix/SpecialFunctions/BesselIdentities.lean` | not-started | 1–3 person-days after integrability |
| BES-05 | Recurrence $K_{\nu+1}=K_{\nu-1}+(2\nu/z)K_\nu$ | Bessel and GIG moment formulas | C–D | Very high | BES-01; integration by parts, zero boundary terms | `formal/lean/Normix/SpecialFunctions/BesselIdentities.lean` | not-started | 1–2 person-weeks |
| BES-06 | Integrable domination on a parameter neighborhood | Order/argument derivative targets | C–D | Very high | Tail estimates with polynomial and hyperbolic factors | `formal/lean/Normix/SpecialFunctions/BesselDomination.lean` | not-started | Uniform local bounds, not pointwise derivatives |
| BES-07 | Argument derivative of $K$ | `BesselMoments.d_arg` mathematical target | C–D | Very high | BES-06; differentiation under integral | `formal/lean/Normix/SpecialFunctions/BesselDerivatives.lean` | not-started | 1–2 person-weeks; shared foundations |
| BES-08 | Order derivative of $K$ | `BesselMoments.d_order` mathematical target | C–D | Very high | BES-06 | `formal/lean/Normix/SpecialFunctions/BesselDerivatives.lean` | not-started | 1–3 person-weeks; second order adds work |
| BES-09 | Half-order formula $K_{1/2}(z)=\sqrt{\pi/(2z)}e^{-z}$ | `NormalInverseGaussian.log_prob` | C | High | Gaussian integral; BES-04 for negative order | `formal/lean/Normix/SpecialFunctions/BesselHalfOrder.lean` | not-started | 1–3 person-weeks |
| BES-10 | $\partial_\nu\log K_\nu(z)=E[U]$ | `BesselMoments.d_order` | C | Very high | BES-02, BES-08; normalized kernel | `formal/lean/Normix/SpecialFunctions/BesselLogMoments.lean` | not-started | Continuous target, not quadrature output |
| BES-11 | $\partial_{\nu\nu}\log K_\nu(z)=\operatorname{Var}(U)$ | `BesselMoments.d2_order` | C–D | Very high | BES-06, BES-10; second derivatives | `formal/lean/Normix/SpecialFunctions/BesselLogMoments.lean` | not-started | Moment integrability explicit |
| BES-12 | $\partial_z\log K_\nu(z)=-E[\cosh U]$ | `BesselMoments.d_arg` | C | Very high | BES-02, BES-07 | `formal/lean/Normix/SpecialFunctions/BesselLogMoments.lean` | not-started | Sign convention |
| BES-13 | $\partial_{zz}\log K_\nu(z)=\operatorname{Var}(\cosh U)$ | `BesselMoments.d2_arg` | C–D | Very high | BES-06, BES-12; second derivatives | `formal/lean/Normix/SpecialFunctions/BesselLogMoments.lean` | not-started | Normalized-kernel covariance |
| BES-14 | $\partial_{\nu z}\log K_\nu(z)=-\operatorname{Cov}(U,\cosh U)$ | `BesselMoments.d2_order_arg` | C–D | Very high | BES-06, BES-10, BES-12 | `formal/lean/Normix/SpecialFunctions/BesselLogMoments.lean` | not-started | Mixed derivative and its sign |
| BES-15 | Finite quadrature log-partition has the discrete moment identities | `_moments_from_weights` | A–B | Very high | EF-07–EF-10; fixed nodes and weights | `formal/lean/Normix/Numerics/BesselFiniteMoments.lean` | not-started | Instantiate finite EF with $(u,-\cosh u)$ |
| BES-16 | Forward value equals the frozen-geometry model at its selection point | `_eval_quad` | A–B | Very high | Explicit geometry selection and fixed-geometry function | `formal/lean/Normix/Numerics/FrozenGeometry.lean` | not-started | Value correspondence only |
| BES-17 | AD output matches fixed-geometry partial derivatives | `_eval_quad`, `log_kv_moments`, `stop_gradient` | B–C | Very high | BES-15, BES-16; explicit AD rule semantics | `formal/lean/Normix/Numerics/FrozenGeometry.lean` | not-started | Not the full adaptive forward function's classical derivative |
| BES-18 | Fixed-order large-$z$ log-Bessel asymptotic with remainder | `log_kv` mathematical reference | D | High | Explicit asymptotic analysis | `formal/lean/Normix/SpecialFunctions/BesselAsymptotics.lean` | not-started | 4–12 person-weeks or more per target |
| BES-19 | Small-$z$ behavior for $\nu=0$ | Bessel/GIG boundary theory | D | High | Separate zero-order asymptotic route | `formal/lean/Normix/SpecialFunctions/BesselSmallArgument.lean` | not-started | Specify limit and remainder |
| BES-20 | Small-$z$ behavior for nonzero fixed $\nu$ | Bessel/GIG boundary theory | D | High | Nonzero-order asymptotic route | `formal/lean/Normix/SpecialFunctions/BesselSmallArgument.lean` | not-started | Uniformity near order zero is separate |
| BES-21 | A specified derivative asymptotic with a justified remainder | Bessel derivative references | D | High | BES-18 or BES-19/BES-20; derivative remainder control | `formal/lean/Normix/SpecialFunctions/BesselDerivativeAsymptotics.lean` | not-started | Do not differentiate an unchecked big-O expression |
| BES-22 | Tail integral bound for the selected quadrature window | `_edge`, `_eval_quad`, `log_drop` | D | High | BES-01, BES-06; explicit parameter region | `formal/lean/Normix/Numerics/BesselTailError.lean` | not-started | Height drop is not an integral error bound |
| BES-23 | Ideal finite-window quadrature error | `_eval_quad` | D | High | Window regularity; exact quadrature rule assumptions | `formal/lean/Normix/Numerics/BesselQuadratureError.lean` | not-started | Part of 2–6 person-month regional error project |
| BES-24 | Stored-node and stored-weight perturbation error | `_GL_T_J`, `_GL_LOGW_J` | D | High | Actual constant encoding; perturbation analysis | `formal/lean/Normix/Numerics/BesselConstantError.lean` | not-started | Stored floats are not exact Legendre roots |
| BES-25 | Value/log-value error for the fixed configuration | `log_kv` | D | High | BES-22–BES-24; relative error below $1$ | `formal/lean/Normix/Numerics/BesselValueError.lean` | not-started | Include stored $\log2$ error; rounding still separate |
| BES-26 | Error for one selected Bessel derivative or moment ratio | `BesselMoments` derivative fields | D | High | Weighted-integral bounds; BES-25; denominator control | `formal/lean/Normix/Numerics/BesselDerivativeError.lean` | not-started | Choose one derivative per card |
| BES-27 | Convergence of an explicit family of quadrature rules | `bessel.py` numerical specification | D | High | Expanding windows and refined rules; tail and quadrature limits | `formal/lean/Normix/Numerics/BesselQuadratureLimit.lean` | not-started | Fixed 192 nodes alone define no limit process |

The derivative formulas to establish are

$$
\partial_zK_\nu(z)=-\frac{K_{\nu-1}(z)+K_{\nu+1}(z)}2,\qquad
\partial_\nu K_\nu(z)=\frac12\int_{\mathbb R}u e^{\nu u-z\cosh u}\,du.
$$

For the argument derivative, differentiate under the integral and use $\cosh u=(e^u+e^{-u})/2$. The recurrence follows by integration by parts, with vanishing terms at both ends. The half-order formula connects to a Gaussian integral; negative half-order follows from symmetry. These are **derivation** routes from standard special-function theory and the chosen integral. Mathlib theorem names remain **to verify**.

For differentiation under the integral, work locally with $|\nu|\le M$ and $z\ge z_0>0$. Here $M,z_0$ are fixed, and a compact parameter box may impose an upper bound on $z$ as well. Required derivatives can be bounded by functions of the form

$$C(1+|u|^m)(1+\cosh^n u)e^{M|u|-z_0\cosh u},$$

where $m,n$ are nonnegative integers and $C>0$. Prove that the bound is integrable and that the chosen parameter box contains a neighborhood of the differentiation point. Pointwise differentiability of the integrand is insufficient.

Let $P_{\nu,z}$ have density proportional to $e^{\nu u-z\cosh u}$, let $U$ be its coordinate random variable, and let $L(\nu,z)=\log K_\nu(z)$. Then the continuous targets are

$$
\partial_\nu L=E[U],\quad \partial_{\nu\nu}L=\operatorname{Var}(U),\quad
\partial_zL=-E[\cosh U],\quad \partial_{zz}L=\operatorname{Var}(\cosh U),
$$

$$\partial_{\nu z}L=-\operatorname{Cov}(U,\cosh U).$$

An order derivative differentiates with respect to $\nu$. It needs no elementary closed form to be formalized. `BesselMoments.d_order`, `d2_order`, `d_arg`, `d2_arg`, and `d2_order_arg` approximate the five quantities above.

**Prove the discrete version first.** For fixed finite nodes $u_i$ and positive weights $w_i$, define

$$A_Q(\nu,z)=\log\sum_iw_i e^{\nu u_i-z\cosh u_i}.$$

Its derivatives satisfy the same moment and covariance relations with normalized finite weights. Let $p_i$ be those weights, $s_i$ the node statistics, and $\bar s=\sum_i p_i s_i$. The covariance $C_Q$ satisfies, for every vector $v$,

$$v^\top C_Qv=\sum_i p_i\bigl(v^\top(s_i-\bar s)\bigr)^2\ge0.$$

This is the centered Gram structure in `_moments_from_weights`. A Gram matrix consists of vector inner products and is positive semidefinite. Strict positive definiteness also requires no degenerate statistic direction on the nodes. Positive weights alone do not establish it. [N5; local check and derivation]

**Frozen geometry needs its own specification.** `_eval_quad` selects a center, window, and nodes from the inputs, then freezes them with `stop_gradient`. AD computes partial derivatives for that fixed geometry. The classical derivative of the full forward function, whose nodes change with input, generally includes geometry derivatives.

Let $g_0$ be the geometry selected at $(\nu_0,z_0)$. Define a fixed-geometry function $L_{g_0}(\nu,z)$. Prove that the returned value equals this function at the selected point, and that AD returns its partial derivatives there. Then bound their differences from the true $L$ and its derivatives. Treating `stop_gradient` as identity in forward semantics does not justify equating AD with the ordinary derivative of the full composition.

**Asymptotics do not by themselves certify error.** For fixed $\nu$, recalled literature gives, as $z\to\infty$,

$$L(\nu,z)=-z+\tfrac12\log\!\frac{\pi}{2z}
+\frac{4\nu^2-1}{8z}+O(z^{-2}).$$

The remainder is eventually bounded by a constant times $z^{-2}$; the constant may depend on $\nu$. This is not a uniform bound for $|\nu|\le300$ and all $z$. Small-$z$ behavior separates $\nu=0$ from $\nu\ne0$. A value asymptotic cannot be differentiated with respect to order or argument without a further proof.

Production code has removed the implementation that switched between asymptotic regions. Proving asymptotics now supports tail bounds, boundary theory, and independent checks. It does not certify a currently used asymptotic branch. [N5, N7; local check]

The fixed 192-node rule and fixed window-drop threshold supply no parameter that automatically tends to a limit. A convergence theorem needs a family of rules with growing windows and changing grid or order. Both tail and quadrature errors must tend to zero. Accuracy of the current fixed configuration needs explicit finite-region bounds.

Gauss–Legendre quadrature approximates a finite-interval integral with particular nodes and weights. NumPy `leggauss` generates the stored floating-point nodes in source. A real semantics may interpret these stored values as exact rationals. That does not make them exact roots of Legendre polynomials. Finite-weight covariance proofs need no exact Gauss–Legendre nodes. Applying the exact rule's error theorem needs separate node and weight perturbation bounds.

Use the following error decomposition. Let $I=2K_\nu(z)$, let $I_W$ be the exact integral over the selected window, let $Q$ use ideal quadrature constants, and let $\widehat Q$ use stored constants but real arithmetic. Then

$$|\widehat Q-I|\le |I-I_W|+|I_W-Q|+|Q-\widehat Q|.$$

These terms are truncation, quadrature, and constant perturbation. They exclude arithmetic rounding. If $|\widehat Q/I-1|\le\varepsilon<1$, then

$$|\log\widehat Q-\log I|\le-\log(1-\varepsilon).$$

Add the stored $\log2$ error separately. For derivatives, bound integrals containing factors such as $u$ and $\cosh u$, then control ratios. This makes the claim “`log_kv` is accurate” a set of reviewable obligations. [derivation]

## 7. Other theory pages

The remaining website theory belongs in the inventory, but not in a single initial delivery. The rows below are a topic-level decomposition, not a formula-by-formula audit. [N8; local check on topics; mathematical statements and library support to verify]

| ID | Item | normix function/page | Difficulty | Value | Dependencies | Lean module (planned) | Status | Notes/PR |
|---|---|---|---|---|---|---|---|---|
| TH-01 | Entropy identity for the specified exponential family | `docs/theory/varentropy.md`; `ExponentialFamily.entropy` | B–C | Medium–high | EF-12; integrability and base-density conditions | `formal/lean/Normix/ExponentialFamily/Entropy.lean` | not-started | Review constant-base assumption |
| TH-02 | Variance of information content | `varentropy.md`; `ExponentialFamily.varentropy` | B–C | Medium–high | EF-13; second information moment | `formal/lean/Normix/ExponentialFamily/Varentropy.lean` | not-started | Existence conditions explicit |
| TH-03 | Density-power integral identity | `varentropy.md`; `log_density_power` | B–C | Medium–high | Partition domain at scaled parameters; base density | `formal/lean/Normix/ExponentialFamily/DensityPower.lean` | not-started | Defines the quantity used for Rényi |
| TH-04 | Rényi formula on its valid order domain | `varentropy.md`; `renyi(alpha)` | B–C | Medium–high | TH-03; order-domain and limit conditions as needed | `formal/lean/Normix/ExponentialFamily/Renyi.lean` | not-started | No implicit moment-existence claim |
| TH-05 | GIG square-root density integral for Hellinger quantities | Hellinger section of `docs/theory/gig.md` | B–C | Medium–high | DM-10; compatible parameter domain | `formal/lean/Normix/Distributions/GIGHellinger.lean` | not-started | Exact normalization needed |
| TH-06 | Joint GH square-root density integral | Hellinger section of `docs/theory/gh.md` | C | Medium–high | DM-13, DM-14; explicit joint density | `formal/lean/Normix/Mixtures/Hellinger.lean` | not-started | Distinguish joint and marginal results |
| TH-07 | Hellinger distance does not increase under marginalization | Hellinger sections of `gig.md`, `gh.md` | C | Medium–high | Normalized joint laws; integral inequality | `formal/lean/Normix/Mixtures/Hellinger.lean` | not-started | Do not assert equality for marginals |
| TH-08 | Fixed KL penalty equals the specified statistic shrinkage objective | `docs/theory/shrinkage.md` | B–C | High | EF identities; exact penalty definition | `formal/lean/Normix/EM/Shrinkage.lean` | not-started | Penalized monotonicity is EM-14 |
| TH-09 | Finite-step online sufficient-statistic recurrence | `docs/theory/online_em.md`; `EtaUpdateRule` | A–B | Medium | Defined step sizes and update rule | `formal/lean/Normix/EM/OnlineRecurrence.lean` | not-started | Algebra before stochastic convergence |
| TH-10 | Telescoping identity for the specified online recurrence | `online_em.md` | A–B | Medium | TH-09 | `formal/lean/Normix/EM/OnlineRecurrence.lean` | not-started | Initial and final terms explicit |
| TH-11 | Review and prove a specified regret identity | `online_em.md` | B–C | Medium | TH-09, TH-10; loss and comparator definitions | `formal/lean/Normix/EM/OnlineRegret.lean` | not-started | Topic-level target; exact statement to verify |
| TH-12 | Stochastic convergence for one specified online update | `online_em.md`; `IncrementalEMFitter` | C–D | Medium | Step-size, noise, stability assumptions | `formal/lean/Normix/EM/OnlineConvergence.lean` | not-started | Months beyond finite recurrence work |
| TH-13 | Factor conditional-moment formula | `docs/theory/factor_analysis.md` | C | Medium | Gaussian conditioning; explicit factor model | `formal/lean/Normix/Mixtures/FactorMoments.lean` | not-started | Select one conditional moment per card |
| TH-14 | Covariance representation under factor constraints | `factor_analysis.md` | B–C | Medium | Factor-model definitions and matrix constraints | `formal/lean/Normix/Mixtures/FactorCovariance.lean` | not-started | Not an unconstrained full EF |
| TH-15 | Factor-model constrained M-step | `factor_analysis.md` | C | Medium | TH-13, TH-14; constrained objective | `formal/lean/Normix/EM/FactorMStep.lean` | not-started | Do not apply unrestricted expectation inversion |
| TH-16 | Linear projection identity for a normal mixture | `docs/theory/mean_risk_optimization.md` | B–C | Medium | Mixture representation; linear algebra | `formal/lean/Normix/Mixtures/Projection.lean` | not-started | Explicit projection and covariance |
| TH-17 | Mean-risk dimension reduction | `mean_risk_optimization.md` | B–C | Medium | TH-16; stated risk functional | `formal/lean/Normix/Finance/MeanRisk.lean` | not-started | One restricted proposition per card |
| TH-18 | Quadratic objective equivalence with specified transaction costs | `docs/theory/transaction_costs.md` | A–B | Medium | Objective and constraint definitions | `formal/lean/Normix/Finance/TransactionCosts.lean` | not-started | Preserve domain and cost convention |
| TH-19 | Conditional-tail expectation formula for CVaR | `docs/theory/cvar_derivatives.md` | C | Medium | Density and tail-integrability assumptions | `formal/lean/Normix/Finance/CVaR.lean` | not-started | Specify tail and probability level |
| TH-20 | A specified CVaR derivative | `cvar_derivatives.md` | C–D | Medium | TH-19; differentiation under integral | `formal/lean/Normix/Finance/CVaRDerivatives.lean` | not-started | Boundary and density conditions |
| TH-21 | A specified ENB risk decomposition | `docs/theory/enb.md`, `generalized_enb.md` | B–C | Medium–low | Risk and decomposition definitions | `formal/lean/Normix/Finance/ENB.lean` | not-started | Exact page statement to verify |
| TH-22 | Homogeneous-function identity used by ENB | `enb.md`, `generalized_enb.md` | B | Medium–low | Homogeneity and differentiability | `formal/lean/Normix/Finance/ENBHomogeneity.lean` | not-started | State the domain |
| TH-23 | A specified matrix-transformation identity for generalized ENB | `generalized_enb.md` | B–C | Medium–low | Matrix domain and risk coordinates | `formal/lean/Normix/Finance/ENBTransforms.lean` | not-started | One transform per card |
| TH-24 | Normalize a discrete quadrature-scenario measure | `docs/theory/quadrature_scenarios.md` | A | Medium–high | Finite nonnegative weights and positive total | `formal/lean/Normix/Numerics/QuadratureScenarios.lean` | not-started | Finite algebra first |
| TH-25 | Conditional-integral identity for quadrature scenarios | `quadrature_scenarios.md` | B–C | Medium–high | TH-24; conditional model and integrability | `formal/lean/Normix/Numerics/QuadratureScenarios.lean` | not-started | Exact identity before approximation |
| TH-26 | Truncation error for a specified scenario construction | `quadrature_scenarios.md` | C–D | Medium–high | TH-25; selected tail bound and metric | `formal/lean/Normix/Numerics/ScenarioError.lean` | not-started | Error work budget separate |

The original incremental estimates remain:

| Page group | Priority and estimate | Main reason for difficulty |
|---|---|---|
| `varentropy.md` | Medium–high; 2–5 person-weeks after EF foundations | Entropy, information variance, and density-power formulas need base-density and moment conditions |
| Hellinger sections of `gig.md`, `gh.md` | Medium–high; 1–3 person-weeks after normalization | Square-root density integrals and contraction under marginalization |
| `shrinkage.md` | High; 1–3 person-weeks after EM foundations | Match a fixed KL penalty to shrunk statistics and the same penalized objective |
| `online_em.md` | Medium; 1–3 person-weeks for algebra, then months for stochastic convergence | Recurrences and telescoping are finite algebra; regret and stochastic limits need further assumptions |
| `factor_analysis.md` | Medium; 4–8 person-weeks | Conditional moments and covariance constraints prevent unrestricted full-EF inversion |
| `mean_risk_optimization.md`, `transaction_costs.md` | Medium; 1–4 person-weeks per restricted proposition | Linear projection, dimension reduction, and quadratic objective equivalence need explicit domains |
| `cvar_derivatives.md` | Medium; 4–10 person-weeks | Tail expectations, densities, and differentiation under integrals |
| `enb.md`, `generalized_enb.md` | Medium–low; 2–6 person-weeks per group | Risk decomposition, homogeneous-function identities, and matrix transformations |
| `quadrature_scenarios.md` | Medium–high; 1–2 person-weeks for finite algebra; error bounds separate | Discrete normalization is finite; conditional integrals and truncation need analysis |

Regret is cumulative loss relative to a specified comparison solution. CVaR is average loss in a specified tail. ENB measures the effective number of risk sources. Proposition review must decide which statements are library contracts and which remain research derivations. Backtest performance does not verify these mathematical identities.

Use [the report's source inventory](01_lean_math_feasibility.md#sources) for the source paths and recalled literature. No Mathlib lemma name, completed proof, or new measurement is asserted by this checklist.
