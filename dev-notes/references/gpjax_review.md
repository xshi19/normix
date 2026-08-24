# GPJax Review: Design, JAX Usage, and Lessons for normix

**Date:** 2026-08-20
**Repo:** https://github.com/thomaspinder/GPJax (org alias: JaxGaussianProcesses/GPJax)
**Reviewed at:** v0.18.0, `main` @ `569c29d` (2026-08); MIT; ~656 stars; JOSS paper (Pinder & Dodd 2022)
**Size:** ~8,900 lines in `gpjax/`; Python ≥ 3.11
**Runtime deps:** `jax`, `equinox`, `paramax`, `lineax`, `numpyro`, `optax`, `jaxtyping`, `beartype`, `scipy`, `numpy`, `tqdm`, `rich`

GPJax is a Gaussian-process library whose stated goal is that "the code should
be as close as possible to the maths we write on paper". It is the closest
thing in the JAX ecosystem to a sibling project of normix: an Equinox-based,
maths-first statistical modelling package with heavy Cholesky linear algebra,
a fitting loop, and executable documentation. This review covers what it does,
how it uses JAX, its design history, and what normix should (and should not)
borrow.

---

## 1. What GPJax does

### 1.1 The model

A Gaussian process is a prior over *functions*: $f \sim \mathcal{GP}(m, k)$
with mean function $m(\cdot)$ and kernel $k(\cdot,\cdot)$. Evaluated at any
finite set of inputs, the function values are jointly Gaussian with covariance
$K_{ij} = k(x_i, x_j)$. Given observations $y = f(x) + \varepsilon$, Bayes'
rule yields a closed-form posterior over $f$ at unseen inputs — mean *and*
calibrated uncertainty bands. Hyperparameters (lengthscale, variance, noise)
are learned by maximising the marginal log-likelihood

$$
\log p(y) = -\tfrac{1}{2}\bigl(y^\top (K_{xx} + \sigma^2 I)^{-1} y
+ \log\lvert K_{xx} + \sigma^2 I\rvert + n\log 2\pi\bigr),
$$

which requires differentiating through a Cholesky factorisation — the thing
JAX automates. Beyond exact conjugate regression, GPJax covers non-Gaussian
likelihoods (Bernoulli, Poisson) via latent-variable posteriors, sparse
variational GPs (collapsed and stochastic ELBOs), heteroscedastic noise,
multi-output GPs (ICM/LCM, OILMM), graph-input kernels, additive
orthogonal kernels (OAK) with Sobol decompositions, and an opt-in
`state_space` subpackage that re-expresses Markovian GPs as SDEs solved by
Kalman filtering in $O(N)$.

### 1.2 Contrast: `np.random` random walk vs a GP library

`np.cumsum(rng.normal(size=n))` simulates one path of a process that is
already fully specified — parameters known, one direction only (forward
simulation), no notion of data. A GP library is the inverse problem, plus the
machinery to make it differentiable:

| | `np.random` random walk | GPJax |
|---|---|---|
| Object | one sampled path | a distribution over functions (a Brownian-motion kernel makes the GP literally the continuous-time random walk) |
| Data | none — parameters assumed | posterior conditioning: $p(f^\star \mid y)$ at arbitrary inputs |
| Uncertainty | none | predictive mean + covariance from the same object |
| Parameters | fixed by hand | learned by `grad`(marginal log-likelihood) through the Cholesky |
| RNG | stateful, non-differentiable, irreproducible under transforms | `jax.random` keys: pure, splittable, `vmap`/`jit`-safe |
| Scaling | one path at a time | `jit` + `vmap` + GPU; sparse/state-space approximations past $O(N^3)$ |

The same contrast describes normix vs "just call `np.random.standard_t`":
the package's value is inference (EM, η→θ solvers, Fisher information), not
simulation.

---

## 2. Architecture

### 2.1 Core pipeline (`gps.py`)

```
Prior(kernel, mean_function)  *  Likelihood  →  Posterior
        prior(x_test) → GaussianDistribution
        posterior(x_test, train_data) → GaussianDistribution
```

`Prior.__mul__(likelihood)` calls `construct_posterior()`, which dispatches on
the likelihood type: Gaussian → `ConjugatePosterior` (closed form),
Bernoulli/Poisson → `NonConjugatePosterior` (whitened latent values as a
learnable field), heteroscedastic → dedicated posteriors. The `*` operator is
the paper equation $p(f \mid y) \propto p(y \mid f)\, p(f)$ rendered as API.
Classes are generic (`AbstractPrior[MeanF, Kernel]`) with `tp.TYPE_CHECKING`
overloads so `prior * Gaussian(...)` type-checks to `ConjugatePosterior`.

`sample_approx()` returns a *function* $x \mapsto \hat f(x)$ (Wilson et al.
2020 pathwise sampling: random Fourier features + canonical features), so one
coherent posterior draw can be evaluated at any number of query points at
constant cost — a functional-programming answer to the $O(N^3)$ cost of exact
joint sampling.

### 2.2 Parameters (`parameters.py`)

Constrained parameters are `paramax.AbstractUnwrappable` pytree nodes storing
the *unconstrained* value; `unwrap()` applies the constraining bijection
(resolved from numpyro's `biject_to` constraint registry):

| Class | Bijection | Example use |
|---|---|---|
| `Real` | identity | mean constants |
| `PositiveReal` / `NonNegativeReal` | softplus | lengthscale, variance |
| `SigmoidBounded` | sigmoid scaled to $[a,b]$ | ArcCosine weights |
| `LowerTriangular` | fill-triangular + softplus diag | variational Cholesky roots |

`paramax.unwrap(model)` resolves every wrapper in the tree; `fit` calls it
*inside* the loss, so optimisers always step in unconstrained space with zero
user-visible ceremony. `paramax.non_trainable(x)` freezes a subtree.
Notable hygiene: a numpyro dtype bug is worked around in a subclass labelled
`TEMPORARY WORKAROUND` with the upstream discussion linked and removal
criteria stated.

### 2.3 Kernels (`kernels/`)

`AbstractKernel.__call__(x, y) → scalar` is the mathematical object —
unbatched, like normix's `log_prob`. Matrix assembly is delegated to a
swappable `compute_engine` (strategy pattern):

- `DenseKernelComputation` — the Gram matrix is literally
  `vmap(lambda x: vmap(lambda y: k(x, y))(ys))(xs)`;
- `DiagonalKernelComputation`, `ConstantDiagonalKernelComputation` (a
  stationary kernel's diagonal is $\sigma^2 \mathbf{1}$ — $O(1)$);
- `BasisFunctionComputation` (RFF low-rank), `EigenKernelComputation` (graph
  Laplacian eigenbasis).

Kernels compose with `+` and `*` into `SumKernel`/`ProductKernel` (nested
combinations are flattened). Every stationary kernel exposes an abstract
`spectral_density` property — its Bochner spectral measure as a numpyro
distribution — which is the single hook that makes the generic `RFF`
approximation work for any stationary kernel. Same shape as normix's DEC-5:
an abstract `log_kernel` unlocks generic PINV sampling.

### 2.4 Linear algebra (`linalg/`)

Since v0.14 GPJax delegates to [Lineax](https://docs.kidger.site/lineax/):
`gram()` returns `lx.TaggedLinearOperator(..., positive_semidefinite_tag)`,
and `cholesky_factor`, `logdet`, `logdet_from_factor` are
`functools.singledispatch` functions with structure-exploiting registrations
per operator type (diagonal, identity, and custom `BlockDiag` / `Kronecker`
operators). The v0.18 changelog documents the payoff: KL divergences on the
ELBO hot path were factorising the same matrices twice (four Choleskys where
two suffice, issue #664); routing log-dets through already-computed factors
made `grad(prior_kl)` 13× faster for the whitened family at 1024 inducing
points.

### 2.5 Objectives (`objectives.py`)

Pure functions `(model, Dataset) → scalar`: `conjugate_mll`,
`conjugate_loocv` (Rasmussen–Williams §5.4.2 closed form), `log_posterior_density`,
`elbo`, `collapsed_elbo`, `heteroscedastic_elbo`. Users negate with a lambda
to minimise. Models never own their loss — the objective is data, so swapping
MLL for LOOCV is a one-line change. Docstrings carry the full display-math
derivation of each bound.

### 2.6 Fitting (`fit.py`)

Three optimisers, one contract (`model, history = fit*(model=..., objective=..., train_data=...)`):

| Function | Engine | Loop | Notes |
|---|---|---|---|
| `fit` | any Optax `GradientTransformation` | `lax.scan` (or `vscan` with tqdm) | mini-batching with replacement via `jr.choice`; `unroll` exposed |
| `fit_scipy` | SciPy L-BFGS-B | Python | `ravel_pytree` flattens the params pytree to one vector; jitted `value_and_grad` behind a NumPy shim |
| `fit_lbfgs` | Optax L-BFGS + zoom linesearch | `lax.while_loop` | gradient-norm stopping *inside* jit via `optax.tree_utils.tree_get(opt_state, "grad")` |

All three: `eqx.partition(model, eqx.is_array)` splits trainable from static,
`paramax.unwrap` inside the loss handles constraints, `eqx.combine`
reassembles. `fit(safe=True)` runs ~70 lines of hand-rolled `_check_*`
validation first.

### 2.7 Variational families (`variational_families.py`)

Sparse-GP posteriors $q(u) = N(\mu, S)$ over inducing values, in **four
parametrisations**: standard $(\mu, \text{chol}(S))$, whitened,
**natural** $\theta = (S^{-1}\mu, -\tfrac{1}{2}S^{-1})$, and **expectation**
$\eta = (\mu, S + \mu\mu^\top)$ — the docstrings write $q(u) =
\exp(\theta^\top T(u) - a(\theta))$ with $T(u) = [u, uu^\top]$ explicitly.
This is normix's exponential-family θ/η duality deployed for optimisation
geometry: stepping in θ approximates natural-gradient descent. GPJax stores
the natural/expectation parameters as raw `Real` fields and lets Adam walk
them (no PSD projection), whereas normix solves η→θ as a Bregman problem with
domain guarantees — same mathematics, different robustness/simplicity
trade-off.

### 2.8 Likelihoods and integrators

`AbstractLikelihood` carries a link function and an `integrator` field.
`expected_log_likelihood` (the ELBO's data term
$\int \log p(y \mid f)\, q(f)\, df$) is delegated to an `AbstractIntegrator`
strategy: `GHQuadratureIntegrator` (20-point Gauss–Hermite default) for
general likelihoods, `AnalyticalGaussianIntegrator` for the conjugate case.
Quadrature-vs-closed-form as a swappable object mirrors normix's
backend/method kwargs on `from_expectation`.

### 2.9 `state_space/` and its Bessel functions

Markovian kernels (Matérn family) admit an exact SDE representation solved by
Kalman filtering/smoothing — $O(N)$ instead of $O(N^3)$, same posterior. The
subpackage ships its own `_bessel.py` for scaled modified Bessel
$\tilde I_k(c) = e^{-c} I_k(c)$ (needed by periodic state-space kernels),
with a three-regime `lax.cond` dispatch: forward recurrence seeded by
`i0e`/`i1e` for $c$ above the truncation order, Miller downward recurrence
(start at $k_{\max} = 2\cdot\text{order} + 20$, rescale by `i0e`) in the
middle, log-space power series below $10^{-8}$. Structurally a small sibling
of normix's 4-regime `log_kv`. Differences: integer orders only, fixed
truncation known at trace time, and — because `i0e`/`i1e` have JAX
derivatives and `lax.scan` is differentiable — **no `custom_jvp` needed**.
normix's harder problem (real order $\nu$, ∂/∂ν with no closed form) is what
forces `log_kv`'s custom JVP; GPJax never hits it.

### 2.10 UX extras

- `Dataset`: a `@dataclass(slots=True)` registered as a pytree; enforces 2-D
  shapes at construction, warns on non-float64 inputs, supports `d1 + d2`
  concatenation.
- `summary.py`: GPflow-style `summarise(model)` — a `rich` table with one row
  per parameter (value, bijector, prior, trainable, shape, dtype), wired into
  `__rich__` and `_repr_mimebundle_` so models render themselves in notebooks.
- `citation.py`: `cite(obj)` is a `singledispatch` returning a formatted
  BibTeX entry for any component (`cite(Matern32())` → Matérn's 1960 thesis);
  calling it on a jitted function raises a pointed `RuntimeError`.
- `vscan`: `lax.scan` with a tqdm bar driven by `jax.debug.callback` under
  `lax.cond` (credited to Jérémie Coullon's blog post).

---

## 3. How GPJax leverages JAX

| JAX feature | Where | Note |
|---|---|---|
| `vmap` | Gram matrices (`vmap(vmap(k))`), diagonal ops, batched sampling | scalar math stays scalar; batching is an engine concern |
| `lax.scan` | `fit` optimisation loop; Bessel forward recurrence | training history is the scan output |
| `lax.while_loop` | `fit_lbfgs` | convergence test on `opt_state` fields inside jit |
| `lax.cond` | Bessel regime dispatch | only the taken branch contributes under `jit`/`grad` |
| `jax.debug.callback` | tqdm progress inside `scan` | host callback gated by `lax.cond` on `step % log_rate` |
| pytrees | `Dataset` (manual registration), everything else `eqx.Module` | `Dataset` passes through `jit`/`scan` boundaries |
| `eqx.partition` / `combine` / `filter_value_and_grad` | all fitters | trainable = `eqx.is_array`; static fields ride along |
| `ravel_pytree` | `fit_scipy` | one flat vector in/out of SciPy; same trick as normix's CPU solver path |
| `jax.random` keys | everywhere; documented contract | "split before you reuse, never reuse after you split" (sharp-bits page) |
| `jaxtyping` + `beartype` | shape annotations `Float[Array, "N D"]` on every signature | **enforced only in tests** via `pytest --beartype-packages=gpjax`; zero production cost |
| x64 | `Dataset` warns on float32; tests force `jax_enable_x64` | same conftest pattern as normix |

Notable **absences**: no `custom_jvp`/`custom_vjp` anywhere (their special
functions are integer-order and autodiff-safe; normix's real-order `log_kv`
is genuinely harder numerics), no `pmap`/sharding, no `checkify` in practice
(mentioned in docs; constructors still do Python `isinstance` validation, so
**GPJax objects cannot be constructed inside `jit`** — a documented
limitation. normix's `from_classical`/`from_natural`/`from_expectation`
being fully traceable is a real advantage worth protecting).

---

## 4. The backend odyssey (design history)

GPJax has changed its module/parameter backend repeatedly: early
parameter-dict/chex era → custom pytree module (`mytree`) → Flax NNX → since
v0.14, **Equinox + paramax + lineax + numpyro constraints** — the stack it
has today, and (minus paramax/lineax) the stack normix chose on day one.
Costs of the churn, visible in the repo:

- A per-release `docs/migration.md` ("one section per release that changed a
  public API; upgrading across two releases means reading two sections").
- Regression #712: the `Zero` mean function silently *drifted away from zero*
  (0.0 → 5.09 on data with mean 5) after the Equinox migration. Root cause,
  quoted from the changelog: "a changed trainability contract rather than a
  lost line: under `nnx`, `fit` optimised only `Parameter` instances, so
  `Zero`'s bare array was inert by construction; under Equinox, `fit`
  partitions on `eqx.is_array`, which makes every array leaf trainable."
  Fixed by wrapping the constant in `paramax.non_trainable`.

Lesson for normix: **trainability must be an explicit contract, not an
artifact of which leaves happen to be arrays.** normix's EM fitters update
via `from_expectation` (no gradient partitioning), so today's exposure is
low — but any future gradient-based fitter that partitions on `eqx.is_array`
inherits exactly this failure mode (e.g. a fixed `mu` that silently becomes
trainable). Worth a design-table row when such a fitter appears.

---

## 5. Structural comparison with normix

Different scientific problems — nonparametric regression over functions vs
parametric heavy-tailed density estimation — but near-isomorphic package
anatomy:

| Concern | GPJax | normix |
|---|---|---|
| Base module system | `eqx.Module`, immutable | same |
| The scalar mathematical core | `kernel(x, y)` | `log_prob(x)` |
| Batching | `compute_engine` objects wrapping `vmap` | `jax.vmap` at call sites / fitter |
| Parameter constraints | paramax wrappers (model-side, softplus) | solver-side φ↔θ chain rule + clamps (`THETA_FLOOR`, `GIG_CLAMP_*`) |
| Fitting | gradient descent on an objective (Optax/SciPy) | EM with closed-form M-steps + Bregman η→θ solves |
| θ/η duality | Natural/Expectation *variational* families, optimised by SGD | first-class parametrisations with `from_natural`/`from_expectation` |
| Linear algebra structure | lineax operators + PSD tags + singledispatch | hand-rolled Cholesky (`L_Sigma`) and Woodbury (`factor.py`) |
| PSD safety valve | `jitter` = 1e-6, documented ("raise the jitter before you suspect your model") | `SIGMA_REG`, `HESSIAN_DAMPING`, constants table |
| Special functions | integer-order $\tilde I_k$, 3 regimes, no custom JVP | real-order `log_kv`, 4 regimes, `custom_jvp` |
| SciPy bridge | `ravel_pytree` → L-BFGS-B | CPU backend Bessel + `scipy.optimize` M-step |
| Numerical audits | issues #662–#675 folded into v0.18 with perf numbers | 2026-07-12 review roadmap (44 items) |
| Scaling escape hatch | sparse VI, state-space $O(N)$ | factor Σ (Woodbury), incremental EM, CPU E-step |

The parameter-constraint row is the deepest design divergence. GPJax
constrains at the *parameter* (any third-party optimiser works on the raw
pytree; cost: wrapper indirection everywhere, `_val()` unwrap calls sprinkled
through math code). normix constrains at the *solver* (distributions store
clean classical values; cost: each new solver must re-implement the
reparametrisation). Both are coherent; normix's fits its EM-centric world
where most updates are closed-form and never risk leaving the domain.

---

## 6. Lessons for normix

### 6.1 Testing and verification

| GPJax practice | normix status | Verdict |
|---|---|---|
| `test_jit_compatibility.py`: dedicated suite asserting jit-vs-eager numerical equality for core ops | jit exercised implicitly across tests; no dedicated equivalence module | **Adopt** — cheap, catches tracing bugs (e.g. `lax.cond` branch divergence) at the API surface |
| `test_numerical_stability.py`: parametrised extreme-parameter sweeps asserting finiteness | `test_extreme_parameters.py` covers this | Already have |
| `test_dependencies.py`: every declared runtime dep is actually imported (born from shipping a dead 14 MB dep to every macOS install, #675) | import *smoke* tests exist (optional deps blocked); no declared-vs-imported check | **Adopt** — trivial |
| Docstring examples run by `xdoctest`; markdown snippets by `mktestdocs`; both in the CI gate | docstring examples not executed (only MyST-NB pages are) | **Consider** — normix docstrings carry examples; a doctest pass would keep them honest |
| `hypothesis` property-based tests (`deadline=None, max_examples=20`) | not used | Consider selectively — good fit for round-trip properties (θ↔η↔classical, `cdf`/`ppf` inverses) |
| `pytest --beartype-packages` runtime shape-checking of jaxtyping annotations, tests only | no shape annotations | See 6.4 |
| `filterwarnings = ["error"]`, `xfail_strict`, pytest-xdist `-n 8` | partial | Consider warnings-as-errors; xdist depends on suite thermals |
| `tests/_reference/`: committed reference implementations (dense Kalman) that the fast path must match | scipy comparisons play this role | Same idea, already have |

### 6.2 Benchmarking

GPJax runs [ASV](https://asv.readthedocs.io/) continuously: per-commit trend
series on `main`, results in an orphan `asv-results` branch (git worktree),
dashboard published into the docs site at `/benchmarks/`, and a
`bench-compare <base> <branch>` regression report for PRs. Their
`benchmarks/README.md` "five rules" are a compact JAX-benchmarking checklist:
(1) always `block_until_ready` — otherwise you time dispatch, not work;
(2) keep beartype/hypothesis out of the bench env; (3) pin machine identity;
(4) cache discipline — warm-up call in `setup()` for steady-state benchmarks,
`jax.clear_caches()` for compile benchmarks; (5) pin the Python version or
reset the series.

**Compile time is a first-class tracked metric** (`track_compile_*`
benchmarks; "variational posteriors carry more pytree structure through the
trace, so the compile gap matters for iteration speed").

normix status: `block_until_ready` used throughout; `bench_jit_solvers.py`
already separates first-call (compile) from cached-call latency — which is
how the GH retrace pathology was found (`tech_notes/jax_overhead_diagnosis.md`).
Missing: trend tracking across commits. Verdict: **skip ASV** (heavy for a
single-maintainer project) but **consider** wiring `benchmarks/compare.py`
into CI as an opt-in PR check, and keep compile-vs-steady-state separation as
a stated convention in `benchmarks/`.

> **Superseded 2026-08-23:** ASV adopted as a scoped trend-tracking layer
> alongside the deep-dive scripts — see
> [`../plans/asv_benchmarking.md`](../plans/asv_benchmarking.md).

### 6.3 Documentation

The docs are GPJax's strongest artefact. Concretely:

| Practice | Detail | Verdict for normix |
|---|---|---|
| `sharp_bits.md` | one page of numerical gotchas: stateless PRNG contract, bijector rationale, Cholesky-jitter failure signature ("NaNs in your loss — raise the jitter before you suspect your model"), $O(N^3)$ wall, objects-inside-jit rule | **Adopt** — normix has the material scattered (VG `b=0` singularity, `alpha_min`, `B_POST_FLOOR`, backend choice, `LOG_EPS` flooring) but no single "when normix bites" page in `docs/user_guide/` |
| Method-selection decision table | "up to a few thousand → `conjugate_mll`; ~50k → `collapsed_elbo`; beyond → `elbo`", with citations | **Adopt** — normix analog: full Σ vs `Factor*` (d, r thresholds), batch vs incremental EM (N), `e_step_backend` jax vs cpu (~15× at large N). The numbers exist in benchmarks; put them in one table |
| `design.md` notation table | on-paper symbol ↔ code name (`Kxx`, `Lx`, …), keyed to Rasmussen–Williams | **Adopt** — normix states θ↔`theta`, η↔`eta` in AGENTS.md (agent-facing); a published table in `docs/` keyed to [Shi2016] would serve users |
| `glossary.md` | MyST `{.glossary}` definition list; every term linkable via `` {term}`jitter` ``; entries are miniature essays (the Cholesky entry includes the $n^3/3$-flop argument) | Consider — moderate effort, high tutorial value |
| Executable examples as jupytext `py:percent` | `.py` under version control, executed by MyST-NB at build; smoke vs full render via one env var (`GPJAX_DOCS_CI`); full render in a separate scheduled workflow | Same architecture as normix docs (cached MyST-NB); normix's `html` vs `html-strict` split matches. Already have |
| `sphinx-codeautolink` | API names inside example code blocks auto-link to reference pages | **Consider** — cheap dependency, real navigation win for tutorial-heavy docs |
| MathJax browser check | `docs/scripts/check_mathjax.py` drives headless Chromium over every built page; MathJax errors are client-side and invisible to `sphinx-build -W` | **Consider** — normix docs are maths-dense; a broken `\gamma` renders as red text no CI currently sees |
| `migration.md` | one section per breaking release, imperative fix-it instructions with before/after code | Adopt the *pattern* when normix next breaks API; not needed retroactively |
| Hand-written CHANGELOG entries | root-cause narratives with issue links and perf numbers (the #712 entry is a model postmortem) | Partial — normix's release-please changelog is a ledger, not a narrative. Keep release-please; for correctness-level fixes, keep writing the story in tech notes (current practice) and link them from the release notes |
| `cite(obj)` | singledispatch BibTeX per component | Optional nicety — normix's `docs/references.md` covers attribution; a `cite()` would add discoverability for Devroye/Shi2016 provenance at ~100 lines |

### 6.4 API and typing

- **jaxtyping shape annotations** (`Float[Array, "N D"]`) double as machine-
  checked shape documentation, enforced only under pytest via the beartype
  hook. For normix — where the unbatched-core convention makes shapes a
  contract (`log_prob` takes a single observation) — annotating the public
  API would encode that contract. **Consider**; cost is a dev-only dep pair
  and annotation churn; do not enforce at runtime.
- **`tp.TYPE_CHECKING` overloads + generics** give GPJax precise static types
  for dispatchy constructors. normix's `from_expectation` dispatching on
  `NormalMixtureEta` vs `jax.Array` could get overloads for the same IDE
  benefit. Minor.
- **Notebook repr**: `summarise()` + `_repr_mimebundle_` render any model as
  a parameter table. normix's `utils/validation.py` prints parameters
  manually in notebooks. A `_repr_mimebundle_` (or plain `_repr_html_`) on
  `ExponentialFamily`/`NormalMixture` showing classical params, θ, η side by
  side would improve every tutorial page at low cost. **Consider** — keep
  `rich` out of core deps (plain HTML string suffices).
- **Operator-overloading composition** (`prior * likelihood`): evocative in
  GP land because kernels/likelihoods combine freely. normix's mixtures have
  a *fixed* composition (subordinator class ↔ marginal class), so an
  equivalent operator (`MultivariateNormal @ Gamma → VarianceGamma`?) would
  be a gimmick. Skip.

### 6.5 Scientific-problem handling worth noting

- **The abstract-hook-unlocks-generic-approximation pattern**:
  `spectral_density` (Bochner) → generic RFF for any stationary kernel;
  exactly parallels normix's `log_kernel` → generic PINV table (DEC-5).
  Validation that the pattern scales; no action.
- **Same model, different representation as an opt-in subpackage**:
  `state_space` re-expresses supported kernels as SDEs for $O(N)$ inference
  without touching the core API. Structural precedent if normix ever grows an
  alternative inference path (e.g. characteristic-function-based estimators):
  a sibling subpackage, not flags on the existing classes.
- **Capability dispatch**: `construct_posterior` routes heteroscedastic
  likelihoods by `likelihood.supports_tight_bound()` — dispatch on a
  capability method, not `isinstance` chains. Tidy pattern where a family
  splits by mathematical property.

### 6.6 Process and CI

- Separate workflows: unit tests / ruff / docs smoke-build (warnings as
  errors) / *scheduled full notebook render* / integration / security scan /
  **commit-lint** (enforces the conventional-commit format normix documents
  but does not enforce). The commit-lint action is a one-file adopt if
  release-please parsing ever gets corrupted by a malformed commit.
- `interrogate` docstring-coverage gate (fail-under 64%) and coverage
  fail-under 50% — floors, not targets. Low value for normix (docstring
  discipline already enforced by review).
- Governance docs, Slack, issue triage via CodeTriage — community-scale
  machinery; not applicable at normix's current scale.

### 6.7 Agent docs (meta-lesson)

GPJax ships a `CLAUDE.md` with a good architecture map — and it references
`gpjax/numpyro_extras.py`, a module that no longer exists (the functionality
now lives in a docs example + integration test). One stale pointer in an
otherwise excellent file. normix's `agent-maintenance` skill and context-map
audits target exactly this failure mode; the lesson is that agent docs rot at
the same rate as any other docs and need the same review triggers.

---

## 7. What not to borrow

| Anti-pattern | Why avoid |
|---|---|
| Heavy runtime dependency set (`numpyro` for bijections, `beartype`+`jaxtyping` imported at runtime, `tqdm`, `rich` as hard deps) | normix core stays at `jax`/`equinox`/`jaxopt`/`numpy`/`scipy`; UX niceties belong in optional extras |
| Constructor-time Python validation (`isinstance` checks in `__init__`) | makes objects non-constructible inside `jit`/`vmap`/`grad` — documented as a GPJax limitation. normix's traceable `from_*` constructors are a feature; guard them in review |
| `fit(safe=True)` — ~70 lines of hand-rolled `_check_model` / `_check_verbose` boilerplate | validation at boundaries is right; a bespoke checker per kwarg is noise. normix's targeted `ValueError`s suffice |
| Parameter wrappers leaking into math code (`_val(x)` unwrap calls sprinkled through kernels/posteriors) | the cost side of model-side constraints; normix's solver-side reparametrisation keeps densities free of unwrap ceremony |
| Backend churn (params-dict → custom pytree → Flax NNX → Equinox) | each hop cost migration docs, contributor retraining, and at least one silent correctness regression (#712) |
| `__init_subclass__` docstring-inheritance metaprogramming | implicit magic to save writing docstrings; normix writes them |
| Growing ruff `per-file-ignores` lists | symptom of rules fighting the codebase; keep lint config minimal |

---

## 8. Adoption candidates, prioritised

| # | Item | Effort | Value | Section |
|---|---|---|---|---|
| 1 | "Sharp bits" page in `docs/user_guide/` + method-selection decision table (Σ structure, EM variant, backend, known singularities) | M | High | 6.3 |
| 2 | JIT-vs-eager equivalence test module | S | High | 6.1 |
| 3 | Declared-deps-are-imported regression test | S | Med | 6.1 |
| 4 | Doctest pass over docstring examples (xdoctest or `--doctest-modules`) | S | Med | 6.1 |
| 5 | Published notation table (symbol ↔ code, keyed to [Shi2016]) in `docs/` | S | Med | 6.3 |
| 6 | `_repr_html_` parameter tables on distributions (θ/η/classical) | M | Med | 6.4 |
| 7 | jaxtyping annotations on the public API + beartype in tests only | M | Med | 6.4 |
| 8 | `sphinx-codeautolink` + MathJax headless check in docs CI | S | Med | 6.3 |
| 9 | Glossary page with `{term}` roles | M | Low-Med | 6.3 |
| 10 | Hypothesis round-trip properties (θ↔η, cdf↔ppf) | M | Low-Med | 6.1 |
| 11 | commit-lint CI action | S | Low | 6.6 |
| 12 | `cite()` helper | S | Low | 6.3 |

Explicit non-adoptions with reasons recorded: paramax-style parameter
wrappers (§5, solver-side reparametrisation already solves it), ASV (§6.2),
operator-overloaded model composition (§6.4), lineax (revisit only if the
covariance-structure zoo grows beyond dense + factor; today's two structures
are well served by `L_Sigma` and Woodbury).

---

## 9. Summary

GPJax validates normix's founding choices from an independent evolution: it
*arrived* at Equinox modules, unbatched scalar cores batched by `vmap`,
pure-function objectives, `lax.scan`/`while_loop` training loops, SciPy
bridges via `ravel_pytree`, regime-dispatched special functions under
`lax.cond`, and numerical audit campaigns — after three backend migrations
normix never had to make. Its genuine leads over normix are concentrated in
the documentation layer (sharp-bits page, decision tables, glossary,
notation table, per-release migration guides) and in test/CI breadth
(jit-equivalence suite, doctest execution, shape-checked tests, dependency
hygiene checks). Its weaknesses — heavy runtime dependency set, constructors
that cannot be traced, wrapper ceremony inside the math — are the mirror
image of normix's lean-core discipline. The highest-value transfers are
documentation patterns, not code.
