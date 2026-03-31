# normix Package Review

Date: 2026-03-30

## Scope and method

This review covered:

- architecture and design docs (`README.md`, `docs/ARCHITECTURE.md`, `docs/design/design.md`, `docs/theory/em_algorithm.rst`, `pyproject.toml`)
- core implementation under `normix/`
- active and skipped tests under `tests/`
- direct runtime probes for several mathematically delicate or user-facing paths

Validation performed:

- `uv run pytest tests/`
  - Result: `260 passed, 246 skipped, 1 warning in 386.07s`
- targeted runtime probes with the project virtualenv to check:
  - public quick-start examples
  - numerical behavior of `InverseGaussian.cdf`
  - behavior of `_subordinator_log_partition` for Normal-Inverse-Gamma
  - `BatchEMFitter._fit_scan` iteration counting

## Executive summary

The package has a strong core idea and a real technical center of gravity: the GH/GIG exponential-family formulation, the log-partition triad, and the CPU/JAX split around Bessel-heavy code are thoughtful and unusually well-motivated. The active test suite also gives decent confidence in the main batch-EM and GH/GIG paths.

The main concern is that the repository is still in a migration state, and that migration boundary now leaks into correctness, API coherence, and documentation quality. The code that looks most mature is:

- univariate exponential-family distributions
- GIG / Bessel infrastructure
- batch EM for the mixture families

The code that looks materially less trustworthy is:

- `OnlineEMFitter` and `MiniBatchEMFitter`
- public documentation and quick starts
- parts of the nominal exponential-family surface for joint distributions
- older but still-present tests and theory docs that describe the previous API

My overall assessment is:

- mathematically ambitious and often quite good
- strongest in the central GH/GIG batch-EM path
- not yet fully consistent as a package-level public library
- still carrying a significant amount of migration debt

## Highest-priority findings

### [P1] `OnlineEMFitter` and `MiniBatchEMFitter` do not implement the algorithms their docstrings claim

Files:

- `normix/fitting/em.py:359`
- `normix/fitting/em.py:406`
- `normix/fitting/em.py:447`
- `normix/fitting/em.py:496`

Why this matters:

- `OnlineEMFitter` claims Robbins-Monro online EM with step size `1 / (tau0 + t)`, but the computed `step` is never used.
- `MiniBatchEMFitter` claims Robbins-Monro averaging of sufficient statistics, but it just runs an ordinary batch-style `e_step`/`m_step` on each sampled minibatch.
- `MiniBatchEMFitter.tau0` is stored but never used.
- `OnlineEMFitter` also returns `converged=True` unconditionally (`normix/fitting/em.py:438`), even though it never checks a convergence criterion.

Impact:

- these classes are algorithmically mis-specified, not just under-tested
- users will get behavior different from the advertised estimators
- results are difficult to interpret scientifically because the estimator being run is not the estimator being documented

Recommendation:

- either implement genuine stochastic sufficient-statistic averaging
- or rename these classes to reflect what they actually do
- add dedicated tests that verify dependence on `tau0` and compare trajectories against batch EM on synthetic data

### [P1] `InverseGaussian.cdf` is numerically unstable and returns `NaN` in high-shape regimes

File:

- `normix/distributions/inverse_gaussian.py:129`

What I observed:

- direct probe: `InverseGaussian(mu=1.0, lam=1000.0).cdf(1.0)` returned `NaN`
- SciPy reference for the same case is about `0.5063062555`

Why it happens:

- the implementation uses `exp(2 * lam / mu) * Phi(-t2)`, which overflows for large `lam / mu`

Impact:

- this is a real mathematical correctness bug, not just a style issue
- the current formula is fine in moderate regimes but fails in precisely the regimes scientific users often care about for concentration

Recommendation:

- rewrite the second term in log-space
- use a numerically stable normal-CDF formulation such as `log_ndtr`
- add extreme-regime tests, not just moderate-parameter comparisons

### [P1] Nearly half the collected test suite is skipped because it still targets the old API

Representative files:

- `tests/test_exponential_family.py:14`
- `tests/test_distributions_vs_scipy.py:29`
- `tests/test_variance_gamma.py:17`
- `tests/test_normal_inverse_gamma.py:19`
- `tests/test_normal_inverse_gaussian.py:18`
- `tests/test_generalized_hyperbolic.py:29`
- `tests/test_em_regression.py:9`

Observed test outcome:

- `260 passed`
- `246 skipped`

Why this matters:

- the green suite materially overstates the confidence level of the package
- many of the skipped files are exactly the ones that would exercise public API expectations, multivariate behavior, round-trips, and regression scenarios
- the current suite validates the modern core, but leaves large parts of the advertised package surface weakly defended

Recommendation:

- treat the migration of skipped tests as a first-class engineering task
- prioritize:
  - public API examples
  - multivariate normal API expectations
  - joint distribution exponential-family round-trips
  - regression tests for EM convergence behavior

### [P2] The `lax.scan` EM path reports `n_iter` incorrectly

File:

- `normix/fitting/em.py:218`

Why this matters:

- `_fit_scan` computes:
  - `valid_mask = param_changes > 0`
  - `n_iter = sum(valid_mask) + (1 if converged else 0)`
- this double-counts the terminating iteration whenever convergence is achieved inside the scan

Observed behavior:

- in a direct probe of the scan path, a run that effectively stopped after one live update reported `n_iter = 2`

Impact:

- convergence diagnostics are inaccurate
- anything downstream that compares solver efficiency between scan and loop paths will be biased

Recommendation:

- store an explicit per-iteration active mask or stop index in the scan carry
- do not reconstruct iteration count from `param_changes > 0`

### [P2] Public examples and theory docs are stale or broken

Files:

- `README.md:29`
- `README.md:94`
- `README.md:101`
- `normix/__init__.py:15`
- `docs/theory/em_algorithm.rst:405`

Concrete problems:

- `README.md:29` uses `normix.fitting.em.BatchEMFitter(...)` without importing `normix`
- `README.md:94` uses `JointGeneralizedHyperbolic` without importing it
- `README.md:102` refers to `m_step_solver`, but the implementation uses `m_step_backend` / `m_step_method`
- `normix/__init__.py:15` shows `GeneralizedHyperbolic.fit(X, key=key, ...)`, but `fit` is an instance method; direct probe raises `TypeError`
- `docs/theory/em_algorithm.rst:405-412` still describes old methods like `joint.set_expectation_params` and `_expectation_to_natural`

Impact:

- first-use experience is currently brittle
- docs no longer describe the code users actually import
- this undermines trust even when the core math is sound

Recommendation:

- make all README and module-docstring examples executable in CI
- update theory docs to current method names or explicitly mark them historical

### [P2] The exponential-family contract is not consistently implemented across the package

Files:

- `docs/ARCHITECTURE.md:39`
- `README.md:68`
- `normix/distributions/variance_gamma.py:134`
- `normix/distributions/normal_inverse_gamma.py:135`
- `normix/distributions/normal_inverse_gaussian.py:143`
- `normix/distributions/generalized_hyperbolic.py:165`
- `normix/distributions/normal.py:34`

Why this matters:

- the docs say every distribution subclasses `ExponentialFamily`
- they also imply the exponential-family constructors are generally available
- in the implementation:
  - all four joint mixture families raise `NotImplementedError` for `from_natural`
  - `MultivariateNormal` does not subclass `ExponentialFamily` at all

My assessment:

- this is less a local bug than an architectural contract mismatch
- if joint families are internal-only, the docs should say that clearly
- if they are true public exponential-family objects, their parameterization round-trip should exist

Recommendation:

- pick one of these two positions and make the package consistent:
  - "joint classes are internal helpers"
  - "joint classes are full public exponential-family distributions"

### [P2] `JointNormalInverseGamma._subordinator_log_partition` is incorrect and returns `NaN`

File:

- `normix/distributions/normal_inverse_gamma.py:50`

What I observed:

- a direct probe of `_subordinator_log_partition` for valid inverse-gamma parameters returned `NaN`
- the equivalent `InverseGamma(...).log_partition()` returned the correct finite value

Cause:

- it passes `theta = [-beta_ig, -(alpha_ig + 1)]`
- but `InverseGamma` uses natural parameters `[beta, -(alpha + 1)]`

Impact:

- today this appears to be dead or unused code
- but dead code that is mathematically wrong is still a maintenance hazard and a sign of drift

Recommendation:

- either fix it and cover it with tests
- or delete it if the architecture no longer needs it

## Category review

### Mathematical rigor

Strengths:

- The central GH/GIG formulation is mathematically coherent and better-than-average for an applied stats package.
- The log-partition triad is an elegant abstraction. It keeps the mathematical source of truth in one place while still making room for analytical and CPU-specialized overrides.
- The GIG implementation is the strongest mathematical component in the repo. The rescaled `eta -> theta` strategy, Bessel-ratio expectations, and CPU/JAX dual path show real numerical awareness.
- The special-case families (VG, NIG, NInvG) generally do use their own formulas rather than lazily delegating everything through GH. That matches the design philosophy in `AGENTS.md`.

Concerns:

- The Inverse Gaussian module docstring has the wrong sign for the log-partition (`normix/distributions/inverse_gaussian.py:23-24`). The code is correct; the math in the header is not.
- The package currently has a split personality between mathematically rigorous core code and mathematically stale surrounding documentation.
- The numerical story is strong in the Bessel-heavy core, but weaker in some "standard" functions like `InverseGaussian.cdf`, which currently fails in concentrated regimes.

Suggestions:

- add a small "mathematical invariants" test layer for each exponential-family distribution:
  - gradient of `psi` equals expectation parameters
  - Hessian is symmetric positive semidefinite in valid regions
  - direct density matches SciPy over moderate and extreme regimes
- add explicit extreme-parameter tests for all functions that use exponentials outside log-space

### Code correctness

What looks good:

- Active tests exercise the modern GIG/Bessel/solver stack reasonably well.
- GH, NIG, VG, and NInvG special-case log densities appear internally consistent.
- The CPU/JAX E-step agreement tests are a real strength.

What worries me:

- The online and minibatch fitters are not just under-tested; they are not implementing what they claim.
- The batch-EM scan path has incorrect iteration accounting.
- Public examples fail even though core internals pass.
- A significant part of correctness confidence is illusory because the skipped tests are exactly the ones that used to test public behavior.

Suggested priority order:

1. fix `InverseGaussian.cdf`
2. either implement or temporarily remove/deprecate `OnlineEMFitter` and `MiniBatchEMFitter`
3. repair scan-path diagnostics
4. restore migrated tests around public API and regression behavior

### Package architecture and high-level design

Strengths:

- The separation between model math (`NormalMixture`, `JointNormalMixture`) and optimization orchestration (`BatchEMFitter`) is good.
- The CPU/JAX split is pragmatic and grounded in measured behavior, not hand-wavy "GPU everywhere" assumptions.
- The design around immutable `eqx.Module` objects is appropriate for this domain.

Weaknesses:

- The public API boundary is not crisp.
  - Some classes are presented as public mathematical objects but only partially implement the advertised contract.
  - `MultivariateNormal` sits outside the exponential-family architecture even though the docs flatten that distinction.
- The repository is carrying both the new architecture and too many textual references to the old one.
- Global JAX configuration is applied in many modules instead of once at a clear entry point.

Suggestions:

- make the public surface explicit:
  - which classes are stable public API
  - which classes are internal implementation machinery
- centralize `jax_enable_x64` setup
- define one documentation truth source for the current API and delete or quarantine stale descriptions

### Detailed design review by module

#### `normix/exponential_family.py`

Strengths:

- strong conceptual center
- good abstraction level for the log-partition triad
- the generic `from_expectation` solver bridge is clean

Concerns:

- the base-class contract is stronger than what some subclasses actually deliver
- the class currently acts as both a rigorous abstraction and a "best effort" interface; that tension leaks into the joint distributions

Suggestion:

- either relax the public claims in the docs or complete the missing round-trip constructors for joint families

#### `normix/distributions/gamma.py`, `inverse_gamma.py`, `inverse_gaussian.py`

Strengths:

- these are relatively clean and readable
- the analytical gradient/Hessian overrides are appropriate

Concerns:

- `InverseGaussian.cdf` needs a stable reformulation
- the Inverse Gaussian module docstring should be corrected to match the implemented math

#### `normix/distributions/generalized_inverse_gaussian.py`

Strengths:

- this is the package's standout module
- the balance of analytical formulas, asymptotics, CPU overrides, and solver integration is impressive

Concerns:

- it is complex enough that it deserves even more property-based and edge-case testing than it already has
- the module is carrying a lot of numerical responsibility; if this grows further, splitting CPU and JAX helper logic may help readability

#### `normix/mixtures/joint.py` and `normix/mixtures/marginal.py`

Strengths:

- the joint/marginal separation is the right architectural move
- the closed-form normal-parameter M-step is nicely centralized

Concerns:

- `_subordinator_log_partition` exists as an abstract hook, but the architecture around it no longer looks fully coherent because some implementations appear unused
- `NormalMixture.joint` lacks a return annotation (`normix/mixtures/marginal.py:40`), which is minor but symptomatic of a somewhat uneven public-surface polish

#### `normix/distributions/variance_gamma.py`, `normal_inverse_gamma.py`, `normal_inverse_gaussian.py`, `generalized_hyperbolic.py`

Strengths:

- good reuse of the common mixture scaffolding
- special-case marginal formulas seem consistent with GH limits

Concerns:

- the joint classes inherit `ExponentialFamily` but do not finish the advertised constructor surface
- several public methods lack return annotations
- GH default initialization is practical, but it is relatively heavy and exception-swallowing; it trades robustness for opacity

#### `normix/fitting/em.py`

Strengths:

- `BatchEMFitter` is well-structured overall
- the split between scan and Python loop is sensible

Concerns:

- `OnlineEMFitter` and `MiniBatchEMFitter` are the weakest part of the package
- scan-path `n_iter` reporting is wrong
- `OnlineEMFitter` reports convergence unconditionally

#### `normix/fitting/solvers.py`

Strengths:

- flexible and well-factored
- clear separation between theta-space math and solver-space transforms
- the bounded reparameterization is sensible

Concerns:

- `jaxopt` is now unmaintained, and the test suite already reports that warning
- that is not an immediate blocker, but it is a dependency-risk signal

#### `normix/utils/bessel.py`

Strengths:

- probably the most technically sophisticated utility in the package
- the JAX/CPU split is well-motivated

Concerns:

- public `log_kv` lacks a return type annotation (`normix/utils/bessel.py:249`)
- given how central this function is, executable documentation examples would be valuable

#### `normix/distributions/normal.py`

Assessment:

- useful, but architecturally out of band
- currently it is a lightweight helper rather than a fully integrated distribution type

Concerns:

- it does not match the rest of the package API very well:
  - no `rvs`
  - no `mean` / `cov` convenience surface matching the rest of the library
  - not an `ExponentialFamily`
- this would be fine if documented as a helper, but it is currently presented more like a peer to the other distributions

### Code style

Strengths:

- naming is generally mathematically disciplined
- the code is more readable than most research-style numerical libraries
- the separation between symbolic math and implementation is usually good

Concerns:

- public type annotations are incomplete in several places, for example:
  - `normix/utils/bessel.py:249`
  - `normix/distributions/generalized_hyperbolic.py:265`
  - `normix/distributions/generalized_hyperbolic.py:390`
  - `normix/distributions/variance_gamma.py:196`
  - `normix/distributions/normal_inverse_gaussian.py:198`
  - `normix/mixtures/marginal.py:40`
- `jax.config.update("jax_enable_x64", True)` is repeated across many modules instead of being centralized
- there are pockets of dead or drifted code, which is the main style smell in this repo

Suggestions:

- enforce return annotations on public methods
- centralize global JAX config
- delete unused compatibility leftovers rather than preserving them indefinitely

### Documentation

Strengths:

- the architecture docs are thoughtful
- the design rationale is unusually explicit
- the mathematical ambition of the project is reflected well in the theory folder

Concerns:

- several public examples are broken today
- theory docs still reference old API names
- README examples do not appear to be tested
- users currently need to infer which docs reflect the new architecture and which are historical leftovers

Suggestions:

- add a docs CI job that executes:
  - README code blocks
  - module-level doctest snippets
  - a very small subset of notebook smoke checks
- mark historical design notes and investigations as historical if they no longer describe the current API

### Packaging and metadata

File:

- `pyproject.toml:36-43`

Concerns:

- `jax[cuda12]` is a hard runtime dependency, which is unusually opinionated for a library package
- `matplotlib` is in core dependencies even though it appears to be notebook/support tooling, not core math
- this packaging story does not match the "core scientific library" shape suggested by the codebase

Suggestions:

- make the base install CPU-neutral
- move notebook/plotting dependencies to optional extras
- keep core runtime as small as possible

## Suggested roadmap

### Immediate

1. Fix `InverseGaussian.cdf` numerical stability and add extreme-regime tests.
2. Decide whether `OnlineEMFitter` and `MiniBatchEMFitter` are:
   - true supported algorithms to implement properly
   - or experimental placeholders to deprecate/remove
3. Fix broken README and `normix.__init__` examples.
4. Fix `BatchEMFitter._fit_scan` iteration accounting.

### Short term

1. Migrate the skipped tests that cover the current public API.
2. Remove or repair dead helpers like `JointNormalInverseGamma._subordinator_log_partition`.
3. Decide the public status of the joint distribution classes and align docs with that decision.
4. Add executable doc checks in CI.

### Medium term

1. Rationalize `MultivariateNormal` relative to the rest of the package API.
2. Centralize JAX runtime configuration.
3. Tighten packaging so optional tooling is not installed as a hard dependency.

## Bottom line

`normix` already contains a very solid mathematical and numerical kernel, especially around GIG/GH and the batch-EM workflow. The biggest gap is not the central theory; it is the package boundary around that theory: public examples, less-used fitters, migrated tests, and API consistency.

If the next round of work focuses on correctness of the peripheral paths and on cleaning up the migration debt, this could become a genuinely strong scientific library rather than a strong research codebase with a library-shaped surface.
