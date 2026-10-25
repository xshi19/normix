# normix software architecture: what TLA+ and Hypothesis can check

2026-10-24

Start with a small set of Hypothesis property tests for the existing code. TLA+ is worth learning; documentation publishing is its best current pilot. Distribution objects, parameter conversions, the Bessel kernel, and most fitting code mainly face numerical, domain, and interface risks. A separate TLA+ model would omit much of that detail and add a description to maintain.

TLA+ can help where applications involve crash recovery, repeated requests, concurrent writes, or work that must eventually finish. normix already has incremental EM. That does not make it a service with persistence, message acknowledgements, and a recovery protocol.

This report records a static review. No network access, dependency installation, Python tests, Lean builds, or TLC runs supported that review or this revision. All code blocks are **illustrative and unrun**. This revision changes only this report.

The evidence labels distinguish the following:

- **local check**: findings recorded from files at normix commit `51e053c57237a29c71b56d3a2a86988277477bd9`. Reading a test shows that it exists, not that it passed during the review.
- **Reported input**: the full CPU suite was reported to take about 71 minutes, and jax-lean was reported to have 12 commits. Neither figure was measured or checked again.
- **derivation**: mathematical test conditions derived here or taken from repository formulas. These are not machine-checked proofs.
- **to verify**: version-specific behavior or coverage that still needs checking. General descriptions of TLA+, TLC, PlusCal, Hypothesis, and alternative tools come from prior knowledge. External documentation was not checked.

CI means continuous integration: automatic checks after code changes. A PR is a proposed change submitted for merging. A backend is the selected computation path, such as `'jax'` or `'cpu'`. Automatic differentiation, or AD, computes derivatives from operation rules. A bundle is a result object holding several moments and derivatives from one computation.

## 1. Three kinds of checking, from first principles

### 1.1 TLA+: specify the allowed steps

TLA+ is a language for specifying system behavior. A **specification** states precisely what a system may do and what it must guarantee. Sets, functions, and logical expressions describe a **state machine**: states and the allowed transitions between them. A state can describe a fitting stage, iteration count, and stopping flag. It need not represent a database.

For documentation publishing, the state might record whether the old and new builds have finished and which version the website shows. The two jobs can finish in either order. Each allowed order produces an **execution trace**, a sequence of states.

Three common TLA+ names are:

- `Init`: the allowed initial states. For example, neither job has started and the website version is 0.
- `Next`: the allowed changes in one step. A job might finish building or publish its artifact.
- `Spec`: the complete behavior specification. A common form is `Init /\ [][Next]_vars`. Here, `vars` collects the state variables and `[]` means "always". The brackets permit a step that leaves the state unchanged, called **stuttering**. Waiting outside the model should not make a behavior invalid.

A prime marks the next state. For example, `version' = 2` constrains the next version to 2; it is not a Python assignment. `UNCHANGED version` leaves that variable unchanged. `/\` and `\/` mean "and" and "or", respectively.

An **invariant** is an assertion that should hold in every reachable state. One example is that the displayed version never falls below the highest version already published. **Safety** means that a bad event never occurs. Invariants are a common way to express safety. **Temporal properties** describe whole traces. "Every started publication eventually finishes" uses "eventually", written `<>`. A property that guarantees progress is a **liveness** property.

A system can avoid publishing the wrong version by never publishing anything. Safety therefore does not imply liveness. Eventual completion usually needs a **fairness assumption**: the scheduler cannot forever ignore a step that remains enabled. Fairness is an assumption of the model, not a proof about GitHub, an operating system, or a network.

**TLC** is a model checker for TLA+. With finite sets of versions, jobs, and other state components, it can explore reachable states and transitions. It checks invariants and supported temporal properties. Liveness checking can inspect cycles in the state graph without executing infinitely many steps. A failure produces a counterexample trace.

A completed exhaustive check of a finite model provides more coverage than a few randomly chosen schedules. Checking two jobs does not prove correctness for every number of jobs. Random simulation also lacks the exhaustive guarantee. TLA+ has a deductive proof route, but a finite TLC check is not that kind of proof.

**PlusCal** is an algorithm language embedded in TLA+ files. It resembles labeled pseudocode and translates to TLA+. It helps describe loops and multiple processes. It does not automatically translate normix Python into a model. The modeler must still decide how labels divide **atomic steps**: steps that no other job can interrupt in the model. Steps that are too large can hide races.

A hand-derived counterexample shows the purpose:

```text
Website version 0
New job finishes building -> new job publishes version 2
Old job finishes building -> old job publishes version 1
Website returns to version 1
```

Both builds succeed. No numerical error occurs. The failure comes from the allowed order. Section 5 gives a complete small model.

TLC checks the **model**. It does not read the actual workflow, Python branches, or deployment implementation. A passing model still needs a mapping from each modeled step to implementation steps. A formal proof of that correspondence is a **refinement proof**. This report contains no such proof. A retry path missing from the model, or an omitted crash, can break the correspondence.

### 1.2 Hypothesis: generate inputs and call real code

Hypothesis is a Python tool for **property-based testing**. An ordinary test supplies specific inputs and expected answers. A property test supplies an input class and a relation that should hold for every generated case. The tool generates concrete inputs and calls the real function.

Hypothesis is not a uniform random sampler. It combines generation, boundary exploration, saved counterexamples, and search. Here, "generative sampling checks" distinguishes it from exhaustive finite model checking.

Three core terms are:

- **Property**: an executable assertion. For example, converting valid Gamma parameters to natural parameters and back should recover the original values.
- **Strategy**: a description of how to generate inputs, such as `st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False)`.
- **Shrinking**: searching for a smaller, clearer failing input after a failure. It does not guarantee a globally smallest counterexample.

The example below calls `Gamma.natural_params()` and `Gamma.from_natural()`. In `normix/distributions/gamma.py`, $\alpha>0$ is the Gamma shape and $\beta>0$ is its rate. The natural parameters are $\theta=(\alpha-1,-\beta)$.

```python
import numpy as np
from hypothesis import given, strategies as st
from normix import Gamma

positive = st.floats(
    min_value=0.5, max_value=5.0,
    allow_nan=False, allow_infinity=False,
)

@given(alpha=positive, beta=positive)
def test_gamma_natural_roundtrip(alpha: float, beta: float) -> None:
    original = Gamma(alpha=alpha, beta=beta)
    restored = Gamma.from_natural(original.natural_params())
    np.testing.assert_allclose(
        [float(restored.alpha), float(restored.beta)],
        [alpha, beta], rtol=1e-10, atol=1e-12,
    )
```

This assumes float64 is enabled as in `tests/conftest.py`. The tolerances apply to this algebraic round trip away from boundaries. They are not suitable defaults for GIG inversion.

Deliberately removing the addition of 1 when `from_natural` recovers $\alpha$ should expose the error for ordinary positive parameters. If both conversion functions contain matching errors, the round trip might still pass. Independent formulas, SciPy, or high-precision reference values are also needed. A reference implementation is often called an **oracle**. It supplies evidence for the answer; it is not infallible.

Hypothesis can also test operation sequences. `RuleBasedStateMachine` lets a test define operations, generate sequences, and check invariants after each step. It suits real APIs used through sequences such as update, read, and update again. It does not exhaust all schedules like TLC, and it does not automatically introduce thread interleavings.

For example, `SampleWeightedUpdate` must combine batches by sample count. Batches `[2]` and `[0, 0]` have combined mean $2/3$. Giving both batch means weight $1/2$ would produce 1. Section 4 gives a stateful test sketch using this class.

`settings` controls the test budget. `max_examples` sets the generated-example budget, not an absolute limit on calls including shrinking. `deadline` checks the time per example. A first JAX compilation can take much longer than later calls. Use `deadline=None` for these tests, then control total cost through CI job timeouts and duration reports.

`derandomize=True` helps reproduce CI runs. It does not promise the same trace across Hypothesis versions, JAX versions, and devices. Record versions and settings.

### 1.3 Where Lean fits

Lean is a proof assistant. Definitions, propositions, and proofs are checked by its kernel. Mathlib is its mathematics library. Lean can express statements over all distributions satisfying stated assumptions, such as the conditional variance formula under finite-moment assumptions.

Lean can also describe program states, but code semantics must be defined explicitly. Proving a real-valued formula does not prove JAX rounding, compilation, device placement, or random-number behavior. This report assumes no particular Bessel theorem is already in Mathlib. Coverage and lemma names remain **to verify**.

TLA+ behavior models, Lean mathematical proofs, and Hypothesis tests of real code can share a specification while checking different parts.

## 2. Comparison and selection

This table compares tool mechanisms from prior literature and tool knowledge. It does not report executed checks.

| Question | TLA+ with TLC | Hypothesis | Lean, for comparison |
|---|---|---|---|
| What is checked directly? | A hand-written or generated design model | Real Python/JAX code called by tests | Propositions and proofs over precise definitions |
| Strength of guarantee | Complete exploration of reachable behaviors within a finite scope, subject to the model, configuration, and checker | Finitely many generated checks; passing does not establish correctness for every input | A deductive conclusion under the given definitions, assumptions, and axioms |
| Main gap | Model-to-implementation correspondence and generalization beyond the checked size | Inputs and operation sequences not explored | Correspondence between mathematical or semantic definitions and actual execution |
| Useful for finding | Duplicate submissions, lost updates, wrong ordering, deadlock, starvation | Missing domain checks, indexing or shape errors, numerical boundaries, interface inconsistencies, state-update errors | Logical gaps, missing assumptions, false identities |
| Poor direct fit | Bessel accuracy and error bounds over continuous parameters | Liveness for arbitrary schedules and universal mathematical theorems | The full JAX/XLA execution chain without suitable semantics |
| Learning curve | State and temporal reasoning; syntax is not the main difficulty | Accessible to Python users; strategies and tolerances need care | Types, propositions, proofs, and the mathematical library |
| Small-pilot cost, estimated here | 1–2 person-days for modeling, plus 1–2 hours of specification review | 0.5–1 person-day for the first small group, plus about 1 hour of specification review | Allow learning time even for small algebraic propositions; no promise of complex analysis in a few days |
| Maintenance | Check the model after implementation or protocol changes | Maintain strategies, references, tolerances, versions, and runtime | Maintain definitions, library versions, and dependent proofs |
| Current fit for normix | Good for publishing; lower return in most other areas | High; works directly with pytest, JAX, and contract tests | A parallel mathematical workstream; does not replace code tests |

Selection rules for review:

1. If the requirement describes a function's output for an input, start with pytest or Hypothesis. For two backend choices, an exhaustive parametrized test is often simpler.
2. If the requirement depends on history, first try short sequences through the real API. Stateful tests usually suit cumulative counts, repeated calls, and recovery after exceptions.
3. If independent participants, retries, crashes, shared resources, or eventual progress matter, consider TLA+. It also applies to sequential programs, but iteration alone does not justify it.
4. If a mathematical result must hold over an infinite domain, consider Lean. First state the finite-moment, differentiability, integrability, and solver-accuracy assumptions correctly.

Architecture checks need not choose between TLA+ and Hypothesis. Dependency directions, forbidden calls, and complete public exports often need only deterministic source checks. jax-lean's `tests/test_architecture.py` checks import relationships to protect layer boundaries without a separate TLA+ state machine.

## 3. The current normix implementation

Paths below are relative to the repository root. Sources at the end provides file links. Write $\theta$ for natural parameters, $\eta$ for expected sufficient statistics, and $\psi$ for the log-partition function. Sufficient statistics are observation functions paired with parameters in an exponential-family density. These are statistical concepts, not a synchronization protocol between mutable objects.

### 3.1 Backend contracts: check real branches first

Decisions S9 and S11 in `dev-notes/design/design.md` require explicit backend choices to be respected. GIG inversion with `backend='jax'` must not hide a host callback. A **host callback** returns from JAX execution to a Python/CPU function. Moving the result back to a device does not make that solve native JAX.

Check four claims separately: numerical agreement, JIT support, differentiability, and the execution path. `jax.jit` compiles a traced computation. `jax.vmap` maps a single-observation function over a batch. A GPU output array does not show that no intermediate solve ran on the CPU. `backend='jax'` should preserve the caller's selected JAX device; it does not force GPU execution.

Relevant entry points include:

- `normix/exponential_family.py`: `expectation_params`, `fisher_information`, and `from_expectation`.
- `normix/utils/bessel.py`: `log_kv` and `log_kv_moments`.
- `normix/distributions/generalized_inverse_gaussian.py`: `GIG.from_expectation`.
- `normix/mixtures/marginal.py`: `NormalMixture.e_step`. `normix/fitting/em.py` passes the E-step and M-step backends.

CPU is an explicit, existing path. The CPU E-step can still use JAX for quadratic forms while GIG/Bessel calculations use NumPy. Its contract does not ban JAX throughout the stack. Verification convenience does not justify changing the selected backend.

`tests/test_no_host_callbacks.py` already includes a lexical scan and a recursive jaxpr check. The latter enters `while`, `cond`, `scan`, `jit`, and custom differentiation bodies. A **jaxpr** is JAX's traced intermediate computation representation. Extend the existing check for cold starts, warm starts, and batching. A new check that inspects only the top level would be weaker. These checks cover traced shapes and static configurations; they do not prove every device compiler correct.

**local check:** `log_kv_moments` rejects unknown backends with `ValueError`. The end of `log_kv` has only a `'cpu'` branch and a default JAX return. Thus, the source routes `log_kv(..., backend='typo')` to JAX. `GIG.expectation_params_batch` has similar "CPU, otherwise JAX" dispatch. The first is a clear public-contract check. The public scope of the second also needs confirmation. This is a static finding, not an executed reproduction or a fix.

A TLA+ model that holds the backend constant would not find this implementation gap. Prefer finite enumeration, invalid-argument tests, path sentinels, and jaxpr inspection. A **path sentinel** temporarily replaces a function that must not run and fails if it is called. Install it before tracing so a cached JIT program cannot bypass it. Runs without a GPU must record the absence of accelerator evidence.

### 3.2 EM: local state, usually without external scheduling

`BatchEMFitter.fit` centers data, selects a loop, runs E/M steps, and restores location. Its pure JAX batch path uses `lax.scan` when both backends are JAX, `algorithm='em'`, `verbose<=1`, and `eta_update` is absent. Other cases use a Python loop. The E-step, M-step, and regularization order can be read in `_em_step` and `_mcecm_step`.

`EMResult` has three distinct flags:

- `converged`: the batch EM Aitken stopping criterion holds. Aitken estimates the remaining improvement from three consecutive objective values. It does not prove a global optimum.
- `diverged`: a normal-parameter change or the objective becomes non-finite. The previous finite model is retained.
- `solver_failed`: mixing-distribution parameters are non-finite, or the Python path catches a corresponding solver exception.

All three flags can be false after budget exhaustion. Incremental EM sets `converged` to `None`. Therefore, `not converged` does not mean `diverged`. After stopping, `_fit_scan` freezes its output state. Its loop body still computes a candidate `_step` before selecting old values. An unchanged model after logical stopping does not imply that hardware performs no more computation.

`EStepResult.log_lik` is the average log-likelihood per observation for the input model of the E-step. The returned model usually includes one further M-step. Recompute its likelihood instead of treating `log_likelihoods[-1]` as its value. On Python failure paths, likelihood records can also be shorter than the number of attempted steps. "Every diagnostic array always has length `n_iter`" is not an established property.

A **warm start** uses existing parameters as the solver's initial point. A **cold start** constructs a new initial point. `GeneralizedHyperbolic.m_step_subordinator` passes the current GIG natural parameters as a warm start. Tests should check that this argument is passed correctly. They should not require identical warm-start and cold-start trajectories.

Distributions are immutable `eqx.Module` objects, but fitters are ordinary Python objects. Both `BatchEMFitter.fit` and `IncrementalEMFitter.fit` temporarily replace `self.eta_update` and restore it in `finally`. Sequential reuse and restoration after exceptions deserve tests. If two threads share one fitter, this field could create an interleaving. Support for that use has not been established, and no race was reproduced. Clarifying the sharing contract is usually cheaper than modeling all of EM.

A compilable `_fit_scan` body also does not make the entire Python `fit` wrapper unconditionally usable inside `jax.jit`. Result construction includes `int(n_iter)`, Boolean conversions, slicing, and timing. Tests should cover the promised transformation boundary.

### 3.3 Incremental EM and future persistence

`IncrementalEMFitter.fit(model, X, key=...)` already supports mini-batch updates. Rules in `eta_rules.py` have explicit state, such as `SampleWeightedUpdate.cumulative_n`. The JAX scan and Python loops use prearranged random subkeys. Their agreement can be tested for the same inputs and key.

The entry point receives the complete array `X`, computes its full-sample mean, and draws batches from it. It does not implement queue consumption, checkpoint recovery, or exactly-once processing. The theory in `docs/theory/online_em.md` makes no such implementation promise.

A future recoverable streaming service would justify a TLA+ model. Acknowledged data offsets must not exceed persisted updates. After recovery, the model, accumulated $\eta$, step count, rule state, and random key must belong to one checkpoint.

A **checkpoint** is a consistent snapshot sufficient to resume execution. Saving only the model, without the step-size count or rule state, cannot reproduce later updates. Full-sample centering would also need an explicit streaming convention. These are future design questions, not implemented features.

### 3.4 Immutable objects, parameter layouts, and derived results

`ExponentialFamily` stores classical distribution parameters. `natural_params()` and `expectation_params()` compute the other representations. `NormalMixture` holds a joint object, and `replace` uses `eqx.tree_at` to return a new object. A **pytree** is JAX's convention for decomposing objects into array leaves and structural information. Equinox makes these classes participate in that convention.

Main risks include wrong parameter slots, confusion between moments and natural parameters, dropped fields during conversion, and updates that alter the original object. These are code properties. The six fields of `NormalMixtureEta` follow the theoretical order. The GIG vector instead uses $E[\log Y],E[1/Y],E[Y]$. The mapping needs checking.

Do not apply $\nabla\psi=\eta$ uniformly to every joint class. Sections 2.4–2.5 of `docs/design/exponential_family.md` explain that Joint VG, NInvG, and NIG return moments in the larger GH statistic space. Gradients of the restricted partition function omit fixed slots. Moment inversion for these classes should use `NormalMixtureEta`. GH also has scale non-identifiability. Distributional equivalence does not always require identical classical parameter entries.

Distinguish three kinds of stored results. `QuantileTable` is an independent, immutable CDF/quantile table. `MeanRiskProblem.__init__` stores a decomposition and scenarios for the current model. JIT caches compiled programs. The first two are snapshots and should be rebuilt when the model changes. Replacing `MeanRiskProblem.model` with `eqx.tree_at` while retaining its old decomposition is not a valid update merely because objects are immutable.

First check that the original object stays unchanged, rebuilding gives consistent results, and tree flattening/unflattening preserves values. No current protocol involves multiple writers invalidating a shared cache. A TLA+ model has little value here.

### 3.5 GIG solvers: the actual fallback behavior

Dispatch occurs in `GIG._from_expectation_unchecked` and `fitting/solvers.py`:

| Condition | Behavior in the reviewed source |
|---|---|
| JAX, supplied `theta0`, Newton, low verbosity | Calls the cached `_gig_jax_newton_jit` |
| JAX, no `theta0` | `_jax_cold_start` tries ordered starts with the selected method and can stop at a finite, converged result |
| CPU, no `theta0` | Multi-start L-BFGS-B; this cold-start strategy is not a response to Newton failure |
| Other warm starts | `solve_bregman` dispatches by backend and method; CPU Newton uses SciPy `trust-exact` |
| Inside Newton | Damping and line-search alternatives remain within that backend's solve |
| GH CPU warm start raises an exception | `m_step_subordinator` currently uses `except Exception`, retains the previous GIG, and continues EM; the normal block may already have changed |
| GH finite result exceeds the allowed parameter range | Reuses the previous subordinator; non-finite JAX results remain available for outer failure reporting |

There is no universal automatic "Newton -> L-BFGS -> CPU" chain. If no solve succeeds, a multi-start solver can return its best finite candidate. The GIG construction entry point still checks whether to accept it. CPU multi-start control flow also differs from JAX's early exit after success.

**local check:** E17 and the method documentation emphasize non-convergence and `ValueError`, but the current `except Exception` catches more. It may also hide implementation errors. Confirm the intended exception scope before writing negative tests. This report neither adopts the broader catch as the contract nor claims an executed reproduction.

This sequential algorithm has a finite set of branches. Enumerate success, non-convergence, non-finite output, and out-of-range output, then test outer handling. A TLA+ model could abstract numerical results into these labels. It would not find a wrong Hessian, a wrong tolerance, or an actual call that ignores `backend`. Small controlled solver problems, failure injection, and property tests are the better current fit.

### 3.6 Finance, public APIs, and compilation boundaries

The reviewed `finance/` code performs risk calculations and optimization. It contains no broker order submission, order acknowledgements, shared position ledger, or concurrent trading service. Its constraints and mathematical transformations still need checks.

- `MeanRiskProblem.weights` should satisfy the budget and two projection constraints for reachable targets. Construction checks rank and reads values on the host. The entire constructor cannot be described as traceable. Unreachable-target exceptions also differ between eager and traced execution and need separate coverage. During tracing, JAX records computations with objects representing arrays; they cannot freely become Python scalars.
- `CVaR.value_w` and `value_grad_hess_w` should use the same scenario measure. Check positive homogeneity of degree one under positive scaling of holdings. CVaR is an average loss in the specified tail; follow the repository's sign convention. Finite moments and fixed scenarios are prerequisites.
- `TransactionCostProblem.solve` calls `solve_transaction_cost_qp`, which explicitly uses SciPy SLSQP. Its guarantee concerns a local quadratic approximation: the returned objective is no worse than holding unchanged. It does not guarantee global improvement of the true risk objective. This is an existing CPU optimization boundary.
- `DiscreteMeasure.from_weights` converts positive nodes and nonnegative weights into a discrete probability measure: nodes with their probabilities. `tail_index` records the original distribution's moment-existence threshold. A finite set of nodes must not conceal a divergent moment of that distribution.

Public APIs also need checks for single observations, batched output shapes, empty samples, invalid backends, and seed reproducibility. Record value agreement, `jit`, `vmap`, and AD separately. Some host checks are skipped during tracing, so eager and traced modes need not raise identical Python exceptions.

### 3.7 Publishing and concurrent workflows

Existing concurrency candidates are in `.github/workflows/`. After creating a release, `release-please.yml` explicitly dispatches `publish.yml` and `docs-full.yml` with the release tag. Both target workflows also declare tag-push and manual triggers. YAML alone does not establish that every release triggers duplicate runs. Token and platform rules affect event propagation; these remain **to verify**.

`docs.yml` groups concurrency by workflow/ref and permits cancellation of older runs in the same group. `docs-full.yml` uses `pages-release` and does not cancel active runs. Both write to `gh-pages`. The reviewed files show no shared deployment lock or common version-acceptance policy between these groups.

The last writer therefore presents a real protocol question. Decide whether the website follows master or the latest release. Those sources cannot be placed in a single arbitrary "newer" order.

Branch protection, environment approval, and required-check settings between CI, release, and publication cannot be inferred from these files alone. This report does not claim an observed production failure or prove platform settings correct.

`dev-notes/plans/loops_and_orchestration.md` describes planned nightly benchmarks, PR-triggered reviews, and other recurring workflows. One proposed constraint is at most one open item per recurring workflow. If independent jobs later retry item creation or accept evaluations and merge results from old commits, TLA+ could check duplication and stale-result acceptance. Current work can use separate paths, recorded commits, and clear ownership. A plan is not evidence of an operating concurrent service.

### 3.8 Expected return by area

These are estimates for future work. One person-day means about six hours of focused work. None of these times measures completed verification.

| Area | State/concurrency complexity | Current choice | Implementation work and required review | Initial effort |
|---|---|---|---|---|
| Backends, jaxpr, invalid configuration | Limited dispatch; no concurrency protocol | Deterministic contract tests with a few generated inputs | Extend existing checks; review S9/S11 execution promises | 0.5–1 day; 0.5–1 hour of review |
| Parameters, pytrees, snapshots | Mostly pure functions | Hypothesis | Write strategies and independent references; review domains and representation equivalence | 1–2 days; 1–2 hours of review |
| Batch EM and GIG failure handling | Local state; finite paths | pytest enumeration and stateful tests | Inject failures and compare paths; review stopping and fallback meanings | 1–2 days; 1–2 hours of review |
| Incremental rules | Ordered accumulation; no current persistence protocol | `RuleBasedStateMachine` | Test counts, weights, and key reproducibility; review statistical weighting | 0.5–1 day; about 1 hour of review |
| Finance | Mainly numerical constraints | Generate feasible problems and check residuals | Write scenario and transformation tests; review risk signs, moment domains, and objectives | 1–2 days; 1–2 hours of review |
| Documentation publishing | Independent jobs share a target | Decide policy, then try a small TLA+ model | Build state graphs and counterexamples; decide which version should win | 1–2 days; 1–2 hours of review |
| Recoverable streaming EM | Future retries and crashes | Model when a concrete need exists | Model recovery; define offsets, snapshots, and atomicity | About 2–4 days for a single-writer model; half a day of review; implementation separate |
| Concurrent recurring workflows | Mostly planned | Defer modeling | Identify shared resources; decide whether to build the service | Half a day to define scope once needed; full estimate pending |

## 4. A practical Hypothesis plan

### 4.1 Define input domains first

At the reviewed baseline, `pyproject.toml` does not include Hypothesis. Add it through uv as a development dependency when implementing this plan. It should not become a runtime dependency. This revision installs nothing and leaves the lockfile unchanged.

Initial strategies should derive target moments from valid distributions. Three uniformly generated real numbers are not automatically a valid GIG $\eta$. Necessary GIG conditions include $E[X]E[1/X]>1$ and $-\log E[1/X]<E[\log X]<\log E[X]$. Those conditions do not prove that every such triple can be inverted without error. First generate moderate $(p,a,b)$, calculate moments, and test inversion. Put invalid moments in separate negative tests.

Generate positive parameters on a log scale, for example $a=e^u$ with $u\in[-3,3]$. A bounded matrix $A$ gives a positive-definite covariance $AA^\top+\lambda I$ for $\lambda>0$, where $I$ is the identity matrix. Also bound its condition number so the first tests do not consist entirely of ill-conditioned problems. Fix a few dimensions and vary values within them.

Avoid rejecting most inputs with `assume`. Excessive filtering wastes generation effort and makes failures harder to interpret.

Separate **tolerance** sources: floating-point rounding, inversion residuals, quadrature truncation, table interpolation, and sampling error. First require expected finite quantities to be finite. Then compare with explicit `rtol` and `atol`. Two NaNs must not count as agreement. Expected divergent moments or NaN derivatives need separate properties. A looser tolerance that hides a failure is not a fix.

### 4.2 Twenty-five candidate properties

In the Bessel properties, $K_v(z)$ denotes the modified Bessel function of the second kind, with order $v$ and positive argument $z$.

This is a candidate list, not a proposal to add 25 full random-test groups at once. Extend existing files where possible and retain fixed regression cases. "Fast" means intended for the default PR set, subject to timing. Use the existing `slow`, `stress`, `integration`, and `gpu` markers for expensive checks, wide domains or large samples, real-data workflows, and real accelerator execution. Use `contract` separately to mark importance.

| # | Entry point and property | Assumptions, references, and test tier |
|---|---|---|
| 1 | `natural_params` followed by `from_natural` recovers classical parameters for `Gamma`, `InverseGamma`, `InverseGaussian`, and `GIG` | Valid interior parameters, away from constructor clipping boundaries; independently check parameter formulas. Fast, `contract` |
| 2 | `expectation_params` followed by `from_expectation` recovers $\theta$ for those classes, with a recomputed $\eta$ residual | Generate from valid models. Separate GIG cold and warm starts; perturb warm starts moderately instead of using only the exact solution. Moderate Gamma/IG cases fast; GIG starts in `slow` |
| 3 | `ExponentialFamily.expectation_params` and `fisher_information` agree with the gradient and Hessian of the class's `_log_partition_from_theta` | Applicable full exponential families, including Gamma, IG, and interior GIG. Exclude restricted joint VG/NInvG/NIG. Use scale-aware symmetry and positive-semidefiniteness tolerances. GIG slow |
| 4 | `NormalMixture.compute_eta_from_model` followed by `m_step_normal` recovers the normal parameter block; the pytree path of `NormalMixture.from_expectation` recovers the same distribution | Use VG/NInvG/NIG domains with the required finite moments. Compare GH densities or moments because of scale freedom. Check the six-field to three-field GIG ordering. Split fast/slow by family |
| 5 | `NormalMixture.replace` leaves every original array leaf unchanged; the new object changes only requested parameters and their derived values | Include `sigma` to `L_Sigma` conversion and subordinator fields. Save value snapshots, not just object identities. Fast |
| 6 | Repeated calls and independent rebuilds of `GIG.quantile_table`, `InverseGaussian.quantile_table`, and `QuantileTable.cdf/ppf/rvs` agree | A table is a snapshot of a fixed model. Rebuild after model updates; an old table does not update itself. Also compare with independent density/CDF references. Construction slow; reused queries fast |
| 7 | `log_kv(v,z)` satisfies $\log K_{-v}(z)=\log K_v(z)$ | $z>0$; begin with moderate ranges and check both backends separately. Add generated inputs to existing Bessel contracts. Fast |
| 8 | `log_kv` satisfies $K_{v+1}(z)=K_{v-1}(z)+(2v/z)K_v(z)$ | $z>0$; initially restrict to $v>0$ and compare with log-sum-exp to avoid overflow from exponentiation. Test $v=0$ separately. Wide-domain `stress` |
| 9 | `log_kv_moments` values and first/second derivatives agree with mpmath references; `jax.grad/hessian(log_kv)` agrees with the bundle | Reuse `tests/data/bessel_reference.json` in PR checks. Generate new high-precision points only in later nightly checks, with doubled-precision self-checks. Retain component-specific tolerances from `test_bessel_contract.py`. New points slow |
| 10 | `Gamma.pdf`, `InverseGamma.pdf`, and `GIG.pdf` integrate approximately to 1 on sufficiently wide, fine grids | On the positive half-line, use $x=e^u$ and integrand $f(e^u)e^u$, where $f$ is the density. Compare wider and finer grids with an independent tail bound or reference integral. A normalized PINV table is not evidence of PDF normalization. Slow |
| 11 | `Gamma.cdf`, `InverseGaussian.cdf`, and `GIG.cdf` are monotone and in $[0,1]$; corresponding `ppf` methods approximately invert interior probabilities | Sort input points and test CDF values outside support separately. Table-based `cdf(ppf(q))` checks internal consistency, not accuracy; use SciPy or another independent reference. Analytical families fast; GIG slow |
| 12 | Sample first and second moments from `Gamma.rvs`, `GIG.rvs`, and `NormalMixture.rvs` agree with `mean/var/cov` | Required higher moments must exist and tails must be moderate. Use the statistical budgets below, not a universal 5%. Fix sample lengths and record seeds. `slow, stress` |
| 13 | `Gamma.rvs`, `GIG.rvs`, `JointNormalMixture.rvs`, and `NormalMixture.rvs` have correct shapes and reproduce the same seed | Fix version, method, and device. Positive-support samples are positive; joint sampling returns `(X,Y)`. Different seeds need not always produce different outputs, and differing outputs do not prove independence. Small samples fast |
| 14 | CPU/JAX values agree for `log_kv/log_kv_moments`, `GIG.expectation_params/fisher_information`, and `NormalMixture.e_step` | Same valid parameters and dtype. Compare $\eta$, likelihood, and derivatives separately. Shared algorithms prevent the two backends from being each other's sole accuracy reference. Kernels fast; complete E-step slow |
| 15 | `GIG.from_expectation`, `ExponentialFamily.expectation_params`, `BatchEMFitter.__init__`, and both Bessel entry points respect backends and reject invalid names | Exhaust both valid choices and use path sentinels. The invalid-name check for `log_kv` is expected to expose Section 3.1's gap. Mark it "to reproduce", not passed. Fast |
| 16 | Cold-start, warm-start, and `vmap` jaxprs for `GIG.from_expectation(backend='jax')` contain no host callbacks | Reuse the recursive traversal in `test_no_host_callbacks.py`. Confirm that solver loops appear in the graph so an empty wrapper cannot pass. Check each static configuration once, not 200 times at the same shape. `contract` |
| 17 | `Gamma.log_prob`, `GIG.from_expectation`, and `NormalMixtureEta` agree under eager/jit/vmap within promised boundaries; tree flattening/unflattening preserves values and structure | Batch single-observation `log_prob` through `vmap`. Check AD separately against S11; do not require differentiable CPU constructors. Small models fast; GIG and real devices slow or `gpu` |
| 18 | Ordinary EM updates through `NormalMixture.e_step` then `m_step`, and through `BatchEMFitter.fit`, do not decrease the objective beyond explicit error | Fixed data, no shrinkage, no active boundary clipping, and sufficiently accurate M-steps. Start with NIG, then GH. Check each actual updated model without misaligning the E-step trace. `slow` |
| 19 | Stopping, failure, and budget exhaustion in `BatchEMFitter.fit` agree with `EMResult`; exceptions restore the fitter's `eta_update` | Inject finite, non-finite, and exceptional outputs through a controlled `_step` or test double, then compare with a small real model. Check model/count freezing after stopping, not absence of computation. Control checks fast; integration slow |
| 20 | `SampleWeightedUpdate.__call__` combines by cumulative sample count; scan/Python paths of `IncrementalEMFitter.fit` agree for a fixed key | Test pure-rule batch sequences first. This does not imply that repartitioning full EM batches preserves the final model. Full incremental fitting is a separate slow check; `converged is None` |
| 21 | Endpoint weights in `affine_combine` preserve the corresponding statistics; `Shrinkage(base, eta0, tau=0)` equals the base rule; scalar and equal-valued per-field $\tau$ paths agree | Valid nonnegative weights and finite moments. Compare returned rule state as well as $\eta$. Use nonzero statistics to avoid vacuous passes. Fast |
| 22 | `NormalMixture.regularize_det_sigma` and `GeneralizedHyperbolic.regularize_a_eq_b` preserve marginal density/moments and are idempotent | Applying twice should leave the result unchanged. Avoid scaling that triggers `GIG_CLAMP` or other clipping. Check the target determinant or $a=b$. Do not claim pointwise joint-density invariance at fixed latent coordinates. Split fast/slow by family |
| 23 | `DiscreteMeasure.from_weights` gives nonnegative probabilities summing to 1; positive weight scaling preserves the measure; joint node/weight permutation preserves expectations | Reject all-zero weights; allow individual zero weights. `coerce` preserves an existing measure and its `tail_index`; GIG `quadrature` preserves the original tail index. Small measures fast |
| 24 | `MeanRiskProblem.weights` satisfies $\mathbf1^\top w=1$, $\mu^\top w=\tilde\mu$, and $\gamma^\top w=\tilde\gamma$; `NormalMixture.project(w)` mean/variance match projection formulas | Generate reachable targets at ranks 1/2/3. Separately check eager errors for unreachable targets. Here $w$ is holdings, $\mathbf1$ is the all-ones vector, $\mu,\gamma$ are model location and skew vectors, and $\tilde\mu,\tilde\gamma$ are reduced targets. Reuse fixed small scenarios. Fast or slow |
| 25 | `TransactionCostQP.weights_from_v` preserves the self-financing budget; `solve_transaction_cost_qp` returns a feasible result with approximate objective no lower than `hold_objective` | Generate small problems where unchanged holdings are feasible. Failures should raise explicitly. Check original constraint residuals. Do not require improvement of the true objective or accept infeasible unchanged holdings as fallback. `slow` |

Component consistency and positive homogeneity for `CVaR.value_grad_hess_w` can extend `tests/finance/test_cvar.py` using generators from items 23–25. Begin with fixed scenarios and positive weights so Monte Carlo noise does not obscure numerical derivatives. Include this in the finance phase, not the first PR checks.

### 4.3 Three properties that need particular care

**EM monotonicity — derivation.** Let $\ell_t$ be the average log-likelihood of model $t$ on a fixed sample. A test can require $\ell_{t+1}-\ell_t\ge-(\varepsilon_a+\varepsilon_r\max(1,|\ell_t|))$. The absolute and relative budgets $\varepsilon_a,\varepsilon_r$ need calibration against reference calculations. This report gives no unmeasured tolerance for every family.

The ordinary exact-EM monotonicity argument comes from prior literature knowledge and repository theory. It does not automatically cover approximate M-steps, posterior moment floors, parameter clipping, mini-batches, or arbitrary shrinkage. Scalar MAP shrinkage should check a penalized objective; MAP means maximum a posteriori estimation. General per-field rules need not correspond to one scalar objective. Rejecting every failing example with `assume` does not establish monotonicity.

**Sampling moments — derivation.** For $n$ independent identically distributed samples, let $\bar X$ be the sample mean, with theoretical mean $\mu$ and variance $\sigma^2$. If the variance is finite, Chebyshev's inequality gives $\Pr(|\bar X-\mu|>\sqrt{\sigma^2/(n\delta)})\le\delta$. Here, $\delta\in(0,1)$ is the permitted false-alarm probability for that assertion.

This bound is conservative. Staying inside a wide bound does not establish sampling quality. For Gamma samples, the known distribution of their sum permits a stronger check. That distributional fact comes from prior knowledge; confirm the parameter convention before use.

A second-moment check needs a fourth moment. Define $M_2=n^{-1}\sum_{i=1}^n X_i^2$. Use $\operatorname{Var}(X^2)=E[X^4]-E[X^2]^2$ to bound its error. If $|\bar X-\mu|\le r_1$ and $|M_2-E[X^2]|\le r_2$, then the sample variance with denominator $n$ differs from $\sigma^2$ by at most $r_2+r_1(2|\mu|+r_1)$.

This gives the tolerance a statistical basis. At heavy-tail boundaries without a fourth moment, use bounded-interval frequencies or quantile tests instead of a Gaussian approximation.

More parameters, coordinates, and backends increase false-alarm opportunities. Assign a total budget to the main assertions, distribute the individual $\delta$ values, and record the actual number of checks. A fixed seed makes failures reproducible; it does not remove false alarms. Preserve failing inputs and seeds instead of changing seeds until tests pass. These bounds assume ideal independent sampling and do not prove independence of a pseudorandom generator.

**PDF normalization and Bessel references.** Two refined integration grids can agree while both miss the same narrow peak. Bound the distribution scale, locate the peak, and check tails. `build_pinv_table` divides cumulative integrals by the final value. Its CDF ending at 1 therefore does not verify the PDF normalization constant.

Bessel identities and backend agreement can also share an error. Frozen high-precision tables provide another source of evidence. mpmath is a high-precision numerical reference, not a formal proof or an automatic rigorous interval bound.

### 4.4 Settings sketch and JAX costs

This is a proposed addition to `tests/conftest.py`. Preserve existing fixtures and x64 settings rather than replacing the file. The Hypothesis API details come from prior knowledge and remain **to verify** for the chosen version.

```python
import os

import jax
from hypothesis import settings

jax.config.update("jax_enable_x64", True)

settings.register_profile(
    "normix_ci",
    max_examples=20,
    stateful_step_count=12,
    deadline=None,
    derandomize=True,
    database=None,
)
settings.register_profile(
    "normix_nightly",
    max_examples=200,
    stateful_step_count=50,
    deadline=None,
    derandomize=False,
    database=None,
)
settings.load_profile(
    os.environ.get("NORMIX_HYPOTHESIS_PROFILE", "normix_ci")
)
```

Disabling the counterexample database keeps CI independent of a historical cache. Failures must then be extracted from logs and retained as fixed regression cases or `@example` inputs. Whether to enable a local database can be decided separately. Do not overwrite system paths such as `$HOME` through environment variables.

JAX adds several constraints:

- Enable x64 before importing test modules that create arrays. `tests/conftest.py` already does this. Test float32 in a separate process with its own tolerances; do not switch global configuration between examples.
- JIT compilation usually depends on shape, dtype, and static arguments. Randomizing shapes, `maxiter`, or method strings can trigger many compilations. Fix these structures first and generate array values.
- Define compiled functions at module scope. Creating a new `jax.jit(lambda ...)` that captures new parameters for every example can turn compilation cost into a misleading solver timing.
- `np.asarray`, scalar reads, and `block_until_ready` can wait for asynchronous results. Performance measurements must wait for completion. Timing only the return from a call can omit device execution.
- pytest fixtures prepare and clean up test environments. A function-scoped fixture is generally not rebuilt for every Hypothesis example. `monkeypatch` temporarily replaces behavior; restore replacements and state per example. Do not simply disable the related health check. Health checks flag problems in generation or test structure.
- Verbosity can switch EM between Python and scan paths. Fix other settings and random keys when comparing paths. A CPU reference must not replace the selected JAX execution path.
- `deadline=None` disables only the per-example time limit. It does not make a test fast or justify disabling all health checks. Investigate recompilation before suppressing a warning, and record any local exception.

### 4.5 An ordinary property-test file sketch

The proposed file is `tests/test_property_contracts.py`; this revision does not create it. The first property calls real Gamma conversions. The second compiles and reuses a Bessel symmetry check. Ranges and tolerances are initial trial values, not calibrated accuracy promises.

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, strategies as st

from normix import Gamma
from normix.utils.bessel import log_kv

pytestmark = pytest.mark.contract

positive = st.floats(
    min_value=0.5, max_value=5.0,
    allow_nan=False, allow_infinity=False,
)
orders = st.floats(
    min_value=-5.0, max_value=5.0,
    allow_nan=False, allow_infinity=False,
)

@jax.jit
def bessel_pair(v: jax.Array, z: jax.Array) -> jax.Array:
    return jnp.stack((log_kv(v, z), log_kv(-v, z)))

@given(alpha=positive, beta=positive)
def test_gamma_roundtrip(alpha: float, beta: float) -> None:
    original = Gamma(alpha=alpha, beta=beta)
    restored = Gamma.from_natural(original.natural_params())
    got = np.array([float(restored.alpha), float(restored.beta)])
    assert np.isfinite(got).all()
    np.testing.assert_allclose(
        got, [alpha, beta], rtol=1e-10, atol=1e-12,
    )

@given(v=orders, z=positive)
def test_bessel_symmetry(v: float, z: float) -> None:
    pair = np.asarray(bessel_pair(
        jnp.asarray(v, dtype=jnp.float64),
        jnp.asarray(z, dtype=jnp.float64),
    ))
    assert np.isfinite(pair).all()
    np.testing.assert_allclose(
        pair[0], pair[1], rtol=1e-10, atol=1e-11,
    )
```

These properties avoid simply copying an implementation formula into the test. They can still miss matching errors that cancel. Retain existing SciPy comparisons, high-precision tables, and fixed regression points.

### 4.6 A stateful test sketch for a real update rule

This sketch uses `SampleWeightedUpdate` to combine one-dimensional statistics. Arrays are pytrees, so a scalar can test cumulative weighting before extending to `NormalMixtureEta`. Each action supplies a batch of integers. Their integer sum provides an independent reference without copying the class's recurrence.

```python
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from normix.fitting.eta_rules import SampleWeightedUpdate

class WeightedMeanMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.updater = SampleWeightedUpdate()
        self.state = self.updater.initial_state()
        self.eta = jnp.asarray(0.0, dtype=jnp.float64)
        self.total = 0
        self.count = 0
        self.step = 0

    @rule(batch=st.lists(st.integers(-5, 5), min_size=1, max_size=4))
    def append_batch(self, batch: list[int]) -> None:
        values = jnp.asarray(batch, dtype=jnp.float64)
        self.eta, self.state = self.updater(
            self.eta, jnp.mean(values), self.step, len(batch), self.state,
        )
        self.total += sum(batch)
        self.count += len(batch)
        self.step += 1

    @invariant()
    def agrees_with_raw_data(self) -> None:
        assert int(self.state["cumulative_n"]) == self.count
        if self.count:
            np.testing.assert_allclose(
                float(self.eta), self.total / self.count,
                rtol=1e-12, atol=1e-12,
            )

TestWeightedMean = WeightedMeanMachine.TestCase
TestWeightedMean.settings = settings()
TestWeightedMean.pytestmark = [pytest.mark.contract]
```

This example is unrun. pytest collection and stateful settings integration remain **to verify**. A test case can contain several actions. `stateful_step_count` and `max_examples` budget sequence length and sequence count, respectively. Shrinking can remove batches or reduce elements to expose a trigger such as unequal batch sizes.

The test checks the actual cumulative rule. It does not establish streaming recovery, concurrent safety, or equality of full EM fits under different batch partitions. An EM E-step depends on the current model, so full EM generally lacks that partition invariance.

### 4.7 Fit the existing test-cost work

The reported 71 minutes refers to the full CPU suite. It is not a measurement of the default PR set or a measurement from this review. At the reviewed baseline, `pyproject.toml` excludes `slow`, `stress`, `integration`, and `gpu` by default. `ci.yml` uses that default set. Preserve these tiers.

Start with 6–8 fast properties and about 8–20 inputs each. Choose inexpensive parts of items 1, 5, 7, 15–17, and 20–21. GIG cold-start inversion, repeated full EM fits, quadrature grids, and large sampling checks should start as `slow`. Move a property into the default set only after measuring it.

Use an **unmeasured target** of about 1–2 extra minutes on an ordinary CI machine for the first group. If it exceeds that target, inspect compilation shapes and repeated fixtures before reducing budgets or changing markers. Changing the backend to CPU is not an acceptable shortcut.

Implementing and calibrating all 25 candidates in stages is estimated at about 2–3 weeks, depending on numerical counterexamples. The work requires more than wrapping existing tests in `@given`.

Possible future commands, **not executed**:

```bash
uv run pytest tests/ --durations=50
NORMIX_HYPOTHESIS_PROFILE=normix_nightly uv run pytest tests/test_property_contracts.py tests/test_property_numerics.py -m "not gpu" --durations=50
```

Both `test_property_*.py` paths are proposed files. The nightly profile changes Hypothesis budgets but does not change pytest marker selection. The example therefore overrides the marker filter explicitly. A filter such as `smoke or contract` can select tests also marked `slow`; exclude slow tiers explicitly when required. With `--strict-markers`, do not use unregistered `property` or `nightly` markers.

Applicable repository conventions:

- Use uv, pytest, existing markers, and the x64 configuration. Respect single-observation APIs and batch through `vmap`.
- Retain independent references and explicit tolerances under `.cursor/rules/testing-guidelines.mdc`. Property tests address contracts and failure mechanisms; they do not replace existing checks.
- Preserve the recursive scanner and scanner self-tests in `test_no_host_callbacks.py`. Its deliberately nested callbacks already provide negative validation.
- If a future bug is confirmed and fixed, retain a failing test first under the repository's TDD workflow. This report claims no failing-then-passing evidence.
- This revision does not update `design.md`, plans, or other files. During implementation, record accepted decisions and rejected alternatives in their governing documents.

### 4.8 Mutation validation and records

For each new property, retain at least one targeted record showing that a deliberate defect makes it fail. **Mutation validation** changes a small part of the implementation to check whether the property detects that error class. It checks usefulness, not completeness.

Do this later in a disposable copy or with controlled test doubles. This revision changes no source code. Mutate one mechanism at a time. Distinguish assertion failures from compilation and installation failures. Failure to compile does not demonstrate detection of the intended numerical error.

| Property | Proposed mutation | Expected detection |
|---|---|---|
| Natural-parameter round trip | Omit the addition of 1 when recovering Gamma $\alpha$ | Parameter mismatch |
| Immutable update | Make a test double overwrite the original snapshot | Original values change |
| Backend propagation | Ignore `m_step_backend` and always use CPU | Path sentinel fails; value comparison alone may miss it |
| No callbacks | Insert a callback only inside a nested conditional | Recursive jaxpr scan fails |
| Cumulative rule | Average batches equally instead of weighting by sample count | Unequal batch sizes produce the wrong mean |
| Stopping state | Remove flag freezing after stopping | Later candidates change the final status |
| PDF normalization | Multiply the density by 1.1 | Integral mass differs from 1; normalized-table checks may miss it |

A future record can use this template. Results in this report are all "not run". No minimal counterexample or timing is invented.

```text
property_id / governing decision:
source commit / file / function:
input domain, exclusions, dtype, device, backend:
reference source and tolerance rationale:
tool versions, profile, command, seed or replay information:
original result and duration: not run
exact mutation:
mutation result and shrunk counterexample: not run
failure came from the intended assertion: to verify
result after restoring the original code: not run
specification review:
```

jax-lean's `tests/test_certification.py` also changes divisors, operators, and variable references, then expects Lean to reject the result. Its method of checking that a checker rejects errors is useful here. The original review read those tests but did not run them.

## 5. A small TLA+ model: old documentation must not replace new documentation

This pilot uses documentation publishing alone. It models the common deployment target of `docs.yml` and `docs-full.yml`. It does not place numerical calculations in the state space.

First state the **example policy**: suppose the two artifacts have an agreed order. The old artifact is version 1, the new artifact is version 2, and the website must not return to a lower number. This is a teaching assumption, not an accepted normix website policy. Whether master or releases take priority remains an open question.

The model has three variables:

- `phase`: each job is `queued`, `built`, or `done`.
- `shown`: the website's displayed version, initially 0.
- `highWater`: the highest version whose publication decision has completed. This is a history summary used to check the property. The implementation need not store a field with that name.

`Build` can occur in either order. `PublishOrSkip` accepts only artifacts at least as new as the displayed version. **Comparing versions and updating the website form one atomic step.** This is the model's main assumption to review.

The TLA+ sketch below has not undergone syntax checking or TLC execution. A future run should first show that the faulty variant fails, then check the corrected variant.

```tla
---- MODULE DocsPublish ----
EXTENDS Naturals

Jobs == {"old", "new"}
Version(job) == IF job = "old" THEN 1 ELSE 2

VARIABLES phase, shown, highWater
vars == <<phase, shown, highWater>>

Init ==
    /\ phase = [job \in Jobs |-> "queued"]
    /\ shown = 0
    /\ highWater = 0

Build(job) ==
    /\ phase[job] = "queued"
    /\ phase' = [phase EXCEPT ![job] = "built"]
    /\ UNCHANGED <<shown, highWater>>

PublishOrSkip(job) ==
    /\ phase[job] = "built"
    /\ phase' = [phase EXCEPT ![job] = "done"]
    /\ shown' = IF Version(job) >= shown THEN Version(job) ELSE shown
    /\ highWater' = IF Version(job) > highWater
                     THEN Version(job) ELSE highWater

Next == \E job \in Jobs : Build(job) \/ PublishOrSkip(job)

TypeOK ==
    /\ phase \in [Jobs -> {"queued", "built", "done"}]
    /\ shown \in 0..2
    /\ highWater \in 0..2

NoRollback == shown = highWater
AllFinished == \A job \in Jobs : phase[job] = "done"
EventuallyFinished == <>AllFinished

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ (\A job \in Jobs :
          WF_vars(Build(job)) /\ WF_vars(PublishOrSkip(job)))
====
```

`\E` means "there exists" and `\A` means "for all". `[Jobs -> {...}]` is the set of functions from jobs to phases. `EXCEPT ![job] = ...` changes only that job's phase.

`WF` denotes weak fairness: if an action remains enabled from some point onward, it must eventually occur. After both jobs finish, the state may remain unchanged forever.

The proposed TLC configuration is also unrun:

```text
SPECIFICATION Spec
INVARIANT TypeOK
INVARIANT NoRollback
PROPERTY EventuallyFinished
CHECK_DEADLOCK FALSE
```

Deadlock reporting is disabled because no work remains after all jobs finish. This setting applies only to the teaching model. Disabling it unconditionally for a continuing service could hide errors. Liveness is checked separately.

Deliberately replace the `shown'` line with the unconditional `shown' = Version(job)`. A hand-derived trace is:

| Step | old phase | new phase | shown | highWater |
|---|---|---|---:|---:|
| Init | queued | queued | 0 | 0 |
| Build(new) | queued | built | 0 | 0 |
| PublishOrSkip(new) | queued | done | 2 | 2 |
| Build(old) | built | done | 2 | 2 |
| Faulty PublishOrSkip(old) | done | done | 1 | 2 |

The final state violates `NoRollback`. This is a **derivation**, not TLC output. The corrected model skips the old artifact in the last step and retains `shown=2`. Under the stated fairness assumptions, both jobs eventually finish. Without fairness, permanent stuttering is allowed, so that liveness conclusion does not follow.

The finite model can check that build/publication interleavings preserve the displayed version, that old jobs still finish, and that phases stay within their declared values. It cannot check the actual build commit, artifact freshness, workflow permissions, network failures, cancellation, or the implementation of `peaceiris/actions-gh-pages`.

Reading the website version and uploading later does not implement the modeled atomic step. Both jobs could read version 0, then the old job could overwrite the new one. If the implementation separates reading, comparing, and writing, the model must allow interleavings between those steps.

A single-writer publication entry point, or an equivalent atomic acceptance mechanism, is needed to realize that abstraction. Adopting one remains a future design decision. Serializing builds alone still requires checking queue order against the version policy.

If the model becomes worth maintaining, map every action to its YAML/job/step, ref, artifact identifier, acceptance condition, and failure behavior. Preserve the TLC configuration, model version, source commit, and hand-derived or machine-produced counterexamples. After changes, check that a supposed fix has not removed real actions, added unrealistic fairness, or enlarged atomic steps.

The pilot is estimated at 1–2 person-days to prepare and run the finite model, organize counterexamples, and map it to implementation. Allow 1–2 hours to review the version policy and atomicity assumptions. Static review and one publishing entry point may resolve the problem without long-term TLA+ maintenance. Future protocol complexity determines that need.

## 6. Relation to the broader verification plan and jax-lean

The earlier recommendation to begin with property tests remains useful. Three broader claims need a narrower scope.

**TLA+ suitability depends on the problem.** A model of a numerical kernel omits much of its accuracy and parameter-domain risk. A publication model can expose real ordering risks. For online EM, distinguish the current array-based fitter from a future recoverable service. File names and an "application layer" label do not decide suitability.

**Finance needs more than backtests and numerical comparisons.** Budget constraints, unreachable targets, scenario weights, and local-objective contracts should become automatic properties. One historical backtest rarely covers deficient rank, zero weights, changes of units, or wrong failure branches. The absence of an order-submission protocol gives a concrete reason to defer TLA+.

**A missing Bessel primitive does not prevent verification.** If the numerical Bessel implementation consists of traceable JAX operations, its jaxpr can be given Lean semantics. No primitive named Bessel is required. Traceability, translation support, translation equivalence, approximation of the true $K_\nu$ function, and floating-point error control are five separate questions. Here, $K_\nu$ is the modified Bessel function of the second kind of order $\nu$.

**local check:** a local copy of https://github.com/srush/jax-lean was read. It provides concrete examples of these boundaries:

- `python/jaxlean/translate.py` generates readable Lean functions. `JaxprImporter` in `python/jaxlean/importers/jaxpr.py` generates a separate explicit intermediate program. `certify.py` produces proof obligations equating them.
- `JaxLean/Core/Jaxpr.lean` defines evaluation of that intermediate program. `Core/RealOps.lean` interprets some transcendental operations as real-valued functions. `examples/certificates/generated/CertifiedMean.lean` gives a mean example of fixed length 2.
- `SEMANTICS_REFERENCES.md` states that correspondence between the Python importer and actual JAX execution lies outside that equality proof. Real semantics omit floating-point rounding. `tests/test_certification.py` includes a case where a floating-point conversion changes the actual value, while the real model still admits an identity certificate.
- `SUPPORTED_JAXPR.md` distinguishes ordinary translation from certificate support. Ordinary `scan` translation has a default unrolling limit of 32. The certificate path does not support `scan` and rejects captured floating-point array constants. Operation support does not establish the desired property.

In normix `bessel.py`, `_edge` uses a fixed-count loop and `_geometry` uses `arcsinh`. The kernel also uses `expm1`, `log1p`, reductions, and floating-point node constants. Check their actual lowered jaxprs with the relevant JAX version. No tracing was performed for this report, so membership in the supported subset remains **to verify**. The 40 window iterations may exceed the default unrolling budget. Even after translation succeeds, the error between 192-node quadrature and the continuous integral needs a separate proof.

`stop_gradient` is the identity in a forward-value model but changes AD rules. Forward equality alone does not validate the derivative interpretation after freezing integration nodes. This is why the property plan checks AD, moment bundles, and independent references.

**local check:** the reviewed copy's `pyproject.toml` pins JAX 0.8.0. normix's uv resolution constraint is 0.9.1; that is not an exact upper bound on its published dependency. The copy has no root `LICENSE` file, although some example directories have separate licenses. Do not infer that every file has the same licensing status. This report copies none of its implementation and does not recommend downgrading normix. The same approach can also be implemented independently.

The maintainer will fork [srush/jax-lean](https://github.com/srush/jax-lean) for study and experiments. Open an issue asking srush for a license. Reuse or redistribution of its code requires the author's permission until licensing is clarified. Do not copy code into normix before that clarification.

Plan D, hand-written Lean models under `formal/lean/`, and the jaxpr-to-Lean bridge proceed **in parallel**. The bridge should cover as much of normix as possible, beyond the Bessel kernel. Parts that are hard to translate from jaxpr will be written by hand in Lean. Keep both workstreams distinct from the Hypothesis and TLA+ pilots so neither becomes a prerequisite for useful tests.

Reports and proposition cards are in English. The mathematical inventory has its own tracking file. The candidate property list in this report is a testing plan, not a replacement for that inventory.

Under the Lean source tree, file, module, and directory names use UpperCamelCase, following Mathlib style. Definitions use lowerCamelCase, theorems use snake_case, and namespaces use UpperCamelCase. Lake maps module names to paths. The lowercase `formal/lean/` is the package directory.

For these testing pilots, jax-lean also offers useful checking practices: state semantic boundaries; list supported and rejected cases; mutate generated output and check certificate rejection; protect import boundaries with small source tests. Developing the bridge remains a parallel workstream, outside the Hypothesis and TLA+ pilot implementations.

## 7. Decisions and rejected alternatives

The entries below use the repository's decision-record format and are dated 2026-10-24. Their evidence is the static review, reported inputs, and explicitly qualified prior tool knowledge. Accepted constraints and the parallel Lean plan are distinguished from recommendations. "REJECTED" otherwise means this report recommends against an option in the stated scope; it does not imply a formal maintainer veto.

| Option | Status and scope | Reason and evidence | Decision authority and reopening condition |
|---|---|---|---|
| Native JAX paths and selected backends | Retain; existing constraint | Explicit requirement, S9/S11, and existing no-callback tests | Existing decision; reopen only after an explicit contract change |
| A few Hypothesis properties with existing pytest tests | Recommended; not implemented | Tests real code across continuous parameters and call sequences; reuses references and markers | Review initial specifications; adjust strategies or scope if costs exceed budget |
| Small TLA+ documentation-publishing pilot | Conditionally recommended | Multiple workflows share the website; acceptance policy remains undecided | Proceed after the website policy is settled; long-term maintenance may be unnecessary at low complexity |
| TLA+ for all non-mathematical code | REJECTED for this phase | Most code is pure functions or local sequential state; models omit numerical risks and can drift | Report recommendation; reopen for persistence, promised concurrent calls, or recovery protocols |
| Hidden CPU execution inside JAX paths for verification or speed | REJECTED | Violates the backend contract; S9 already rejects this offload | Existing constraint; this report does not reopen it |
| Porting the runtime core to another language for formalization | REJECTED for the current project direction | Adds correspondence obligations between JAX interfaces and another runtime implementation | Existing constraint; reconsider for an independent subsystem with its own migration need |
| 200 examples for every property in PR checks | REJECTED | Compilation and inversion add to existing test costs; the reported 71-minute full suite supports retaining tiers | Report recommendation; expand individual budgets after measurement shows they are cheap |
| Round trips and CPU/JAX comparisons as the only evidence | REJECTED as sole evidence | Backends can share errors; paired mistakes can cancel in round trips | Report recommendation; retain independent references or different derivations |
| Adopting jax-lean unchanged | Deferred as an integration choice | Version compatibility, certificate coverage, floating-point semantics, and licensing remain unresolved | A study fork proceeds; integration requires resolving the relevant gaps, including permission for code reuse |
| Plan D and the jaxpr-to-Lean bridge | Accepted parallel workstreams | Hand-written models cover difficult translations; the bridge aims for broad normix coverage | Maintainer decision; choose specific propositions and translation targets through the separate inventory |

Select other tools against concrete problems too. The descriptions below come from prior tool knowledge. Current versions, maintenance status, and extensions were not checked. These limitations do not establish that a tool is obsolete or unusable.

| Tool | Explanation from first principles | Reason to reject or defer in this phase | Reopening condition |
|---|---|---|---|
| Alloy | Describes structures through relations and constraints, then searches a finite scope for instances or counterexamples. Some versions support temporal descriptions; details **to verify** | normix has no complex permission, object-relation, or configuration-graph problem. Relational models do not remove numerical-error difficulties | Complex model registration or configuration combinations where structural constraints matter more than step order |
| P language | Describes programs and concurrent behavior with event-driven state machines and can explore schedules | No current actor/message service layer; an event-program model adds cost for pure array code | An independent asynchronous fitting service or multi-participant job protocol |
| Quint | A state-machine specification language and tools intended to ease modeling and checking; exact execution/verification backends **to verify** | Easier syntax would not remove model-to-Python correspondence work. Two specification languages would divide the first pilot's effort | If TLA+ syntax becomes the main obstacle, compare the same publication example within one day |
| stateright | A tool for describing systems in Rust and model checking them | normix is not a Rust project. It does not automatically read Python/JAX, so a model still needs rewriting | An existing Rust service needs a behavioral model maintained in its own language |
| Dafny / Verus | Programs include preconditions, postconditions, and loop properties, checked by logical tools. Verus targets Rust | Direct verification of Python/JAX/Bessel is not their natural route. Porting adds maintenance and correspondence obligations | A small critical control component independently requires the corresponding language |
| CrossHair | Symbolically executes supported Python code, representing inputs as symbols and solving constraints for violations | Coverage depends on supported operations. JAX tracers, array runtimes, compilation, and special functions cannot be assumed covered | A high-risk pure Python configuration or validation module warrants a half-day capability probe |

These choices rank effort. They do not assert that only Lean and Hypothesis are valid. For state-modeling languages, model boundaries and specification review usually matter more than syntax.

## 8. Ranked recommendations and next steps

1. **Start with small backend and call-contract checks.** Prioritize invalid backends, existing no-callback jaxpr checks, immutable updates, pytrees, and cumulative weights. The first 6–8 properties are estimated at 3–5 person-days, including initial timing, shrinking, and mutation validation, plus 3–5 hours of specification review. Writing strategies and tests does not authorize changing failure paths to CPU or weakening mathematical assumptions. Reproduce the `log_kv` gap through TDD first; its fix is a separate implementation task.

2. **Expand numerical properties and finance constraints.** Put GIG inversion, EM objectives, sampling moments, and PDF integrals in slow tiers. Reuse the Bessel high-precision table. Estimate another 5–10 person-days and 4–8 hours of review; numerical fixes need separate estimates. Review acceptable domains, error budgets, and heavy-tail exceptions alongside generators, logs, and references. Budget about 2–3 weeks for the full candidate list, without assuming every item should be adopted.

3. **Try a 1–2 person-day TLA+ publishing pilot.** First decide which version the website should follow. Then run the faulty and corrected models in Section 5 and map actions to workflows. Allow 1–2 hours to review the policy and why comparison plus writing can be atomic. Until that policy is clear, the result remains a research model, not verified deployment.

4. **Defer recoverable streaming EM and shared-state workflow protocols until needed.** If requirements emerge, model recovery and shared state before implementing them. Allow 2–4 person-days and half a day of review for an initial single-writer streaming model. Multiple writers, leases, and cross-process recovery need new estimates. Review whether replay is allowed or exactly-once processing is required, and define persistence boundaries and costs.

Plan D and the jaxpr-to-Lean bridge proceed in parallel with this ranking. Their scope includes broad normix coverage and hand-written Lean models where translation is difficult. Neither is a prerequisite for merging property tests.

## 9. Open questions for the maintainer

1. Should the website show the latest master state, the latest release, or separate versions of both? This determines the publication property.
2. Is concurrent use of one `BatchEMFitter` or `IncrementalEMFitter` instance part of the contract? If not, sequential reuse and exception recovery are the first checks.
3. Does near-term incremental EM need only in-memory mini-batch fitting, or resumable data streams? For recovery, can data be reread, and must random keys and rule state persist across processes?
4. What additional PR runtime is acceptable for the first property tests? This report proposes a 1–2 minute target. Can slow nightly checks run separately?
5. Should the first accuracy contract explicitly cover only x64 and moderate parameter domains? Which boundary distributions, extreme tails, and GPU cases require separate support?
6. Should every public entry point, including `log_kv` and batch helpers, consistently reject unknown backends? This follows E17's direction, but the reviewed source has gaps.

## Sources

Repository links are relative to this report. Source and test references support the recorded static review; they do not show tests passed during it. External works retain known titles and years. Unconfirmed versions, dates, and page numbers are not supplied.

**Repository guidance and research inputs**

- [AGENTS.md](../../../AGENTS.md), [dev-notes/README.md](../../README.md), and [ARCHITECTURE.md](../../ARCHITECTURE.md).
- Earlier background material dated 2026-10-19 included social-media posts, industry statistics, and remote papers. These were not verified and are not independent evidence for this report's conclusions.
- [Design decision table](../../design/design.md): E3, E9–E11, E16–E17, S4–S12, and FIN-1–FIN-5; [design index](../../design/index.md).
- [Test-running guidance](../../../.cursor/skills/running-tests/SKILL.md), [testing guidelines](../../../.cursor/rules/testing-guidelines.mdc), [coding conventions](../../../.cursor/rules/coding-conventions.mdc), and [design-record rules](../../../.cursor/rules/maintain-design-docs.mdc).
- [pyproject.toml](../../../pyproject.toml) and [tests/conftest.py](../../../tests/conftest.py): dependencies, JAX constraints, markers, and x64. The reported full CPU-suite time of about 71 minutes was not remeasured.

**normix source and existing tests**

- [exponential_family.py](../../../normix/exponential_family.py), [gamma.py](../../../normix/distributions/gamma.py), [generalized_inverse_gaussian.py](../../../normix/distributions/generalized_inverse_gaussian.py), and [generalized_hyperbolic.py](../../../normix/distributions/generalized_hyperbolic.py).
- [mixtures/marginal.py](../../../normix/mixtures/marginal.py), [mixtures/joint.py](../../../normix/mixtures/joint.py), and [fitting/eta.py](../../../normix/fitting/eta.py): objects, public moments, statistics, and E/M interfaces.
- [fitting/em.py](../../../normix/fitting/em.py), [fitting/eta_rules.py](../../../normix/fitting/eta_rules.py), and [fitting/solvers.py](../../../normix/fitting/solvers.py): state, backend dispatch, warm starts, and failures.
- [utils/bessel.py](../../../normix/utils/bessel.py), [utils/rvs.py](../../../normix/utils/rvs.py), and [utils/quadrature.py](../../../normix/utils/quadrature.py): numerical kernels, quantile snapshots, and scenario measures.
- [finance/optimization.py](../../../normix/finance/optimization.py), [finance/risk.py](../../../normix/finance/risk.py), and [finance/transaction_costs.py](../../../normix/finance/transaction_costs.py): reduced holdings, CVaR, and local transaction-cost optimization.
- [test_no_host_callbacks.py](../../../tests/test_no_host_callbacks.py), [test_bessel_contract.py](../../../tests/test_bessel_contract.py), [Bessel reference table](../../../tests/data/bessel_reference.json), [test_distribution_conversions.py](../../../tests/test_distribution_conversions.py), [test_incremental_em.py](../../../tests/test_incremental_em.py), and [transaction-cost tests](../../../tests/finance/test_transaction_costs.py).
- Related test entry points: [test_em_cpu_vs_jax.py](../../../tests/test_em_cpu_vs_jax.py), [test_expectation_autodiff.py](../../../tests/test_expectation_autodiff.py), [test_regularizations.py](../../../tests/test_regularizations.py), [test_cdf_ppf.py](../../../tests/test_cdf_ppf.py), and [test_gig_rvs.py](../../../tests/test_gig_rvs.py). The report does not claim a complete audit of every assertion in these files.

**Repository theory, design, and workflows**

- [Exponential-family design](../../../docs/design/exponential_family.md), [EM design](../../../docs/design/em_framework.md), and [solver/Bessel design](../../design/solvers_and_bessel.md).
- [GIG theory](../../../docs/theory/gig.md), [EM theory](../../../docs/theory/em_algorithm.md), [online EM theory](../../../docs/theory/online_em.md), and [repository references](../../../docs/references.md). Theory pages supply statements to verify, not machine-checked proofs.
- [Bessel moment-kernel note](../../tech_notes/bessel_moment_kernel.md) and [older GIG inversion note](../../tech_notes/gig_eta_to_theta.md). The latter covers part of CPU multi-start behavior. Its statement about lacking JIT support does not describe current JAX inversion; use the source and S9/S11 for that.
- [Recurring workflow and orchestration plan](../../plans/loops_and_orchestration.md).
- [ci.yml](../../../.github/workflows/ci.yml), [release-please.yml](../../../.github/workflows/release-please.yml), [publish.yml](../../../.github/workflows/publish.yml), [docs.yml](../../../.github/workflows/docs.yml), and [docs-full.yml](../../../.github/workflows/docs-full.yml). Remote protection settings and actual run logs were not read.

**jax-lean source**

A local copy of https://github.com/srush/jax-lean was read. The paths below are within that project; they are not links to files in normix.

- `README.md` is short and mainly links to a project page, which was not visited. `ARCHITECTURE.md`, `SUPPORTED_JAXPR.md`, and `SEMANTICS_REFERENCES.md` were the main explanatory sources.
- `JaxLean/Core/RealOps.lean` and `JaxLean/Core/Jaxpr.lean`: real semantics and the intermediate program with shapes and types.
- `python/jaxlean/certify.py`, `python/jaxlean/importers/jaxpr.py`, and `python/jaxlean/translate.py`: importing, generation, and certificate boundaries. `pyproject.toml` pins the JAX version.
- `examples/certificates/code.py`, `examples/certificates/generated/CertifiedMean.lean`, `tests/test_certification.py`, and `tests/test_architecture.py`. No Lean build or tests were run.

**External works and tool knowledge: offline, from memory or the local bibliography**

- Leslie Lamport, *Specifying Systems*, 2002. Background on TLA+ specifications, states, and behaviors; recalled from the literature without page-level verification.
- Koen Claessen and John Hughes, *QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs*, 2000. Background on property testing, not evidence for a specific Hypothesis API version.
- Hypothesis documentation on strategies, settings, and stateful testing; TLA+ documentation on TLC and PlusCal. Descriptions use prior tool knowledge. Exact current behavior is **to verify**; no documentation year is invented.
- Dempster, Laird, and Rubin, *Maximum Likelihood from Incomplete Data via the EM Algorithm*, 1977; Cappé and Moulines, *On-line Expectation-Maximization Algorithm for Latent Data Models*, 2009; McLachlan and Krishnan, *The EM Algorithm and Extensions*, 2008. Titles and years can be checked against `docs/references.md`. External full texts were not read.
- NIST DLMF: modified Bessel symmetry and recurrence relations. Formulas came from mathematical knowledge and earlier background material. No external page was visited; no section number or Mathlib lemma name is invented.
- Lean/Mathlib, Alloy, P, Quint, stateright, Dafny, Verus, CrossHair, and mpmath descriptions use prior tool knowledge. Current compatibility and release status were not checked.

## Revision notes

- Made the positive-argument domain $z>0$ explicit in the Bessel recurrence candidate, as already required for the kernel's real-valued contract.
- Updated the decisions to record parallel hand-written Lean models and broad jaxpr translation, a separate mathematical inventory, English proposition cards, and the Lean naming convention.
- Distinguished a jax-lean study fork from code reuse. Recorded the license request and permission requirement before copying code into normix.
- No executable examples, timing figures, or technical findings were revalidated during this English revision.
