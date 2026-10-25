# Feasibility of formalizing normix mathematics in Lean 4 and Mathlib

2026-10-24 · Offline research report

Use an independent Lean project under `formal/lean/` in the normix repository. Start with finite weighted moments, parameter conversions, and mixture representations. Then extend the proofs to continuous exponential families, GIG normalization, and EM. Bessel functions can be defined mathematically in Lean. Error bounds for the current quadrature implementation are a separate research task. Tracing a JAX function to jaxpr supplies a program representation; it does not prove that the program computes Bessel functions correctly.

The accepted direction is to develop plan D, the hand-written Lean models, and the jaxpr-to-Lean bridge in parallel. The bridge should eventually cover as much of normix as possible. Bessel is one target, not the scope limit. Parts that are hard to translate from jaxpr will be written by hand in Lean, with their correspondence to the implementation stated explicitly. The mathematical inventory is maintained separately in [05_math_checklist.md](05_math_checklist.md). Reports and proposition cards use English.

The original investigation used static reading and mathematical analysis. It did not access the network, install dependencies, run tests, or run Lean. This revision also makes no new execution claims. The normix reading baseline is `51e053c57237a29c71b56d3a2a86988277477bd9`. A local copy of https://github.com/srush/jax-lean was read. That copy had no `.git` directory. Its reported history of 12 commits was supplied with the input and was not independently verified. No root `LICENSE` file was present; some examples had separate licenses.

The maintainer will fork [jax-lean](https://github.com/srush/jax-lean), by srush, for study and experiments. Reuse or redistribution of its code requires the author's permission while licensing remains unclear. Open an issue asking for a license before copying code into normix. This report does not claim that the fork or issue already exists.

Three evidence labels apply throughout this report and the checklist:

- **local check**: supported by the source or documentation read for the original investigation. Source presence does not imply a successful build.
- **derivation**: a calculation given here, a proposed proof route, or recalled mathematical literature. External originals were not reread.
- **to verify**: requires a pinned Mathlib audit, compilation, or literature verification. Proposed names are not presented as existing library lemmas.

Work estimates are planning estimates, not measurements. One person-week means five working days. They assume a proof engineer familiar with Lean and a reviewer who checks the mathematics. A sole contributor learning Lean from scratch should allow at least twice the calendar time. Special functions may take longer.

## 1. Understanding Lean

**Formalization** expresses mathematical objects, assumptions, and conclusions in a precise language. **Verification** checks that a proof establishes its stated conclusion. For example, a claim that GIG variance is nonnegative must identify the parameter domain, the random variable's law, and existence of its second moment.

**Lean 4** is a programming language and an interactive proof assistant. Here it supplies mathematical definitions and proofs. JAX remains the numerical runtime.

**Mathlib** is Lean's mathematics library. It has infrastructure for real numbers, linear algebra, measures, integration, and calculus. It does not turn a paper into proofs automatically. Infrastructure for a subject does not establish that every theorem needed by normix already exists.

A **specification**, or spec, is the exact statement to prove. One example is that a specified GIG density integrates to $1$ on $(0,\infty)$ for $a,b>0$ and $p\in\mathbb R$. Review the specification independently of its proof.

A **tactic** helps construct a proof. `ring` handles polynomial identities, `simp` rewrites with known equalities, and `positivity` handles common positivity goals. Ordinary tactics produce proof terms checked by the **kernel**, the small core responsible for the final logical check. The kernel does not decide whether a definition captures the intended probability or finance problem.

An **axiom** is a premise accepted without a proof in this project. `sorry` is a Lean placeholder that skips a proof. A file containing it may build successfully. A successful build alone is not evidence that a proof is complete.

### 1.1 A small theorem

This proposed example has **not been compiled**. `NormixProposal` is a proposed namespace, not an existing library.

```lean
import Mathlib

namespace NormixProposal

theorem gig_b_roundtrip (b : ℝ) : -2 * (-b / 2) = b := by
  ring

end NormixProposal
```

`(b : ℝ)` introduces an arbitrary real number $b$. The conclusion follows the colon; the proof follows `by`. It models the algebraic relation $\theta_2=-b/2$, $b=-2\theta_2$ in `GeneralizedInverseGaussian.natural_params` and `from_natural`. [N2; local check]

The identity also holds for negative inputs, so it needs no assumption $b>0$. Interpreting the result as a valid GIG parameter requires a separate domain proof. The theorem proves neither density normalization nor exact floating-point multiplication and division.

A probability specification must define the density. The following sketch uses real exponentials and logarithms and restricts the support explicitly.

```lean
import Mathlib
open MeasureTheory Set

namespace NormixProposal

noncomputable def gigKernel (p a b x : ℝ) : ℝ :=
  Real.exp ((p - 1) * Real.log x - (a * x + b / x) / 2)

noncomputable def gigMass (p a b : ℝ) : ℝ :=
  ∫ x in Ioi (0 : ℝ), gigKernel p a b x

end NormixProposal
```

This sketch is also **uncompiled**. `noncomputable` marks a mathematical definition; it does not promise an executable real integration algorithm. `Ioi 0` is $(0,\infty)$. Subsequent proofs must establish integrability, positive mass, and equality of that mass to the explicit Bessel formula. Defining a density as its kernel divided by its integral does not verify the normalizing constant used in the code.

Derivative specifications should include existence. The jax-lean file `examples/autodiff/proofs/Jvp.lean` uses `HasDerivAt`, which states both differentiability and the derivative's value. Its `gradient_correct` theorem concerns a quadratic polynomial. It does not prove correctness of every use of `jax.grad`. [J3; local check]

### 1.2 Four verification targets

| Layer | normix example | Proof obligation |
|---|---|---|
| Mathematical theorem | $\nabla\psi=E[t]$ | Derive the derivative formula from the integral definition under explicit assumptions |
| Exact real algorithm | A weighted exponential sum on fixed nodes | Prove algebraic, derivative, and moment identities over the reals |
| Numerical approximation | A 192-node approximation to a Bessel integral | Bound truncation, quadrature, and constant approximation errors |
| Machine execution | JAX/XLA on a selected device | Relate rounding, compilation, and primitive implementations to an error specification |

XLA is the compilation system used by JAX. The first layer does not establish the other three. The current jax-lean approach mainly connects program representations to the second layer. The third needs numerical analysis. The fourth also needs models of machine numbers and compiler behavior.

## 2. Findings from local reading

The initial assessment, dated 2026-10-19, made useful distinctions. Static reading narrowed several of its conclusions.

| Initial assessment | Revised assessment and evidence |
|---|---|
| No Bessel primitive means normal mixtures cannot be handled | This limits the importer's existing support. Mathematical formalization can define Bessel functions by integrals and mixture maps without an importer. [N5, J1] |
| Exponential-family duality and EM can be proved in a few weeks | This may hold for abstract consequences with the analysis assumed. Continuous integrals, differentiation under the integral, and GH instances usually require months. |
| Floating-point numerics and solvers are not worth proving | A complete machine execution proof is not the first target. Real quadrature error, local inverse maps, and restricted convergence results remain useful. Deferral is not impossibility. |
| TLA+ has no role in normix | It is unsuitable for numerical identities. It can model stopping flags, failure recovery, and backend selection; see Section 7. |
| The mathematics website can directly supply Lean specifications | Formulas, domains, and claim strength need review first. Formalization can expose documentation errors. |

The following specification review items came from static reading. They are not a complete documentation audit or a code repair report.

1. In `docs/theory/gig.md`, the alternative parameters are $\delta=\sqrt{b/a}$ and $\eta=\sqrt{ab}$. The density prefactor is written as $\delta^p$. Substitution into the original definition on that page gives $\delta^{-p}$. This algebraic discrepancy needs a separate documentation correction. [N1; derivation]
2. `docs/theory/em_algorithm.md` says that both algorithms converge to the MLE. General EM does not guarantee a global maximum-likelihood solution. Likelihood convergence, stationary accumulation points, and parameter-sequence convergence are distinct. The page also calls Gamma and InverseGamma `from_expectation` closed form. Gamma calls `_newton_digamma`; InverseGamma also requires shape inversion. [N3, N6; local check]
3. Flat GIG statistics use $(\log Y,1/Y,Y)$. `NormalMixtureEta` fields use $(1/Y,Y,\log Y,\ldots)$. Documentation switches between these orders. Lean should use named fields and explicit conversions, without identifying the arrays by position. [N2, N6; local check]
4. The ideal mathematical `_rescale` preserves the marginal law of $X$. Replacing $Y$ by $sY$ changes the joint density in the original coordinates and requires a Jacobian factor. GH `_build_rescaled` also clamps $a,b$. Exact marginal invariance therefore requires the clamps to be inactive. [N4; local check and derivation]
5. `dev-notes/tech_notes/gig_eta_to_theta.md` retains an old claim that the solver cannot be JIT compiled. Current source and decisions S5/S11 include a JAX cold start and implicit derivatives. Current source and dated decision records take precedence over that old limitation. [N6, N7; local check]

## 3. Mathematical inventory, difficulty, and value

The inventory is now a separate tracking document: [05_math_checklist.md](05_math_checklist.md). It contains the former Sections 3.1–3.7: exponential families; densities, moments, and mixtures; CDFs; sampling; EM; Bessel functions; and other theory pages. Each item has an ID, dependencies, a planned Lean module, status, and space for proof or PR notes.

Difficulty ranges from A, finite algebra, to D, special-function or numerical-error theory. The checklist retains the derivations, domain restrictions, effort estimates, and reasons behind those grades. All items initially have status `not-started`; inclusion is not evidence of a proof.

The main dependencies remain finite exponential sums, continuous exponential-family analysis, Bessel integrals, GIG, mixture densities, and EM instances. Finite moment proofs can support the numerical kernel before continuous Bessel analysis is complete. Hand-written models and the jaxpr bridge proceed in parallel and meet at reviewed specifications.

## 4. Mathlib coverage and unresolved questions

**The original investigation had no complete Mathlib source tree. It cannot establish current coverage as of 2026-10-24.** The local jax-lean configuration records Lean `v4.33.0` and Mathlib commit `db584cd6d46c92f209a44c0f1c829460d327499d`. These are file contents, not installation, build, or remote-version checks. [J1; local check]

The Bochner integral generalizes the Lebesgue integral to vector-valued functions. Dominated convergence permits exchanging a limit and an integral when a common integrable bound controls the functions.

| Area | Supported evidence | Unverified scope |
|---|---|---|
| Reals, exp/log/sqrt, elementary calculus | Local `Core/RealOps.lean` imports related Mathlib modules. `Jvp.lean` uses `HasDerivAt`, `deriv`, and derivative composition. | These imports were not compiled here. |
| Measures, measurable maps, independence | `Stdlib/BatchProbability.lean` uses `Measure`, `Measurable`, `iIndepFun`, and measurable composition rules. | This does not establish continuous GH or continuous sampling proofs. |
| Lebesgue/Bochner integration, dominated convergence, differentiation under the integral | Infrastructure is known from Mathlib usage and recalled literature. | Exact modules, hypotheses, and applicable theorem names are **to verify**. |
| Gamma functions, Gaussian integrals, convex analysis | Related definitions and many theorems are recalled. | Incomplete Gamma, digamma/trigamma, GIG transformations, and Legendre interfaces are **to verify**. |
| Bessel $K_\nu$ | Lean can express the integral definition used by normix. | Arbitrary real order, recurrences, order derivatives, and asymptotics in the pinned Mathlib are all **to verify**. Neither presence nor absence has been established. |
| GIG, GH, and EM | They can be built from analysis infrastructure. | No local evidence establishes that these complete developments already exist in Mathlib. |

jax-lean's `FiniteLaw` and `Rand` are finite probability models. Their finite-sum proofs are design references, not continuous GIG integration results. [J4; local check]

The first executable phase should spend 2–4 person-days auditing pinned dependencies. Record each target, existing definition, usable theorem, gap, and minimal compilation example. Review normalization and parameter conventions. A search for “Bessel” alone is insufficient: check the kind, real or integer order, real or complex argument, and normalization.

The costly obligations often precede the final formula:

- **Almost-everywhere equality.** Densities that differ on a null set define the same law. API values at $x=0$ or $x=\mu$ may need separate pointwise conventions.
- **Integrability.** Real-valued Bochner integrals have totalized behavior for nonintegrable functions. This does not represent an ordinary finite expectation. Prove integrability explicitly, or express divergence with a nonnegative integral that permits $+\infty$.
- **Derivative existence.** `HasDerivAt` is stronger evidence than a bare equality for `deriv`. Some totalized definitions return a value even at nondifferentiable points.
- **Improper integrals.** Boundary terms require limits at both ends and integrability proofs.
- **Differentiation under an integral.** A parameter neighborhood needs uniform integrable control. Pointwise finiteness is insufficient.
- **Matrix coordinates.** Symmetric matrices, positive-definite domains, Cholesky factors, and flat arrays need explicit correspondences. Repeated coordinates can invalidate invertibility assumptions.

These obligations can be addressed incrementally. Start with a theorem of explicit scope, then generalize it. Do not remove the mathematical difficulty by assuming the desired conclusion.

## 5. jaxpr to Lean: a parallel route

A **jaxpr** is the intermediate program representation produced by tracing a JAX function. It records inputs, constants, primitive operations, and outputs. A primitive is a basic operation recognized by the system, such as addition, exponentiation, or array reduction. Many Python structures disappear after shapes, dtypes, and static parameters are fixed.

jax-lean provides a useful design for relating generated functions to an intermediate representation. The study fork will support experiments while plan D develops hand-written mathematics. Neither route must wait for the other. [J1, J2; local check on the design]

```text
JAX function --trace--> ClosedJaxpr
                            |
                   Python import/translation
                       /             \
                  Lean IR       Readable Lean function
                     |                 |
                    eval ---- Equality certificate
                                       |
                              Mathematical specification
```

IR means intermediate representation. `eval` is its interpreter, a function that executes the program definition within the model. A **certificate** here is an equality proof checked by Lean.

The local files divide the work as follows:

- `python/jaxlean/translate.py` generates readable functions.
- `python/jaxlean/importers/jaxpr.py` generates an IR with shape and type checks.
- `JaxLean/Core/Jaxpr.lean` defines the IR and its interpreter.
- `python/jaxlean/certify.py` generates equality proof goals.
- `JaxLean/Verification/Certificate.lean` composes local equality certificates.
- `examples/certificates/generated/CertifiedMean.lean` gives a concrete shape `[2]` example.

The certificate proves

$$\operatorname{eval}(\text{imported IR},x)=\text{generated Lean function}(x).$$

This equality does not prove that the Python importer read the original jaxpr faithfully, that shared layout decoding is correct, or that JAX tracing is correct. Shared errors in the two generated paths can still yield a true equality. `SEMANTICS_REFERENCES.md` states this boundary explicitly. [J2; local check]

### 5.1 Why JAX Bessel can be translated in principle

normix expands Bessel evaluation into quadrature, exponentials, arithmetic, and loops. A Lean semantics can describe the whole jaxpr if its operations and control flow have definitions. The absence of a primitive named Bessel is not a fundamental barrier.

The supplied jax-lean copy does not establish immediate compatibility.

| Current normix structure | Local jax-lean boundary | Required work |
|---|---|---|
| JAX 0.9.1 is locked | The copy locks JAX 0.8.0 | Define import rules for the current version; do not downgrade normix for verification |
| `_edge` uses 40 bisection steps and `_fori` | Ordinary translation defaults to at most 32 unrolled `scan` steps; certificates exclude `scan` | Define loop semantics and induction rules, or explicitly restrict the first experiment |
| `_GL_T_J` and `_GL_LOGW_J` are captured floating-point arrays | `JaxprImporter` rejects captured floating-point array constants | Extend constant encoding and record exact stored values |
| Calls include `expm1`, `log1p`, hyperbolic functions, and `hypot` | The support list does not cover every possible expansion | Trace and inspect actual primitives later; no jaxpr was run here |
| `stop_gradient` | The forward model treats it as identity | Specify frozen geometry and AD separately from the derivative of an ordinary composition |
| GIG inversion uses `custom_jvp` and `while_loop` | Ordinary translation inlines only the primal of `custom_jvp_call`; general `while` is unsupported | Separate primal semantics, derivative rules, and termination |
| Mixture sampling and TDR | Random rewrites support only particular `_randint` boundaries | Continuous random models and real PRNG behavior need additional work |

This is a static difference list, not a failed compatibility experiment. `SUPPORTED_JAXPR.md` distinguishes translation support from certificate support. One does not imply the other. [J1; local check]

### 5.2 First bridge milestone and wider coverage

Start a small bridge experiment alongside the mathematical foundations. Suitable targets are the fixed-shape core of `_moments_from_weights` or the pure algebra in `InverseGaussian.from_expectation`. Weight normalization and the centered Gram structure give concrete specifications. Importing the entire GIG solver first would combine too many unresolved obligations.

Allow 1–2 person-weeks for a primitive inventory and one small bridge prototype. Rewriting the importer core requires a separate estimate. Review real-number semantics, clamping conditions, and coverage. Include checks that deliberately corrupt an operation or target and require certificate failure. If two weeks produce only generated Lean text without a checked specification tied to a real function, reassess that implementation approach before expanding it.

The small milestone does not restrict the long-term scope to Bessel. Extend the jaxpr route across distributions, parameter conversions, mixture algebra, moments, fitting, and other traceable normix functions. Write difficult parts by hand in Lean. Record which parts are translated, which are hand-written, and what proves their connection. A hand-written model alone is not an importer certificate.

Translation still leaves approximation proofs. A certificate for all of `_eval_quad` describes a finite-sum algorithm. Connecting it to $\log K$ requires the tail and quadrature analysis in [the Bessel checklist](05_math_checklist.md#6-bessel-functions). Derivative accuracy requires comparison of the corresponding moments, not only function values.

The licensing boundary concerns reuse, not mathematical feasibility. Study and experiment in the fork. Open the license issue and obtain permission before copying or redistributing code whose license is unclear. No code from jax-lean is added to normix by this report.

## 6. Project architecture and review

### 6.1 An independent project in the same repository

Use `formal/lean/` for the hand-written Lean project. normix formulas, coordinate conventions, and approximations can then change in the same commit as theorem statements and coverage mappings. Lean stays outside Python installation dependencies and does not change the JAX runtime path.

**Lake** is Lean's package manager and build tool. **Continuous integration**, or CI, runs automated checks after changes.

| Option | Decision | Reason |
|---|---|---|
| Independent Lake project in the same repository | Selected direction; setup and CI estimated at 2–4 person-days | Code and proofs share a commit; Python and Lean builds remain independent |
| Separate `normix-lean` repository | Defer | Independent releases require synchronization of commits and documentation; reconsider with a separate maintenance team |
| Proofs inside the Python package and runtime | Reject | This confuses mathematical proof with numerical execution and adds installation and device-path costs |

The following tree is planned. These files were not created during the report revision.

```text
formal/lean/
  lean-toolchain
  lakefile.toml
  lake-manifest.json
  Normix.lean
  Normix/
    Foundations/       Integrability, moments, finite-dimensional coordinates
    ExponentialFamily/ Definitions, convexity, derivatives, duality
    SpecialFunctions/  Bessel integrals and identities
    Distributions/     Gamma, IG, GIG, and boundary distributions
    Mixtures/          Mixture measures, GH density, moments, conditional laws
    EM/                Q decomposition, M-steps, restricted convergence
    Numerics/          Finite quadrature, frozen geometry, error bounds
    Bridge/            Correspondence with selected jaxpr real models
  AxiomAudit.lean
  coverage.toml
  decisions.md
  Scratch/             Unfinished work; production modules must not import it
```

**Lean naming conventions.** Lean source files, modules, and directories use UpperCamelCase, following Mathlib style. Definitions use lowerCamelCase, theorems use snake_case, and namespaces use UpperCamelCase. Lake maps module names to paths: `Normix.Numerics.FiniteMoments` corresponds to `Normix/Numerics/FiniteMoments.lean`. This is why the source tree has capitalized names. The lowercase `formal/lean/` is only the package directory; `lean-toolchain`, Lake metadata, and non-Lean tracking files are not module names.

Dependencies run from foundations to distributions, then to mixtures and EM. `Numerics` can use mathematical objects and finite-moment theorems. `Bridge` uses mathematical theorems and generated programs. Mathematical definitions must not depend on a particular trace shape or generated file.

Fix sample spaces and statistic conventions first. An initial model can use finite-dimensional real vectors, explicit Lebesgue reference measures, and named parameter structures. It need not define a class hierarchy for every probability model. Treat the positive GIG interior first and define boundary distributions separately. Since ordinary Lean reals cannot represent the negative infinity returned by `log_prob`, model exact probability with nonnegative densities. Restrict log-density identities to positive-density domains.

### 6.2 Pinned versions, builds, and caches

`lean-toolchain` pins Lean. `lakefile.toml` declares dependencies. `lake-manifest.json` records resolved commits. A moving Mathlib branch should not change the proof environment every day.

The following configuration sketch is **unexecuted**. Its version and SHA come from the local jax-lean configuration. They are not a verified recommendation for the new project. Check compatibility before freezing them.

```toml
name = "normixformal"
version = "0.1.0"
defaultTargets = ["Normix"]

[[require]]
name = "mathlib"
scope = "leanprover-community"
rev = "db584cd6d46c92f209a44c0f1c829460d327499d"

[[lean_lib]]
name = "Normix"
```

The corresponding toolchain example is `leanprover/lean4:v4.33.0`. Initial dependency retrieval requires network access or complete local dependencies. No retrieval was performed here.

A separate Lean CI job could use these commands. Availability of the cache command for the pinned version remains **to verify**.

```bash
cd formal/lean
lake exe cache get
lake build
lake env lean AxiomAudit.lean
```

`lake exe cache get` commonly retrieves compiled Mathlib artifacts; this is recalled tool usage, not a local execution result. It saves dependency compilation time but does not replace the project build. Cache keys should include the operating system, toolchain, and dependency-manifest digest. Proof changes must trigger a new build.

`Normix.lean` should import every production module. CI should compare production files with the coverage inventory so that default builds do not omit files. Run `AxiomAudit.lean` explicitly. Upgrade Mathlib in separate changes and record repair effort. Review version pins, changes in proof scope, and unexpected axioms.

### 6.3 Mapping functions to theorems

Mappings must identify a function and a property. All Lean names below are **proposed**, not existing Mathlib or normix theorems.

| ID | Existing code | Proposed theorem | Covered scope |
|---|---|---|---|
| NM-EF-001 | `ExponentialFamily.expectation_params` | `Normix.EF.grad_log_partition` | Mathematical formula for a regular continuous family; excludes JAX AD correctness |
| NM-EF-002 | `ExponentialFamily.fisher_information` | `Normix.EF.fisher_eq_cov` | Full natural coordinates; excludes restricted-subfamily interface ambiguity |
| NM-GIG-001 | `GIG.raw_moment` | `Normix.GIG.moment_formula` | Exact moments for $a,b>0$; excludes numerical error |
| NM-MIX-001 | `JointNormalMixture.rvs`, `NormalMixture.mean/cov` | `Normix.Mixture.pushforward_law`, `mean_cov` | Ideal independent inputs and required moments |
| NM-GH-001 | `GeneralizedHyperbolic.log_prob` | `Normix.GH.density_eq_integral` | Mathematical density before numerical floors |
| NM-EM-001 | `BatchEMFitter._em_step` | `Normix.EM.likelihood_monotone` | Abstract step with exact posterior and Q improvement |
| NM-BES-001 | `_moments_from_weights`, `BesselMoments.cov` | `Normix.FiniteEF.cov_psd` | Real covariance for finite positive weights |
| NM-BES-002 | `_eval_quad`, `log_kv_moments` | `Normix.Quadrature.frozen_derivatives` | Partial derivatives with fixed geometry; approximation is separate |

A mapping file is sufficient; a new Python decorator is unnecessary. A proposed `coverage.toml` entry is:

```toml
[[claim]]
id = "NM-BES-001"
python_file = "normix/utils/bessel.py"
python_symbol = "_moments_from_weights"
theorem = "Normix.FiniteEF.cov_psd"
lean_file = "Normix/Numerics/FiniteMoments.lean"
theory_page = "docs/design/solvers_and_bessel.md"
scope = "exact-real finite positive weights; no rounding bound"
status = "not-started"
spec_review = "pending"
source_commit = "51e053c57237a29c71b56d3a2a86988277477bd9"
```

Later entries should include domains, assumptions, theorem dependencies, axiom-audit results, and the checked commit. Task status uses the checklist's six states. Coverage is a separate field: conditional theorem, distribution instance, code bridge, and error bound are distinct claims. None is established by this report.

A future docstring label can go in `Notes` without affecting execution:

```text
Formal specification: NM-BES-001.
Exact-real covariance identity; excludes rounding error.
See :doc:`/design/solvers_and_bessel`.
```

The label does not claim complete source verification. A website page can name the theorem, scope, and checked commit, then link to an existing public source revision or generated Lean documentation anchor. Do not guess anchors or display a passing badge before checks exist. `docs/theory` and docstrings must not link into `dev-notes`; the reports remain repository research material outside the Sphinx website.

Allow 2–3 person-days for the initial mapping and public-link mechanism. Review what each label promises. Source or mathematical-page changes should require reconfirmation of affected mappings. An unchanged symbol name does not establish unchanged semantics.

### 6.4 Proposition cards and specification review

Write each proposition card in English and review it before proof work. Use this template:

| Field | Required content |
|---|---|
| ID and status | Checklist ID and current task status |
| Statement in prose | Exact mathematical claim, including quantifiers and conclusion |
| Lean-style signature | Proposed typed statement; mark uncompiled notation and proposed helper definitions |
| Assumptions | Parameter domain, support, integrability, independence, regularity, and nondegeneracy as applicable |
| Why this is the intended spec | Explain how the statement captures the mathematical contract without assuming its conclusion |
| Covered normix function | Exact path and symbol, source baseline, and mathematical or implementation layer |
| Risks of a weakened spec | State what would be lost by adding assumptions, shrinking the domain, or omitting branches |
| Evidence and review | local check / derivation / to verify; reviewer notes, proof dependencies, and PR |

For example, the parameter-roundtrip proposition has the following card. It is a draft example; the corresponding checklist item remains `not-started` until work is recorded there.

| Field | Content |
|---|---|
| ID | EF-01, GIG $b$ parameter roundtrip |
| Statement in prose | For every real $b$, converting to $\theta_2=-b/2$ and back gives $b$. |
| Lean-style signature | `theorem gig_b_roundtrip (b : ℝ) : -2 * (-b / 2) = b` |
| Assumptions | Real arithmetic only. Valid GIG parameters additionally require $b>0$ and a separate domain proof. |
| Why this is the intended spec | It checks the sign and factor used by both conversion formulas. |
| Covered normix function | `normix/distributions/generalized_inverse_gaussian.py`: `natural_params`, `from_natural`; baseline `51e053c57237a29c71b56d3a2a86988277477bd9`. It covers the algebra before constructor clamping. |
| Risks of a weakened spec | Checking one positive input would lose the universal identity. Treating the unclamped formula as the full constructor would omit boundary behavior. |
| Evidence and review | Formula correspondence: local check. Algebra: derivation. Lean compilation and implementation correspondence: to verify. |

A second example captures the finite-moment target without hiding its structure in an assumed covariance predicate. The following signature is a **Lean-style sketch**, not compiled Lean. Its normalization assumption excludes an empty index set.

```lean
theorem centered_gram_nonneg
    {n d : ℕ} (p : Fin n → ℝ) (s : Fin n → Fin d → ℝ)
    (hp : ∀ i, 0 ≤ p i) (hMass : ∑ i, p i = 1)
    (v : Fin d → ℝ) :
    0 ≤ ∑ i, p i *
      (∑ j, v j * (s i j - ∑ k, p k * s k j)) ^ 2
```

| Field | Content |
|---|---|
| ID | EF-09, centered finite covariance is positive semidefinite |
| Statement in prose | A normalized finite distribution with nonnegative weights has a nonnegative centered quadratic form in every direction. |
| Assumptions | Finite real statistics, nonnegative weights, total mass $1$. Normalizing exponential weights also needs positive total mass. |
| Why this is the intended spec | The sum of squares exposes the Gram structure. A separate equality must connect this form to the covariance matrix returned by the modeled function. |
| Covered normix function | `normix/utils/bessel.py`: `_moments_from_weights`, `BesselMoments.cov`; same baseline as above. Exact real finite arithmetic only. |
| Risks of a weakened spec | Assuming positive semidefiniteness would remove the target. Proving only one direction or one shape would not establish the general covariance claim. Floating-point or strict positive-definiteness claims would exceed this statement. |
| Evidence and review | Centered structure: local check. Sum-of-squares argument: derivation. Signature compilation and matrix correspondence: to verify. |

Review at least these questions for every card:

- Has “all valid parameters” become a convenient small interval while the coverage label stayed unchanged?
- Is normalization, independence, or EM monotonicity now an assumption? Such assumptions may define an abstract lemma, but they do not complete a concrete instance.
- Are there examples satisfying the assumptions? Requiring all moments while claiming the divergent InverseGamma region can give an invalid or vacuous scope.
- Does the mapping omit floors, finite iteration limits, failure values, or boundary points?
- Is Fisher information defined as the Hessian to avoid proving its relation to score covariance?
- Does a one-dimensional or length-two proof claim arbitrary-dimensional coverage?

Proof authors may change proofs without silently changing reviewed statements. If a proof stalls, distinguish a mathematical gap, a library gap, and engineering cost. Changed assumptions require a new review and a record of why the original target remains incomplete.

**`sorry` budget.** Production exports allow none. Exploratory files may contain identified gaps, but production modules must not import them. A pilot limit of five open holes is a management suggestion, not a Lean restriction. Text search alone is insufficient because incomplete proofs can enter through imports.

**Axiom audit.** Inspect transitive dependencies of every exported theorem. These proposed commands were not executed:

```lean
#print axioms Normix.FiniteEF.cov_psd
#print axioms Normix.GIG.moment_formula
```

Reject `sorryAx` and unapproved project axioms. Recalled Lean practice allows ordinary classical mathematics to depend on foundational axioms such as `propext`, `Classical.choice`, and `Quot.sound`. Define an explicit acceptable set and explain additions. Also record trust in computation outside the kernel.

An axiom list cannot detect a wrong definition or an overly strong premise. `#print axioms` does not replace specification review. Coverage records should retain summaries of definitions and statements. Replacing a definition must not silently count as proving the original target.

Bridge checks should deliberately corrupt operations, coordinates, or branches and require certificate failure. jax-lean includes such test source in `tests/test_certification.py` and `test_review_regressions.py`. Those files were read, not executed. [J5; local check]

### 6.5 Proof work and mathematical review

| Work | Proof engineering | Mathematical review | Estimated effort |
|---|---|---|---|
| Pinned-library audit | Find definitions, write minimal compilation examples, record gaps | Normalization and parameter conventions | 2–4 person-days; 2–3 review hours |
| Proposition card | Extract assumptions and identify counterexamples | Usefulness and correctness of the statement | 30–60 minutes for a simple target; 2–4 hours for analysis |
| Proof implementation | Small lemmas, tactics, repeated compilation | Main argument and non-obvious assumptions | See the separate checklist |
| Code correspondence | Maintain mappings; mark clamps and approximations | Whether the API label is justified | 1–2 review hours per group |
| Special-function error | Implement the chosen analysis and auxiliary bounds | Dominating functions, error budget, domain | Usually 2–4 review hours weekly; specialist review when needed |

Reviewers need not inspect every tactic line. They must read each statement and its definitions. Proof difficulty does not authorize lowering the specification.

Record constraints, proposals, rejected alternatives, evidence date, decision authority, and reopening conditions. Section 8 distinguishes accepted directions from remaining recommendations. Implementation details and estimated budgets are not automatically approved by accepting the direction.

## 7. Roles of TLA+ and Hypothesis

**TLA+** specifies states and state transitions. Concurrency is not a prerequisite. A sequential program with stopping flags, failure recovery, and execution-path selection can also be modeled as a state machine.

A **safety property** says that a bad event never occurs, such as overwriting the accepted model after stopping. A **liveness property** says that an event eventually occurs, such as eventual return under scheduling and termination assumptions. Liveness often needs fairness: an enabled step cannot be ignored forever.

TLC can enumerate states and transitions in a finite TLA+ model to find violations. Passing a finite configuration does not prove all system sizes or establish that Python implements the model. TLA+ also supports deductive proof, which need not be an initial project requirement. These descriptions are recalled tool knowledge; no version or execution was checked here.

**Hypothesis** is a Python property-testing tool. It generates inputs under supplied rules, runs real code, and tries to shrink failures. Finite executions are not universal proofs. Stateful tests can exercise operation sequences; explicit state modeling and exhaustive finite exploration remain separate tasks.

| Tool | Suitable normix targets | Not established by these targets | Pilot effort and review |
|---|---|---|---|
| Lean | GIG normalization, mixture moments, EM assumptions | Automatic correctness of floating-point code | See the roadmap; review mathematical statements |
| TLA+ | `_fit_scan` freezes after stopping, retains the old model on failure, preserves backend paths | Bessel accuracy, Q improvement, numerical solver convergence | 2–5 person-days; review state abstraction and protocol |
| Hypothesis | Real conversion roundtrips, domain boundaries, backend agreement, stopping sequences | Properties for all inputs | 2–5 person-days for a small set; review input domains and tolerances |

`BatchEMFitter._fit_scan` carries the model, `converged`, `diverged`, `solver_failed`, counters, and likelihood history. Abstract numerical results as finite/nonfinite candidates and stop/continue inputs. Start with three properties: [N6; local check]

1. Once a stopping flag takes effect, padding iterations preserve the model, flags, and effective iteration count.
2. A nonfinite mixture candidate returns the last usable model and marks solver failure.
3. Failure recovery preserves the requested backend and device path.

Finiteness is a reasonable state-machine input. It does not imply parameter validity. Eventual stopping does not imply mathematical convergence. Review the mapping between actual branches and the model, or add stateful tests.

Preserve S9/S11: `backend='jax'` executes on the caller's chosen JAX device, which may be a CPU. `backend='cpu'` selects the explicit NumPy/SciPy path. Proof convenience does not justify automatic offloading. `tests/test_no_host_callbacks.py` already contains source and nested-jaxpr checks that address this implementation constraint directly. [N7; local check]

There is no need to model the entire class hierarchy in TLA+. Algebraic relations in pure functions belong in Lean or tests. Stopping, recovery, batch state, and future concurrent scheduling or checkpointing are more suitable state-machine targets. Keep this pilot independent of the Lean mathematics project.

## 8. Decisions and exclusions

These records are dated 2026-10-24. “Accepted” identifies directions selected by the maintainer for this revision. “Recommendation” identifies remaining proposals. Rejection here concerns the stated project scope, not permanent limits of a tool.

| ID / status | Scope and decision | Reason and evidence | Reopening condition |
|---|---|---|---|
| FV-00 / existing constraint | Preserve JAX-native execution and caller-selected backend/device | Existing S9/S11 and the task constraints | An explicit change to the execution contract |
| FV-01 / accepted direction | Plan D uses `formal/lean/` in this repository, without a Python runtime dependency on Lean | Code and mathematical mappings can share commits; Lake builds independently | A separate maintenance team or cross-project mathematics library |
| FV-02 / recommendation | Start with finite moments, then continuous probability, then algorithmic error | `_moments_from_weights` has finite-sum structure; continuous Bessel analysis still needs foundations | A pinned Mathlib audit establishes that key analysis is already available |
| FV-03 / recommendation | Label mathematics, exact algorithms, approximation, and machine execution separately | jax-lean documents real semantics and importer boundaries | Add coverage only with corresponding machine or compiler proofs |
| FV-04 / accepted licensing boundary | Fork jax-lean for study and experiments; open an issue asking for a license; obtain permission before reuse or redistribution | No root `LICENSE` in the read copy; examples may have separate licenses | Clear authorization and evidence that reuse reduces maintenance cost |
| FV-05 / reject | Switch to CPU/SciPy or replace the runtime with Lean solely for proof convenience | Conflicts with FV-00; mathematical proofs do not require a backend change | A separately authorized reference implementation that leaves production behavior explicit |
| FV-06 / accepted scope, staged delivery | Develop plan D and the bridge in parallel; aim for broad normix coverage; hand-write difficult Lean parts | Selected direction; current importer gaps do not define the long-term scope | Reassess individual milestones using costs and proof evidence; no all-at-once delivery claim |
| FV-07 / defer | End-to-end proof of all floating-point, XLA, GPU, and PRNG behavior | Machine semantics exceed the initial scope | A concrete short kernel requiring strict bounds and a dedicated budget |
| FV-08 / reject | One abstraction treating every distribution as a full EF | Joint VG/NIG/NInvG are restricted subfamilies; marginal mixtures do not carry the same interface meaning | Extend only after the structures are distinguished accurately |
| FV-09 / defer | TLA+ coverage of the whole architecture | Pure functions and mathematical layers offer little state-space benefit | A specific stopping/recovery or concurrent protocol needs verification |
| FV-10 / accepted | English reports and proposition cards; separate mathematical checklist | Selected documentation and tracking policy | An explicit documentation-policy change |
| FV-11 / accepted | Mathlib-style Lean names under the source tree | Lake maps capitalized module components to paths; see Section 6.1 | A deliberate project naming-policy change |

Alternative tools remain possible. Coq/Rocq and Isabelle/HOL are proof assistants. Dafny expresses program contracts with preconditions, postconditions, and loop invariants. Verus applies related methods to Rust. Preconditions constrain calls, postconditions constrain returns, and loop invariants must hold throughout iterations.

| Alternative | Why it is not selected now | When to reconsider |
|---|---|---|
| Coq/Rocq | Lean and Mathlib are selected; changing libraries requires a new dependency and training assessment | A skilled contributor and relevant existing analysis library justify a 1–2 person-week comparison; its formal mathematics ecosystem is established |
| Isabelle/HOL | Higher-order logic and analysis are viable, but no GIG/Bessel advantage was verified here; two stacks add maintenance | Existing related proofs justify a 1–2 person-week comparison on a continuous-integral target |
| Dafny/Verus | They suit contracts, loops, and data structures. Verus targets Rust; normix uses Python/JAX. Special functions still require mathematics infrastructure. | An independent Rust kernel or discrete state component; do not migrate the main implementation for the tool |
| Documentation alone | Readable derivations remain necessary but do not mechanically check hypotheses, substitutions, or differentiation conditions | Low-priority research may remain on paper, without a formalized label |
| Hypothesis alone | It tests real code cheaply but finite samples cannot prove integrals or universal parameter claims | Retain it as a complement to proofs |

These comparisons use general knowledge and project-fit judgments. No comparative experiment was performed. Nonselection does not mean that another tool cannot formalize probability.

Initial claims must also respect these limits:

- Do not claim that a floating-point Hessian is positive semidefinite for every valid input. Prove the real structure, then analyze rounding.
- Do not claim that every EM run is monotone and reaches a global MLE. Separate exact algorithms, approximate branches, and objectives.
- Do not equate `backend='jax'` with GPU execution or use CPU-path success as evidence for JAX transformations and device behavior.
- Do not generalize a fixed-shape certificate to arbitrary dimension or treat primal `custom_jvp` translation as proof of its derivative rule.
- Do not declare Bessel identities as project axioms and label dependent GIG results unconditionally complete.
- Do not replace specialized Gamma, NIG, or other runtime formulas with GH merely because mathematical lemmas are shared.

## 9. Phases and exit conditions

Each phase has its own deliverable and stopping condition. Pausing a branch does not invalidate proved foundations. All time estimates concern a future environment that can install dependencies and run Lean. None of these phases was executed for this report.

| Phase | Deliverable and acceptance condition | Budget / review | Stop or reduce scope when |
|---|---|---|---|
| P0: specifications and dependencies | Pin Lean/Mathlib; audit analysis gaps; review 5–8 cards, statistic order, and documentation discrepancies | About 1 person-week; 3–5 review hours | No buildable pinned environment or no agreement on definitions |
| P1: useful foundations | Conversions, positive finite-weight normalization, Gram positivity, fixed-geometry first moments; at least three theorems tied to real functions; no production `sorry` | 2–3 person-weeks; 1–2 review hours weekly | Week 3 still yields only abstract demonstrations or weakened statements; reassess proof-engineering support |
| P2: first continuous probability chain | Gamma normalization and moments; ideal mixture map, mean, covariance; at least one continuous instance | 4–8 person-weeks; 2 review hours weekly | Four weeks remain blocked on generic integrability without an instance; retain finite results and avoid opening Bessel and EM analysis together |
| P3: Bessel and GIG | Integrability, positivity, symmetry; GIG normalization and moments; then order/argument derivatives | 6–12 person-weeks; 2–4 review hours weekly | No tail-bound route after four weeks, or a new analysis theory is required; conditional GH/EM lemmas may continue with explicit labels |
| P4: GH and EM | GH density equals mixture integral; conditional GIG; normal M-step; abstract one-step monotonicity; implementation gaps recorded | 4–8 person-weeks; 2–3 review hours weekly | Floors and approximate solves cannot be matched to the hypotheses; retain mathematical results without a complete fitter certificate |
| P5: bounded numerical target | Explicit bound for a Bessel value or derivative on a specified compact parameter domain | 2–6 person-months; regular mathematical and numerical review | No computable tail/quadrature bound after 2–3 weeks, or bounds are too loose for the use case |
| Parallel bridge | First real jaxpr linked to a proved spec; importer trust recorded; wider primitive coverage tracked | First milestone: 1–2 person-weeks; 2–4 review hours | No checkable spec within budget; reassess that importer approach and use explicit hand-written models where needed |
| Optional state machine | Three `_fit_scan` state properties and their code mapping | 2–5 person-days; 1–2 review hours | The model only repeats branch conditions without addressing sequences that tests struggle to cover |

P0–P1 and the first bridge milestone are the initial work packages for the parallel direction. Selected continuous results through P2–P4 require a budget measured in months. Adding asymptotics, sampler details, EM stationarity, and numerical error could exceed a year of skilled work. A fixed total estimate before P0 would not be justified.

At each phase boundary, record proved statements, unproved assumptions, source-code gaps, and actual hours. Use that evidence to choose the next milestone rather than counting theorems.

## 10. Priority recommendations

1. **Start P0–P1 in `formal/lean/` alongside the bridge.** Allow 3–4 person-weeks for the mathematical foundation. Prioritize parameter conventions, finite exponential sums, and `BesselMoments` covariance structure. Review 5–8 cards before coverage labels. These targets are close to existing code and need less analysis infrastructure.
2. **Build a complete continuous probability chain through GIG/GH.** Stage P2–P4 over months. Prioritize mixture representations, moments, normalization, posterior conjugacy, and M-steps. Review definitions, boundaries, and integrability.
3. **Prove conditional one-step EM monotonicity first.** Allow about 3–6 person-weeks after density foundations. Treat stationary-point convergence as a separate project. Select numerical approximations for further error analysis explicitly.
4. **Grow the jaxpr bridge across normix.** Use the initial 1–2 person-week milestone to establish readable functions, an independent IR, and equality certificates. Track unsupported primitives and hand-written replacements. Preserve importer, floating-point, and PRNG boundaries and the JAX backend contract.
5. **Select Bessel error bounds, concrete samplers, and TLA+ by need.** Error bounds require months of research. A state-machine pilot can test value in 2–5 person-days. Hypothesis can add boundary tests in parallel, without becoming a prerequisite for the mathematics project. None was implemented here.

## 11. Open questions for the maintainer

These questions concern unresolved priorities and review capacity. They do not reopen the accepted English, checklist, fork, naming, or parallel-development decisions.

1. Which first mathematical acceptance target should lead: finite Bessel moments or the Gamma/ideal-mixture probability chain? Finite moments remain the recommended starting point.
2. Should the first domain be $a,b>0$, with exact Gamma/InverseGamma boundaries handled separately? Are limit theorems also required?
3. Should early coverage emphasize formula traceability or explicit error bounds? Error bounds require a parameter range, precision, and error metric.
4. Is 1–2 hours of weekly proposition review available? Is additional Lean or numerical-analysis review available for special functions?
5. Is the long-term EM target mathematical monotonicity, stationary accumulation points, or guarantees for the actual fitter with floors and fallbacks? Their costs differ substantially.
6. Which normix areas should follow the first bridge milestone? If the Lean project later moves to a separate repository, who will maintain commit mappings and theorem links?

## Sources

These paths record the original static-reading evidence. Function names locate source; they do not report execution. N8 is a topic inventory, not a formula-by-formula audit. No external links were accessed. Earlier social-media claims, recent-paper claims, and success-rate figures were not reverified or used to estimate proof costs.

- **Repository guidance**: [AGENTS.md](../../../AGENTS.md), [dev-notes/README.md](../../README.md), [project catalog](../../../.cursor/rules/project-overview.mdc), [cross-link conventions](../../../.cursor/rules/docs-cross-links.mdc), [design documentation rules](../../../.cursor/rules/maintain-design-docs.mdc), [theory documentation rules](../../../.cursor/rules/maintain-theory-docs.mdc), and [Python conventions](../../../.cursor/rules/coding-conventions.mdc).
- **N1 / distribution theory**: [gig.md](../../../docs/theory/gig.md), [gh.md](../../../docs/theory/gh.md), [em_algorithm.md](../../../docs/theory/em_algorithm.md).
- **N2 / exponential families**: [exponential_family.py](../../../normix/exponential_family.py), [normal.py](../../../normix/distributions/normal.py), [generalized_inverse_gaussian.py](../../../normix/distributions/generalized_inverse_gaussian.py), [exponential-family design](../../../docs/design/exponential_family.md). Focus: triad, `natural_params`, `_implicit_natural_params`, `raw_moment`.
- **N3 / concrete distributions**: [gamma.py](../../../normix/distributions/gamma.py), [inverse_gamma.py](../../../normix/distributions/inverse_gamma.py), [inverse_gaussian.py](../../../normix/distributions/inverse_gaussian.py), [generalized_hyperbolic.py](../../../normix/distributions/generalized_hyperbolic.py), [normal_inverse_gaussian.py](../../../normix/distributions/normal_inverse_gaussian.py), [variance_gamma.py](../../../normix/distributions/variance_gamma.py), [normal_inverse_gamma.py](../../../normix/distributions/normal_inverse_gamma.py). Focus: densities, moments, CDFs, sampling, expectation inversion.
- **N4 / mixtures**: [joint.py](../../../normix/mixtures/joint.py), [marginal.py](../../../normix/mixtures/marginal.py), [mixture design](../../../docs/design/mixtures.md). Focus: `rvs`, `_ambient_moment_contract`, `_posterior_gig_params`, `_mstep_normal_params`, `_rescale`, and the univariate mixin.
- **N5 / Bessel and sampling**: [bessel.py](../../../normix/utils/bessel.py), [rvs.py](../../../normix/utils/rvs.py), [bessel_moment_kernel.md](../../tech_notes/bessel_moment_kernel.md), [gig_rvs.md](../../tech_notes/gig_rvs.md). Historical performance and accuracy figures were not remeasured.
- **N6 / fitting**: [em.py](../../../normix/fitting/em.py), [eta.py](../../../normix/fitting/eta.py), [solvers.py](../../../normix/fitting/solvers.py), [EM design](../../../docs/design/em_framework.md), [gig_eta_to_theta.md](../../tech_notes/gig_eta_to_theta.md). Focus: `_fit_scan`, `_em_step`, `_mcecm_step`, named statistics, solver boundaries.
- **N7 / architecture decisions**: [design.md](../../design/design.md), [solvers_and_bessel.md](../../design/solvers_and_bessel.md), [public solver design](../../../docs/design/solvers_and_bessel.md), [ARCHITECTURE.md](../../ARCHITECTURE.md), [test_no_host_callbacks.py](../../../tests/test_no_host_callbacks.py). S5/S9/S10/S11 are local records; associated external PR links were not opened.
- **N8 / other theory topics**: [varentropy.md](../../../docs/theory/varentropy.md), [online_em.md](../../../docs/theory/online_em.md), [shrinkage.md](../../../docs/theory/shrinkage.md), [factor_analysis.md](../../../docs/theory/factor_analysis.md), [mean_risk_optimization.md](../../../docs/theory/mean_risk_optimization.md), [cvar_derivatives.md](../../../docs/theory/cvar_derivatives.md), [transaction_costs.md](../../../docs/theory/transaction_costs.md), [enb.md](../../../docs/theory/enb.md), [generalized_enb.md](../../../docs/theory/generalized_enb.md), [quadrature_scenarios.md](../../../docs/theory/quadrature_scenarios.md).

A local copy of https://github.com/srush/jax-lean, by srush, was read. The following are paths within that copy, not links to files in normix:

- **J1 / scope and versions**: `README.md`, `ARCHITECTURE.md`, `SUPPORTED_JAXPR.md`, `SEMANTICS_REFERENCES.md`, `lakefile.toml`, `lake-manifest.json`, `lean-toolchain`, `pyproject.toml`. normix constraints are in [pyproject.toml](../../../pyproject.toml) and [uv.lock](../../../uv.lock).
- **J2 / translation and certificates**: `python/jaxlean/translate.py`, `python/jaxlean/importers/jaxpr.py`, `python/jaxlean/certify.py`, `JaxLean/Core/Jaxpr.lean`, `JaxLean/Core/RealOps.lean`, `JaxLean/Verification/Certificate.lean`, `examples/certificates/generated/CertifiedMean.lean`.
- **J3 / derivative example**: `examples/autodiff/code.py`, `examples/autodiff/proofs/Jvp.lean`.
- **J4 / random models**: `JaxLean/Stdlib/Random.lean`, `JaxLean/Stdlib/BatchProbability.lean`, `python/jaxlean/rewrites/random_spec.py`, `examples/random_program/proofs/RandomProgramProofs.lean`.
- **J5 / test source, read only**: `tests/test_certification.py`, `tests/test_review_regressions.py`, `tests/test_random_translation.py`.

External mathematical background and unresolved checks:

- *Maximum Likelihood from Incomplete Data via the EM Algorithm* (1977). EM decomposition and monotonicity are recalled literature; title and year also appear in local [references.md](../../../docs/references.md).
- *On the Convergence Properties of the EM Algorithm* (1983). Title and year are recalled. Exact hypotheses, theorem numbers, and applicability to GH are **to verify**.
- *Graphical Models, Exponential Families, and Variational Inference* (2008). Exponential-family and convex-duality background, from recalled literature.
- *NIST Handbook of Mathematical Functions* (2010). Bessel identities and asymptotics, from recalled literature. The current online DLMF version and exact formula locations are **to verify**.
- *Random Variate Generation for the Generalized Inverse Gaussian Distribution* (2014). Recorded in local technical material and references. The original proof and its correspondence to the current discrete envelope are **to verify**.
- *On-line Expectation-Maximization Algorithm for Latent Data Models* (2009). Title and year occur in local references. Stochastic convergence assumptions and their relation to custom normix updates are **to verify**.
- General descriptions of Lean, Mathlib, TLA+/TLC, Hypothesis, Coq/Rocq, Isabelle/HOL, Dafny, and Verus come from tool knowledge or recalled literature. Current capabilities were not externally checked. No unconfirmed Mathlib theorem names are supplied.
