# Integrated verification roadmap

Updated 2026-10-31 against master `d557399905c533b3f15825941be6f04a6e897584`.
Historical planning baseline: `51e053c57237a29c71b56d3a2a86988277477bd9`.

Start with properties of real normix code, while building hand-written Lean models and a jaxpr-to-Lean bridge in parallel. Property tests offer the earliest feedback on backend routing, parameter conventions, and numerical failures. Lean supplies precise mathematical statements; the bridge connects selected programs to those statements. Extend the bridge across normix, using hand-written Lean for parts that are difficult to translate. Keep the mathematical inventory in [05_math_checklist.md](05_math_checklist.md). Treat the first three months as a sequence of bounded deliveries, with one maintainer reviewing specifications and evidence. A small TLA+ publishing model is optional. Continuous Bessel analysis, numerical error bounds, and complete GH/EM proofs require longer work.

## Evidence and decisions

This roadmap combines [report 01](01_lean_math_feasibility.md),
[report 02](02_jaxpr_to_lean_feasibility.md), and
[report 03](03_tla_vs_hypothesis.md). All three reports were read before this
roadmap was written. Repository guidance came from [AGENTS.md](../../../AGENTS.md)
and [dev-notes/README.md](../../README.md). This update checks documentation,
TOML, and the workflow snippet. No normix tests, JAX traces, Lean builds, TLC
checks, or benchmarks were run.

The evidence labels have these meanings:

| Label | Meaning here |
|---|---|
| **local check** | A fact found in the local reports or repository guidance. A source-code finding inherited from a report is marked **local check (reported)**. It has not been independently reproduced here. |
| **derivation** | A mathematical argument from explicit assumptions. It is not a checked Lean proof. |
| **literature memory** | Mathematical or tool knowledge carried by the reports. External sources and current tool versions have not been checked again. |
| **to verify** | A future executable check, library lookup, source review, or unresolved condition. |

W2 M0 implementation merged in PR #161 (`80073f3`): 31 exported theorems
across three modules, 31 axiom-audit entries, and six statement cards.
This includes part of W2 M1. Build evidence is recorded in
[decisions.md](../../../formal/lean/decisions.md); no build was rerun for this update.
**Merged** and **statement-reviewed** are separate statuses: all six cards
remain `statement-drafted`, with specification review **pending**. The remaining
milestones and their acceptance criteria are **to verify**. Effort ranges are
planning estimates, not measurements.

The following decisions govern the plan:

- Reports and proposition cards use English. The mathematical inventory is a separate tracking file: see [05_math_checklist.md](05_math_checklist.md).
- Plan D, the hand-written project under `formal/lean/`, proceeds alongside the jaxpr bridge. Neither waits for the other to finish.
- The bridge uses the existing [xshi19/jax-lean fork](https://github.com/xshi19/jax-lean). Inspect its revision, capabilities, pins, and permission record before the first pilot. Broad normix coverage remains a backlog.
- The fork is for study and experiments while licensing is unresolved. Open an issue asking for a license. Reuse or redistribution of its code requires the author's permission; do not copy its code into normix until clarified.
- Hand-written mathematical definitions and proofs belong in W2. W3 adds generated programs, translation certificates, and explicit connections to W2.
- Lean source files, modules, directories, and namespaces use UpperCamelCase. Definitions use lowerCamelCase; theorems use snake_case. Lake maps module names to paths. The lowercase `formal/lean/` is the package directory.

These decisions supersede the reports' earlier preference for an independent
exporter and their treatment of the bridge as an optional successor to W2.
They do not remove the reports' technical gaps or evidence limits.

**Local check (reported):** a local copy of https://github.com/srush/jax-lean was read
for the source reports. That copy had no root `LICENSE` file. Some examples had
separate licenses. Its upstream revision and reported commit count were not
independently established. No current upstream state is claimed here.

## Priority order

1. **First, review the merged baseline and add Lean CI.** Review the six W2 cards and enforce the axiom policy. Keep W1 backend, round-trip, and no-callback checks active. Audit the existing W3 fork and its first target within the diagnostic cap.
2. **Second, connect `Gamma.var` to its exact formula.** Trace the real method with both parameters varying, certify the translation, and prove equality to $\alpha/\beta^2$ for shape $\alpha>0$ and rate $\beta>0$. Derive positivity afterwards. W2 completes finite-dimensional EF-08/EF-10 in parallel. After the pilot, choose one bounded follow-up below. Issue #136 already landed as `05b0785`; later Bessel work uses current `_geometry`.
3. **Third, expand along demonstrated dependencies.** W2 develops continuous exponential families, distributions, mixtures, and EM. W3 broadens coverage by reusable primitive families, including non-Bessel code. W1 supplies independent numerical checks throughout. Admit W4 only for a defined publishing policy. Defer expensive error proofs until a short analysis study identifies a useful bound.

This order ranks review priority, not exclusive execution slots. W1 can deliver
tests while W2 and W3 develop proofs. A blocked Bessel translation does not block
Gamma properties, finite mathematics, or a non-Bessel bridge target.

## Effort and review assumptions

One engineering day means about six focused hours of AI-assisted work. Five days
make an engineering week. Estimates assume access to the required dependencies
and competence with Lean proof development. They exclude unrelated numerical
repairs, hardware waits, and unresolved licensing delays.

One maintainer reviews all streams. Review hours are additional to engineering
effort. Statements and definitions require human review. Within a reviewed Lean
specification, AI edits proofs only; definition, domain, assumption, and source-slice
changes require visible specification revisions and renewed review. AI can also
prepare tests, source mappings, and evidence records. Learning Lean from scratch
needs extra calendar time; the reports suggest at least twice the allowance.

The tables state a review allowance for each milestone. The calendar allocates
only the portions planned for its first 12 weeks. Later milestone estimates do
not fit inside that calendar merely because implementation can run in parallel.

## W1: property tests of real code

W1 extends existing pytest coverage with a small Hypothesis suite. It retains
deterministic enumeration where there are only two backends or a few failure
states. Hypothesis executes actual Python/JAX code and reduces failing inputs;
it does not prove a universal property (**literature memory**).

**Local check (reported):** existing entry points include
`tests/test_distribution_conversions.py`, `tests/test_bessel_contract.py`,
`tests/test_no_host_callbacks.py`, and `tests/test_em_cpu_vs_jax.py`.
The reports found inconsistent handling of unknown backend names in `log_kv`
and `GIG.expectation_params_batch`. These are candidates for reproduction, not confirmed
runtime failures or completed fixes.

| Milestone | Deliverable and acceptance criterion | Effort | Independent AI work; maintainer review | Dependencies and kill criteria |
|---|---|---|---|---|
| M0: domains and execution budget | A short property plan, development dependency proposal, and fixed shape/dtype profiles. Each property names its real entry point, domain, oracle, exclusions, and tolerance source. Acceptance: reviewed first batch and a recorded timing baseline when execution becomes available. | 0.5–1 day; review 1 hour. | AI extracts existing assertions and prepares strategies. Review domains, independent references, and CI cost. | No W2/W3 dependency. Stop expansion if no independent oracle or meaningful invariant can be stated. |
| M1: contracts and round trips | Six to eight small properties covering backend routing, invalid names, classical/natural round trips, immutable updates, and nested no-callback jaxpr. Cover JAX GIG cold/hot starts and a fixed batched case. Acceptance: real-code checks pass; a targeted mutation fails for the intended reason; failures retain a reproducible example. | 3–5 days; review 3–5 hours. | AI writes tests, reduces examples, and checks traversal coverage. Review routing, constructor clamps, and coordinate conventions. | W1 M0. Share the graph inventory with W3 M0. Stop increasing generated cases if compilation dominates; fix test structure before adding cases. Confirmed bugs become separate failing-test-first repairs. |
| M2: Bessel reference and derivative checks | Compare values and first/second derivatives with the mpmath reference pipeline. Keep AD/moment consistency, symmetry, and recurrence checks separate. Acceptance: stable high-precision references, component-specific tolerances, finite-value checks, and results for eager/JIT/vmap on recorded devices. | 3–5 days; review 2–3 hours. | AI extends strategies and reference records using `scripts/gen_bessel_reference.py` and `tests/data/bessel_reference.json`. Review precision stability, parameter regions, and absolute tolerances near zero. | W1 M0–M1. Check affected cases against landed #136 (`05b0785`). No Lean prerequisite. Stop widening the domain if the oracle is unstable or failures remain unexplained. |
| M3: inverse maps and EM | Slow properties for expectation round trips, residuals, and ordinary batch EM objective changes. Start with NIG, then GH. Acceptance: evaluate each accepted model on the same data; check the stated tolerance; cover stop, failure, and iteration-budget outcomes. | 3–5 days; review 3–5 hours. | AI prepares valid moment generators and failure injection. Review floors, approximate solves, objective definitions, and the distinction between batch and incremental EM. | W1 M1–M2. W2 cards inform assumptions but completed proofs are not required. Stop a claimed monotonicity test if the update lacks a justified objective; record that gap instead of filtering failures away. |
| M4: broader contracts and stable operation | Selected cumulative-weight, CDF/table, sampling, and finance properties from report 03. Acceptance: independent references where needed, recorded statistical error budgets, correct markers, and a separate workflow within its measured budget. Retain useful counterexamples as fixed regressions. | 2–4 days for one selected group; review 2–4 hours. | AI tests `SampleWeightedUpdate`, table snapshots, or a selected finance constraint. Review moment existence, sampling assumptions, and local versus global optimization claims. | W1 M1–M3. Prioritize properties that support W2/W3 targets. Stop broadening if review or runtime cost exceeds the phase budget. Keep the accepted subset. |

### Property contracts

Generate valid model parameters before deriving expectation parameters. Do not
treat arbitrary real triples as feasible GIG moments. Keep dimensions, dtype,
method, and iteration limits fixed in each profile to avoid repeated compilation.
Enable x64 before importing modules that create constant arrays. Test float32
separately with its own tolerance contract.

Round trips need an independent formula or reference. A wrong forward map and a
wrong inverse can cancel. CPU/JAX agreement also needs an independent reference
because both paths can share the same algorithm. mpmath gives numerical evidence,
not a formal interval bound. These limits are **derivation / literature memory**.

For batch EM, let $\ell_t$ be the mean log-likelihood of the actual model after
update $t$, evaluated on fixed data. Use the proposed test condition
$\ell_{t+1}-\ell_t\ge-(\varepsilon_a+\varepsilon_r\max(1,|\ell_t|))$.
Here $\varepsilon_a$ and $\varepsilon_r$ are calibrated absolute and relative
tolerances, respectively. Their values remain **to verify**. A test tolerance
is not a proved approximation bound.

**Local check (reported):** `EStepResult.log_lik` describes the input to the E-step.
The returned model may include a further M-step. Recompute its objective instead
of treating the last history entry as its likelihood. Floors, clamps, approximate
solves, arbitrary shrinkage, and minibatches can invalidate ordinary EM
monotonicity assumptions. Test a penalized objective only when the penalty is
defined. Incremental EM does not inherit full-data stepwise monotonicity.

No-callback checks must traverse nested `scan`, `while`, `cond`, `jit`, and custom
derivative programs. Include a negative example with a callback inside a nested
branch. Check that the intended solver is present, so an empty wrapper cannot
pass. Record value agreement, JIT/vmap support, derivatives, and device behavior
as separate results.

## W2: hand-written Lean mathematics under `formal/lean/`

PR #161 supplies the distribution-parameter and finite-moment modules. The tree
below includes planned extensions; this roadmap creates no Lean files.

```text
formal/lean/
  lean-toolchain
  lakefile.toml
  lake-manifest.json
  Normix.lean
  Normix/
    Foundations/
    ExponentialFamily/
    SpecialFunctions/
    Distributions/
    Mixtures/
    EM/
    Numerics/
    Bridge/
  AxiomAudit.lean
  coverage.toml
  decisions.md
  Scratch/
```

Lake control files and package metadata keep their conventional names. The
UpperCamelCase rule applies to the Lean source tree, including generated source
directories. For example, module `Normix.Numerics.FiniteMoments` maps to
`Normix/Numerics/FiniteMoments.lean`. A proposed definition may use
`finiteLogPartition`; a proposed theorem may use `finite_covariance_psd`.
These are naming examples, not existing Mathlib declarations.

`Foundations` contains integrability, moments, and finite-dimensional coordinates.
`ExponentialFamily` contains definitions, convexity, derivatives, and duality.
`SpecialFunctions` supplies Bessel analysis. `Distributions` builds concrete laws;
`Mixtures` builds mixture measures, densities, moments, and posteriors. `EM`
depends on those definitions. `Numerics` handles finite sums, frozen geometry,
and approximation bounds. `Bridge` connects W3 programs to these results.

The dependency order is foundations, exponential families, distributions,
mixtures, then EM. Finite numerics can start early. Bessel foundations must precede
the GIG integral instance. Mathematical definitions must not depend on generated
programs, trace shapes, or a particular JAX version.

Track mathematical propositions and their identifiers only through
[05_math_checklist.md](05_math_checklist.md). This roadmap uses milestone numbers
for scheduling, not a second proposition-ID scheme. The existing `coverage.toml`
records implementation mappings and refers to that inventory; it must not become
a competing mathematical checklist.

For the Bessel goals, $K_\nu(z)$ denotes the modified Bessel function of the
second kind, with real order $\nu$ and argument $z>0$. Report 01 proposes its
whole-line integral as the starting definition. Matching that definition to
any existing Mathlib definition remains **to verify**.

| Milestone | Deliverable and acceptance criterion | Effort | Independent AI work; maintainer review | Dependencies and kill criteria |
|---|---|---|---|---|
| M0: merged build and statement baseline | PR #161 (`80073f3`): 31 theorems, three modules, six cards, pinned Lean/Mathlib. Statement review remains pending. Remaining acceptance: human review of definitions, nonempty domains, source mappings, and exclusions; enforced CI audit. | Rebaseline: 0.5–1 day; review 2–4 hours. CI budget below. | AI prepares evidence; a human reviews statements. | Independent of W1/W3. Reuse delivered proofs. Do not infer specification acceptance from merge status. |
| M1: remaining finite calculus | Extend EF-08 to multidimensional statistics and prove EF-10: the finite log-partition Hessian equals covariance. Reuse merged normalization, scalar derivative, and centered-Gram results; then instantiate BES-15 with the relevant fixed statistics. Acceptance: derivative existence and matrix equality, with fixed nodes and weights; zero proof holes. | 5–10 days; review 3–5 hours. Replaces the unfinished part of the original 10–15 days. | AI edits proofs of reviewed statements. Review coordinate domains and the distinction between PSD and strict positivity. | W2 M0 statement review. Supplies W3 M3. A scalar second derivative is an intermediate result; stop expansion if no useful calculus result fits the budget. |
| M2: continuous exponential families | Definitions on an explicit natural-parameter domain; convexity, gradient/expectation and Hessian/covariance results under stated analytical conditions. Acceptance: derivative existence and integrability appear in checked statements; Fisher information is related to the score, not defined to make the result tautological. | 20–40 days; review 8–16 hours. | AI develops integral and derivative lemmas. Review support, local domination, minimality, and identifiable coordinates. | W2 M0–M1. Special-function preparation can proceed independently. If four weeks yield no reusable analytical lemma, narrow one theorem and defer duality rather than adding conclusion-shaped assumptions. |
| M3: first continuous distribution | Gamma normalization and moments, then the InverseGamma reciprocal relation. Acceptance: at least one concrete law discharges the analytical assumptions; rate/scale and moment-existence domains are explicit. IG normalization, moments, and CDF form a later increment. | Gamma/InverseGamma: 10–20 days after usable foundations; review 4–8 hours. IG: another 10–25 days; review 4–8 hours. | AI implements substitutions and library connections. Review density constants, support, and divergent moments. | Relevant W2 M2 lemmas; no need to finish all abstract duality first. W3 Gamma formulas connect when available. If four weeks produce no distribution instance, retain finite results and stop opening additional continuous families. |
| M4: special functions | Bessel integral, integrability, positivity, and symmetry; then recurrence and parameter derivatives. Acceptance: the chosen definition matches $K_\nu(z)$; derivative proofs include neighborhood-wide domination and boundary terms. | Foundations: 20–40 days; identities: another 10–20 days. Review 2–4 hours per active week. | AI proves auxiliary integral bounds and substitutions. Review the main analytical route, conventions, and compact parameter domains. | W2 foundations. Independent of W3 translation and #136. Pause if four weeks produce no credible tail-bound route. Keep conditional results labelled as conditional. |
| M5: GIG and mixtures | GIG normalization and moments, ideal normal-mixture transformation, mean/covariance, then GH density and conditional GIG. Acceptance: concrete normalization and posterior results discharge their assumptions; named statistic mappings and all constants agree. | 20–40 days after prerequisites; review 8–16 hours. Separate NIG/VG/NInvG formula increments: 5–15 days each. | AI builds measure transformations, square completion, and moment proofs. Review independence, positive-definite domains, exact boundaries, and scale changes. | W2 M3–M4 and applicable M2 lemmas. W3 connects supported components. Stop at conditional mixture theorems if special-function prerequisites are unavailable; do not label the GH instance complete. |
| M6: EM | Exact conditional-density decomposition, normal/IG M-step results, and one-step likelihood nondecrease when $Q$ improves. Here $Q$ is the expected complete-data log-likelihood under the old posterior. Acceptance: posterior normalization and log-integrability are checked; stationary equations are distinguished from optimality; implementation gaps are listed. | 15–30 days after density foundations; review 6–12 hours. | AI implements decomposition and optimization proofs. Review finite-data assumptions, singular cases, penalties, and the relation to actual E/M steps. | W2 M5 for the GH instance. An abstract conditional theorem can begin earlier. W1 M3 tests implementation behavior. Stop the implementation claim if floors, residuals, or fallback behavior cannot be related to the theorem. |
| M7: numerical error | One bounded-domain Bessel value-error theorem, then separate derivative-error goals. Acceptance: an explicit useful bound includes truncation, quadrature, stored constants, and the logarithm. State separately whether machine rounding is covered. | Initial analysis: 5–10 days. Full restricted-domain study: about 2–6 engineering months; review 4–8 hours per active week. | AI develops inequalities and proof implementations. Review the numerical-analysis method, domain, and useful error threshold. | W2 M1/M4; W3's actual-program connection is required before claiming an implementation bound. Stop after the initial study if no nonempty useful bound is plausible. |

Local inverse maps, information measures, CDFs, ideal sampling algorithms,
factor-analysis results, finance identities, and EM stationary-point convergence
remain tracked in [05_math_checklist.md](05_math_checklist.md). They are not
silently included in M6 or the first three months. The reports estimate months
for several of these extensions. General EM monotonicity does not prove global
MLE convergence, and ideal independent inputs do not verify a real PRNG
(**derivation / literature memory**).

### Specification issues to settle early

The following findings are **local check (reported)** and require focused review.
They are not fixes delivered by this roadmap.

- For a positive mixing variable $Y$, GIG's flat order is $(\log Y,1/Y,Y)$. `NormalMixtureEta` starts with $(1/Y,Y,\log Y)$. PR #162 centralized the map in `_pack_gig_eta` / `_unpack_gig_eta`; verify named slots and both compositions in W3 M2. EF-05 stays dropped from the mathematical inventory.
- `GIG.from_natural` now rejects positive rate coordinates via `eqx.error_if` and delegates to `_from_natural_unchecked`. Zero rates are accepted; that sign check does not establish normalizability for every boundary tuple. EF-06's strict-interior theorem and historical clamp model do not cover the public rejection path. Keep this gap explicit in coverage and any future certificate.
- Repeated symmetric-matrix coordinates prevent an unrestricted positive-definite Fisher claim. Restricted joint subfamilies return ambient moments; these are not automatically gradients in their own coordinates.
- `_rescale` concerns the marginal law. Changing the latent coordinate needs a Jacobian for joint densities. Exact invariance also needs inactive clamps.
- `dev-notes/tech_notes/gig_eta_to_theta.md` contains an old non-JIT description. Use current code and S9/S11 for backend and derivative contracts.

### Known doc issues

Record only; leave these pages unchanged in this update:

- `docs/theory/gig.md` uses $\delta^p$ where $\delta=\sqrt{b/a}$ implies $\delta^{-p}$ for GIG parameters $a,b>0$ and shape $p$.
- `docs/theory/em_algorithm.md` claims convergence to the MLE. Separate likelihood nondecrease, likelihood-value convergence, stationary accumulation points, and parameter convergence; global MLE convergence needs stronger assumptions.

## W3: jaxpr-to-Lean bridge

The existing [xshi19/jax-lean fork](https://github.com/xshi19/jax-lean) is the
implementation base. Its current revision and capabilities remain to be checked.
Keep study work separate from normix until reuse permission is clear.
While waiting, prepare normix trace inventories, statement cards, and W2 proofs.
An unresolved license pauses code integration, not the entire verification plan.

**Historical local check (reported):** the audited jax-lean snapshot has distinct readable-code and IR import
paths. `certify.py` generates equality obligations between the resulting Lean
function and `Program.eval` of the imported IR. Ordinary translation has broader
coverage than certification. The reviewed certificate path excludes `scan`,
`custom_jvp_call`, captured floating arrays, and other required structures.
Ordinary `scan` expansion defaults to 32 iterations. Raising that limit alone
does not supply a loop certificate.

**Historical local check (reported):** the reviewed base requires JAX 0.8.0. normix's
repository resolution constraints select JAX/jaxlib 0.9.1. This does not mean its
published dependency range has an exact 0.9.1 upper bound. The report's JAX 0.8.0
and Lean/Mathlib 4.33 pins describe that snapshot, not today's fork. Inspect the
fork first, reproduce its certificate/application example, then measure any
adaptation needed for normix's JAX 0.9.1 and Lean/Mathlib 4.35. Do not downgrade normix.

Every trace record must include the source commit, entry point, static options,
shape, dtype, backend, device, transformation, nested primitives, and constants.
Record the importer and Lean/Mathlib versions too. A fixed-shape certificate
cannot be advertised as arbitrary-dimensional coverage.

| Milestone | Deliverable and acceptance criterion | Effort | Independent AI work; maintainer review | Dependencies and kill criteria |
|---|---|---|---|---|
| M0: fork audit and selected traces | Record the existing fork revision, pins, permission status, and current certificate support. Reproduce one certificate and application proof. Inventory `Gamma.var`, then helpers and the selected update/derivative target; retain the broad inventory as backlog. | 1–3 days; review 1–2 hours. | AI prepares diagnostics and source mappings. Review import/layout trust and supported primitives. | No W2 completion dependency. Stop open-ended environment repair after three days; record the gap and continue W2. Integration still requires reuse permission. |
| M1: actual `Gamma.var` | Trace the real method with both $\alpha,\beta$ as inputs. Prove IR evaluation equals the generated function, then that function equals W2's `gammaVar`, $\alpha/\beta^2$, for $\alpha,\beta>0$; derive positivity. Acceptance: mutate one generated side to fail certification; mutate the source formula to fail the fixed specification connection. Defer Gamma log-density and its `lgamma`/support obligations. | 4–8 days including adapter work; review 3–5 hours. | AI generates obligations and edits proofs. Review domains, source wrapper, constants, and importer trust. | W3 M0 and reuse permission; reviewed W2 formula. Capturing one fixed instance or proving positivity alone is insufficient. At two weeks, generated text without a certificate and application theorem does not pass. Distributional variance remains DM-01/DM-02. |
| M2: selected follow-up and reusable rules | Choose one target from the table below. Add only its required array/primitive rules, preserving shapes and stored constants. Include PR #162's helper contract when selected. Reject unsupported cases explicitly. | Per-target costs below; broader array support remains 5–10 days, review 4–6 hours. Do not count shared adapter work twice. | AI extends rules and checks; review named slots, domains, and failure behavior. | W3 M1 and relevant W2 statements. Stop a tensor redesign that consumes the budget without another accepted function. |
| M3: Bessel finite mathematics on current `_geometry` | Certificates for arithmetic subgraphs of the landed batched window loop and fixed-geometry quadrature. The enclosing loop waits for M4. Connect normalization, covariance, and frozen-geometry derivatives to W2. Preserve constants and protective branches; make no continuous-integral accuracy claim. | 5–10 days; review 4–6 hours. | AI connects finite lemmas; review the derivative object, tilt dominance, and component invariants. | #136 landed in `05b0785`; refresh traces at current master. W3 M2 and W2 M1. If the graph changes again, refresh provenance before continuing. |
| M4: loops and complete forward Bessel | Loop certificate rules and a modular real-semantics connection for `log_kv(..., backend='jax')`. Acceptance: preserve landed iteration limits, clipping, constants, and return behavior; prove a conditional bracket invariant; connect GIG's outer formula without assuming exact Bessel evaluation. | 10–20 days; review 6–10 hours. | AI builds loop rules, certificates, and mutation checks. Review the loop bound, bracket preconditions, component states, and all excluded failures. | W3 M3; W2 finite lemmas. Numerical accuracy waits for W2 M7. If a scalar certificate repeatedly exceeds a proposed 10-minute limit or the agreed memory limit, try modular proofs. Stop whole-graph expansion after two failed redesigns. |
| M5: breadth beyond Bessel | One accepted slice per increment: distribution formulas, mixture linear algebra, GIG outer log-density, CDF table operations, deterministic EM updates, or finance reductions. Acceptance: a named source slice has a certificate, a useful property, and explicit gaps to its enclosing API. | 5–15 days per bounded slice; review 3–6 hours. | AI ranks primitive reuse and prepares certificates. Review positive-definite assumptions, support branches, and the meaning of each API claim. | W3 M1/M2 and relevant W2 statements. A non-Bessel slice need not wait for M3/M4. Stop each slice at its budget if a new semantic subsystem is required; retain it in the coverage inventory. |
| M6: solver, AD, and random-program extensions | Separate studies for bounded `while`/`cond`, custom JVPs, and ideal random programs. Acceptance: one narrow study produces a checked property, or a precise unsupported-case report. Successful solver state, residual accuracy, derivative correctness, and sampling law remain distinct claims. | 20–50 days per selected subsystem, after a 2–5 day scope study; review 2–4 hours per active week. | AI builds small semantic models and proof attempts. Review termination measures, failure values, residual assumptions, independence, and the relation to the actual API. | Mature W3 loop/array rules and relevant W2 mathematics. Defer full solver and sampler coverage if the scope study cannot isolate a useful result. Never equate finite integer sampling support with verified continuous sampling. |

### Post-pilot: choose one

After the first accepted `Gamma.var` certificate, select one increment. Costs
assume its primitives are supported; stop and record any larger semantic gap.

| Target | Acceptance boundary | Engineering / review cost |
|---|---|---|
| PR #162 GIG statistic helpers (W3 M2) | Certify actual `_pack_gig_eta` / `_unpack_gig_eta` for declared scalar and batched shapes. Prove slots 0, 1, 2 preserve `E_log_Y`, `E_inv_Y`, `E_Y`, plus both round trips. If multiple outputs are unsupported, certify projections and state that limit. Connect a selected caller only when checked. | 1–3 days; 1–2 review hours. |
| Incremental-EM weighted mean (W3 M2/M5; TH-09/TH-10) | Connect an actual scalar-leaf `SampleWeightedUpdate` step using `affine_combine`; prove cumulative count and sample-weighted mean for supplied batch summaries. Full fitting and invariance under changed minibatch boundaries are excluded. | 3–5 days; 2–3 review hours. A later aggregation slice needs a separate 3–5 days. |
| Inverse Gaussian log-partition gradient (W3 M2) | Certify primal, `jax.grad`, and analytical `_grad_log_partition` outputs; prove derivative existence and equality. For natural coordinates $\theta_0,\theta_1$, require $a=-2\theta_0$ and $b=-2\theta_1$ above stored `LOG_EPS`; retain guards and prove them inactive. | 3–7 days; 2–3 review hours. General custom-JVP work stays in M6. |

For batch sizes $n_A,n_B>0$ and supplied statistic means $\eta_A,\eta_B$, the
update target is $(n_A\eta_A+n_B\eta_B)/(n_A+n_B)$. Subsequent E-steps use updated
models, so this identity does not make the fitted model invariant to batching.
Exact-real sum identities also give no bitwise floating-point guarantee.

### Bessel baseline after issue #136

Issue #136 landed in commit `05b0785`. At current master, `_geometry` batches the
$k=+2$ right edge and $k=-2$ left edge on one leading axis. Re-inventory that
implementation; the older five-tilt search is historical. Budget 0.5–1 day within
M0/M3 for provenance and graph refresh; actual traces remain **to verify**.

There is a simple supporting **derivation**. Let $g(x)$ be the centered exponent,
$x$ the centered integration coordinate, and $k\in[-2,2]$ the tilt. For $x\ge0$,
$g(x)+kx\le g(x)+2x$. For $x\le0$, $g(x)+kx\le g(x)-2x$.
Thus the outer tilt controls the other tilted exponents on its respective side.
The actual search still needs bracket, clipping, and finite-iteration arguments.
The batched loop must preserve both component invariants. Exponent domination
alone is not a tail-integral or relative-error bound.

Preserve the current 8 expansion steps, 40 bisections, and 192 quadrature nodes,
including clipping and conditional-bracket assumptions. Use componentwise loop
invariants and reusable iteration lemmas. Landing #136 does not establish the
cost of whole-Bessel certification or a numerical error bound.

### Breadth and trust boundaries

Prioritize primitive families by the number of useful targets they enable:

| Target group | Semantic work to assess | First useful acceptance boundary |
|---|---|---|
| Distribution moments and parameter maps | Arithmetic, powers, reductions, positive-domain special functions | Actual formula plus domain property; distributional meaning comes from W2. |
| Normal and mixture calculations | `dot_general`, triangular solves, Cholesky, named-statistic layouts | A quadratic form or deterministic moment/M-step block under explicit matrix assumptions. |
| GIG/GH densities and inverse maps | Bessel composition, branches, bounded loops, linear solves | Conditional outer formula first; residual and implicit-derivative guarantees separately. |
| CDF and quantile tables | Cumulative sums, interpolation, indexing, positive weights | Table monotonicity and range; distance from the true CDF separately. |
| EM and finance | State tuples, loops, branch semantics, deterministic reductions | One update or constraint identity before a full fitting or optimization wrapper. |
| Sampling | Random keys, distributions, retries, failure states | Ideal transformation under stated independent inputs; actual PRNG coverage separately. |

This table is a demand inventory from the reports, not a verified JAX 0.9.1
primitive list. Exact lowerings are **to verify**. A Python API name is not
necessarily a primitive name. Unknown primitives must cause explicit rejection.
Unsupported code gets a hand-written W2 model where useful, with the missing
source correspondence recorded. It does not receive a translation label.

A certificate proves equality inside the chosen Lean semantics. The Python
importer, shared layout decoding, JAX tracing, XLA, and hardware remain separate
trust boundaries. Floating literals can retain their exact stored binary values
while intermediate real arithmetic omits rounding. Real-valued `log`, division,
and square root also need domain conditions to match the intended computation.

Forward equality does not establish AD correctness. `stop_gradient` is an
identity for values but changes derivatives. Bessel AD must be related to a
function with geometry fixed at the reference input. A custom JVP for expectation
inversion needs an interior exact solution, local smoothness, and an invertible
Hessian for the ideal inverse-map theorem. A finite solver residual is a further obligation.
These boundaries are **local check (reported) / derivation**.

## W4: optional TLA+ protocol studies

**Local check (reported):** `.github/workflows/docs.yml` and
`.github/workflows/docs-full.yml` write to `gh-pages` under different concurrency
groups. This motivates a policy review. It does not establish that a deployment
failure occurred. The website's intended precedence between branch and release
builds remains a decision for the maintainer.

| Milestone | Deliverable and acceptance criterion | Effort | Independent AI work; maintainer review | Dependencies and kill criteria |
|---|---|---|---|---|
| M0: publishing policy | A version-acceptance rule and a map of build, artifact, and publication steps. Acceptance: define whether the site follows a branch, releases, or separate destinations; identify the real serialization point. | 0.5 day; review 1 hour. | AI extracts workflow steps and possible interleavings. The maintainer chooses the site policy and reviews atomicity. | Independent of W1–W3. Stop before model expansion if no policy is defined. |
| M1: finite publishing model | A small TLA+ model, TLC configuration, an intentionally faulty variant, and an implementation map. Acceptance: the faulty model yields a rollback counterexample; the corrected model passes the stated finite safety checks. State fairness assumptions for any liveness claim. | 1–2 days; review 1–2 hours. | AI prepares models and trace explanations. Review whether compare-and-publish is atomic in reality, plus retries, cancellation, and source/artifact identity. | W4 M0. The maintainer edits workflow files. Stop maintaining the model if a simple reviewed publishing path resolves the risk and further modelling adds no value. |
| M2: later streaming EM | A single-writer crash/replay model for a concrete future service. Acceptance: acknowledged offsets never exceed durable updates; model, moments, counter, update-rule state, centering state, and random key recover from one consistent checkpoint. | 2–4 days for the first model; review 3 hours. Multi-writer recovery needs a new estimate. | AI enumerates crash and replay traces. Review persistence, delivery semantics, replay rules, and atomic writes. | A real resumable-streaming requirement is mandatory; W1 state tests and W2 update identities can support it. Stop if the requirement remains array minibatch fitting without persistence. |

TLC checks a finite model, not workflow YAML or Python automatically. Keep a map
from every action to implementation steps. If reality separates reading,
comparing, and publishing, allow interleavings between those steps in the model.
Do not repair a counterexample by inventing atomicity or fairness. These limits
come from **literature memory** and report 03's **derivation** of a stale-publish
trace; no TLC result is claimed here.

## First three months

Week 1 starts when executable environments and review time are available.
The table shows concurrent work. Its 68 review hours are a proposed allocation,
not recorded time. Review totals include statement decisions and acceptance
checks; they do not include authoring workflow changes or repairing numerical bugs.

| Phase | W1 | W2 | W3 | W4 | Maintainer review budget |
|---|---|---|---|---|---|
| Week 1 | Backend and round-trip checks; preserve counterexamples | Review merged M0 cards; Lean CI | Audit existing fork revision, permission, and selected traces | Deferred | 7 hours: W1 3, W2 2, W3 2. |
| Week 2 | Finish first M1 checks; preserve failures | Begin remaining EF-08/EF-10 calculus | M1 `Gamma.var` formula connection or diagnosed adapter gap | Deferred | 8 hours: W1 3, W2 2, W3 3. |
| Weeks 3–4 | M2 Bessel reference checks | Complete finite calculus; begin M2 analysis | Finish first certificate; choose one post-pilot target | Deferred | 10 hours: W1 3, W2 3, W3 4. |
| Weeks 5–6 | M3 inverse-map and EM checks | Start M3 Gamma when integral lemmas permit; abstract duality need not finish | Finish selected increment; extend shared rules as needed | M0 only if publishing policy needs it | 12 hours: W1 3, W2 4, W3 4, W4 1. |
| Weeks 7–8 | Finish selected M3 cases; one M4 group | M2/M3; assess special-function prerequisites | M3 on current `_geometry` if prerequisites pass; otherwise non-Bessel M5 | Optional M1 | 12 hours: W1 2, W2 4, W3 4, W4 2. |
| Weeks 9–10 | M4 and counterexample maintenance | First distribution instance or documented analytical gap | M3; begin M4 only after subgraph acceptance | Review pilot outcome | 10 hours: W1 2, W2 4, W3 4. |
| Weeks 11–12 | Stable selected suite and cost report | Accept the continuous instance if complete; decide M4 scope | Accept bounded certificates; decide loop and breadth budget | Continue or archive publishing model | 9 hours: W1 1, W2 4, W3 4. |

At week 12, expect a small real-code property suite, useful finite Lean results,
and at least one accepted program/property connection if licensing and the
prototype permit. A first continuous distribution is a target, not a promise.
Complete Bessel translation may extend past week 12. Full Bessel error bounds,
GH instances, and EM convergence are outside this calendar.

When review time is exhausted, finish the current evidence package before opening
another statement. Keep W1 failures visible and W2/W3 progress independent. A
license delay redirects effort to W1, W2, and normix-owned trace preparation.
Further Bessel source changes require a fresh inventory. Do not recover schedule
by weakening claims or skipping review.

## Rules across all streams

### Public repository and documentation links

Use English in reports, statement cards, generated explanations, and status
records. Omit personal names, private material, and correspondence drafts.
Credit jax-lean by its repository URL. Keep paths, source commits, and
evidence provenance exact.

The repository is public, but `dev-notes/` is not built into the documentation
website. Public theory pages must not link into this directory. Once a result
is checked, add a concise statement to the relevant `docs/theory/` page in a
separate change. Link that statement to the actual Lean source at a fixed public
commit or a verified generated theorem anchor. Link W2 coverage metadata back
to the theory page and Python symbol. Record assumptions and excluded execution
layers next to the link. Do not create badges or theorem links before the target
exists. Use `docs/references.md` for established bibliography entries.

### Lean CI plan

Add a separate Lean job before expanding the proofs: 1–2 engineering days and
1–2 review hours. Pin Lean `leanprover/lean4:v4.35.0` and Mathlib commit
`56a005855ea17e0b0958f73ec801ae140b0a1679`, as recorded in
[`lake-manifest.json`](../../../formal/lean/lake-manifest.json).
Build all production imports with `lake build`, run `AxiomAudit.lean`, and
**fail on every non-allowlisted axiom**, including `sorryAx`. Printing alone
does not enforce the policy.

Ready-to-paste GitHub Actions workflow; documentation only. The audit checks the
current single-namespace modules and direct root imports. Extend that inventory
check when the layout changes. The [elan installer](https://github.com/leanprover/elan)
supplies the toolchain manager; `lean-toolchain` selects the pinned Lean release.

```yaml
name: Lean
on:
  pull_request:
    paths: ['formal/lean/**', '.github/workflows/lean.yml']
  push:
    branches: [master]
    paths: ['formal/lean/**', '.github/workflows/lean.yml']
  workflow_dispatch:
permissions:
  contents: read
jobs:
  lean:
    runs-on: ubuntu-24.04
    defaults:
      run:
        working-directory: formal/lean
        shell: bash
    steps:
      - uses: actions/checkout@v4
      - name: Check pins
        run: |
          python3 - <<'PY'
          import json
          from pathlib import Path
          assert Path('lean-toolchain').read_text().strip() == 'leanprover/lean4:v4.35.0'
          packages = json.loads(Path('lake-manifest.json').read_text())['packages']
          mathlib = next(p for p in packages if p['name'] == 'mathlib')
          assert mathlib['rev'] == '56a005855ea17e0b0958f73ec801ae140b0a1679'
          PY
      - name: Install Lean
        run: |
          curl -sSfL https://raw.githubusercontent.com/leanprover/elan/v4.2.4/elan-init.sh | sh -s -- -y --default-toolchain none
          echo "$HOME/.elan/bin" >> "$GITHUB_PATH"
          "$HOME/.elan/bin/elan" toolchain install leanprover/lean4:v4.35.0
      - uses: actions/cache@v4
        with:
          path: formal/lean/.lake/packages
          key: lean-${{ runner.os }}-${{ runner.arch }}-${{ hashFiles('formal/lean/lean-toolchain', 'formal/lean/lake-manifest.json') }}
      - name: Build
        run: |
          lake exe cache get
          lake build
      - name: Enforce axiom allowlist and audit coverage
        run: |
          lake env lean AxiomAudit.lean > "$RUNNER_TEMP/axioms.log"
          cat "$RUNNER_TEMP/axioms.log"
          python3 - <<'PY'
          import os
          import re
          from pathlib import Path
          allowed = {'propext', 'Classical.choice', 'Quot.sound'}
          files = list(Path('Normix').rglob('*.lean'))
          modules = {str(p.with_suffix('')).replace('/', '.') for p in files}
          imports = set(re.findall(r'^public import (Normix\.[\w.]+)$', Path('Normix.lean').read_text(), re.M))
          assert modules == imports, 'Production module missing from root imports'
          exported = set()
          for path in files:
              source = path.read_text()
              namespaces = re.findall(r'^namespace ([\w.]+)$', source, re.M)
              assert len(namespaces) == 1, f'Review inventory parser for {path}'
              exported.update(namespaces[0] + '.' + n for n in re.findall(r'^theorem (\w+)\b', source, re.M))
          entries = re.findall(r'^#print axioms ([\w.]+)$', Path('AxiomAudit.lean').read_text(), re.M)
          assert exported and set(entries) == exported and len(entries) == len(exported), 'Incomplete or duplicate theorem audit'
          output = Path(os.environ['RUNNER_TEMP'], 'axioms.log').read_text()
          pattern = r"'([^']+)' (?:depends on axioms:\s*\[([^\]]*)\]|does not depend on any axioms)"
          records = re.findall(pattern, output)
          assert len(records) == len(entries) and {n for n, _ in records} == exported, 'Missing audit output'
          for name, body in records:
              unexpected = {a.strip() for a in body.split(',') if a.strip()} - allowed
              assert not unexpected, f'{name}: forbidden axioms {sorted(unexpected)}'
          PY
```

Before enabling the job, check that a forbidden axiom, missing audit entry, and
missing root import each fail it. Source review still enforces the ban on
unapproved computation mechanisms and checks statement meaning. No workflow is
installed or CI run claimed by this document.

Once W3 exports a real target, add regeneration checks (another 1–2 days and
1–2 review hours). Trigger on mapped Python, constants, dependencies, or generator
changes; regenerate traces, reject stale generated files, and build certificates
**and application proofs**. Record source/fork commits, shapes, static options,
constant digests, and tool versions. Run source/formula and translation mutations
when accepting or changing a rule; two paths can share an importer error.

### CI cost and versions

Keep Lean, bridge, property experiments, and optional TLC checks in separate
workflows. Do not add them to, or slow, the existing PR test job. New property
profiles need explicit collection boundaries so default pytest collection does
not accidentally adopt their full randomized budgets. Workflow file edits must
be done by the maintainer. This roadmap does not authorize publication or release.

Pin Lean, Mathlib, JAX/jaxlib, and bridge revisions. Use a Mathlib build cache,
keyed by toolchain, platform, and dependency manifest. Rebuild normix's proof
modules after changes. Explicitly include all formal modules and the axiom audit;
an unimported file must not escape checking. The merged W2 build record reports
successful `lake exe cache get` use on the pinned versions; CI execution remains
**to verify**.

Begin property experiments with roughly 8–20 cases per cheap property. A separate
fast job may target 1–2 minutes of added CI compute, subject to measurement.
Expensive inversion, fits, reference generation, and sampling stay in slow or
scheduled jobs. Preserve existing `slow`, `stress`, `integration`, and `gpu`
markers. A larger Hypothesis profile does not change pytest marker selection.
Keep JAX compilation outside per-example construction where possible.

For JAX upgrades, regenerate the primitive inventory and certificate manifests,
then run primitive, rejection, and application checks. Review changed lowering
and wrapper structure, not just primitive names. Check whether the historical
0.8.0/0.9.1 mismatch persists in the fork. Compatibility claims require fresh evidence.

### Statement review, axioms, and proof holes

Each English statement card contains the formula, quantifiers, domain, support,
integrability, relevant derivatives, source symbol, theory page, and excluded
cases. Every acceptance record carries the four assurance labels below, marking
each as checked, pending, or not covered. Use the inventory identifiers from
[05_math_checklist.md](05_math_checklist.md); do not invent a second scheme.

Human-review the card, Lean signature, and definitions before proof work. AI
edits proofs only against that reviewed specification. Changes to definitions,
domains, assumptions, or the source slice must appear as visible specification
revisions and receive renewed human review. A blocked proof returns a missing
lemma or counterexample. Include examples satisfying the
assumptions and counterexample attempts. Defining a quantity so the desired
identity becomes tautological is not acceptance. Conditional lemmas remain useful,
but their unresolved premises must remain visible.

The formal exported `sorry` budget is zero. `Scratch/` may hold at most five open
proof holes during the pilot, each linked to an unresolved target. Formal modules
must not import it. This limit is a proposed management rule, not a Lean limit.

Use `#print axioms` for every exported theorem and record transitive dependencies.
Reject `sorryAx` and unapproved project axioms. Review the accepted logical
foundations explicitly; ordinary classical proofs may use `propext`,
`Classical.choice`, and `Quot.sound` (**literature memory; selected-version audit
to verify**). Record any additional trusted computation mechanism. A successful
build alone does not prove that statements are meaningful or free of proof holes.

Generated certificates need negative checks: change an operation, literal, index,
or branch and require the relevant certificate to fail. Property tests need a
targeted failing mutation too. An installation or compilation error is not
evidence that a numerical assertion detected the intended fault.

### Execution and mathematical boundaries

`backend='jax'` keeps execution on the caller's selected JAX device, including
JAX CPU when selected. It must not silently switch to NumPy/SciPy, use a hidden
host callback, or fall back to another backend. `backend='cpu'` selects the
existing explicit CPU path. Do not redefine that path to prohibit its documented
JAX components. Respect S9/S11 and keep normix JAX-native.

Verification infrastructure is outside runtime execution. Do not replace kernels
with Lean code, shorten loops to fit an importer, or route specialized
distributions through GH for proof convenience. Mathematical lemma reuse does
not require runtime implementation reuse.

Every acceptance record must include these four assurance labels separately:

| Assurance label | Required evidence and boundary |
|---|---|
| Math theorem | Reviewed real-valued statement, checked proof, and axiom audit. The 31 merged exports belong here; statement review remains pending. |
| Program-translation certificate | Imported trace equals generated function, plus a connection to the reviewed property; record shapes, primitive semantics, and trusted import/tracing boundary. |
| Approximation-error bound | Explicit domain and error bound relative to the intended integral or distribution, including relevant table/constant errors. |
| Floating-point guarantee | A stated machine-arithmetic property with rounding, overflow, underflow, and execution assumptions addressed. |

No label implies the next. Record numerical tests separately; tolerances are not
proved bounds. Real covariance PSD does not ensure nonnegative computed
eigenvalues. Bessel frozen-geometry derivatives do not differentiate the adaptive
geometry, and a forward certificate does not verify AD, XLA, hardware, or a PRNG.

## First two weeks: ordered checklist

The first two entries record delivered artifacts, not statement acceptance.

- [x] W2 M0 merged: PR #161, `80073f3`, 31 theorems and six cards; review pending.
- [x] Source baseline updated to `d557399905c533b3f15825941be6f04a6e897584`; #136 landed as `05b0785`; retain `51e053c` as historical provenance.
- [ ] Days 1–2: reserve review time, review the six existing cards, and record all four assurance labels.
- [ ] Days 1–3: check `xshi19/jax-lean` revision, capabilities, pins, and permission; reproduce one certificate/application example.
- [ ] Days 1–3: implement the separate Lean CI plan with failing axiom and inventory checks.
- [ ] Days 2–5: keep W1 counterexamples and numerical checks active; reuse PR #162 helper and constructor tests.
- [ ] Days 3–5: inventory `Gamma.var` and selected next targets; defer repository-wide graph work.
- [ ] Week 2: connect the actual `Gamma.var` trace to $\alpha/\beta^2$ and run translation/source mutation checks.
- [ ] Week 2: start the remaining multidimensional EF-08/EF-10 work; refresh Bessel provenance against current `_geometry`.
- [ ] End of week 2: accept evidence or record an adapter gap under the stop rules. After pilot acceptance, choose one follow-up and budget regeneration checks.

The high estimates exceed two weeks. A diagnosed adapter gap is a valid checkpoint
outcome, but does not complete the bridge milestone.

## Risk register

| Risk | Signal and response | Review responsibility |
|---|---|---|
| Wrong or weakened specification | Missing domains, circular assumptions, or changed definitions. Freeze reviewed cards and require explicit revisions. | Maintainer, at each card and acceptance review. |
| License remains unresolved | No reuse permission. Continue study and independent W1/W2 work; keep upstream code out of normix. | Maintainer, at W3 integration decisions. |
| JAX or importer drift | Changed nested graphs, constants, or wrapper semantics. Pin versions and regenerate inventories with negative checks. | W3 prepares evidence; maintainer reviews the boundary. |
| Bessel provenance drift | #136 landed as `05b0785`; an inventory of the old five-tilt search is stale. Use current `_geometry` and refresh after further changes. | Maintainer, at W3 M3 entry. |
| Analysis exceeds budget | No usable integral lemma, tail bound, or continuous instance by a milestone checkpoint. Preserve finite results and pause the blocked branch. | Maintainer, at W2 stop points. |
| Tests share errors or hide failures | Round trips and backend agreement pass without an independent reference; tolerances widen after failures. Add independent evidence and keep counterexamples. | Maintainer, at W1 acceptance. |
| CI or review becomes the bottleneck | Recompilation, growing proof time, or an unreviewed card queue. Reduce active scope; keep workflows separate. | Maintainer, at each calendar phase. |
| Model and implementation diverge | TLA+ atomicity is unreal, or a Lean wrapper omits clamps and failure paths. Revisit the implementation map and downgrade the claim. | Maintainer, at W3/W4 acceptance. |
| GIG historical clamp mistaken for current validation | At `d557399`, `GIG.from_natural` uses `eqx.error_if` and `_from_natural_unchecked`. The old clamp model does not cover rejection or all accepted zero-rate boundaries. Name the unchecked method when certifying only its algebra. | Maintainer, at EF-06/W3 acceptance. |
| Statistic order loses verification coverage | EF-05 is dropped as mathematics, but PR #162's order helpers still need implementation verification. Follow [EF-05](05_math_checklist.md#1-exponential-families) to W3 M2; require named slots as well as round trips. | Maintainer, at helper acceptance. |
| Real-semantics results are overstated | A certificate is described as floating-point, PRNG, or whole-library verification. Use the separate coverage categories. | Maintainer, before public documentation links. |

## Rejected alternatives for this plan

These are scope decisions for this roadmap, dated 2026-10-24. Tool descriptions
come from **literature memory** in the reports, not a fresh comparison. Current
library coverage and compatibility remain **to verify**. Rejection here is not a
claim that a tool cannot express probability or numerical mathematics.

| Alternative | Reason for rejection in this phase | Reopening condition |
|---|---|---|
| Coq/Rocq | A second proof ecosystem duplicates dependency audits and proof-maintenance work. Lean/Mathlib and the jax-lean fork are the selected route. | A concrete needed library and sustained expertise justify a bounded comparison. |
| Isabelle/HOL | No established GIG/Bessel advantage was verified in the reports. Maintaining two mathematical libraries would consume review time. | Existing relevant proofs offer a demonstrable reduction in analytical work. |
| Dafny/Verus | Program contracts do not supply the missing continuous mathematics. Porting Python/JAX code adds a separate implementation-correspondence obligation; Verus targets Rust. | An independent component already needs the corresponding implementation language. |
| CrossHair | Its support cannot be assumed to cover JAX tracing, arrays, compiled execution, or special functions. W1 directly exercises those paths. | A risky pure-Python validator warrants a short capability study. |
| Only documentation | Readable derivations remain necessary, but they do not mechanically check assumptions or connect formulas to code. | Low-priority propositions may remain documented, with an unproved status. |
| Only property tests | Finite executions do not prove integral identities or universal parameter claims. | Keep them as evidence for a bounded runtime contract, without a formal-proof claim. |
| Unmodified jax-lean or a new independent exporter | The unmodified snapshot has known coverage and version gaps. A separate exporter duplicates work before the chosen fork has been tested. | Revisit the exporter design after a failed bounded prototype; do not change the accepted fork direction silently. |
| Full floating-point/XLA/PRNG verification now | This exceeds the first useful targets and lacks a checked semantic foundation in the reports. | A short critical kernel has a required machine-error contract and a separately reviewed budget. |

At every milestone, record accepted results, unresolved premises, implementation
gaps, and actual effort. Use those records to choose the next increment. A stopped
translation or analysis study leaves its accepted tests and mathematical results
available without expanding their claims.

## References for this update

- [Lean Verified Transformers](https://srush.github.io/lean-transformer/): finite sums, folds, and composition proofs; its rational model does not verify normix floating-point execution.
- [Provably Correct Tensor Puzzles](https://srush.github.io/jax-lean/): separate translation certificates, application properties, and derivative connections.
- [Lean Verified Tokenizers](https://srush.github.io/verified-tokenizer/): independent representation specifications and explicit success/failure behavior.
