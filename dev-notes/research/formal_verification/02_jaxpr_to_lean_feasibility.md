# From jaxpr to Lean: scope, design, and roadmap for normix

2026-10-24.

Explore a study fork of [jax-lean](https://github.com/srush/jax-lean), by srush, while developing hand-written Lean models under `formal/lean/` in parallel. The jaxpr route should cover as much of normix as possible. Start with small distribution formulas, finite sums, and mixture moments. Use Bessel as a demanding application, not as the limit of the project. Write difficult mathematical components by hand and connect them to translated code when the bridge supports them.

The local copy has no project-wide LICENSE file. The fork is for study and experiments. Reuse or redistribution of its code requires the author's permission. Open an issue asking for a license, and do not copy its code into normix until this is clarified. This report records the plan; it does not report a fork, issue, or external review as completed.

This report uses static reading only. No network access, dependency installation, JAX execution, tests, or Lean builds were performed. Evidence labels have these meanings:

- **local check**: a statement checked against local source or documentation. It does not mean a test passed.
- **derivation**: a mathematical argument given here, without a machine-checked proof.
- **to verify**: a proposed proof, an untraced program feature, or an external claim that still needs checking. Work estimates are planning estimates, not measurements.

The normix reading baseline is commit `51e053c57237a29c71b56d3a2a86988277477bd9`. A local copy of https://github.com/srush/jax-lean was read; this report calls it JL. The supplied description reported 12 commits. The copy had no independent Git metadata, so its commit count and upstream revision were not confirmed. The parent repository's Git metadata cannot establish either fact.

Repository guidance: [AGENTS.md](../../../AGENTS.md) and [dev-notes/README.md](../../README.md). Reports and proposition cards use English. The math inventory belongs in a separate tracking file. The pilot list below concerns translation experiments; it does not replace that inventory. The fork and parallel Lean work are accepted directions. Detailed interfaces, estimates, and proof targets below remain proposals.

## 1. Corrections to the initial assessment

The distinction between real arithmetic and floating-point execution remains central. Several initial conclusions need narrower wording.

`scan` is a loop with a fixed iteration count and carried state. `custom_jvp` lets a function supply its directional derivative rule; `custom_jvp_call` is the corresponding call node. TLA+ specifies states and transitions. These mechanisms have different verification boundaries.

| Initial assessment | Conclusion after static reading |
| --- | --- |
| No Bessel primitive means normix cannot be translated. | **local check:** current `log_kv` expands into ordinary JAX operations. The gaps are primitive coverage, loops, constant tables, and mathematical error proofs. A primitive named Bessel is not required. |
| jax-lean translates jaxpr and generates certificates. | **local check:** ordinary translation and certification cover different subsets. Some `scan` and `custom_jvp_call` programs translate, but the certificate path rejects them. |
| Floating-point properties are not worth proving. | **derivation:** separate the investment by layer. Local error bounds can be useful. Full floating-point proofs for Bessel, solvers, and compiled execution are much larger projects. |
| All mathematics on the website can be verified in Lean. | **to verify:** statements can in principle be formalized after definitions, parameter domains, and analytic hypotheses are made explicit. This gives no short schedule and does not establish Mathlib's Bessel coverage. |
| TLA+ has no use in normix. | **derivation:** Lean is more direct for formulas. TLA+ can model stopping, failure recovery, and state preservation. It cannot replace numerical analysis. |

Two older descriptions also need care. **local check:** current [bessel.py](../../../normix/utils/bessel.py) has no `custom_jvp`, piecewise asymptotic branches, or finite differences. `custom_jvp` remains in [ExponentialFamily._implicit_natural_params](../../../normix/exponential_family.py) for derivatives of expectation-parameter inversion.

The older [gig_eta_to_theta.md](../../tech_notes/gig_eta_to_theta.md) says multistart control flow prevents JIT. Current code and design rows S5 and S11 provide a JAX path. JIT compiles a function for repeated execution. The sampling discussion in [solvers_and_bessel.md](../../../docs/design/solvers_and_bessel.md) still says “no `while_loop`”, while `_gig_rvs_tdr` uses `lax.while_loop`. This report follows the source. Those files are outside this revision.

## 2. What jaxpr and Lean each do

### 2.1 Gamma variance as a small example

jaxpr is the intermediate representation produced by tracing a JAX function. It records inputs, constants, primitive operations, and their dependencies.

Tracing usually follows the function with shape and dtype information and records array operations. It does not verify all Python behavior. Python branches controlled by static configuration have already been selected. In particular, `backend='jax'` and `backend='cpu'` identify different programs to verify.

A primitive is an operation treated as basic at this representation level. `add` and `log` are primitives; `scan` can be a primitive containing a subprogram. A Python API name need not be its primitive name. For example, `gammaln` usually lowers to `lgamma`; the pinned JAX version's actual graph remains **to verify**.

**local check:** the body of [Gamma.var](../../../normix/distributions/gamma.py) is:

```python
return self.alpha / self.beta**2
```

The shape parameter is $\alpha>0$, and the rate parameter is $\beta>0$. The following jaxpr is an illustration written by hand, not output from a JAX run:

```text
{ lambda ; alpha:f64[] beta:f64[]. let
    b2:f64[] = integer_pow[y=2] beta
    variance:f64[] = div alpha b2
  in (variance,) }
```

`f64[]` denotes a float64 scalar. Each intermediate variable is bound once; this is static single assignment, or SSA. Object fields become inputs or captured constants depending on the tracing wrapper. Here both fields are inputs.

Lean checks mathematical proofs. Mathlib is its mathematical library. We can interpret this program as the real function $V(\alpha,\beta)=\alpha/\beta^2$ and prove $V>0$ on the stated domain. This proves a property of the exported real model. It does not prove float64 rounding, underflow, or overflow properties. It also does not establish that a Gamma distribution has this variance. That statement needs a moment or integral proof.

### 2.2 Three proof targets

Let $J$ be a jaxpr, $F$ its readable Lean translation, and $A$ the algorithm specified by a human.

1. Translation correctness: the Lean interpreter of the imported $J$ returns $F$.
2. Algorithm correctness: $F=A$, or $F$ approximates the intended mathematical quantity with a stated error on a stated domain.
3. Machine accuracy: actual JAX execution differs from the real result by a bounded amount.

A translation certificate is a proof of the first claim. Lean's kernel checks the proof object. It is not a certificate of every property of the original Python function. The second claim needs a separate theorem. The third needs a floating-point model and assumptions about compilation and execution.

The initial scope is a limited part of the first two layers. JAX tracing remains trusted to record the intended computation.

### 2.3 Can a function be turned into a theorem automatically via jaxpr?

Yes, within a declared semantic model and supported fragment. The generator can emit a translation-certificate theorem. It cannot infer the intended mathematical specification from the program alone.

For a supported graph, the generated theorem has the form $\forall x,\ \operatorname{eval}(\operatorname{import}(J),x)=F(x)$. Here $x$ ranges over modeled inputs at the traced shapes and dtypes. The statement becomes a certificate only after Lean checks a proof with no `sorry` and no unapproved axioms. The importer and its connection to JAX remain part of the trust boundary.

Several further results can use reusable theorem templates:

| Candidate result | What can be generated | What still needs review or proof |
| --- | --- | --- |
| Shape facts | Input/output shape statements and well-typed coordinate maps for the imported graph. | The facts concern the traced shapes. They do not establish arbitrary-shape correctness or verify the Python layout decoder. |
| Vectorized-versus-loop equivalence | Certificates for both graphs, followed by an equality proof using finite-sum or tensor rules. | The intended pairing, matching shapes, and a proof that both readable functions express the same calculation. A loop graph must be supported first. |
| `jax.grad` versus a mathematical derivative | Trace the primal and derivative programs separately. Certify both, then use derivative rules to prove that the derivative translation equals Mathlib `deriv` of the scalar readable function. | Differentiability and domain conditions. Multivariate log-partitions need coordinate or Fréchet derivatives. `stop_gradient` and custom derivative rules need explicit treatment. This is a per-function result, not verification of JAX's entire AD transformation. |
| Closed forms | Simplify supported arithmetic and finite sums to a human-checkable formula. | The formula's intended meaning, domain, and treatment of stored constants. Equality to a formula does not by itself prove a probability law. |
| Log-partition gradients | For selected families, instantiate derivative rules and connect $\nabla\psi$ to the implemented expectation-parameter formulas. | The sufficient-statistic order, inactive guards, special-function derivatives, and the proof that these formulas are actual expectations. |

**local check:** JL includes vectorized/loop examples in `examples/tensor_puzzles/proofs/TensorLoopProofs.lean` and a quadratic derivative example in `examples/autodiff/proofs/Jvp.lean`. These show the proof structure. They do not provide automatic proofs for arbitrary normix functions. New normix templates and their discharge remain **to verify**.

The two steps for normix are explicit:

1. Certify that the imported `Gamma._log_partition_from_theta` equals a readable Lean function with the same constants and guards.
2. Prove the human specification about that function: on the Gamma natural-parameter domain, its gradient equals the expectation parameters for $(\log X,X)$.

The second step can use hand-written mathematics from option D. Automation may close algebra or derivative subgoals after reusable rules exist. It cannot choose and validate the specification itself. “The PDF integrates to 1”, “EM is monotone”, and “the Bessel error is below a tolerance” require human statements and review. Probability measures, integrability, parameter domains, approximation assumptions, and error tolerances are not supplied by tracing.

### 2.4 A theorem-template generator

**Proposed design; to verify:** for each exported function and trace configuration, emit a readable definition, typed imported IR, a certificate theorem, and a human-specification stub. Record the source commit, shape, dtype, backend, transformations, and constants beside them.

The certificate must be `sorry`-free and Lean-checked before it is marked complete. The specification stub should be a commented declaration outline or a separate tracking entry. It must not become a theorem accepted through `sorry` or an axiom. An unsupported graph produces a named gap, not a placeholder certificate. A human then states the domain and property, proves it, and composes it with the certificate.

For example, proposed names are `gammaLogPartition`, `gammaLogPartitionIr`, `gamma_log_partition_translation_correct`, and `gamma_log_partition_gradient_eq_expectation`. These are proposed normix names, not claims about existing Mathlib declarations. Reports and proposition cards remain in English. The separate math inventory tracks the specification and proof status.

Under the Lean source tree, file, module, directory, and namespace names use UpperCamelCase. Definitions use lowerCamelCase; theorem names use snake_case. For example, a module `Normix.Generated.Gamma` maps through Lake to `Normix/Generated/Gamma.lean`. The lowercase `formal/lean/` is the package directory, not a Lean module name. Existing upstream paths cited in this report retain their actual spelling.

### 2.5 Six pilot groups

**local check:** support below follows JL's `SUPPORTED_JAXPR.md`, including its separate certificate table. **to verify:** no normix jaxpr was generated here. Primitive lists inferred from source must be checked recursively under JAX 0.9.1. “Automatic certificate” below means after version adaptation and any listed extensions, followed by Lean checking.

| Pilot and source | jaxpr features; current JL support and gaps | Automatic target | Human specification |
| --- | --- | --- | --- |
| 1. Gamma log-partition and interior logpdf: `Gamma._log_partition_from_theta`, `Gamma._grad_log_partition`, and the interior composition of `ExponentialFamily.log_prob` | Scalar arithmetic, `log`, static indexing, and shape operations are covered. `lgamma` and `digamma` are missing; Hessians also need `polygamma` or its lowering. The full API contains nonfinite support branches, outside the current model. | A certificate for each supported graph; formula equality for the interior logpdf; a derivative-template proof after special-function rules exist. `Gamma.var` is the arithmetic-only entry test. | For $\alpha=\theta_1+1>0$ and $\beta=-\theta_2>0$, prove $\psi=\log\Gamma(\alpha)-\alpha\log\beta$ and $\nabla\psi=(\operatorname{digamma}(\alpha)-\log\beta,\alpha/\beta)$. Then identify these with $E[\log X]$ and $E[X]$. Density normalization is a separate integral theorem. |
| 2. InverseGaussian log-partition and interior logpdf: `InverseGaussian._log_partition_from_theta` and `_grad_log_partition` | Arithmetic, `max`, `sqrt`, `log`, static `slice/squeeze`, and array assembly are covered when the actual lowering fits the table. The scalar log-partition is a promising early certificate. Full logpdf still has nonfinite branches. Its stored approximation to $\pi$ must be retained. | Certificates and a closed-form derivative proof on a domain where all relevant guards are inactive. A scalar coordinate restriction can compare translated `jax.grad` with Mathlib `deriv`. | With $a=-2\theta_1>0$, $b=-2\theta_2>0$, prove $\psi=-\tfrac12\log b-\sqrt{ab}$ and $\nabla\psi=(\sqrt{b/a},1/b+\sqrt{a/b})$. Matching the guarded analytical code requires strict bounds above `LOG_EPS` where used. Prove these are expectations of $(X,1/X)$. For logpdf, distinguish stored constants from exact mathematical constants. |
| 3. Bessel centered exponent: `_g_k` at fixed integer $k$, with `_ab_coeffs` connected later | Arithmetic, `min/max`, comparisons, and Boolean `select_n` are covered. `expm1` is missing. `_ab_coeffs` also needs the actual `hypot` lowering; its tuple output exceeds the certificate subset. Use scalar projections only with explicit provenance. | A certificate preserving the clip to $[-700,700]$ and a readable formula. Under an inactive-clip hypothesis, rewriting can expose the untruncated expression. | Prove the centering identity in section 5.1 and strict concavity for positive coefficients. The real identity $\operatorname{expm1}(x)=e^x-1$ does not prove the machine implementation's accuracy. |
| 4. Bessel window function: `_edge` and the tilt dependence of `_g_k` | The previous features, `log1p`, selection, and fixed `lax.fori_loop` state updates. `log1p` is missing. Static `fori_loop` normally lowers to `scan`; ordinary translation unrolls it with a default limit of 32. Certificates reject `scan`; both paths reject `while`. The 40-step bisection needs an extension. | Certificates for the exact 8-step doubling and 40-step bisection, after bounded-loop rules exist. A reusable algebra template can instantiate the tilt-difference identity for two exported static tilts. | Let $c(x)=\min(700,\max(-700,x))$. The exact clipped formula satisfies $g_{k_2}(x)-g_{k_1}(x)=(k_2-k_1)c(x)$. Thus it is nondecreasing in $k$ for $x\ge0$ and nonincreasing for $x\le0$. Prove this sign distinction and the conditional bracket invariant. Neither establishes successful bracketing for every input. |
| 5. `_logsumexp` over fixed nodes, followed by `_moments_from_weights` | `exp/log`, `reduce_sum/max`, broadcasting, and `dot_general` are covered. The actual JAX `logsumexp` lowering may contain unsupported `is_finite` or nonfinite branches. Captured floating arrays are rejected by certification. Moments also need `expm1`, the `sinh/cosh` lowering, and treatment of multiple outputs. | A certificate for a supported fixed-size call; equality to a finite exponential sum after its algebraic proof; vectorized-versus-loop equality from certified graphs. Input arrays can isolate the first proof, but do not certify captured tables. | For a nonempty finite vector $q$, prove the result is $\log\sum_i e^{q_i}$, weights are positive and sum to one, and the centered covariance is positive semidefinite. These are finite-sum facts, not Bessel quadrature error bounds. |
| 6. Mixture mean and covariance: `_normal_mixture_mean` and `_normal_mixture_cov` in `mixtures/marginal.py` | On the finite-moment branch, addition, multiplication, broadcasting, and the outer-product lowering are covered. Actual helpers also use `is_finite`, infinity branches, and subordinator methods. The mean calls the half-order raw moment, which can need special functions even when the desired mean formula is simple. | Certificates for the finite formula and fixed traced dimensions; coordinate formulas or vectorized/loop equivalence. Connecting them to the full helpers requires a justified branch reduction or richer semantics. | For $X=\mu+\gamma Y+\sqrt{Y}Z$, assume $Y\ge0$, suitable finite moments, independent centered $Z$, and $\operatorname{Cov}(Z)=\Sigma$. Prove $E[X]=\mu+\gamma E[Y]$ and $\operatorname{Cov}(X)=E[Y]\Sigma+\operatorname{Var}(Y)\gamma\gamma^\top$. In one dimension this is the variance formula. Divergent-moment branches need separate specifications. |

The clipped `_g_k` identity is a **derivation**: subtract the two expressions; all terms except the tilt cancel. Since the production tilt is a Python integer, tracing fixes it. A theorem for arbitrary $k$ belongs in the hand-written model, with separate connections to the exported static instances.

Fixed iteration counts make loops finite, but do not remove data-dependent choices inside the body. A bounded `while` also needs condition and state semantics. Unrolling until an observed run stops is not a proof. Preserve the actual trip count, counter updates, and failure behavior. Sections 5.2 and 7 describe the required extension.

## 3. How jax-lean works

**local check:** this section uses the local copy's `README.md`, `ARCHITECTURE.md`, `SUPPORTED_JAXPR.md`, `SEMANTICS_REFERENCES.md`, and implementation files. Its README only links to the project page. The other documents contain the main technical account. Upstream paths below identify files in that copy; they are not repository-local links.

### 3.1 From Python to proof files

Each arrow has a separate trust boundary:

```text
Python function + input shapes/dtypes + static configuration
    -> JAX make_jaxpr -> ClosedJaxpr
                         |-> translate.py -> readable Lean function
                         |-> importers/jaxpr.py -> Lean program IR
    -> certify.py -> equality proof obligation
    -> Lean checking -> translation certificate
    -> separate application proof -> mathematical specification
```

`ClosedJaxpr` pairs the program with captured constants. Bessel's node tables can be captured constants rather than function arguments.

| File or function | Responsibility |
| --- | --- |
| `python/jaxlean/translate.py::transpile` | Calls `jax.make_jaxpr`, then `translate`. |
| `translate.py::translate` | Requires JAX 0.8.0 and calls JAX `check_jaxpr` for structural checks. It consumes jaxpr objects, not printed text. |
| `translate.py::Emitter.program` | Emits Lean `let` expressions, inlines supported calls, and statically unrolls `scan`. |
| `python/jaxlean/jaxpr.py` | Checks static shapes and dtypes and supplies names and type notation. It does not evaluate primitives. |
| `python/jaxlean/importers/jaxpr.py::JaxprImporter.run` | Independently reads the same jaxpr into Lean IR. It does not parse the readable function's generated text. |
| `python/jaxlean/certify.py::certify` | Combines the readable function, IR, and equality obligation for all modeled inputs. Returning source text is not verification. |
| `certify_module` | Processes pure, single-output `jit` children and composes their certificates in the parent while preserving function boundaries. |

`translate` and `transpile` do not automatically produce certificates. `certify` accepts fewer programs than ordinary translation. Single-body `certify` rejects ordinary nested `jit`; supported call graphs use `certify_module`.

### 3.2 Tensors and the interpreter in Lean

`JaxLean/Core/Tensor.lean` represents tensors as functions from bounded coordinates to values. A vector of length $n$ can be understood as $\{0,\ldots,n-1\}\to\mathbb R$. Shapes occur in types, so coordinate construction must exclude out-of-bounds indices.

`JaxLean/Core/Jaxpr.lean` defines `DType`, `Var`, `Atom`, `Env`, `Op`, and `Program`: dtypes, variables, input atoms, environments, operations, and programs. `Program.eval` evaluates an operation, binds its result, and continues. Real, Boolean, and signed Int32 values share a typed environment.

Floating dtypes map to $\mathbb R$. Int32 indices retain signed integer and wraparound semantics. The interpreter therefore does not treat every value as a real number. `JaxLean/Core/RealOps.lean` maps `exp`, `log`, `sqrt`, and other functions to Mathlib's real functions. Its real instance is `noncomputable`: it gives exact mathematical meaning without promising executable decimal evaluation for arbitrary real expressions.

`JaxLean/Verification/JaxprRules.lean` supplies local evaluation rules. The `jaxpr_certificate` tactic in `JaxLean/Verification/Certificate.lean` composes them. A tactic constructs a proof; the kernel still checks that proof.

`examples/certificates/generated/CertifiedMean.lean` defines `certified_mean_ir` and `certified_mean_translation_correct`. It compares IR evaluation with a mean function of length 2. **local check:** these definitions and the proof structure exist in the file. Their compilation was not rerun.

### 3.3 What remains trusted

The certificate states $\operatorname{eval}(\operatorname{import}(J),x)=F(x)$ for all modeled inputs of fixed shape. Its external assumptions include:

- Correct import of the Python jaxpr. Both paths share shape and index decoding in `layout.py` and `static_index.py`; a shared error can affect both sides.
- The choice of real semantics as the intended abstraction of each JAX primitive. Lean checks the chosen definitions, not their adequacy for JAX.
- JAX tracing, compiler lowering, compilation, and hardware execution.
- Lean's kernel, the logical foundations used, and the assumptions in application theorems. Proof dependencies must be audited for `sorry` and unapproved axioms.

Floating literals preserve their stored binary rational value. A stored float32 `0.1` is not replaced by $1/10$. Intermediate real operations do not round. Division, logarithms, and other functions use Mathlib's total definitions, including values outside the usual formula domains. These are not JAX NaN or infinity semantics. Application theorems need assumptions such as nonzero denominators and positive logarithm arguments.

`tests/test_certification.py` includes corrupted examples that change a divisor from 2 to 3 or replace multiplication with addition and require proof failure. It also illustrates that float32-to-float16-to-float32 conversion changes a machine value while a real identity certificate remains valid. **local check:** the tests document the boundary. They were not run here.

### 3.4 Fixed shapes and primitive coverage

Tracing fixes shapes, static parameters, and many Python loop counts. A certificate for length 2 does not cover length 3. Generic library lemmas can quantify over lengths, but a new generated program still needs a connection to them. `vmap` batches function application; JL processes its lowered graph rather than reconstructing source `vmap` syntax.

**local check:** ordinary deterministic translation supports the following restricted cases. Forward values mean returned values without any derivative claim. `stop_gradient` preserves a forward value while blocking AD propagation; section 5.5 explains the consequence.

| Category | Local support inventory |
| --- | --- |
| Real arithmetic | `add`, `add_any`, `sub`, `mul`, `div`, `neg`, `square`, and `integer_pow`, including negative integer exponents. |
| Ordering and selection | `abs`, `min`, `max`; `eq/ne/lt/le/gt/ge`; Boolean `and/or/xor/not`; Boolean two-branch `select_n`. |
| Elementary functions | `exp/log/sqrt/rsqrt/sin/cos/tanh`. |
| Shape operations | Fixed-shape `reshape/transpose/broadcast_in_dim/squeeze/slice/rev/concatenate`. |
| Reductions and products | `reduce_sum/reduce_prod`; `reduce_min/reduce_max` on nonempty domains; `dot_general` with batch and multiple contraction axes. A reduction combines values along selected axes. |
| Indexing and updates | `gather` with explicit clipping, signed Int32 indices, and restricted dimension configurations. `scatter/scatter-add` with static in-bounds destinations and at most 256 update elements. Replacement destinations cannot conflict; additive destinations may repeat. |
| Other operations | Floating-result `iota`, restricted conversions, `copy`, and forward-value `stop_gradient`. |
| Calls and loops | Pure `jit`; primal-only `custom_jvp_call`; statically unrolled `scan`, including reverse scans. The default limit is 32 iterations per scan and is configurable. |

The certificate subset rejects `scan`, `custom_jvp_call`, `reduce_prod`, captured floating arrays, multiple outputs, and random operations. Both paths reject `while` and `cond`. Partial runtime Int32 support does not cover all integer widths, bit operations, or float-to-integer conversion. General NaN, infinity, and effects such as printing or external callbacks are rejected.

A narrow exception erases the identities `max(-inf,x)` and `min(+inf,x)` in standard softmax lowering. This is not an extended-real model and does not admit all normix support branches. The support document also records real-valued `iota` certificates; integer `iota` remains outside that claim.

### 3.5 Randomness and licensing

`python/jaxlean/rewrites/random_spec.py` is not a PRNG verifier. A PRNG is a deterministic generator controlled by a seed. With `random_model="uniform"`, JL structurally matches JAX 0.8.0 `_randint` calls and replaces them with an ideal uniform integer distribution.

The accepted forms mainly use one typed scalar key and one sampling call, or one fixed-size split followed by batched scalar sampling. Matching checks primitives, parameters, shapes, dtypes, literals, and effects, not just a function name. Key reuse, multiple sampling calls, dynamic bounds, ordinary array `randint(shape=...)`, and other distributions are excluded.

`JaxLean/Stdlib/FiniteLaw.lean` models a mass function on a finite sample space. `JaxLean/Stdlib/Random.lean` and `Independent.lean` provide ideal sampling, expectation, variance, and independence rules. Independence after splitting is an explicit model assumption, not a conclusion about Threefry. Reusing one sampled value does not create independent draws.

These results do not directly cover `Gamma.rvs`, GIG rejection sampling, PINV continuous uniform sampling, or normal draws in mixtures. Rejection sampling repeatedly proposes and accepts samples. The PINV path samples through an inverse numerical distribution table. A theorem about output at a fixed seed differs from a theorem about its distribution under random seeds.

**local check:** the JL root has no project-wide LICENSE, and `pyproject.toml` does not specify a project license. `examples/TENSOR_PUZZLES_LICENSE` and `examples/noether/LICENSE` cover local material; they do not license the whole project. The accepted plan is a study fork and an issue requesting a license. Reuse or redistribution requires the author's permission. No JL code should enter normix before clarification. This report does not infer an intended future license.

## 4. Gaps between normix and JL

This is a source-based requirements list, not a measured jaxpr inventory. logpdf means log probability density; CDF means cumulative distribution function. Exact primitives and nested graphs remain **to verify**. NumPy or JAX API names must not be treated as primitive names.

| normix location and requirement | Candidate primitives or structure | JL status and consequence |
| --- | --- | --- |
| `Gamma.var` and simple moments | `integer_pow`, `div`, and arithmetic | Small entry point; JAX version adaptation is still required. |
| `Gamma._log_partition_from_theta`, gradient, Hessian | `lgamma`, `digamma`, `polygamma`, or their lowering | Unsupported. Gamma logpdf is not an arithmetic-only example. |
| `Gamma.cdf`, `InverseGamma.cdf`, `utils/gammainc.py::gammaincinv` | Incomplete Gamma functions and inverse-normal-CDF lowering; possible `igamma`, `erf`, `erfc`, and inverse error functions | Special-function semantics and derivative theorems are missing. Exact `ndtri/log_ndtr` lowering needs tracing. |
| Bessel `_ab_coeffs/_g_k/_edge/_geometry` and moment bundle | `expm1`, `log1p`, `asinh`; `hypot/sinh/cosh` or their lowering; comparisons and selection | No direct rules for these functions in the support inventory. Some may lower to supported operations or introduce wrappers. Inspect complete subgraphs. |
| Bessel `_fori`; PINV `_bisect_w` | Static `fori_loop` usually becomes `scan` | Bessel bisection uses 40 steps and PINV uses 200, above the default limit of 32. Raising the limit does not add certificates. |
| `fitting/solvers.py::_jax_newton_raw/_backtrack/_multistart_jax`; `distributions/gamma.py::_newton_digamma` | `while`, nested `cond`, counters, and multiple state values | Unsupported. Bounded-loop and termination-state semantics are needed. |
| `fitting/em.py::_fit_scan` and incremental EM | `scan`, multiple outputs, state tuples; random selection in the incremental path | Whole fitters cannot be exported directly. Outer timing, Python conversions, and result slicing also fall outside pure jaxpr. |
| GIG `_cdf_or_ppf/_gig_rvs_devroye/from_expectation` | `cond` | Unsupported. Branch calls cannot simply be treated as elementwise selection. |
| `ExponentialFamily._implicit_natural_params` | `custom_jvp_call`; linear algebra after differentiation | Ordinary translation inlines only the primal. This does not prove the derivative rule. |
| Cached GIG and Newton calls | `jit`, version-dependent wrappers such as `pjit` | JL explicitly handles `jit`, not generic `pjit`. Inspect the actual node and semantics; renaming it is insufficient. |
| Bessel `logsumexp`, moments, and EM statistics | `reduce_sum/max`, `exp/log`, broadcasting, `dot_general`; possible `is_finite` | Most arithmetic is covered. Finiteness checks, infinities, and guard branches may block translation. Softmax support does not establish this version's logsumexp support. |
| Bessel `_GL_T_J/_GL_LOGW_J` | Captured float64 arrays | Ordinary translation can emit arrays; certification rejects them. Preserve every stored value. |
| PINV `jnp.interp/cumsum`; mixture data indexing | `gather`, dynamic slicing, `scan`, or cumulative reductions | Only clipped gather is supported. `reduce_sum` support does not imply `cumsum` support. |
| Matrix `.at[...].set` in `mixtures/joint.py`; GIG initialization | `scatter/scatter-add` or update lowering | Small static in-bounds updates may fit. Runtime indices, modes, and update counts need individual checks. |
| Normal quadratic forms, GIG inversion, and implicit derivatives | Possible `triangular_solve/cholesky/eigh` and `custom_linear_solve` | Unsupported. Linear algebra rules are needed; a solver is not ordinary division. |
| Support sets, degeneracies, and failure values | `select_n/cond`, `is_finite`, NaN, and $\pm\infty$ | Finite real semantics is insufficient. Start with successful interior domains and separately specify failure or extended values. |
| `rvs` and incremental EM | Random keys, splitting, bit operations, Gamma/normal/uniform/choice | The ideal randint model is not a proof of these distributions. |

**local check:** normix's [pyproject.toml](../../../pyproject.toml) pins JAX/jaxlib 0.9.1 in its repository constraints. Its public dependency remains `jax>=0.4.38`. JL's translator explicitly rejects versions other than 0.8.0. Adapt the experimental fork to normix's pinned version; do not downgrade production dependencies to satisfy JL.

Track coverage by function, shape, dtype, backend, differentiation, and batching. A scalar `log_kv` proof does not verify its full `vmap`, GIG `log_prob`, or GPU execution. The nested-jaxpr traversal in [test_no_host_callbacks.py](../../../tests/test_no_host_callbacks.py) is a useful starting pattern for a future inventory. That test does not prove other semantics.

## 5. Bessel: translation and mathematical proof

### 5.1 The mathematical object in the current algorithm

**local check:** this decomposition follows [bessel.py](../../../normix/utils/bessel.py), [constants.py](../../../normix/utils/constants.py), and [bessel_moment_kernel.md](../../tech_notes/bessel_moment_kernel.md). The published design cites DLMF, the NIST reference for mathematical functions, for the integral representation. The external source and corresponding Mathlib coverage remain **to verify**.

For real order $\nu$ and $z>0$, the target is the modified Bessel function of the second kind, $K_\nu(z)$. The representation is $2K_\nu(z)=\int_{\mathbb R}\exp(\nu u-z\cosh u)\,du$.

The integrand's mode is $u_0=\operatorname{asinh}(\nu/z)$. Set $\kappa=\sqrt{\nu^2+z^2}$ and $x=u-u_0$. Define $a_c=(\kappa+\nu)/2$ and $b_c=(\kappa-\nu)/2$; the subscripts distinguish these coefficients from GIG parameters. The centered exponent is $g(x)=\nu x-a_c(\exp x-1)-b_c(\exp(-x)-1)$.

**derivation:** then $K_\nu(z)=\tfrac12\exp(\nu u_0-\kappa)I$, where $I=\int_{\mathbb R}\exp(g(x))\,dx$.

The code has these steps:

1. `_as_arrays` converts to float64, broadcasts, and floors $z$ at the smallest normal float64 value. A proof of the unmodified formula must assume this guard leaves the input unchanged.
2. `_ab_coeffs` computes $\kappa$ with `hypot`. It forms the smaller coefficient using $z(z/(\kappa+|\nu|))$, avoiding direct subtraction and the intermediate $z^2$.
3. `_geometry` finds left and right windows for integer tilts $k=-2,-1,0,1,2$. The tilted exponent $g(x)+kx$ accounts for exponential moments.
4. `_edge` initializes an upper bound with `log1p`, performs 8 conditional doublings and 40 bisections, and targets $g(x)+kx\le-40$. The search bound is capped at 800. Coordinates inside the search exponent are clipped to $[-700,700]$.
5. `_nodes_and_logw` places 96 Gauss–Legendre nodes on each of $[x_{\rm lo},0]$ and $[0,x_{\rm hi}]$. These quadrature nodes and weights approximate interval integrals. The node count is fixed, but intervals and node positions depend on input. This is not a global uniform grid.
6. `_eval_quad` freezes the windows, nodes, and geometry with `stop_gradient`, then uses `logsumexp`. Mathematically this is $\log\sum_i\exp(q_i)$; a stable implementation subtracts a maximum before exponentiation.
7. `_moments_from_weights` normalizes weights to $p_i$ and computes the mean $\bar s$ of $s_i=(x_i,e^{-x_i}-1,e^{x_i}-1)$. It forms rows $R_i=\sqrt{p_i}(s_i-\bar s)$ and constructs covariance as $R^\top R$, then symmetrizes it. It also computes parameter derivatives.

NumPy `leggauss(96)` and `np.log` create nodes and log weights at import time. These calculations are outside the traced JAX program. A proof can fix their stored values without verifying NumPy's table-generation algorithm.

Float64 is also a configuration assumption. [normix/__init__.py](../../../normix/__init__.py) warns if `jax_enable_x64` is disabled; it does not enable it. Future experiments must enable x64 before importing normix and record captured-array dtypes. A source annotation alone does not establish 64-bit runtime values.

### 5.2 Fixed iteration counts still permit data dependence

The `fori_loop` counts are fixed. Inputs still determine branch choices, endpoints, and windows. The outer Python loop over five tilts is unrolled during tracing. **local check:** the source implies 10 `_edge` calls, each with 48 state updates, for 480 updates. This is a source count, not a runtime measurement.

Let $c_j$ be state after step $j$ and $B$ the body. A fixed loop can be modeled as $c_{j+1}=B(c_j)$ for a specified number $N$ of steps. It can use finite recursion or explicit expansion. Lean can prove an invariant by induction on $j$. Keeping the body and count usually makes proofs more readable than expanding every step and can limit generated size.

**derivation:** a first `_edge` theorem can be conditional. Assume the initial bisection interval brackets the target crossing and the clipped comparison agrees with the intended exponent there. Each step preserves the bracket. After 40 steps its width is at most $2^{-40}$ times the initial width, hence at most $800\cdot2^{-40}$.

This does not prove that a bracket is found. Separate proofs must establish that 8 doublings suffice, the cap at 800 does not obstruct the search, and clipping preserves the needed comparisons. `_edge` returns after the iteration limit without a success flag for bracketing. Its comment about the exponent at 700 is not a theorem for all parameters.

For the untruncated exponent, $g(0)+k0=0$ and its second derivative is $-a_c e^x-b_c e^{-x}<0$. **derivation:** strict concavity helps establish the structure of the two threshold crossings. Formalization and the connection to clipping remain **to verify**.

Newton, backtracking, and GIG rejection sampling have data-dependent exits. For a bounded `while_loop`, define semantics with a remaining-step budget. Check the condition at each step, return when it is false, and distinguish termination from budget exhaustion. Prove that the budget follows from the source counter and stopping conditions. It cannot be chosen from one observed run.

For example, a successful precheck in `_jax_newton_raw` sets a stop flag without increasing `num_steps`. The loop bound must account for that check iteration. The numerical-step limit need not equal the number of body calls. A termination proof is still separate from a proof that Newton finds a root.

### 5.3 First layer: the program expresses the intended algorithm

The following targets are proposed and **to verify**:

- The imported program equals a real algorithm preserving actual constants, clips, selections, and loop counts.
- For a fixed window and positive weights, the output is the specified finite sum and its logarithm.
- Normalized weights are positive and sum to one.
- `_moments_from_weights` returns a positive semidefinite centered covariance. For every vector $h$, this means $h^\top C h\ge0$, since the quadratic form is a nonnegative weighted sum of squares.
- Given a valid bracket, bisection preserves the crossing and bounds endpoint separation.

These are plausible targets for work measured in weeks. They can expose errors in signs, scales, node maps, loop state, or derivative projections without requiring a complete Lean Bessel library first.

The importer remains partly trusted. Exact positive semidefiniteness does not ensure that every computed machine eigenvalue is nonnegative. A finite loop bound does not establish a tail-error target.

### 5.4 Second layer: the finite sum approximates $K_\nu$

Translation provides a program expression. Bessel accuracy also requires integral existence, valid changes of variables, and bounds on tails and quadrature error.

Let the returned window be $[L,R]$ with $L<0<R$. Define the omitted tail $T=\int_{-\infty}^{L}e^{g(x)}dx+\int_R^{\infty}e^{g(x)}dx$. If $g'(L)>0$ and $g'(R)<0$, concavity bounds $g$ by its endpoint tangent lines. **derivation:** integrating their exponentials gives $T\le e^{g(L)}/g'(L)+e^{g(R)}/(-g'(R))$. This argument has not been formalized.

An endpoint exponent drop of 40 therefore does not imply relative integral error $e^{-40}$. The bound also needs tail slopes and a lower bound for $I$. Tilts $k=\pm1,\pm2$ help control exponential-moment tails. Order derivatives containing $x$ and $x^2$ need corresponding polynomial-weight bounds.

| Error component | Required proof |
| --- | --- |
| Window and tails | The actual finite search returns endpoints satisfying the tail-bound assumptions. If bracketing is assumed, label the result conditional. |
| Ideal quadrature error | Bound the difference between exact Gauss–Legendre quadrature and the window integral. From literature memory, the classical remainder uses high derivatives; 96 nodes per panel involve derivative order 192. Conditions, useful bounds, and library coverage are **to verify**. |
| Table constants | Stored nodes are not exact polynomial roots, and stored `logw` values are not exact logarithms of ideal weights. Bound node and effective-weight perturbations. |
| Logarithmic constants | `_LOG_TWO=np.log(2.0)` is rounded. In the real model it still differs from exact $\log2$. |
| Runtime floating point | Bound actual rounding in `hypot/asinh/expm1/logsumexp`, reductions, and matrix products. The other rows do not include this error. |

A reviewable target can expose each approximation. Let $Q_{\rm tab}>0$ be the centered quadrature sum computed over the reals with stored nodes and log weights. Let $c_2$ be stored `_LOG_TWO`. Where guards preserve the mathematical inputs and the geometry identities hold, the real algorithm returns $L_{\rm alg}=\nu u_0-\kappa+\log Q_{\rm tab}-c_2$.

**derivation:** suppose $I\ge m>0$ and the combined tail, ideal quadrature, and table error is at most $\varepsilon m$, with $0<\varepsilon<1$. Then $|L_{\rm alg}-\log K_\nu(z)|\le-\log(1-\varepsilon)+|c_2-\log2|$.

Importing a floating table as rationals does not make it an exact Gauss–Legendre table. An axiom asserting $Q_{\rm tab}=I$ would hide the required analysis.

A possible initial domain is $|\nu|\le10$ and $0.1\le z\le100$. This is a proposed experiment, not a new package guarantee. **local check:** the code describes tests for $|\nu|\le300$, a declared numerical domain $z\ge10^{-10}$, and an unresolved case $(\nu,z)=(0,10^{-300})$ with 192 points. These statements do not supply a uniform theorem for arbitrarily large $z$.

Finite-sum, weight, and covariance properties are practical engineering targets. Useful error bounds covering every guard, stored table value, the full parameter domain, and second derivatives are research work. A classical remainder may be valid but too large to help. Validated integral enclosures on subintervals are another possibility, also requiring substantial analysis.

Accuracy at 192 points differs from convergence as the node count tends to infinity. Refining nodes on a fixed window only approaches the truncated integral. Convergence to the whole $K_\nu$ also requires vanishing tails and treatment of the fixed window cap. Current constants do not directly support that asymptotic claim.

### 5.5 `stop_gradient` and implicit derivatives need separate proofs

AD constructs derivative programs from local rules. A JVP is the derivative matrix applied to a direction. A custom rule is not correct merely because the user supplied it.

`stop_gradient` returns its input but gives it zero derivative under AD. For example, `stop_gradient(x) * x` has forward value $x^2$. JAX's rule gives derivative $x$, while the ordinary mathematical derivative is $2x$. Translating `stop_gradient` as identity and differentiating the readable function can therefore prove a property of the wrong derivative object.

For Bessel, introduce explicit geometry $\gamma$ containing reference inputs, the center, nodes, and weights. Define $F(\gamma;\nu,z)$. The production forward value is $F(\gamma(\nu,z);\nu,z)$, while AD freezes $\gamma$ at the reference point and differentiates the last two arguments.

When the geometry identities hold, $F(\gamma;\nu,z)=\log\sum_i w_i\exp(\nu u_i-z\cosh u_i)-c_2$, where $w_i>0$ and $u_i$ are fixed. Let $p_i$ be the normalized weights. **derivation:** $\partial_\nu F=\sum_i p_i u_i$ and $\partial_z F=-\sum_i p_i\cosh u_i$. Second derivatives are covariances of the corresponding statistics.

This is a discrete identity connecting `_eval_quad` to the moment bundle. It is not automatically the ordinary derivative of the function that recomputes its geometry. It is also not automatically the derivative of the continuous integral. That connection needs interchange conditions and weighted integral error bounds. Small value error does not imply small derivative error.

**local check:** `examples/autodiff/proofs/Jvp.lean` separately proves the derivative of a quadratic function and connects generated `grad` and JVP programs to it. It does not verify JAX's entire AD transformation. Normix can use the same proof decomposition without copying implementation code.

`ExponentialFamily._implicit_natural_params` needs another argument. Let $\theta$ be natural parameters, $\eta$ expectation parameters, and $\psi$ the log-partition. Suppose $\psi$ is twice continuously differentiable near an interior point, $\nabla\psi(\theta)=\eta$ holds exactly, and $H=\nabla^2\psi(\theta)$ is invertible. **derivation:** the local inverse derivative should satisfy $H\,d\theta=d\eta$. The exact Lean theorem needed is **to verify**.

The code stops differentiation through optimization history. It then uses diagonal scaling, eigenvalue checks, and a Cholesky solve. Failure or an active boundary returns NaN derivatives. A first theorem can say that, under successful interior and invertibility conditions, the returned tangent solves this linear system.

That does not make a finite Newton result an exact root. Nor does it make the custom derivative the ordinary derivative of the truncated iteration program. For GIG, $H$ comes from a discrete moment bundle. The proof must also address its difference from the continuous Hessian and the nonzero equation residual.

### 5.6 Consequences for logpdf, CDF, and sampling

For $x,\alpha,\beta>0$, the Gamma interior logpdf is $\alpha\log\beta-\log\Gamma(\alpha)+(\alpha-1)\log x-\beta x$. For $x,a,b>0$ and real $p$, the GIG formula is $(p-1)\log x-(b/x+ax)/2-\log2-\log K_p(\sqrt{ab})-(p/2)(\log b-\log a)$.

Replacing GIG's true $\log K$ by an approximation transfers that term's error directly to logpdf. It does not make the approximate density integrate exactly to one. The `_gig_degeneracy_flags` Gamma/InverseGamma branches and `_unpack_safe` guards also need treatment. Positive $a,b$ alone do not ensure the interior Bessel branch is selected.

`ExponentialFamily.log_prob` includes infinities outside the support and at endpoints. A theorem restricted to $x>0$ does not remove unselected infinity constants from the raw jaxpr. The first result can cover an interior formula. Connecting the actual API needs a proved branch reduction under the assumptions, or explicit extended-value semantics.

For CDF work, `utils/rvs.py::build_pinv_table` provides a trapezoidal integral table, cumulative sums, and interpolation. Initial proofs can cover monotonicity, endpoints, and nonnegative weights. Distance to the true CDF still requires integration, truncation, and interpolation error bounds. `tail_eps` is a density-height threshold in this file; it is not a tail-probability bound without proof.

Sampling needs a probability space. A first theorem can assume ideal independent uniforms and prove an inverse-CDF or rejection-sampling distribution. Actual PRNG behavior is another layer. GIG rejection sampling has a maximum number of rounds and returns NaN on exhaustion. A useful specification includes the distribution conditional on success and a failure-probability bound. It cannot claim an unconditional exact GIG sample. JL's finite randint examples do not automatically establish continuous-distribution results.

## 6. The floating-point gap

The following tool descriptions come from literature memory and remain **to verify** for compatibility and available library coverage. Installing one tool does not verify compiled JAX Bessel execution.

| Route | Result | Cost and limits |
| --- | --- | --- |
| Real proofs plus high-precision differential tests | Exact model properties and observed errors on finitely many inputs. | Preferred starting point. Comparing implementations at the same inputs gives evidence, not a global rounding guarantee. |
| Interval arithmetic | Bounds containing a true result if every operation has valid outward rounding. | Useful for fixed domains, tables, and local expressions. Requires sound exponential and logarithm bounds. Parameter dependence may make intervals too wide. A small experiment is estimated at 2–6 person-weeks. |
| A Flocq-style floating-point model | Explicit formats, rounding, error propagation, and exceptional cases. | Flocq is a Coq library, not a Lean/JAX plugin. Its approach is relevant. A small kernel may take months; full coverage has no reliable estimate here. |

Use the existing structure of [scripts/gen_bessel_reference.py](../../../scripts/gen_bessel_reference.py) and [tests/data/bessel_reference.json](../../../tests/data/bessel_reference.json). The generator uses mpmath for high-precision Bessel values and derivatives. mpmath supplies arbitrary-precision numerical calculations, not automatic rigorous enclosures. Raising precision and checking stability still gives numerical evidence.

**local check:** [test_bessel_contract.py](../../../tests/test_bessel_contract.py) separates reference comparisons from AD/moment consistency. Most fields use relative thresholds of $10^{-11}$; some order derivatives use $10^{-7}$. Symmetry zeros use absolute thresholds. These are source test criteria, not errors measured in this report.

A future 3–5 person-day numerical extension could cover window caps, small and large $z$, $\nu=0$, half-integer orders, and degeneracy boundaries. Compare eager execution, JIT, vmap, and the selected device. Reviewers must assess the domain, reference precision, and tolerances. This estimate excludes waiting for GPU resources.

Floating-point analysis must handle changed branch comparisons, reduction order, device-specific elementary functions, and compiler fusion. A symbolic error term on each addition is insufficient. State formats, subnormal behavior, overflow, special-function error contracts, and allowed execution modes.

Existing constraints S5, S8, S9, S10, and S11 remain. The JAX path runs on the user-selected device. Verification tools run separately during development. Lean, NumPy, and CPU callbacks do not replace production JAX operations. A JAX theorem also does not verify the CPU path.

## 7. Four design options and the selected direction

One person-day means roughly six hours of effective work, including review. Estimates assume help from someone familiar with Lean. Learning Lean and maintaining the project alone would lengthen elapsed time. None of these estimates includes a full-domain Bessel error proof.

| Option | Estimate | Trust boundary and benefit | Current position |
| --- | --- | --- | --- |
| A. Use JL directly | Reproduce examples in 1–3 person-days. No reliable short estimate for current normix targets. | Existing interpreter and certificates, but JAX 0.8.0, special functions, loop certificates, and floating tables block direct use. | Keep as a comparison baseline. |
| B. Fork JL and extend it | An initial useful subset was estimated at 3–6 person-weeks; Bessel forward connection could require a further 4–8 person-weeks. These estimates need validation. | Reuses tensor, indexing, and certificate structure. The importer and layout remain trusted. Version migration and upstream architecture add maintenance work. | Preferred exploration because broad normix coverage is the goal. The fork is for study and experiments. Reuse or redistribution requires permission; no copying into normix until licensing is clarified. |
| C. Write an independent minimal exporter | Scalar certificate prototype in 1–2 person-weeks; fixed vectors and a Bessel forward target in roughly 5–10 person-weeks total. | Controls a narrow JAX 0.9.1 model but must supply its own importer, semantics, proof rules, and rejection checks. New code does not remove proof obligations. | Alternative if the fork's licensing, migration, or maintenance cost blocks useful progress. C does not depend on jax-lean code. |
| D. Hand-written Lean models plus differential tests | Useful finite-sum and covariance models in roughly 1–3 person-weeks. Broader mathematical work needs separate estimates. | Mathematics can advance immediately. Source/model correspondence initially rests on review and tests rather than a translation certificate. | Proceed in parallel with B. This is the `formal/lean/` plan in [report 01](01_lean_math_feasibility.md), including components difficult to translate. |

B and D are complementary. D supplies specifications and reusable mathematics. B connects supported normix graphs to those definitions. A theorem about a hand-written model must retain its source-correspondence gap until that connection is proved.

### 7.1 Extending the fork through primitive rules

**local check:** JL's primitive index points to `Jaxpr.Op` and `Op.eval` in `JaxLean/Core/Jaxpr.lean`. Python import dispatch is in `JaxprImporter.run`; readable emission is in `translate.py`. `RealOps` and tensor definitions supply mathematical operations. `Verification/JaxprRules.lean` and `Certificate.lean` supply local rules and proof composition. These are dispatch points, not evidence of a ready-made plugin registry.

**Proposed design:** maintain a primitive table connecting each accepted source case to a Python decoding/emission rule and a Lean semantic definition with a checked local lemma. Most pointwise additions can be small Lean + Python pairs. Update the importer and readable emitter separately, then extend the support inventory and acceptance/rejection tests. Do not make the importer read the emitted function. Record shape, dtype, literal, effect, and domain restrictions per entry.

| Extension | Python side | Lean side and limits |
| --- | --- | --- |
| Special functions | Recognize the actual primitive and parameters; emit and import `expm1`, `log1p`, and later positive-domain `lgamma`/derivative cases. | Define real semantics, local evaluation lemmas, and derivative rules where needed. Exact Mathlib definitions and lemma names are **to verify**. Machine special-function accuracy remains separate. |
| Fixed `fori_loop` and bounded `while` | Decode the actual `scan` or `while` body, counter, carry, and bounds. Preserve source state and exit behavior. | Add finite iteration or fuel semantics and composition rules. Prove any bound and loop invariant. This needs more infrastructure than a pointwise primitive pair. |
| `custom_jvp_call` | Keep the primal call boundary and separately identify transformed derivative graphs. | Certify forward semantics first. A custom JVP needs its own specification and proof; primal inlining cannot supply it. |
| `pjit` or version-specific call wrappers | Inspect JAX 0.9.1 parameters, nested graphs, and effects; reject unmodeled cases. | Connect supported pure call semantics to child certificates. Sharding or device behavior is not proved by treating a call as its body. |
| `gather` | Reuse the existing clipped Int32 subset where applicable; validate any new mode or dimension-number pattern. | Extend coordinate semantics and rules only for explicitly modeled cases. Existing coverage is restricted, so this is an extension rather than a missing primitive in full. |
| Captured floating tables | Decode exact stored values, shapes, and dtypes; record a content digest. | Add typed constant semantics and certify the imported table's use. Prove table conditions separately. |

The version adapter must distinguish JL's JAX 0.8.0 assumptions from normix's JAX 0.9.1 graphs. Removing a version guard is not a migration proof. Inspect changed primitive parameters and nested graphs, and retain unsupported-case rejection. The table should let broad coverage grow incrementally without making every primitive change a redesign.

### 7.2 What an independent exporter would contain

Option C means independently written code with no dependency on JL's implementation. It is not a thin wrapper around the fork and not code copied into normix under another name.

The Python side would read `ClosedJaxpr` and emit typed Lean expressions. The Lean side would define a small program syntax and real interpreter. A separate equality proof would connect the readable function and interpreter. Initially accept scalar reals, Booleans, and confirmed counter types.

Next add fixed vectors, represented by `Fin n → ℝ`, where `Fin n` contains nonnegative indices below $n$. Bessel has scalar inputs and outputs but 192 intermediate nodes and a moment bundle. A permanently scalar-only model would expand arrays into large expressions. Arbitrary-rank tensors, dynamic shapes, and general scatter would remain outside the initial scope.

Each primitive needs input conditions, mathematical meaning, and rejection conditions. For real inputs, `expm1` can mean $e^x-1$, and `log1p` can mean $\log(1+x)$. These definitions do not prove numerical stability of the machine functions. Positive-domain `lgamma` can mean $\log\Gamma$; other domains and derivative properties are separate obligations. Mathlib names and coverage remain **to verify**.

Import constant tables as rationals preserving stored binary values, with shapes and content digests. A first theorem can quantify over positive weights. A later connection must show that the captured table satisfies those hypotheses. Turning constants into inputs proves only a parameterized function until those inputs are tied to the source constants.

`cond` selects a subprogram to execute; `select_n` selects already computed values. They cannot be interchanged without a semantic argument, especially with invalid inputs or exceptional branches. Initially support `while` only when a pure loop has a proved bound. Reject unknown primitives, effects, unsupported dtypes, and unmodeled failure cases.

For either B or C, record source commit, entry point, static configuration, JAX/Lean/Mathlib versions, shapes, dtypes, nested primitives, captured constants, hypotheses, and open obligations. Regenerate after source or constant changes. Reviewable definitions and provenance show which program a theorem covers.

Two generation paths can still share mistakes. A certificate constrains emission relative to imported IR; it does not prove Python-object decoding. Independent primitive comparisons and deliberately corrupted certificates are needed. Moving and verifying more layout decoding in Lean could later reduce the trusted boundary.

### 7.3 Decision record

The record date is 2026-10-24. Accepted directions below come from the current project decisions. Other entries state this report's recommendations, not completed implementation results.

| Item | Status, reason, and evidence | Reconsider when |
| --- | --- | --- |
| Preserve JAX execution and backend choice | Existing requirement; local design S5/S9/S11 excludes hidden host callbacks. Verification does not alter production execution. | The execution contract is explicitly changed. |
| B and D proceed in parallel | Accepted direction. Broad coverage motivates the fork; hand-written mathematics covers current translation gaps. | Measured licensing, migration, or maintenance barriers justify changing the bridge route. D continues independently. |
| English reports and proposition cards; separate math inventory | Accepted direction. Translation coverage and mathematical proof status remain separate records. | The documentation organization is explicitly revised. |
| Study fork with license clarification | Accepted direction. No project-wide license was found locally. Ask for a license; obtain permission before reuse or redistribution; do not copy code into normix meanwhile. | The author grants suitable permission or adds a license. |
| A directly covers Bessel | Not recommended. Increasing the scan limit does not add certificates, missing functions, tables, or error analysis. | Actual coverage changes and version adaptation is checked. |
| Change production formulas or reduce loop counts for export | Not recommended. This changes the program and numerical contract being proved. | A separate algorithm change has a numerical rationale and explicit approval. |
| Add an axiom saying Bessel returns the correct value | Rejected as implementation verification. Conditional outer theorems are useful only with the assumption visible. | The boundary has an independent proof, or the claim is explicitly conditional. |
| Full IEEE floating-point and compiler correctness | Deferred; no experiment performed. The scope exceeds the first bridge and models. | Small real proofs are useful and a concrete use case requires machine guarantees. |
| Call hand-written models “normix is formally verified” | Rejected wording. The source-correspondence gap must remain visible. | Source connections are proved or the claim is narrowed to the models. |

## 8. Milestones and stopping conditions

All milestones are future work. No commands or proofs below were executed here. Start a roughly two-week M0/M1 diagnostic period for B while D develops the selected mathematical models. The two-week checkpoint diagnoses feasibility; it is not a promise of broad fork coverage. The six pilots extend beyond Bessel. Their full cost is not established by the earlier Bessel-focused estimates.

| Stage | Deliverable and acceptance | Estimate and review | Stop or redirect when |
| --- | --- | --- | --- |
| M0: reproduce the baseline and inventory graphs | In an isolated study environment, reproduce JL's mean, quadratic AD, and short scan examples. Separate translation from certificates. `lake build JaxLeanExamples` is needed for application proofs; plain `lake build` is insufficient. Inventory nested primitives and constants for the six pilots under JAX 0.9.1. Start license clarification. In parallel, state D's propositions. | 1–3 person-days; 1–2 hours of boundary review. Record version differences. Keep JL implementation code outside normix pending permission. | After 3 person-days without a clear build/version diagnosis, stop open-ended environment repair. Preserve findings and continue D; assess C as an independent alternative. |
| M1: small distribution certificates | Connect actual `Gamma.var` and try InverseGaussian's scalar log-partition. Extend positive-domain `lgamma` for Gamma's interior logpdf and log-partition. Produce at least one certificate and one reviewed property. Corrupt a divisor or Gamma term and require proof failure. State the infinity-branch gap. | Original scalar/Gamma scope: 4–8 person-days, with 3–5 hours of mathematical review and a Lean reviewer for certificate structure. Added pilot coverage may increase this estimate. | At the two-week checkpoint, generated text without a checked certificate and property is insufficient. Narrow B or assess C while D continues. Do not weaken assumptions or add axioms to pass. |
| M2: GIG's outer logpdf | Parameterize the Bessel term explicitly by $\ell_K$. Prove interior formula composition and logpdf error propagation. Map to `GeneralizedInverseGaussian._log_partition_from_theta` and `ExponentialFamily.log_prob`. Label the result conditional; connect the implementation after M4. | 3–5 person-days; 2–3 hours to review parameter order, signs, degeneracy flags, and floors. D can prove the outer theorem before the bridge exists. | If all degeneracies, CDFs, and solvers become prerequisites, keep the initial theorem on the interior. Do not identify abstract $\ell_K$ with a verified Bessel implementation. |
| M3: fixed nodes and mixture formulas | For `_g_k` on the unclipped interior and `_eval_quad` with fixed geometry, prove centering, normalized weights, and covariance positivity. Add captured-array import. Connect frozen-geometry AD to moments. Develop the finite-moment mixture pilot alongside this work. | Original Bessel scope: 5–10 person-days; 4–6 hours of geometry, scale, and derivative review. The mixture extension needs its own estimate after tracing. | If only hand-written formulas are proved, record D results with no source certificate. If tensor work grows too large, keep the node model one-dimensional while retaining wider goals in the inventory. |
| M4: loops and full forward Bessel | Add certificates for fixed iteration, preserving 8 doublings, 40 bisections, and 192 nodes. Prove `_edge` invariants conditionally, then study bracketing on a small domain. Connect `log_kv(..., backend='jax')` and return to M2. Treat one bounded dynamic `while` separately. | 10–20 person-days for the earlier Bessel target; 6–10 hours of joint mathematical and Lean review. General loop infrastructure may add work. | If a scalar forward certificate repeatedly exceeds about 10 minutes or its memory budget, use modular proofs. After two unsuccessful designs, stop whole-graph expansion. This threshold is a planning proposal, not a measurement. |
| M5: useful error bounds | On a stated small domain, prove tail, quadrature, table, and logarithmic-constant bounds. Start with values, then derivatives. Assess expansion to the declared domain afterward. | Research estimate: 30–80 person-days, possibly longer. First allocate 5–10 person-days to analysis. Mathematical direction may require half a day to one day per week, plus numerical-analysis and Lean review. | If bounds are empty or useless, stop promising production accuracy. Retain completed real structural results. |

The earlier M0–M4 decomposition estimated roughly 5–10 person-weeks for a controlled narrow engineering experiment. It excluded all CDFs, continuous samplers, and complete EM. Option B's migration and broader coverage may exceed that estimate. Do not treat M5 as part of the same budget.

After the pilots, expand the source inventory across distributions, parameter conversions, mixture moments, linear algebra, solvers, and EM. Select the next primitive extensions from actual gaps. The aim remains broad normix coverage. Unsupported components receive hand-written models and explicit connection obligations instead of disappearing from scope.

A new Lean reader can first spend about 2–4 half-days learning to inspect quantifiers, hypotheses, definitions, and proof dependencies. Long proof scripts need not be the starting point. Review must decide whether each proposition matters and whether its hypotheses cover the intended use.

Acceptance records must distinguish checked certificates, checked mathematical properties, assumed algorithm conditions, measured numerical errors, and uncovered APIs or devices. A proof file's existence, passing Python tests, or Lean accepting a file with proof holes is insufficient.

## 9. Questions and possible work with the jax-lean author

JL's typed IR, per-program certificates, and function boundaries fit normix's needs. Normix supplies concrete cases involving special functions, fixed loops, and custom derivative semantics. This makes a small technical discussion useful even before any code reuse.

Possible contribution order, subject to licensing:

1. Independently written examples for scalar `expm1/log1p` and captured floating arrays. Preparation estimate: 1–3 person-days.
2. Bounded-loop certificates using fixed-step bisection. Research estimate after agreeing on an interface: 1–3 person-weeks.
3. A minimal example separating forward equality from frozen-geometry derivatives, followed by custom JVP and implicit-inversion cases. Preparation estimate: 2–4 person-days; proof-rule work needs a separate estimate.
4. Bessel application proofs above the translator. These need not make upstream responsible for all special-function analysis.

These estimates overlap the roadmap and should not be added twice. Maintenance commitments remain unknown. JAX and Lean changes can require migration. General tensor applications may have different priorities from special functions. Accepting a primitive does not establish agreement on its mathematical specification. The unverified 12-commit description says nothing about willingness to collaborate or technical quality.

Open an issue asking for a license. Keep the technical questions neutral:

- Is a project-wide license planned, and what permission is available for code reuse and redistribution?
- What is the supported-primitive roadmap, including captured floating constants, certificates, and call boundaries?
- Is there interest in special functions, fixed or bounded loops, and `custom_jvp` examples that distinguish forward semantics from derivative specifications?
- What is the JAX version policy, and how should a JAX 0.9.1 adaptation relate to the current 0.8.0 requirement?

No issue or message was sent during this offline revision. There is no completed external review to report.

## 10. Roles of Lean, Hypothesis, and TLA+

The main route here is jaxpr to Lean. The following tool descriptions come from prior tool knowledge and remain **to verify** in an executable setup.

Hypothesis generates inputs, executes Python code, and shrinks failing cases. Lean proves statements for all inputs in a specified mathematical model under stated assumptions. TLA+ describes states and transitions. TLC explores reachable states of a chosen finite model. Finite model checking is not a proof for every system size.

| Question | Suitable entry point | Limit of the conclusion |
| --- | --- | --- |
| Why is a fixed-weight covariance positive semidefinite? What bounds Bessel quadrature error? | Lean | Does not automatically cover JAX rounding or the exporter. |
| Do CPU/JAX, JIT/vmap, and mpmath disagree on some parameters? | Hypothesis or existing parameterized numerical tests | No counterexample found does not mean global correctness. |
| Does EM preserve the accepted model after stopping? Does an inner failure retain the last finite model? | A small TLA+ model, with Hypothesis state tests as a possible code connection | Abstracting a numerical step as success/failure does not prove its convergence. |

**local check:** [BatchEMFitter._fit_scan](../../../normix/fitting/em.py) has `converged/diverged/solver_failed` flags and uses `done` to freeze the model and history. A TLA+ model could check that termination is not reversed and failed updates do not overwrite the accepted model. Sequential code can be modeled this way. For this small flow, Lean or direct state tests may cost less to maintain.

A large TLA+ project is not required for the jaxpr experiment. An optional failure-recovery model could be limited to 2–4 person-days plus about two hours of abstraction review. Stop if it merely restates code without an independent safety property. A safety property excludes a bad reachable state, such as returning a failed update as successful.

EM likelihood monotonicity still requires mathematical E-step and M-step conditions. Finite solvers, approximate Bessel values, regularization, and fallback behavior must appear in the scope. State safety cannot replace numerical convergence proofs. EM monotonicity alone does not establish parameter convergence or a global optimum.

## 11. Recommended order

1. Explore B through the study fork and a bounded M0/M1 diagnostic period. Inventory all six pilot groups under JAX 0.9.1. Seek license clarification before reuse or redistribution, and keep JL code out of normix until clarified.
2. Develop D under `formal/lean/` in parallel, as in report 01. Start with finite sums, covariance, family-specific log-partitions, parameter conversions, mixture formulas, and conditional bisection invariants. Hand-written models cover gaps that translation cannot yet handle.
3. Add primitive rules and checked certificates incrementally. Start with simple actual functions and connect them to D's specifications. Expand toward as much of normix as possible; Bessel is one application. Keep C as a truly independent alternative if B is blocked by licensing or engineering cost. A remains a baseline.
4. Reassess M5 after useful source connections exist. Begin with 5–10 person-days on one domain's error bounds. Full floating-point execution, continuous samplers, and whole-EM export remain later work with separate specifications and budgets.

Reviewers must check propositions, domains, approximation layers, and excluded cases. Lean acceptance cannot replace that judgment. The separate math inventory tracks these obligations; reports and proposition cards remain English.

## 12. Remaining scope questions

The language, separate inventory, fork, parallel D, broad coverage goal, and Lean naming convention are settled. Remaining questions concern implementation scope:

- Which checked property should lead the pilots: InverseGaussian's log-partition gradient, fixed-node moment/derivative consistency, or Gamma's interior logpdf?
- Which first specifications should cover only successful interior domains, and which require explicit nonfinite, degenerate, or failure states?
- What domain, quantity, and tolerance should the first Bessel error theorem cover? Values, order derivatives, and parameter Hessians have different costs.
- What budget and Lean review capacity should follow the M0/M1 checkpoint? The earlier 5–10 person-week estimate does not price broad coverage.
- After licensing is clarified, which generic primitive and loop extensions should be maintained in the fork, and which should be proposed upstream?
- When should the scope expand beyond deterministic calculations? Sampling distributions and actual PRNG behavior need separate probability specifications and budgets.

## Sources

These are static-reading sources. They support claims about code, documentation, and written tests, not claims that tests were executed. A local copy of https://github.com/srush/jax-lean was read. Its paths below are plain source references because that copy is not part of normix. External papers, current tool status, and Mathlib coverage were not independently verified.

| Source | Use in this report |
| --- | --- |
| [AGENTS.md](../../../AGENTS.md), [dev-notes/README.md](../../README.md), [ARCHITECTURE.md](../../ARCHITECTURE.md) | Repository rules, documentation boundaries, modules, and interfaces. |
| JL `README.md`, `ARCHITECTURE.md`, `SUPPORTED_JAXPR.md`, `SEMANTICS_REFERENCES.md`, `docs/primitive-index.md` | Architecture, primitive coverage, and trust boundaries. Semantic references are not conformance proofs. |
| JL `python/jaxlean/translate.py`, `jaxpr.py`, `certify.py`, `importers/jaxpr.py`, `layout.py`, `static_index.py` | Emission, independent import, version checks, constants, and layout/index handling. |
| JL `python/jaxlean/rewrites/random_spec.py`, `JaxLean/Stdlib/FiniteLaw.lean`, `Random.lean`, `Independent.lean` | Ideal random models and their limits. |
| JL `JaxLean/Core/Jaxpr.lean`, `Tensor.lean`, `RealOps.lean`, `JaxLean/Verification/JaxprRules.lean`, `Certificate.lean` | Types, interpreter, real semantics, and certificate rules. |
| JL `examples/basics/code.py`, `examples/certificates/generated/CertifiedMean.lean`, `examples/autodiff/proofs/Jvp.lean`, `examples/random_program/proofs/RandomProgramProofs.lean`, `examples/tensor_puzzles/proofs/TensorLoopProofs.lean` | Short scan, certificate, AD, probability, and vectorized/loop examples. |
| JL `tests/test_translation.py`, `tests/test_certification.py`, `tests/test_common_primitives.py`, `tests/test_random_translation.py` | Written coverage, rejection, and deliberately corrupted-certificate tests. Not run here. |
| JL `pyproject.toml`, `lean-toolchain`, `lakefile.toml`, `examples/TENSOR_PUZZLES_LICENSE`, `examples/noether/LICENSE` | Pinned tools and local licenses. The copy declares Lean/Mathlib v4.33.0; availability and execution remain unverified. |
| [normix/utils/bessel.py](../../../normix/utils/bessel.py), [constants.py](../../../normix/utils/constants.py), [exponential_family.py](../../../normix/exponential_family.py), [normix/__init__.py](../../../normix/__init__.py) | Current Bessel steps, constants, support handling, implicit JVP, and x64 configuration. |
| [gamma.py](../../../normix/distributions/gamma.py), [generalized_inverse_gaussian.py](../../../normix/distributions/generalized_inverse_gaussian.py), [inverse_gamma.py](../../../normix/distributions/inverse_gamma.py), [inverse_gaussian.py](../../../normix/distributions/inverse_gaussian.py) | Distribution APIs, special functions, degeneracies, sampling, and pilot formulas. |
| [mixtures/joint.py](../../../normix/mixtures/joint.py), [mixtures/marginal.py](../../../normix/mixtures/marginal.py), [fitting/solvers.py](../../../normix/fitting/solvers.py), [fitting/em.py](../../../normix/fitting/em.py), [utils/rvs.py](../../../normix/utils/rvs.py), [utils/gammainc.py](../../../normix/utils/gammainc.py) | Linear algebra, loops, state, CDFs, interpolation, and mixture moments. |
| [docs/theory/gig.md](../../../docs/theory/gig.md), [docs/theory/em_algorithm.md](../../../docs/theory/em_algorithm.md), [docs/design/solvers_and_bessel.md](../../../docs/design/solvers_and_bessel.md) | Published mathematical targets and algorithm descriptions. They still need review before becoming formal specifications. |
| [dev-notes/design/design.md](../../design/design.md), [solvers_and_bessel.md](../../design/solvers_and_bessel.md), [bessel_moment_kernel.md](../../tech_notes/bessel_moment_kernel.md), [gig_eta_to_theta.md](../../tech_notes/gig_eta_to_theta.md) | S5/S8–S11, rejected backend changes, current Bessel design, and some older descriptions. |
| [test_bessel_contract.py](../../../tests/test_bessel_contract.py), [test_no_host_callbacks.py](../../../tests/test_no_host_callbacks.py), [gen_bessel_reference.py](../../../scripts/gen_bessel_reference.py), [bessel_reference.json](../../../tests/data/bessel_reference.json), [normix pyproject.toml](../../../pyproject.toml) | Test thresholds, graph traversal, high-precision references, and JAX version constraints. |
| [Report 01](01_lean_math_feasibility.md) | Parallel hand-written mathematics and the `formal/lean/` project. |

External background retained from literature memory; original texts are **to verify**:

- *Flocq: A Unified Library for Proving Floating-Point Algorithms in Coq* (2011). Used only to identify a floating-point modeling approach.
- *Specifying Systems* (2002). Used for TLA+'s state and behavior specification role.
- *Maximum Likelihood from Incomplete Data via the EM Algorithm* (1977). Also cited in the local theory documents. This does not establish that the current implementation satisfies every condition.

DLMF's Bessel representation, Gauss–Legendre error analysis, and external Lean/Mathlib, Hypothesis, and mpmath documentation were not checked online. Exact Mathlib definitions and lemmas for Bessel functions, quadrature remainders, special-function derivatives, and inverse functions remain **to verify**.
