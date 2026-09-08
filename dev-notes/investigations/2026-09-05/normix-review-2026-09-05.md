# Normix review — 5 September 2026

Reviewed `xshi19/normix`, `master` commit `7b6c4dfc784567c9f5f6b67a152a49eca6ed8488` (31 August 2026), package version 0.3.0. This is a source audit with independent numerical reproductions, not a proof of correctness for the entire package. Package source was not changed.

**Overall assessment: keep the architecture, strengthen its mathematical and numerical contracts before optimizing it.** The combination of exponential-family mathematics, immutable Equinox models, conditional GIG posteriors, and separate fitting orchestration is a good foundation. CPU and JAX paths are useful complementary implementations. The reviewed revision nevertheless returns materially wrong answers for some valid inputs, including misleading convergence statuses. I would prioritize those defects over new features or a broad C++/Rust rewrite.

Verification used Python 3.12.13, JAX/jaxlib 0.9.1, Equinox 0.13.6, NumPy 2.4.2, SciPy 1.17.0, float64, and CPU. Bessel checks used SciPy and 60-digit mpmath reference calculations; other checks used independent conditional-moment identities and deterministic synthetic data. GPU performance was not measured. Full-suite execution status is recorded at the end.

**Priorities.** “High” means a valid input can produce a materially wrong value, sample, or success status. It does not mean every ordinary use of the package fails.

| Priority | Area | Confirmed issue | First action |
|---|---|---|---|
| High | Bessel derivatives | Regime seams and fixed-step finite differences corrupt gradients and curvature | Add independent seam/curvature tests; replace fragile order derivatives |
| High | GIG normalization | Tiny-argument branch can select an invalid Gamma/InverseGamma limit | Require mathematically valid limits; preserve the positive-parameter interior |
| High | GIG sampling | Valid parameters produce NaNs; exhausted rejection attempts are not checked | Stabilize setup and implement explicit acceptance/failure handling |
| High | Solvers | Transformed-gradient stopping can falsely report convergence | Report and test the intended residual in natural coordinates |
| High | EM | Convergence excludes mixing parameters and depends on data units | Check the full fitted distribution and likelihood/stationarity |
| High | Joint EF API | Special-case joint expectations and Fisher matrices are wrong | Distinguish intrinsic constrained coordinates from ambient GH moments |
| High | Heavy-tail moments | Nonexistent moments become finite or negative values | Enforce moment-existence conditions, including symmetric mixture cases |
| Medium | Density powers | Rényi calculations miss integrability restrictions | Validate powered distribution domains |
| Medium | Finance | Rank deficiency, mislabeled minimum variance, unchecked quantile brackets | Handle supported ranks and use actual covariance/valid brackets |
| Medium | Public contracts | Support handling, differentiation claims, API inconsistencies | Publish and enforce operation-specific contracts |

**1. Bessel accuracy needs repair at the derivative level.**

The dispatch in [bessel.py](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/utils/bessel.py) switches among quadrature, small-argument, Hankel, and Olver approximations. The custom JVP obtains the order derivative from two evaluations separated by a fixed `1e-5`. Second differentiation inherits additional finite-difference cancellation. A small disagreement between adjacent approximations becomes a large derivative error when divided by that step.

| Quantity | Parameters | Normix JAX | Independent reference |
|---|---|---:|---:|
| `log K_v(z)` | v=25, z=1e-6 | 416.752819689 | 416.808025681 |
| Order derivative | v=25, z=1e-6 | 2778.0034 | 17.70740025 |
| Order derivative | v=25, z=0.1 | 7.6218002 | 6.1944791 |
| Second order derivative | v=25, z=1 | -150.0017177 | +0.04077454 |
| Second order derivative | v=1, z=10000 | 0.00454747 | 0.000099995 |

The negative curvature is particularly serious: log K is convex in real order. A negative result can corrupt the GIG Fisher matrix and a Newton step. Increasing Hessian damping hides symptoms rather than repairing the derivative.

There is a separate CPU fallback defect. At v=1000, z=500, `log_kv(..., backend='cpu')` returns **383.066358166**, while mpmath and the JAX path return **322.317651492**. `kve` overflows because it contains exponential scaling, but the fallback assumes overflow implies the leading small-z approximation is appropriate. That assumption is false. These large-order values are outside many ordinary fits, but the public kernel currently advertises arbitrary real order.

Internal floors also change valid inputs: JAX `log_kv(1,1e-50)` returns approximately 69.07755 instead of 115.12925 because an internal argument is floored at 1e-30. Define and test a supported numerical domain explicitly; avoid silently replacing a representable positive argument with a much larger one.

Recommended fixes:

- Treat value, order derivative, argument derivative, and curvature accuracy as separate requirements. Test both sides of every regime boundary, including boundaries crossed by v±1, v±2, and derivative stencils.
- Replace the blanket CPU overflow fallback with a validated large-order/log-domain method. A finite fallback value is insufficient evidence that the approximation is valid.
- Give quadrature a demonstrated accuracy envelope, error estimate or convergence comparison, and suitable centering/scaling. A fixed 64-node rule over a long interval can miss a narrow integrand peak.
- Preserve the exact argument-derivative recurrence, while using accurately evaluated ratios. Avoid claiming that an exact identity makes an approximate numerical implementation exact.
- Use analytic/asymptotic order derivatives or quadrature moments instead of repeatedly differencing large log values. Merely choosing a different fixed epsilon will not solve seams and large-z cancellation together.

The relevant primary references are [DLMF integral representations](https://dlmf.nist.gov/10.32), [recurrences and derivatives](https://dlmf.nist.gov/10.29), [large-order expansions](https://dlmf.nist.gov/10.41), and [SciPy's scaled K implementation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.special.kve.html).

**2. GIG boundary logic and sampling fail on valid positive parameters.**

In [generalized_inverse_gaussian.py](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/distributions/generalized_inverse_gaussian.py), `_log_partition_from_theta`, `_log_partition_cpu`, and related boundary logic select a limiting distribution from `sqrt(a*b)` and the relative sizes of a and b. They then clamp the limiting shape to a tiny positive number.

For `GIG(p=-2, a=1, b=1e-22)`, all parameters are valid in the GIG interior. The correct log partition is **102.700038453**. JAX returns **69.077552790** and CPU returns **690.775527898**. The disagreement reflects their different tiny constants. For p=0 at the same a,b, the correct value is **3.929641584**, but the same invalid limiting branches are selected.

Gamma limits require the appropriate positive shape, and InverseGamma limits require the appropriate negative GIG order. A very small product alone does not justify substituting either boundary distribution. Keep finite positive a,b in a stable interior calculation unless a limit approximation is explicitly justified with its domain and error.

The default TDR sampler also fails: both `GIG(-2, 1e-8, 1e-8).rvs(1000)` and `GIG(0, 1e-8, 1e-8).rvs(1000)` produced 1000 NaNs. In `_gig_tdr_setup`, `(p + sqrt(p*p+a*b))/a` cancels for negative p; for p near zero and tiny sqrt(ab), the local-curvature interval can be enormous and overflow its exponentials.

Use a sign-aware rationalized mode expression or work directly with the log-mode, for example `0.5*(log(b)-log(a)) + asinh(p/sqrt(a*b))` with additional stable handling of extreme ratios. Stabilize the envelope in log coordinates and adapt its width by the actual log-density drop, not only local curvature. The density mode has the analogous cancellation issue with p replaced by p-1.

Finally, `_gig_rvs_devroye` takes `argmax(ok)` after 20 attempts. If a column contains no accepted proposal, argmax returns zero and the code returns an unaccepted proposal. Check acceptance explicitly; retry unresolved samples in controlled batches or return an explicit failure. Precomputing 20 proposals for every sample also spends substantially more work and memory than a typical high-acceptance sampler needs. Benchmark chunked rejection on CPU and GPU separately.

This is a demonstrated silent sampling bias, not just a hypothetical rare exhaustion: `GIG(0,1e-4,1e-4).rvs(10000,seed=42)` produced **12.38%** of draws with `abs(log X)>20`. For this model the true tail probability is bounded above by `exp(-24270.59)`, using the convexity of cosh in the log-X density. Returning finite numbers can therefore conceal an invalid sample distribution. This is one of the first defects I would repair.

**3. Solver success must mean the requested equations have been solved.**

[solvers.py](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/fitting/solvers.py) documents `grad_norm = ||grad psi(theta)-eta||_inf`, but JAX Newton's stopping calculation uses the reparameterized gradient. Near a bound, the Jacobian of theta(phi) can be tiny: the transformed gradient vanishes even when the natural-coordinate moment residual is large.

A one-dimensional bounded quadratic reproduction starts at theta=-1e-12 with a target solution -1. The solver reports a residual of approximately 1e-12 and success while the actual natural-coordinate residual is approximately 1. This defect exists independently of Bessel accuracy.

There is also a direct GIG failure at ordinary parameters. Invert the moments of `GIG(.5,1,1)` with the JAX Newton backend and a warm start from `GIG(0,10,10)`. Even with the default 500 iterations, the returned distribution remains approximately `(p=0,a=10,b=10)`, with maximum moment residual **0.95114**. The constructor discards the solver status. A successful return from `from_expectation` is therefore not currently evidence of a successfully inverted expectation parameter.

Use the intended natural-coordinate residual for interior moment inversion. If intentionally optimizing a constrained problem with an active bound, report a projected/KKT residual separately. Expose both transformed and natural residuals for diagnostics; do not use the same field name for different quantities. Return actual iterations, the last accepted finite state, and a meaningful reason for stopping. Reject nonfinite line-search candidates explicitly and ensure proposed steps are descent directions. A fixed small diagonal addition does not guarantee a usable Hessian.

CPU solver methods also need an explicit bounds contract: accepting a `bounds` argument must not imply that methods which do not enforce it are constrained. Domain clipping inside psi is not an equivalent replacement for constrained optimization.

**4. EM currently stops before the mixing distribution has converged.**

The convergence hook in [marginal.py](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/mixtures/marginal.py#L721) returns normal-component parameters, excluding the mixing distribution. [em.py](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/fitting/em.py#L230) uses their change as its stopping and nonfinite-iterate guard. Accurately solving a conditional M-step does not establish convergence across outer EM iterations.

A deterministic 2,000-observation VG example gives:

| Fit | Reported iterations | Mixing shape alpha | Mean log likelihood |
|---|---:|---:|---:|
| Default tolerance | 3; `converged=True` | 1.587310 | -1.316958 |
| Continue the returned model for 50 iterations at tighter tolerance | 50 | 0.674369 | -1.287771 |

The improvement after nominal convergence is about **58.37 total log-likelihood units**. Scaling the same observations by 0.001 makes the default fit stop after one iteration, with alpha=1.849064; continuation reaches approximately 0.670849. Thus the stopping rule is also sensitive to measurement units.

Include subordinator changes in a fixed latent-scale convention, normalize changes using a fixed data scale, and supplement them with likelihood improvement or a stationarity check. Inspect all model parameters for validity before accepting an update. Preserve distinctions among outer convergence, inner solver failure, and divergence.

The JAX scan has an additional efficiency issue: it computes `_step` before masking already-stopped states. Freezing the output does not skip the E/M work. Guard the expensive step with scalar control flow or select an appropriate while-loop/implicit-differentiation design. This optimization must follow correction of the stopping rule.

**5. The joint exponential-family abstraction needs a precise coordinate contract.**

Joint VG, NInvG and NIG expose the ambient GH sufficient-statistic blocks `[log Y, 1/Y, Y, X, X/Y, vec(XX'/Y)]`. Their special-case partition implementations omit a constrained coordinate, while inherited `expectation_params()` and `fisher_information()` take unconstrained derivatives of that extension. The partition value can be correct on the constrained family without those derivatives being the advertised expectations or covariance.

For a one-dimensional VG joint model with mu=1, gamma=0.3, Sigma=1, alpha=3, beta=2:

| Quantity | Returned | Independently calculated |
|---|---|---|
| E[t] | `[.229637, 0, 1.5, 1.45, .3, 1.735]` | `[.229637, 1, 1.5, 1.45, 1.3, 2.735]` |
| Smallest Fisher eigenvalue | -0.709363 | Must be nonnegative for a covariance matrix |

NInvG similarly returns E[Y]=0 in a case where it is positive and finite; NIG returns E[log Y]=0 by omitting the fixed-order coordinate. Sources: [VG](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/distributions/variance_gamma.py), [NInvG](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/distributions/normal_inverse_gamma.py), [NIG](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/distributions/normal_inverse_gaussian.py).

Choose between explicit ambient sufficient-statistic moments/covariances and an intrinsic constrained-coordinate Fisher information. If keeping both, name them separately. For intrinsic coordinates, use the appropriate score/pullback; for ambient moments, assemble the actual conditional moments and boundary limits. Do not promise generic flat-eta inversion for a constrained family without defining admissible eta.

Also, full `vec(XX')` is nonminimal: off-diagonal symmetric duplicates imply singular directions for d>1. Positive semidefinite is the right ambient covariance invariant; strict positive definiteness requires independent coordinates such as an appropriately weighted half-vectorization.

Scope is important: this particular bug does **not** invalidate every joint method. EM constructs its own sufficient statistics. Public KL/Hellinger routes lift to GH with explicit boundary handling, protecting them from this derivative defect. NIG density powers likewise have a separate lift. Those protections should be retained.

**6. Moment and density-power domains must be enforced.**

[InverseGamma](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/distributions/inverse_gamma.py#L107) evaluates formulas outside their moment-existence domains:

- `InverseGamma(.5, 1).mean()` returns -2 and `.var()` returns -2.666667.
- `InverseGamma(1.5, 1).var()` returns -8.
- At alpha=.5, the first four raw moments return finite positive values, although all four diverge.

The mean requires alpha>1, variance alpha>2, and positive raw moment order k requires k<alpha. `gammaln(alpha-k)` outside that domain is analytic continuation of a special function, not the probability integral.

Propagate fixes carefully into NInvG. A zero skew coefficient can remove terms requiring higher moments of Y; multiplying infinity by zero mechanically will create NaNs. Symmetric NInvG is a Student-t case with degrees of freedom 2alpha: its mean exists for alpha>1/2, variance for alpha>1, and fourth moment for alpha>2. Those are weaker than the generic nonzero-skew requirements. Add dedicated symmetric/partially symmetric tests.

Rényi entropy has an analogous problem in [exponential_family.py](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/exponential_family.py). `Gamma(.2,1).renyi(2)` returns about 1.324736 even though the squared density is nonintegrable at zero; the extended-real answer is -infinity. `InverseGamma(.5,1).renyi(.5)` returns about 2.260212 although the square-root density is nonintegrable in the tail; the answer is +infinity. For positive Rényi order q, require q(alpha-1)+1>0 for Gamma and q(alpha+1)-1>0 for InverseGamma. Validate analogous joint density-power domains.

Separately, `jax.grad(lambda beta: Gamma(3., beta).renyi(2.))(2.)` raises `UnexpectedTracerError`: the nested custom JVP captures traced model parameters through `self`. Pass differentiable dependencies explicitly and test derivatives with respect to both distribution parameters and entropy order. Successful differentiation with respect to order alone is not a complete autodiff contract.

**7. Additional robustness and finance findings.**

The closed-form normal M-step algebra is structurally sound, but raw second-moment subtraction is not translation robust. Constructing exact-model VG sufficient statistics with Sigma=1 gives a recovered Sigma near 1 at mu=0, about 1.865 at mu=1e8, and NaN at mu=1e10. Use centered sufficient statistics/affine preprocessing with a tracked reference center. Fixed absolute jitter cannot recover precision already lost in subtracting large outer products. Source: [joint.py normal M-step](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/mixtures/joint.py).

Public positive-support log densities can return NaN at negative inputs because adding a -infinity base measure does not cancel `log(x)` NaNs. Mask the input before unsafe operations; return -infinity off support and deliberate one-sided endpoint values. Reject empty/nonfinite datasets and invalid static configuration at public boundaries. `Gamma.fit_mle([])` currently returns NaN parameters and an unknown expectation backend silently chooses JAX.

Three specific finance errors are worth fixing if that module is intended for use:

| Finding | Reproduction | Recommendation |
|---|---|---|
| Reduced-coordinate matrix assumed rank three | Valid symmetric 3D VG with gamma=0 yields all-NaN inverse and weights | Handle rank-deficient coordinates or explicitly reject unsupported rank; a pseudoinverse alone also needs feasibility checks |
| `min_variance_point` minimizes Sigma dispersion | With EY=VarY=1, Sigma=I, gamma=[0,10,0], returned weights have variance 11.4444; true minimum is .497537 | Use `Cov(X)` or rename to `min_dispersion_point` |
| `CVaR.value_reduced` uses an unchecked ±20-standard-deviation bracket | alpha=.0001, mu=0, gamma=-1, sigma=.01, Y=[1]*999+[1e6]: CVaR=10000000 vs correctly bracketed 1000017.55 | Use a guaranteed component-quantile bracket and check quantile residual |

Sources: [optimization.py](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/finance/optimization.py#L105), [risk.py](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/finance/risk.py#L185).

For the empirical conditional-normal mixture, a guaranteed q-quantile bracket is the minimum and maximum over i of `mu + gamma*Y_i + sigma*sqrt(Y_i)*ndtri(q)`. At the first endpoint every component CDF is at most q; at the second each is at least q. This supplies a simple exact bracketing argument for the numerical problem actually being solved.

**8. The best Bessel performance opportunity is shared work.**

Introduce an optional fused evaluator in natural coordinates, such as `partition_bundle(theta, order=2, backend=...)`, returning psi, eta and Hessian, plus numerical status. Keep the solver responsible for its phi-to-theta chain rule. The published [design rationale](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/docs/design/exponential_family.md) rejects fusion because it supposedly forces that chain rule into distributions; it does not. Fusion and coordinate separation are independent choices.

The JAX GIG gradient uses five Bessel evaluations; its Hessian has an eleven-evaluation stencil that already contains those five. Reusing that stencil can remove duplicated evaluations before any new numerical algorithm is introduced. Allow value-only line-search evaluations and reuse accepted-point bundles.

The CPU Hessian instead uses a fixed natural-coordinate finite-difference step and 19 partition evaluations for a 3-by-3 matrix. That is another candidate for the same stable derivative bundle; its stencil can also cross parameter-domain boundaries. The inner JAX Newton scan, like the outer EM scan, continues calculating expensive updates after its convergence flag is set. Eliminating that work can improve CPU warm-start costs without changing the underlying special-function algorithm.

A more substantial prototype can compute derivatives as moments of one stabilized integral. Set L(v,z)=log K_v(z), and define a normalized density on the whole real line proportional to `exp(v*u-z*cosh(u))`. Then:

$$
L_v=E[u],\qquad L_z=-E[\cosh u],
$$

$$
L_{vv}=\operatorname{Var}(u),\qquad
L_{zz}=\operatorname{Var}(\cosh u),\qquad
L_{vz}=-\operatorname{Cov}(u,\cosh u).
$$

These follow by differentiating the [Bessel integral representation](https://dlmf.nist.gov/10.32). A single set of positive quadrature weights can supply the partition, gradient, and centered covariance. Center near `asinh(v/z)` and scale/adapt the integration region. Centered variance evaluation avoids obtaining tiny curvature by subtracting nearby huge log values. An alternative integrates GIG directly in log-x and accumulates Cov[log X,1/X,X]. This is a proposed implementation direction, not a demonstrated universally faster replacement. Validate tails, truncation, parameter-dependent quadrature, and higher derivative rules against an independent oracle.

Keep asymptotic expansions where they outperform quadrature, with analytic derivatives and explicit accuracy criteria. The existing [Takekawa work](https://arxiv.org/abs/2108.11560) is a relevant starting point for parallel integral evaluation; adapting an idea still requires validating this implementation's domain.

**Batched dispatch is an independently measured waste.** `vmap(lax.cond)` becomes selection over batched predicates, generally evaluating the alternatives. That contradicts the module's blanket claim that only one branch runs for arrays. This behavior is documented by [JAX](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html).

In a local float64 CPU microbenchmark with 4,096 dynamic inputs all in the Hankel regime, the jitted public dispatcher took approximately **6.056 ms**, while the jitted existing Hankel formula alone took **0.197 ms**, with identical outputs. Both were synchronized; 30 repetitions were used. The approximately 31-fold difference isolates avoidable work in this one case. It is not an end-to-end package speedup or a GPU measurement.

Possible remedies include a uniform-batch fast path guarded by a scalar predicate, static regime-specialized kernels when the domain is known, or chunked mapping that preserves conditional execution. Mixed batches need a measured strategy; dynamic partitioning introduces its own shape and scatter overhead. Compare alternatives at equal accuracy.

**9. Improve GIG optimization in stages.**

Keep natural-coordinate convex inversion as the reference solver. Rescale sufficient statistics before solving, use consistent derivative bundles, test the true residual, and warm-start from the previous EM iteration. Inspect actual solver progress rather than assuming a small dimensional problem needs a more elaborate optimizer.

Check necessary expectation-domain conditions before an expensive solve: positive forward/inverse moments, `m_+*m_- >= 1`, and `-log(m_-) <= ell <= log(m_+)`. Degenerate equality cases need an explicit policy; these checks alone are not a complete proof of finite interior solvability.

An optional exact scale profile reduces the three-parameter optimization to two. Let

$$
z=\sqrt{ab},\quad s=\tfrac12\log(b/a),\quad
\ell=E[\log X],\quad m_-=E[1/X],\quad m_+=E[X].
$$

The expected negative log likelihood, up to a constant independent of p,z,s, is

$$
F=\log(2K_p(z))+p(s-\ell)
+\frac z2\{m_+e^{-s}+m_-e^s\}.
$$

For fixed p,z, its unique minimizing scale is

$$
s^*=\frac12\log\frac{m_+}{m_-}
-\operatorname{asinh}\left(\frac{p}{z\sqrt{m_+m_-}}\right).
$$

Thus one can optimize p and log z and recover a=z exp(-s*), b=z exp(s*). Evaluate products/ratios stably. The profile follows directly from `a*m_+ - b*m_- = 2p`; balancing the empirical forward/inverse moments does not imply fitted a=b unless p=0. This profile is worth benchmarking, but it changes coordinates and does not retain a blanket convexity guarantee in the resulting two variables. Use the existing convex formulation to verify solutions and boundary behavior.

For differentiable expectation inversion, use the implicit relation `grad psi(theta)=eta`. In a regular identifiable interior, the derivative solves `H(theta)*dtheta=deta`. Implement a custom derivative through that linear solve rather than backpropagating through all optimizer iterations. Handle singular representations, boundary solutions, failed solves, and conditioning explicitly.

**10. C++ or Rust should be a small optional kernel, chosen after profiling.**

| Approach | Potential benefit | Recommendation |
|---|---|---|
| Improve existing JAX | Fusion, device execution, corrected branches, fewer derivatives | First choice for the JAX path |
| Improve NumPy/SciPy path | Reuse compiled special functions; reduce repeated work and callbacks | First choice for CPU fitting |
| C++ kernel with JAX FFI | Fused evaluation and direct buffer access; fits the documented C++ FFI interface | Preferred initial native prototype if profiling justifies one |
| Rust kernel | Comparable opportunity to implement a fast CPU numerical kernel | Reasonable if Rust is a project preference; not an intrinsic accuracy/speed advantage |
| Rewrite package/solver stack | Much larger maintenance and validation burden | Not justified by the current evidence |

SciPy's `kve` already wraps a compiled AMOS implementation; Python syntax around it is not evidence that the special-function evaluation is interpreted. JAX also compiles array programs. The language change must eliminate measured overhead or enable a better algorithm to pay for itself. [SciPy reference](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.special.kve.html)

If pursuing a native implementation, define a batched kernel that returns log K and its needed partial derivatives/ratios together, with error/status outputs. Avoid one Python/native call per scalar. C++ has the most direct match to JAX's documented header-based FFI. Rust/PyO3 can expose Python extensions, but that alone does not make an operation JIT-able, differentiable, or GPU-enabled; JAX lowering, batching, and derivative rules remain separate work. A CPU kernel does not automatically run on a GPU. [JAX FFI](https://docs.jax.dev/en/latest/ffi.html), [PyO3 guide](https://pyo3.rs/)

Boost's real-order Bessel implementation is a useful comparison candidate, not a complete log-K-plus-order-derivatives solution by itself. Check error envelopes, overflow behavior, packaging, licensing, and supported platforms before adopting any native backend. [Boost Bessel documentation](https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/bessel/mbessel.html)

Measure a native prototype against both warmed JAX and vectorized SciPy at equal error targets. Include the actual end-to-end EM workload, parameter regimes, first-call compilation, steady-state runtime, memory, and transfer costs. Use `block_until_ready` for asynchronous JAX results. Do not claim a performance win if it changes convergence or numerical accuracy. [JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)

**11. Design, style, documentation, and validation improvements.**

Keep the existing model/fitter separation and immutable PyTrees. Treat mathematical correctness, supported parameter domains, and honest diagnostics as release requirements; optimize elegance within those requirements. The code's main problems are numerical contracts and a few overextended abstractions, not formatting. Avoid a broad class-hierarchy rewrite; first make public operations trustworthy and numerical kernels reusable.

- **Consistent fit results:** standardize `maxiter`/`max_iter` and distribution-returning versus result-returning fit methods. Expose solver residuals, termination reasons, accepted steps and approximation warnings through result objects. Constructors used as numerical inverses need an explicit failure policy.
- **JAX-native sampling:** provide `sample(key, sample_shape=...)` with `jax.random.split`, retaining `rvs(n, seed=...)` as a convenience adapter. Repeated default seeds are deterministic and do not compose naturally with JAX workflows.
- **Capability matrix:** label each operation's eager CPU, JIT, vmap, JVP, VJP and second-derivative support. `jax.jit(lambda eta: Gamma.from_expectation(eta).alpha)` works, but `jax.grad` of that function raises the dynamic-while-loop reverse-mode error. The homepage's “differentiable end to end” and “lossless” conversions therefore overpromise.
- **Teach the mixture convention immediately:** `Sigma` is dispersion, not generally covariance. Put `E[X]=mu+gamma*E[Y]` and `Cov(X)=E[Y]*Sigma+Var[Y]*gamma*gamma.T` beside the first constructor. Align latent scale before comparing recovered raw parameters; fitted densities/moments are often the better first tutorial target.
- **Correct GIG descriptions:** for a,b>0 the density vanishes at zero for every p and has a unique interior mode. The current gallery and mode docstring incorrectly transfer Gamma-like claims about monotonicity/divergence for p<=1.
- **Explain approximation contracts:** CDF/PPF interpolation tables, numerical tails, derivative availability and table reuse belong in user-facing method documentation. Backend selection should be workload-specific rather than “CPU/JAX is always faster.”
- **Keep the documentation layers:** tutorials, user guide, theory, reference, and research are already a sensible organization. Add concise domain/capability tables and cross-links; tighten inaccurate statements instead of reorganizing everything.
- **Factor fitting should preserve low-rank complexity:** Woodbury solves help density evaluation, but the E-step still materializes a dense weighted XX' matrix, and convergence materializes FF'+D. For high-dimensional training, accumulate diagonal and X-factor statistics directly and use low-rank trace identities for covariance-change norms. State separate costs for log density and the complete fitting loop.

Validation should move from mostly same-implementation consistency toward independently grounded contracts:

1. Independent values and derivatives on a logarithmic parameter grid, plus deliberate seam neighborhoods and exact special cases. Do not avoid x=0 or shape=1 simply because they are difficult.
2. Correct support, moment existence, powered-density integrability, and endpoint behavior.
3. E[t] and Cov[t] from independent conditional algebra or reliable Monte Carlo, especially for joint special cases; PSD and symmetry with conditioning-aware tolerances.
4. Fitting continuation tests: a converged model should not materially improve after additional well-resolved iterations. Check invariance under translation, unit changes, and equivalent latent-scale conventions.
5. Invalid-domain, empty-input, nonfinite-iterate, failed-line-search, and rejected-sampling paths with explicit statuses.
6. Paired runtime/error benchmarks, iteration counts, and residuals. The existing deep benchmark utility catches all exceptions during timed calls, which can make failures look fast; fail or label the case explicitly. Existing GIG warm-start benchmarks start at the exact parameter solution, so add small, moderate and difficult perturbations representative of successive EM targets.
7. Release tests on the installed wheel from outside the checkout. The current wheel smoke runs `python -c` in the repository directory and can accidentally import source instead of the wheel. Check the imported path. Require numerical validation for the exact release artifact, with a modest supported Python/OS/JAX matrix and periodic extended tests.

Relevant implementation/configuration sources: [CI](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/.github/workflows/ci.yml), [publish workflow](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/.github/workflows/publish.yml), [testing guidelines](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/.cursor/rules/testing-guidelines.mdc), [benchmark utility](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/benchmarks/utils.py), [GIG ASV benchmark](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/asv_bench/benchmarks/gig.py), [factor implementation](https://github.com/xshi19/normix/blob/7b6c4dfc784567c9f5f6b67a152a49eca6ed8488/normix/mixtures/factor.py).

**Suggested implementation order:** first add the independent reproductions below as regressions and repair silent wrong-answer paths; next correct solver/EM statuses and coordinate contracts; then fuse kernels and remove wasted batch/scan work; finally profile the corrected end-to-end workloads and decide whether a native kernel is worth maintaining. Rebenchmark after convergence fixes, because the present fit timings can reward premature stopping.

**Minimal reproduction examples.** Run against the reviewed commit with float64 enabled. These intentionally show current failures; they are not proposed corrected implementations.

```python
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import mpmath as mp
from normix import log_kv, GIG, Gamma, InverseGamma, VarianceGamma

mp.mp.dps = 60
v, z = 25.0, 1.0
print('order curvature:', jax.grad(jax.grad(lambda v: log_kv(v, z)))(v))
print('reference:', mp.diff(lambda v: mp.log(mp.besselk(v, z)), v, 2))
print('CPU overflow fallback:', log_kv(1000., 500., backend='cpu'))
print('reference:', mp.log(mp.besselk(1000, 500)))

g = GIG(-2., 1., 1e-22)
print('GIG JAX psi:', g.log_partition())
print('GIG CPU psi:', type(g)._log_partition_cpu(np.asarray(g.natural_params())))
print('finite samples:', jnp.isfinite(GIG(-2., 1e-8, 1e-8).rvs(1000)).sum())
draws = GIG(0., 1e-4, 1e-4).rvs(10000, seed=42)
print('impossible-tail fraction:', jnp.mean(jnp.abs(jnp.log(draws)) > 20))

ig = InverseGamma(.5, 1.)
print('mean, variance:', ig.mean(), ig.var())
print('Renyi Gamma:', Gamma(.2, 1.).renyi(2.))
print('Renyi InvGamma:', ig.renyi(.5))

m = VarianceGamma.from_classical(mu=jnp.array([1.]), gamma=jnp.array([.3]),
    sigma=jnp.eye(1), alpha=3., beta=2.)
print('joint eta:', m.joint.expectation_params())
print('joint Fisher eigenvalues:', jnp.linalg.eigvalsh(m.joint.fisher_information()))
```

EM continuation reproduction:

```python
rng = np.random.default_rng(13)
y = rng.gamma(.7, 1/.7, 2000)
x = (np.sqrt(y) * rng.normal(size=2000))[:, None]
X = jnp.asarray(x)  # repeat with x * .001
m = VarianceGamma.from_classical(mu=X.mean(axis=0), gamma=jnp.zeros(1),
    sigma=jnp.atleast_2d(X.var()), alpha=2., beta=2.)
r = m.fit(X, max_iter=80, track_ll=True,
    e_step_backend='cpu', m_step_backend='cpu')
r2 = r.model.fit(X, max_iter=50, tol=1e-8, track_ll=True,
    e_step_backend='cpu', m_step_backend='cpu')
print(r.converged, r.n_iter, r.model.joint.subordinator().alpha,
      r.model.marginal_log_likelihood(X))
print(r2.model.joint.subordinator().alpha,
      r2.model.marginal_log_likelihood(X))
```

**Test execution record:** `.venv/bin/python -m pytest tests/ -q --tb=short` completed with **1,111 passed, 85 deselected, one warning**, in 1,481.69 seconds (24 minutes 41 seconds). The warning is the installed JAXopt package's unmaintained-project deprecation notice. The deselected cases follow the default exclusion of slow/stress/integration/GPU tests; those extended suites were not run. The independent reproductions above are additional checks outside the existing suite. Passing the default suite therefore does not resolve these counterexamples. Report code examples were also syntax-checked, and the underlying numerical examples were executed in the accompanying audit work. Final `git diff --stat` was empty.
