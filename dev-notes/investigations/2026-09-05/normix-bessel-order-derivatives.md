# Normix: Bessel order derivatives, evidence and repair options

Verification of these numbers and prototypes of the recommended oracles: `bessel-derivative-verification.md`.

Review date: 2026-09-05. Source reviewed: Normix commit `7b6c4dfc784567c9f5f6b67a152a49eca6ed8488`. This is a numerical review and implementation proposal; no package code was changed.

## 1. What the independent reference actually was

The original audit used two references, which should be distinguished:

- `original_probe_bessel.py` compared values against SciPy's `log(kve(v,z))-z`. Its first order-derivative comparison used a centered finite difference of **SciPy's log-scaled Bessel values**, with step `1e-5`. This is an independent implementation of the function, but still a double-precision finite difference. It is a useful diagnostic, not a strong derivative oracle.
- `original_probe_derivatives_timing.py` used **mpmath with `mp.mp.dps=60`** for selected values and second order derivatives. The essential reference expression was `mp.diff(lambda nu: mp.log(mp.besselk(nu, mp.mpf(z))), mp.mpf(v), 2)`. The Normix comparator was nested `jax.grad`, evaluated in float64.

These two original files are preserved verbatim, including their original absolute import path and timing code. Change that path to run them elsewhere. The new `check_bessel_derivatives.py` is a portable, stronger follow-up, not the original script renamed.

The word “independent” means that mpmath does not call Normix's piecewise approximations or custom JVP. It does **not** mean an exact symbolic answer. By default, `mp.diff` uses numerical differences with extra working precision. High precision makes the cancellation much less problematic, but setting 60 digits alone is not a proof of 60 correct digits. [mpmath differentiation documentation](https://mpmath.org/doc/current/calculus/differentiation.html).

The enhanced script compares mpmath differentiation at 60 and 90 digits and checks a second construction: direct integration of probability moments, without calling either `besselk` or `diff`. It checks values, first derivatives and second derivatives at seven selected points. Its integral has a precision-dependent tail cutoff; agreement is strong numerical evidence, not a certified interval bound or a guarantee over the entire domain. It intentionally reports Normix discrepancies rather than asserting that Normix passes.

Run in an environment containing Normix, JAX and mpmath:

```bash
python check_bessel_derivatives.py --repo /path/to/normix --output results.json
```

The script enables JAX float64 before importing Normix, records dependency versions and the checkout commit, and converts the same binary64 inputs to mpmath. Its output distinguishes the nested-AD second derivative from a direct second-difference stencil.

## 2. Which derivative is fragile?

Write

$$
L(\nu,z)=\log K_\nu(z),\qquad z>0,\quad \nu\in\mathbb R.
$$

Here $\nu$, called `v` in the code, is the **order**; $z$ is the **argument**. The problematic quantities are primarily

$$
L_\nu=\frac{\partial L}{\partial\nu},\qquad
L_{\nu\nu}=\frac{\partial^2L}{\partial\nu^2},
$$

and potentially mixed derivatives built from the same rules. “Fragile” describes the numerical implementation, not a singularity in these mathematical derivatives. For real order and positive argument they are well defined.

Normix's `normix/utils/bessel.py`, in `_log_kv_jax_jvp`, supplies

$$
D_hL(\nu,z)=\frac{\widehat L(\nu+h,z)-\widehat L(\nu-h,z)}{2h},
\qquad h=10^{-5},
$$

where $\widehat L$ is the implemented approximation. Consequently, `jax.grad(log_kv)` uses this finite difference. JAX does not replace it with an exact derivative of the underlying mathematical function.

Differentiating that rule again essentially composes two first differences:

$$
D_hD_hL=
\frac{\widehat L(\nu+2h)-2\widehat L(\nu)+\widehat L(\nu-2h)}{4h^2}.
$$

Actual floating-point evaluation can differ from evaluating this simplified expression. Separately, the GIG explicit Hessian uses the familiar $h$-spaced stencil with denominator $h^2$. These are different numerical paths; their reported errors need not match.

For the argument derivative, the exact recurrence is available:

$$
L_z=-\tfrac12\left(\frac{K_{\nu-1}(z)}{K_\nu(z)}+
                         \frac{K_{\nu+1}(z)}{K_\nu(z)}\right).
$$

That is a useful foundation, although errors in the Bessel values and cancellation in subsequent derivatives can still matter.

## 3. There are analytical order derivatives

The premise “there is no analytical derivative with respect to order” is too strong. There are exact series, special-value identities and integral representations. What is missing is a comparably convenient general-order formula using only a few neighboring Bessel values. NIST explicitly tabulates order derivatives, including formulas at integer and half-integer orders. Some identities involve cancellation or removable singularities near integers, so existence alone does not make them a good float64 algorithm. [DLMF §10.38](https://dlmf.nist.gov/10.38).

A particularly useful starting point is

$$
K_\nu(z)=\int_0^\infty e^{-z\cosh t}\cosh(\nu t)\,dt.
$$

For positive $z$, the tail decays rapidly enough to differentiate under the integral for any fixed real order. [DLMF equation 10.32.9](https://dlmf.nist.gov/10.32#E9).

The following moment identities are derived directly from that representation. Let $q(t)$ be its normalized integrand on $[0,\infty)$, and define $s(t)=t\tanh(\nu t)$, $c(t)=\cosh t$. Then

$$
L_\nu=E_q[s],\qquad
L_{\nu\nu}=\operatorname{Var}_q(s)+E_q[t^2\operatorname{sech}^2(\nu t)],
$$

$$
L_z=-E_q[c],\qquad L_{zz}=\operatorname{Var}_q(c),\qquad
L_{\nu z}=-\operatorname{Cov}_q(s,c).
$$

The second-order formula is preferable to subtracting $E[t^2]-E[s]^2$: it adds nonnegative terms and permits a centered variance calculation. One set of quadrature nodes and normalized weights can supply all these quantities.

An even simpler order-derivative interpretation follows by extending the integral to the real line:

$$
2K_\nu(z)=\int_{-\infty}^\infty e^{\nu u-z\cosh u}\,du,
\quad q(u)=\frac{e^{\nu u-z\cosh u}}{2K_\nu(z)}.
$$

Thus

$$
\boxed{L_\nu=E_q[u],\qquad L_{\nu\nu}=\operatorname{Var}_q(u)>0.}
$$

Higher order derivatives are the corresponding cumulants of $u$. This also proves useful invariants: $L$ is even in order, $L_\nu$ is odd, $L_\nu(0,z)=0$, and $L_{\nu\nu}$ is strictly positive. A substantially negative second derivative is unequivocally wrong.

For the standard GIG parameterization

$$
f(x)\propto x^{p-1}e^{-(ax+b/x)/2},\quad a,b>0,
$$

its log normalizer is $\log2+\tfrac p2\log(b/a)+L(p,\sqrt{ab})$. Therefore its curvature in $p$, with $a,b$ fixed, is $L_{\nu\nu}=\operatorname{Var}(\log X)$. An incorrect sign here damages the geometry used by Newton updates and Fisher information calculations. It is a correctness issue, not just a performance issue.

## 4. Two distinct failure mechanisms

### Approximation seams

Normix dispatches among Hankel, Olver, small-argument and quadrature approximations. In the reviewed implementation, the large-order threshold is $|\nu|>25$; the quadrature has 64 Gauss–Legendre nodes. At $\nu=25$, the order finite difference can sample different approximations on opposite sides. Their value errors need not agree.

A mismatch $\Delta$ across a boundary contributes approximately $\Delta/(2h)$ to a first difference, and can contribute on the scale of $\Delta/h^2$ to a second difference. A seemingly small value discrepancy can therefore dominate a derivative. Decreasing the step can make this worse. Under-resolved quadrature also causes value errors away from a seam.

The original value check at $(25,0.1)$ gave approximately `128.984756343362` against `128.984784889971`. The absolute log-value error is only about $2.85\times10^{-5}$, yet the first derivative was about `7.6218` against `6.19448`. Near $(25,10^{-6})$, the first-derivative discrepancy was much larger.

Moving the threshold merely moves the potential seam. A repair must verify overlapping approximations for values **and derivatives**. Smooth blending also needs care: differentiating a blend introduces terms containing the derivative of the blending weight times the approximation mismatch.

### Cancellation, even within a good branch

For central differences, a useful schematic error model is

$$
|D_hL-L_\nu|\lesssim C_1h^2+O(\delta/h),\qquad
|D_h^2L-L_{\nu\nu}|\lesssim C_2h^2+O(\delta/h^2),
$$

where $\delta$ includes floating-point and approximation error. These are explanatory scales, not certified bounds for Normix.

At large $z$, $L$ contains a large term $-z$ that is independent of order. A second difference subtracts nearly equal large values to extract a very small order curvature. At $z=10^4$, float64 rounding of a value near $-10^4$ is on the order of $10^{-12}$. Dividing such noise by $h^2=10^{-10}$ can produce noise on the order of $10^{-2}$, while the true curvature is about $10^{-4}$.

For fixed order, the large-argument expansion gives

$$
L=-z+\tfrac12\log\frac{\pi}{2z}
  +\frac{4\nu^2-1}{8z}-\frac{4\nu^2-1}{16z^2}+O(z^{-3}),
$$

$$
L_{\nu\nu}=\frac1z-\frac1{2z^2}+O(z^{-3}).
$$

This explains why large-argument second derivatives are especially demanding for raw-log finite differences. This expansion is a fixed-order statement, not a uniform approximation for arbitrarily large order.

## 5. Recommended numerical design

My recommendation is a **hybrid derivative-aware kernel**: compute derivatives from stable integral moments in the difficult region, and differentiate validated series/asymptotic kernels in their reliable regions. Keep a high-precision reference in tests. There is no single universally best quadrature size, finite-difference step or asymptotic threshold.

### A practical improvement to the current quadrature

Use the same log-domain quadrature infrastructure to accumulate normalized weights and moments. Compute centered variances, not differences of large raw moments. Check convergence by increasing node count and expanding the integration interval. Positive quadrature weights preserve a nonnegative variance, but that property alone does not guarantee accuracy.

For the whole-line representation, the mode and local curvature are

$$
u_0=\operatorname{asinh}(\nu/z),\qquad
\kappa=\sqrt{\nu^2+z^2}.
$$

Here $u_0$ is the mode, not the order. Integrate in $y=\sqrt\kappa(u-u_0)$ to resolve concentrated densities. With $x=u-u_0$, the centered exponent is exactly

$$
g(u_0+x)-g(u_0)=\nu(x-\sinh x)-\kappa(\cosh x-1).
$$

Use stable small-$x$ expressions in float64, such as $\cosh x-1=2\sinh^2(x/2)$, and a series for $x-\sinh x$ near zero. Adapt the integration interval using tail decay; a fixed number of local standard deviations is inadequate in every regime, especially when both order and argument are small. The supplied reference script uses high precision and numerical tail expansion, not a production float64 implementation.

### Differentiate the approximation before losing small terms

For the Hankel region, compute the log of the correction series directly and differentiate that smooth expression with AD or explicit coefficient derivatives. The order-independent $-z$ then contributes exactly zero, without subtracting rounded evaluations. Use a direct scaled-log kernel

$$
S(\nu,z)=\log(e^zK_\nu(z)),\qquad S_\nu=L_\nu,\quad S_{\nu\nu}=L_{\nu\nu}.
$$

Computing `log_kv(v,z) + z` after `log_kv` has already rounded cannot recover lost bits. The scaling must happen inside the evaluation. For an Olver kernel, differentiate all order dependence, including its ratio $z/\nu$, and validate derivative truncation errors separately from value errors.

In the small-argument, fixed positive-order leading approximation,

$$
L\approx\log\Gamma(\nu)+\nu\log(2/z)-\log2,
\quad L_\nu\approx\psi(\nu)+\log(2/z),
\quad L_{\nu\nu}\approx\psi_1(\nu).
$$

These are inexpensive derivatives but the approximation is not uniform near order zero. Use a separate near-zero method and validate correction terms. Do not differentiate a formula valid only at a single half-integer order as if it described a neighborhood in order.

### Keep derivative paths consistent

Expose an internal routine returning a value, gradient and Hessian together, with shared intermediate work and one error-controlled regime decision. Use it both in the custom JVP and in GIG optimization. Define and test the higher derivative behavior explicitly: replacing a first derivative with a custom moment rule does not automatically guarantee that every nested derivative is implemented correctly. Parameter-dependent quadrature nodes and bounds also need consistent differentiation and truncation control.

For the GIG Hessian, compute the covariance of sufficient statistics where feasible. This preserves the mathematical structure and avoids reconstructing tiny curvatures by subtracting large log normalizers. A line search or trust region is still useful, but cannot repair a wrong derivative oracle.

## 6. If finite differences must remain temporarily

1. Compute a genuinely scaled log function before subtraction in the large-argument region.
2. Use separate, adaptive steps for first and second derivatives, with a step sweep and Richardson comparison. For an ideal smooth, accurately evaluated float64 function, common starting scales are machine-epsilon to the powers $1/3$ and $1/4$, respectively, times a suitable parameter scale. They are not universal prescriptions here: approximation error can dominate rounding.
3. Detect stencils that cross a regime boundary. Use a single validated approximation throughout that stencil only if its overlap region is demonstrably accurate. Otherwise use a moment-based fallback.
4. Check convergence of the underlying quadrature before trusting finite-difference convergence. Agreement between two step sizes can be misleading if both sample the same biased approximation.
5. Check positivity and symmetry and report unresolved accuracy instead of silently using a bad Hessian. Clipping a negative curvature to zero hides the numerical failure and does not restore a correct Newton direction.

Complex-step differentiation is not a drop-in fix. It requires an implementation analytic in complex order; Normix's real-valued absolute values and branching do not meet that requirement. SciPy's `kve` interface accepts real order, even though its argument may be complex. Moreover, a naive complex-step second derivative can still suffer cancellation. High-precision complex contour methods or Taylor-series arithmetic are useful reference options with appropriate backends. [SciPy kve documentation](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.special.kve.html).

## 7. Packages, code and papers worth using

| Resource | Best role for Normix | Qualification |
|---|---|---|
| [mpmath](https://mpmath.org/doc/current/calculus/differentiation.html) | High-precision test references and precision escalation | Too costly for the optimization hot path; numerical accuracy still needs checks. |
| [Takekawa's logbesselk](https://github.com/tk2lab/logbesselk) | Study existing JAX/TensorFlow implementations designed for log Bessel calculations and derivatives | Already cited in your development survey. Revisit its actual derivative algorithms; independently test accuracy, higher derivatives and current JAX compatibility before reuse. |
| [BesselK.jl](https://github.com/cgeoga/BesselK.jl) and [Geoga et al., Fitting Matérn Smoothness Parameters Using Automatic Differentiation](https://arxiv.org/abs/2201.00090) | Study algorithms designed to support order differentiation through a Bessel implementation | Especially relevant to avoiding black-box finite differences. Validate Normix's log-scale, negative-order and extreme-argument requirements before porting. |
| [FLINT/Arb complex hypergeometric functions](https://flintlib.org/doc/acb_hypgeom.html) | Potential stronger reference using ball arithmetic and Taylor coefficients | The documented `acb_hypgeom_bessel_k_0f1_series` accepts polynomial order and argument. With order `nu0+t`, a series logarithm yields order derivatives as factorial times coefficients. Handle its documented integer-order cases and verify finite enclosures. Not exercised in this audit. |
| [DLMF order derivatives](https://dlmf.nist.gov/10.38) | Exact identities and special cases for independent tests | A mathematically exact expression can still be numerically unstable near removable singularities. |

SciPy's scaled Bessel values are useful for value and ratio checks, but are not themselves an order-derivative implementation. Do not confuse derivatives with respect to argument, such as `kvp`, with derivatives with respect to order.

### C++ or Rust?

A native rewrite of the current finite-difference formula would reproduce its mathematical weaknesses. First choose and validate the derivative algorithm. Then profile an implementation that returns values and derivatives together.

For a CPU-oriented native kernel, C++ offers a direct route to existing numerical code and the JAX FFI examples; Rust is also viable through a C ABI if that is your preferred maintenance language. Neither automatically provides order derivatives. A useful native interface would process a batch and return `L`, `L_v`, `L_z` and the required Hessian entries in one call, amortizing dispatch and sharing quadrature or series work. JAX integration needs explicit differentiation and batching behavior, and a CPU-only kernel may introduce transfers in GPU workloads. [JAX FFI documentation](https://docs.jax.dev/en/latest/ffi.html).

I would prototype the moment/asymptotic hybrid in JAX first, verify the difficult cases, then compare end-to-end GIG fitting time against a fused native CPU implementation if profiling justifies it. No native speedup factor has been measured in this follow-up.

## 8. Regression coverage and implementation order

First add the reproduced failures and independent references to tests. Include orders at zero, positive and negative orders, half-integers, and points immediately on both sides of every dispatch boundary. Sweep argument logarithmically and include concentrated large-argument cases. Use an absolute tolerance near zero derivatives and a relative tolerance elsewhere; a single relative tolerance is inappropriate at order zero.

Then replace the order finite differences and the separate GIG Hessian stencil with the shared derivative-aware kernel. Verify parity, strictly positive order curvature, mixed-derivative agreement, and covariance/Fisher structure. Compare values and derivatives against the reference independently; a recurrence check against the same underlying approximation is not an independent accuracy test.

Finally benchmark complete GIG optimization, recording objective evaluations, derivative evaluations, iterations, convergence quality and wall time. A faster Bessel call is valuable only if the optimizer retains correct derivatives and reliable convergence.

## 9. Results of the enhanced check

The accompanying JSON records the actual run, including full reference strings, dependency versions, precision comparisons and integral comparisons. The table below is generated from that run. “Nested AD” refers specifically to differentiating Normix's custom JVP twice; the JSON separately records the direct second-difference result.

| Order | Argument | Normix first derivative | Reference first derivative | Normix second derivative, nested AD | Reference second derivative |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0 | 0 | 0.7311018457 | 0.7311001812 |
| 0.5 | 1 | 0.3613286169 | 0.3613286169 | 0.7061579099 | 0.7061585585 |
| 25 | 0.1 | 7.621800228 | 6.194479127 | 71366.90343 | 0.04081030157 |
| 25 | 1 | 3.889322633 | 3.892323422 | -150.0017177 | 0.04077454478 |
| 25 | 1e-06 | 2778.003407 | 17.70740025 | 138015335 | 0.04081066326 |
| 1 | 10000 | 9.995346772e-05 | 9.999500037e-05 | 0.004547473509 | 9.999500004e-05 |
| 5 | 1e+06 | 5.820766091e-06 | 4.9999975e-06 | 0.2910383046 | 9.999995e-07 |

Maximum precision-escalation discrepancy: `5.01e-62`. Maximum direct-integral discrepancy: `1.19e-61`. Each discrepancy is `abs(a-b)/max(1,abs(b))`, maximized over the value and both derivatives. These are observed agreements, not certified error bounds.

A narrow diagnostic also differentiates the existing internal Hankel approximation directly, bypassing the custom finite-difference rule:

| Order | Argument | Direct AD of Hankel kernel, second order derivative |
|---:|---:|---:|
| 1 | 10000 | 9.999500004172915e-05 |
| 5 | 1e+06 | 9.999994999880418e-07 |

These two direct-kernel results agree with the reference to approximately float64 precision. This is evidence that the finite-difference rule is avoidable in this regime, not validation of direct AD through every existing branch.

Environment: Python 3.12.13; jax 0.9.1, jaxlib 0.9.1, numpy 2.4.2, mpmath 1.4.1, scipy 1.17.0.
