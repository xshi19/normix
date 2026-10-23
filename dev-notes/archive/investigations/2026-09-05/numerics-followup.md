# Numerics follow-up

> **ARCHIVED — 2026-10-23.** This note landed in [#134](https://github.com/xshi19/normix/pull/134) (`bc8b24e`). Later follow-ups are [#135](https://github.com/xshi19/normix/pull/135) (`0bc11fc`) and [#145](https://github.com/xshi19/normix/pull/145) (`37914d0`).

Scope: review items 6.4, 6.5, 9.4, and 9.2. Reproduction:
`.venv/bin/python /workspace/reports/scratch/repro_numerics.py`.
Targeted pytest files run serially on the shared 15 GB host.

## 6.4 Joint density powers

Let $g(y)$ be the subordinator density and
$X\mid Y=y\sim N_d(\mu+\gamma y,y\Sigma)$, with $\Sigma$ positive definite.
For $q>0$, integrating the powered conditional Gaussian gives

$$
\int_{\mathbb R^d}p(x\mid y)^q\,dx
=q^{-d/2}(2\pi)^{(1-q)d/2}|\Sigma|^{(1-q)/2}y^{(1-q)d/2}.
$$

Thus only $\int_0^\infty g(y)^q y^{(1-q)d/2}\,dy$ controls existence.
For VG the integrand is proportional to
$y^{q(\alpha-1)+(1-q)d/2}e^{-q\beta y}$.
The tail is integrable; the origin requires
$q(\alpha-1)+(1-q)d/2>-1$.
For NInvG it is
$y^{-q(\alpha+1)+(1-q)d/2}e^{-q\beta/y}$.
The origin is integrable; the tail requires
$q(\alpha+1)-(1-q)d/2>1$.
Equality gives logarithmic divergence in each case.

For GH, define $p_q=q(p-1)+(1-q)d/2+1$. The remaining integral is
$\int_0^\infty y^{p_q-1}e^{-q(ay+b/y)/2}\,dy$.
It is finite for every $q>0$ if $a,b>0$, including NIG ($p=-1/2$).
At $b=0$ it requires $p_q>0$; at $a=0$ it requires $p_q<0$.
GH evaluates this expression directly in subordinator coordinates.
Recovering a zero rate from scaled joint natural parameters can introduce
roundoff that changes the boundary law. In the new three-dimensional VG
quadrature case, that changed $R(2)$ from $-0.562049$ to $-0.823855$.

The guards follow PR #118 / decision MD1: evaluate a safe interior order,
then select $+\infty$ for $R(q)=\log\int p^q$ outside the domain.
Rényi entropy uses $R(q)/(1-q)$, including within its Taylor window when
the requested order is nonintegrable. The finite Taylor branch applies
only within the integrability domain.

Audit: `NormalMixture.joint_renyi` delegates to the joint and inherits these
conditions. Factor mixtures have no joint information-measure API.
KL and Hellinger use partition chords, not `log_density_power`; their
existing gauge lifting is unchanged. GIG already guards its boundary
partition domain. InverseGaussian uses the interior GIG embedding;
MultivariateNormal is integrable for every positive order.

Evidence: `test_density_power_domains.py` integrates in log scale using
SciPy after exact Gaussian integration. It covers both sides of the
boundary, equality, finite orders below and above one, GH boundary lifts,
NIG/GH interiors, `jit`/`vmap`, and a nonintegrable order inside the Taylor
window. Truncated tail integrals stay constant or grow in divergent cases.
Before the fix: 8 of the initial 22 cases failed; both Taylor-window cases
also failed. The supplied repro gave VG $H_2=1.2775267168058138$ and
NInvG $R(1/2)=1.438798918624631$; the corrected results are $-\infty$ and
$+\infty$.

## 6.5 Information-measure autodiff

The nested custom JVP captured `self` while declaring only the order as a
primal. Parameter tracers therefore escaped its differentiation boundary.
The Taylor formula already supplies the limiting derivative
$\partial_q H_q|_{q=1}=-V_H/2$. Removing the custom rule lets ordinary AD
differentiate both branches, including their distribution parameters.
The unused exact denominator is masked to remain safe under `vmap` at one.
Entropy and varentropy use ordinary nested derivatives and had no such
capture defect. NIG inherits the fix through its GH embedding.

`test_information_gradients.py` compares parameter and order derivatives
with central differences for Gamma, InverseGamma, joint VG, and joint NIG
at $q=0.7,1,2$. It also checks entropy/varentropy parameter derivatives,
Gamma closed forms, and batched derivatives at one. The first regression
failed with `UnexpectedTracerError` before the change. For Gamma(3,2),
$\partial_\beta H_2=-0.5$ and
$\partial_\alpha H_2=2\log 2-7/6=0.219627694453224$;
$\partial_q H_q|_{q=2}=-0.106770228755435$ is unchanged.

## 9.4 Implicit expectation inversion

Design constraints: F2 keeps the existing constructors; F4 supplies the
Hessian triad; S2 keeps all derivatives in natural coordinates; E14 keeps
Gamma's optional shape projection. No new class, module, or public API.
The shared private classmethod
`ExponentialFamily._implicit_natural_params(eta, theta)` attaches the
derivative to a solved natural parameter vector. Gamma, InverseGamma,
GIG, and the generic constructor solve with stopped targets and initial
guesses, then call this method. Closed-form inverses remain ordinary AD.

Differentiate $\nabla\psi(\theta)=\eta$ to obtain
$H(\theta)d\theta=d\eta$. The custom JVP solves this linear system;
JAX transposes it for reverse mode. It uses the distribution's Hessian
triad, never the optimizer's damped matrix or its reparametrization.

Let $D_{ii}=\sqrt{H_{ii}}$ and $C=D^{-1}HD^{-1}$.
The tangent is $D^{-1}C^{-1}D^{-1}d\eta$, computed by Cholesky.
This removes units before checking identifiability. Require finite
entries, positive diagonal, and
$\lambda_{\min}(C)>n\epsilon\lambda_{\max}(C)$ for $n$ natural coordinates
and machine epsilon $\epsilon$. Otherwise the derivative is NaN.
Nonfinite solutions and active natural-coordinate bounds also give NaN.
No ridge or pseudoinverse substitutes a different derivative. Keeping the
NaN as a multiplicative tangent factor makes both JVP and VJP propagate it.

Gamma's shape projection occurs after the implicit inverse, so an active
`alpha_min` has zero shape derivative and the rate still follows the target
mean. Joint pytree M-steps inherit subordinator derivatives. Redundant flat
joint coordinates are not promised a unique inverse; a singular Fisher
matrix gives NaN derivatives rather than an invented gauge choice.

GIG cold starts formerly entered a CPU multi-start even with
`backend='jax'`. Python `float` conversions made this fail under tracing.
JAX Newton now uses traceable Gamma/InverseGamma/InverseGaussian seeds and
the existing JAX multi-start solver. Its selected metadata stays in arrays.
The CPU path remains eager-only and does not support JAX transformations;
All three JAX methods support transformations.

Before: the Gamma and InverseGamma Jacobian tests raise the dynamic-loop
reverse-mode error; the GIG cold-start test raises `ConcretizationTypeError`.
`test_expectation_autodiff.py` checks forward/reverse Jacobians, scalar
reverse gradients, JIT, and central differences at three points per family.
References are analytical trigamma Hessians for Gamma/InverseGamma and
40-digit mpmath derivatives of the Bessel normalization for GIG. A GIG
with $a=10^{-4},b=10^4$ tests unequal units. Additional cases cover cold
starts, failed solves, a singular representation, the closed-form inverse
Gaussian, Gamma's projection, and joint VG/NInvG M-steps.

The first 20 inversion cases passed. Two additional regressions exercise
the generic base constructor with JAX L-BFGS/BFGS under JIT. The first
failed at a Python `float` conversion in the solver result. Selected
quasi-Newton metadata now remains in arrays, as for Newton. This change
is required by the derivative path; it does not change convergence tests.
At Gamma(3,2), the inverse Fisher matrix is
$\left[\begin{smallmatrix}16.23357293&-10.82238195\\
-10.82238195&8.54825464\end{smallmatrix}\right]$.
The InverseGamma(3,2) shape gradient is
$(10.82238195,-16.23357293)$ instead of a reverse-mode exception.

The resumed validation adds
`test_gig_cold_start_quasi_newton_derivative_under_jit` for L-BFGS and
BFGS. It exposed a remaining CPU fallback in JAX quasi-Newton cold starts:
`np.array(t0)` raised `TracerArrayConversionError` before solving.
That change made all JAX cold starts use the requested method: Newton
used `vmap`, and quasi-Newton used sequential `lax.map`. Round 5 below
replaces both with early stopping over ordered seeds. A finite converged
solution still outranks a lower unconverged objective. The selected
iteration count is preserved.

## 9.2 GIG target-domain boundary

The statistic order is $(\log X,1/X,X)$, so write
$\eta=(\ell,m_-,m_+)$. For a positive nonconstant random variable,
strict Jensen gives $-\log m_-<\ell<\log m_+$;
Cauchy–Schwarz gives $m_-m_+>1$. Both moments must be positive and every
entry finite. The public constructor checks these necessary conditions
before solving. Equality is rejected without a tolerance: a finite
interior GIG is nonconstant. Gamma and InverseGamma boundaries with finite
moments need not saturate these inequalities; they remain admissible to
this cheap check. Existence of an interior GIG still depends on the solve.

Eager failures raise `ValueError` naming the first violated condition.
A scalar traced call uses `lax.cond` to return a NaN model without running
the solver. The invalid branch also has NaN derivatives. Under `vmap`,
JAX can evaluate both branches; the invalid result is still masked.
The numerical Newton kernel is unchanged. The public boundary delegates
valid targets to `_from_expectation_unchecked`, which retains the solve
and shared implicit derivative.

TDD: replacing the solver with a failure sentinel showed that an invalid
negative inverse moment reached the solver before the fix.
`test_gig_expectation_domain.py` checks each inequality, zero/negative
moments, NaN/infinity, equality, both eager backends, traced/vmapped NaN
models, invalid-target gradients, valid fits, and finite-moment Gamma /
InverseGamma limits.

After: all 31 domain tests pass. The supplied invalid targets now raise
condition-specific `ValueError` instead of spending a solve and raising
`RuntimeError`. A 30-pair warmed JIT timing on GIG(0.5,1,1), a 0.9-scaled
warm start, and 20 Newton steps gave median 2.799 ms through the public
guard versus 2.804 ms through the unchecked solve. This shows no measurable
overhead in that workload; it is not a general performance claim.
Reproducer: `/workspace/reports/scratch/time_gig_boundary.py`;
output: `/workspace/reports/scratch/gig_boundary_timing.txt`.

## Round 5: cold-start performance

The default JAX cold start changed from CPU L-BFGS-B to a traceable JAX
solve in `b9db28c`. At `96fe9ca`, Newton vmapped 15 seeds through a
500-iteration `lax.scan`. Converged parameters froze, but each lane still
computed gradients, Hessians, line searches, and Bessel quadrature.

Instrumentation of the original scan recorded the first strict KKT
convergence check for GIG$(0.5,1,1)$, with float64 enabled before import:

| Original lane | Seed | Checks at $10^{-10}$ / $10^{-12}$ |
|---|---|---|
| 0, 3, 4 | Gamma and duplicate perturbations | 19 / 19 |
| 1, 7, 8 | InverseGamma and duplicate perturbations | 19 / 19 |
| 2, 11–14 | InverseGaussian and duplicate perturbations | 6 / 7 |
| 5, 6 | Other Gamma perturbations | 19 / 19; 16 / 16 |
| 9, 10 | Other InverseGamma perturbations | 18 / 18; 16 / 16 |

Every lane executed 500 iterations. None needed the full budget.
Quasi-Newton used `lax.map` and likewise solved every seed, even after
finding a finite converged result.

The Newton kernel now uses `lax.while_loop` and reports its actual
iteration count. Its Newton step, damping, line search, strict stopping
tolerance, and final inversion acceptance test are unchanged. The cached
warm-start entry point retains its four-array return contract.

All three JAX methods try ordered seeds in a cached `lax.while_loop`.
The first finite converged result ends the search. If none converges,
the lowest finite objective wins; if every objective is nonfinite, the
first result remains the failure result. GIG tries the interior
InverseGaussian seed first, before the Gamma/InverseGamma boundary seeds.
Its cached classmethod includes seed generation and fixed rate bounds.
The seed loop contains one solver graph, avoiding a second compiled copy
for the first seed. CPU multistart still tries every seed.

The implicit derivative remains attached after the stopped primal solve:
$H(\theta)d\theta=d\eta$. No optimizer history is differentiated.
This change is a measured solver scheduling choice under S2/S3/S11;
it introduces no class boundary or derivative formula.

Seven regression cases failed before the fix: a quadratic exhausted its
reported Newton budget; all three methods visited an unnecessary later
seed; three standard GIG targets reported 30 steps. The guards assert
iteration counts or observed objective evaluations, never elapsed time.
Additional cases check cache reuse and selection after nonfinite or
unconverged starts. `GIGFromExpectationCold` in
`asv_bench/benchmarks/gig.py` measures eager and compiled cold starts for
Newton, L-BFGS, and BFGS after compilation.

Cold-start tolerance is a dynamic JIT argument, so changing it reuses the
executable. The multistart JIT is inlined into the cached GIG entry point
so compilation can see the fixed rate bounds. Seed perturbations use
array broadcasting rather than unrolling separate updates for each seed.

CPU measurements on the shared review box, JAX 0.9.1, float64:

| Measurement | `96fe9ca` | Round 5 |
|---|---:|---:|
| Supplied eager repro, first call, $10^{-10}$ | 50.73 s | 3.31 s |
| Supplied eager repro, next call, $10^{-12}$ | 44.79 s | 0.01 s |
| Cold Newton, compiled median | 19.8267 s | 0.001976 s |
| Cold L-BFGS, compiled median | 0.3964 s | 0.001967 s |
| Cold BFGS, compiled median | 2.4786 s | 0.001298 s |
| Perturbed warm Newton, compiled median | 3.041 ms | 1.095 ms |
| GH M-step, compiled median | 17.149 ms | 1.901 ms |
| Five GH EM steps, compiled median | 85.077 ms | 13.446 ms |

The supplied master baseline was 2.67 s / 3.25 s. The first JAX call still
pays compilation; the second tolerance no longer compiles another solve.
Compiled inversion medians use two before and three after samples; the
GH measurements use 61 samples. No GPU is available on this box.

For GIG$(0.5,1,1)$, GIG$(-1.5,2,0.7)$, and GIG$(2.5,0.5,3)$, the selected
Newton counts at $10^{-10}$ / $10^{-12}$ are 6 / 7, 6 / 6, and 9 / 9.
The corresponding L-BFGS counts are 18 / 19, 17 / 17, and 21 / 23;
BFGS counts are 17 / 18, 16 / 17, and 22 / 23. All are finite and converged.

The external review report at
`/workspace/reports/codex-numerics-summary.md` records validation results,
commands, raw artifact paths, and the local commit. Profilers and timing
scripts live under `/workspace/reports/scratch/round5_*`; the paired GH
benchmark is `/workspace/reports/scratch/time_em_forward.py`.

Final validation: `test_expectation_autodiff.py` 24 passed,
`test_gig_expectation_domain.py` 31 passed, `test_solvers.py` 85 passed
and 6 default-deselected, `test_gig_log_partition.py` 24 passed.
Each file ran in its own serial pytest process with `-p no:cacheprovider`.
Ruff F401/F811/F841 passed for `normix/` and the changed tests/benchmark.
All six ASV setup/time smoke cases passed. GH M-step and five-step fit
outputs match the pre-fix snapshot exactly, with maximum absolute
change zero. No full suite, GPU suite, or docs build ran.
