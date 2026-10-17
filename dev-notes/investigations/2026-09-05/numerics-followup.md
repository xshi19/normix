# Numerics follow-up

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
GIG's documented transformation contract is the JAX Newton path.

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
