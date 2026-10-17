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
