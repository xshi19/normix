# Bessel moment-quadrature kernel (S10)

Production `log_kv` / `log_kv_moments` in `normix/utils/bessel.py`. One
centered whole-line Gauss–Legendre kernel, two array backends (`jax`, `cpu`).
No regimes, no `lax.cond`, no `custom_jvp`, no finite differences.

- **192-node kernel** — $2K_\nu(z)=\int_{\mathbb R}e^{\nu u-z\cosh u}\,du$
  replaced by a 192-point weighted sum (two Gauss–Legendre panels).
- **$u_0$** — mode of the integrand, $\operatorname{asinh}(\nu/z)$. Nodes
  live in $x=u-u_0$.
- **AD** — `jax.grad` / `jax.hessian` of that sum. Same derivatives as
  `log_kv_moments(...).d_arg` / `d_order` / `d2_*` (not finite differences).

User-facing walkthrough: `docs/tutorials/core/03_bessel_and_log_kv.md`.
Investigation that forced the rewrite: `dev-notes/investigations/2026-09-05/`.

## Representation

$$
2K_\nu(z)=\int_{\mathbb R}e^{\nu u-z\cosh u}\,du.
$$

Mode $u_0=\operatorname{asinh}(\nu/z)$, $\kappa=\sqrt{\nu^2+z^2}$, $x=u-u_0$.
The exponent is evaluated cancellation-free:

$$
g_c(x)=\nu x-\Bigl[\tfrac{\kappa+\nu}{2}\,\mathrm{expm1}(x)+\tfrac{\kappa-\nu}{2}\,\mathrm{expm1}(-x)\Bigr],
\qquad \kappa-\nu=z\bigl(z/(\kappa+\lvert\nu\rvert)\bigr)
$$

($\nu\ge 0$; mirror for $\nu<0$). The grouping $\kappa\cdot 2\sinh^2(x/2)+\nu\sinh x$
cancels in the left tail when $\kappa\approx\nu$ and is not used.

Two GL panels share the mode: $[x_\mathrm{lo},0]\cup[0,x_\mathrm{hi}]$. The
window is the smallest interval that drops $g_c(x)+kx$ by `BESSEL_QUAD_LOG_DROP`
for every tilt $|k|\le$ `BESSEL_WINDOW_TILT`. Nodes, weights, and window are
frozen under `lax.stop_gradient`. The log-sum-exp identity is then exact for
that geometry, so

$$
\partial_\nu\log K=E[u],\qquad
\partial_{\nu\nu}\log K=\mathrm{Var}(u)>0,
$$

and every higher derivative is a cumulant of the discrete measure. $\nu=0$ is
an ordinary point.

## `BesselMoments`

`log_kv_moments(v, z, backend)` returns `log_k`, $u_0$, the mean/covariance
of $s=(x,\mathrm{expm1}(-x),\mathrm{expm1}(x))$, and stored argument jets.
`d_order` / `d2_order` are $u_0+E[x]$ and $\mathrm{Var}(x)$. `d_arg`,
`d2_arg`, `d2_order_arg` are projections of
$w=2\sinh(u_0+x/2)\sinh(x/2)=\cosh(u_0+x)-\cosh u_0$, not affine images
of `cov`.

`cov` is the centered Gram $R^\top R$ with $R_i=\sqrt{p_i}\,(s_i-\bar s)$:
relative-ε entries, PSD to rounding. Reconstructing $\mathrm{Var}(e^{\pm x})$
from $\mathrm{expm1}(\log E[e^{2x}]-2\log E[e^x])$ loses the $O(1/z^2)$
even part of $\mathrm{Var}(\cosh u)$ once $z\gtrsim 10^6$; at
$(\nu,z)=(1/2,10^8)$ that path returned $L_{zz}<0$ against the exact
$1/(2z^2)$ (DLMF 10.39.2).

The grouping $\cosh u_0\cdot 2\sinh^2(x/2)+\sinh u_0\sinh x$ is the same
left-tail cancellation the exponent already avoids when $\kappa\approx\nu$;
do not use it for $w$.

GIG in $u=\log X-\tfrac12\log(b/a)$ is this family with $\nu=p$, $z=\sqrt{ab}$.
One call gives $\eta$ and $H=D\,\mathrm{cov}\,D$. Dense $H$ inherits an
$O(\varepsilon z)$ relative error in its smallest eigenvalue (the
$q=(0,1,1)$ direction at $a=b$). PSD until $z\sim 10^{16}$. Newton uses
relative Tikhonov $\lambda\,\mathrm{tr}(H_\theta)/n$ applied to the Fisher
before the bound sandwich. No $K_{p\pm 1}/K_p$ ratios,
no $p\pm 1$ evaluations.

NIG $E[\log Y]=\texttt{log\_kv\_moments}(-0.5,\lambda/\mu).\texttt{d\_order}+\log\mu$.

## Constants (sweep-frozen)

`BESSEL_QUAD_NODES=192`, `BESSEL_QUAD_LOG_DROP=40`, `BESSEL_WINDOW_TILT=2`,
`BESSEL_WINDOW_ITERS=40`. Sweep: `benchmarks/bench_bessel.py --accuracy`.
128 nodes fails the 1e-12 scaled-error contract at $\partial_z(0,10^{-10})$.

Stated domain: $z\ge 10^{-10}$, $|\nu|\le 300$. $(0,10^{-300})$ unresolved at
192 nodes. `log_kv(1,10^{-50})\approx 115.129$ (the old small-$z$ branch
clipped $z$ to `LOG_EPS` and returned $\sim 69$).

## What was deleted

Four-regime `lax.cond` (Hankel / Olver / small-$z$ / 64-point half-line GL),
`@jax.custom_jvp` with FD $\partial_\nu$, CPU `scipy.kve` plus a wrong overflow
fallback (383 vs 322 at $(1000,500)$), `BESSEL_EPS_V`, `FD_EPS_FISHER`,
`BESSEL_SMALLZ_THRESHOLD`.

Those FD stencils sampled shifted orders across regime seams: $L_{\nu\nu}=-600$
at GIG$(25,1,1)$; differencing $L\approx -z$ at $z=10^4$ lost $10^{-4}$
curvature. After S10, GIG$(25,1,1)$ has $H_{11}=\mathrm{Var}(\log X)>0$;
`cpu/newton` at that point goes from `grad_norm≈0.27` to $\sim 10^{-14}$.

`kve` remains a test oracle (`tests/test_jax_bessel.py`, mpmath table in
`tests/data/bessel_reference.json`).

## Benchmarks (2026-09-05, same HEAD)

Iterations and final LL: non-increasing / non-decreasing (gates pass).
Newton at the old seams converges instead of stalling.

Wall time vs the previous kernel/`kve` path (gates **miss** $\le 1.2\times$):

- Default fitter: GH $524\,\mathrm{ms}\to 5.18\,\mathrm{s}$; similar $5$–$10\times$
  on VG / NIG / NInvG.
- CPU E-step batch ($N=2552$): $2.3\to 41.5\,\mathrm{ms}$.
- SP500 GH E-step $z=\sqrt{a_\mathrm{post}b_\mathrm{post}}$ is $O(1)$ — Hankel
  ($z>\max(25,v^2/4)$) and Olver ($|v|>25$) do not fire there.

128 nodes was tried as the plan's first mitigation; it breaks the 1e-12 table.
Hankel/Olver were not re-added: they do not earn the E-step domain, and a
`lax.cond` seam is how the FD Hessian went negative. The residual wall time is
192-node quadrature vs AMOS `kve`, paid for a correct Fisher.

An FFI kernel would implement the `BesselMoments` contract under the JAX tier
by explicit opt-in (not shipped). `_BACKENDS` has two live entries.

## Tests

`tests/test_bessel_contract.py` — table (1e-11 relative; 1e-7 for $d_v,d_{vz}$
at large $z$; absolute at symmetry zeros), identities including $2z^2 L_{zz}=1$ at $\nu=1/2$,
AD ≡ bundle via `_log_kv_quad_jax`, large-$z$ finiteness,
former-seam Taylor consistency (not a global Lipschitz bound).
`tests/test_gig_properties.py::TestGIGMomentHessian` — $H$ PSD, $H_{11}>0$,
CPU ≡ JAX, sample $\mathrm{Cov}[t(X)]$, concentrated $q^\top H q=4L_{zz}$;
vs `jax.hessian(ψ)` is `slow`.
