# Bessel order derivatives — verification of the 2026-09-05 audit

Source still matches the reviewed tree for `normix/utils/bessel.py` and GIG (current HEAD is a later docs-only commit). Package code was not changed. Prototype: `prototype_bessel_derivatives.py`. Numbers: `prototype-bessel-results.json`.

## Verdict

The Bessel numbers in the Astra review are right. The order-derivative math is right, with one stencil-scale clarification. The recommended repairs work; the ranking is more specific than “use moments everywhere.”

| Claim | Status |
|---|---|
| Values / $L_\nu$ / $L_{\nu\nu}$ at the seven audit points | Reproduced on this machine; JAX vs the audit JSON is bit-identical |
| mpmath 60-digit table | Spot-checked at $(25,1)$, $(25,10^{-6})$, $(0.5,1)$; matches |
| Independent half-integer identity (DLMF 10.38.7) | $L_\nu(1/2,1)=e^{2}E_1(2)$ agrees with the mpmath table to $1$ ULP |
| CPU overflow fallback $(1000,500)$ | CPU $383.066358166$, JAX and mpmath $322.317651492$ |
| Internal $z$ floor: `log_kv(1, 1e-50)` | $69.07755$ vs leading small-$z$ $115.12925$ |
| `GIG(-2,1,1e-22)` log-partition | JAX $69.07755$, CPU $690.77553$, mpmath $102.700038453$ |
| `GIG(-2, 1e-8, 1e-8).rvs(200)` | $0$ finite draws |
| `InverseGamma(0.5,1)` mean/var; Rényi headlines | Match |

EM continuation, solver false-convergence, and joint-EF Fisher were not re-run here.

## Why the current $\partial_\nu$ is wrong

Two mechanisms, both real.

**Seam.** Dispatch is `|v| > 25` for Olver, else (here) 64-node quadrature. The custom JVP uses $h=10^{-5}$. At $\nu=25$:

```
v-h = 24.99999  → quadrature
v+h = 25.00001  → Olver
```

The nested second derivative is the $2h$ stencil

$$
\frac{L(\nu+2h)-2L(\nu)+L(\nu-2h)}{4h^2}.
$$

The GIG Hessian uses the $h$ stencil with denominator $h^2$, four times larger. At $(25,0.1)$ the quadrature/Olver value mismatch is $\Delta\approx 2.85\times 10^{-5}$, and $\Delta/(4h^2)\approx 7.1\times 10^{4}$, matching the nested result $71366$. GIG $H_{11}$ at `GIG(25,0.1,0.1)` is $285466=\Delta/h^2$.

The existing $\partial_\nu$ tests cannot catch this: they compare against SciPy finite differences with the **same** $h=10^{-5}$, and they never sit on `|v|=25`.

**Large-$z$ cancellation.** Inside a single Hankel branch the primal is accurate, but $L\approx -z$ is independent of order. Differencing logs rounded at $z=10^{4}$ cannot recover $L_{\nu\nu}\sim 10^{-4}$. Direct AD of `_hankel_log_kv` recovers the mpmath curvature to float64.

$L_{\nu\nu}<0$ is impossible: on the whole-line representation it is $\mathrm{Var}(u)>0$, and for GIG it is $\mathrm{Var}(\log X)$. The Newton $H_{11}=-600$ at `GIG(25,1,1)` is a broken oracle, not a geometry of the model.

## Math review

The identities in `normix-bessel-order-derivatives.md` check out.

- DLMF 10.32.9: $K_\nu(z)=\int_0^\infty e^{-z\cosh t}\cosh(\nu t)\,dt$. Differentiating under the integral is legitimate for $z>0$.
- Half-line: $L_\nu=E_q[t\tanh(\nu t)]$, $L_{\nu\nu}=\mathrm{Var}_q(s)+E_q[t^2\mathrm{sech}^2(\nu t)]$. The second form equals $E[t^2]-E[s]^2$; it is the stable evaluation, not a different theorem.
- Whole line: $2K_\nu(z)=\int e^{\nu u-z\cosh u}\,du$, so $L_\nu=E[u]$, $L_{\nu\nu}=\mathrm{Var}(u)$, $L_z=-E[\cosh u]$, $L_{zz}=\mathrm{Var}(\cosh u)$, $L_{\nu z}=-\mathrm{Cov}(u,\cosh u)$.
- Mode $u_0=\mathrm{asinh}(\nu/z)$, curvature $\kappa=\sqrt{\nu^2+z^2}$, and
  $g(u_0+x)-g(u_0)=\nu(x-\sinh x)-\kappa(\cosh x-1)$
  are exact. The small-$x$ rewrites $\cosh x-1=2\sinh^2(x/2)$ and a series for $x-\sinh x$ are required in float64.
- Hankel $L_{\nu\nu}=1/z-1/(2z^2)+O(z^{-3})$ matches $(1,10^4)$ and $(5,10^6)$.
- DLMF 10.38 order derivatives exist, including $\partial_\nu K_\nu|_{\nu=0}=0$ and the half-integer $E_1$ formula. They are not a general float64 algorithm: (10.38.2) has $\csc(\nu\pi)$.
- Complex-step is not a drop-in: the implementation uses `abs` and `lax.cond`.

One scale note the audit left implicit: nested AD through the JVP is a $2h$ stencil ($4h^2$ in the denominator). The GIG 11-Bessel Hessian is the $h$ stencil. Same $\Delta$, different blow-up.

## Do the recommendations improve the current method?

Yes. Prototypes vs the same mpmath table, **scaled** error $|g-r|/\max(1,|r|)$ as in the audit script (so $L_{\nu\nu}\sim 10^{-4}$ is not hidden by a relative scale of $1$):

| $(\nu,z)$ | current $L_{\nu\nu}$ | kernel AD | moments $n=128$ | Gauss–Hermite $n=64$ | Olver AD |
|---|---:|---:|---:|---:|---:|
| $(0,1)$ | $1.7\times 10^{-6}$ | $3\times 10^{-15}$ | $4\times 10^{-16}$ | $3\times 10^{-10}$ | n/a ($v=0$) |
| $(0.5,1)$ | $6.5\times 10^{-7}$ | $2\times 10^{-15}$ | $0$ | $2\times 10^{-10}$ | $1.1$ (wrong regime) |
| $(25,0.1)$ | $7.1\times 10^{4}$ | $1.3\times 10^{-6}$ | $3\times 10^{-17}$ | $7\times 10^{-18}$ | $4\times 10^{-12}$ |
| $(25,1)$ | $150$ (wrong sign) | $1.9\times 10^{-8}$ | $1\times 10^{-17}$ | $0$ | $4\times 10^{-15}$ |
| $(25,10^{-6})$ | $1.4\times 10^{8}$ | $3.5\times 10^{-4}$ | $0$ | $7\times 10^{-18}$ | $4\times 10^{-12}$ |
| $(1,10^{4})$ | $4.4\times 10^{-3}$ | $0$ | $8\times 10^{-20}$ | $3\times 10^{-20}$ | $3\times 10^{-12}$ |
| $(5,10^{6})$ | $0.29$ | $0$ | $8\times 10^{-22}$ | $4\times 10^{-22}$ | $2\times 10^{-11}$ |

Relative to $|L_{\nu\nu}|$ itself, current errors at the last two Hankel points are $\sim 45\times$ and $\sim 3\times 10^{5}$. Kernel AD of Hankel is exact at float64.

**Kernel AD** = `jax.grad` of `_log_kv_scalar` (the selected branch, no order FD). This is the cheapest repair. It already removes the seam spike and the large-$z$ curvature. It is **not** enough at $(25,10^{-6})$: the primal is still the unresolved 64-node quadrature ($L$ itself is $0.055$ off), so the derivative of a bad approximation is only moderately less bad ($4\times 10^{-4}$ on $L_\nu$).

**Olver AD** at $|\nu|=25$ is already at $10^{-12}$, including $(25,10^{-6})$. The threshold `|v| > 25` places the seam exactly where Olver is a better kernel than quadrature. Moving the threshold only moves the seam. Differentiating the selected kernel, or preferring Olver in its overlap, is the actual fix.

**Small-$z$ leading term** at $(25,10^{-6})$: $L_\nu=\psi(\nu)+\log(2/z)$, $L_{\nu\nu}=\psi_1(\nu)$ match mpmath to float64. The branch is not taken because the test is `z < 1e-6`, so $z=10^{-6}$ falls through to quadrature.

**Centered whole-line moments** in $y=\sqrt{\kappa}(u-u_0)$: essentially exact at every audit point, including the two that break kernel AD of quadrature and the Hankel FD. A first prototype that integrated in raw $u$ with an $O(1)$ window failed at large $z$ (peak of width $1/\sqrt{\kappa}$ lost in $[-1,1]$). That is the audit’s “integrate in $y=\sqrt{\kappa}(u-u_0)$” point, and it is necessary.

**Gauss–Hermite** on the same centered exponent is at float64 on the concentrated cases and $\sim 10^{-10}$ at $(0,1)$ (the density is not Gaussian). Fine as a large-$\kappa$ option, not as the only quadrature.

**Takekawa-style moments on the current 64-node half-line rule** match kernel AD of quadrature: good when the primal is resolved, $1.6\times 10^{-2}$ on $L_{\nu\nu}$ at $(25,10^{-6})$. Differentiating the integrand does not repair an under-resolved integral. Recenter and adapt the window.

**Failure of naive Laplace moments:** extra point $(0,10^{-4})$, $L_{\nu\nu}$ error $1.4\times 10^{-4}$. Both $\nu$ and $z$ small: Laplace width $1/\sqrt{z}$ is far larger than the true $\cosh$ cutoff $\sim\log(1/z)$. Kernel AD of the existing quadrature is fine there ($10^{-15}$). The audit already said a fixed number of local standard deviations is not universal.

GIG Hessian vs the same oracles:

| GIG $(p,a,b)$ | $H_{11}$ now | nested AD of `log_kv` | kernel AD | moments |
|---|---:|---:|---:|---:|
| $(0.5,1,1)$ | $0.706155$ | $0.706158$ | $0.706159$ | $0.706159$ |
| $(25,1,1)$ | $-600.12$ | $-150.00$ | $0.040775$ | $0.040775$ |
| $(25,0.1,0.1)$ | $2.85\times 10^{5}$ | $7.14\times 10^{4}$ | $0.040809$ | $0.040810$ |
| $(1,10^{4},10^{4})$ | $0.01819$ | $0.00455$ | $1.000\times 10^{-4}$ | $1.000\times 10^{-4}$ |

Fixing only the custom JVP does **not** fix Newton: `_hessian_log_partition` has its own FD stencil.

## Hybrid that the numbers support

1. **Hankel region:** AD the Hankel series in scaled-log form (already `_hankel_log_kv`). Do not FD `log_kv`.
2. **Olver region, and its overlap down through $|\nu|=25$:** AD the Olver kernel. Accurate at the present seam and at $(25,10^{-6})$.
3. **Small $z$, $|\nu|\gtrsim 1/2$:** leading $\psi/\psi_1$ (and keep the value formula). Widen or close the `z < 1e-6` hole so $z=10^{-6}$ is not quadrature.
4. **Moderate $(v,z)$:** centered moments, or native AD of a *resolved* quadrature. Same 64-node Takekawa rule is enough for $(0,1)$ and $(0.5,1)$; it is not enough for large $\nu$ and tiny $z$.
5. **One bundle** $(L, L_\nu, L_z, L_{\nu\nu}, L_{zz}, L_{\nu z})$ shared by the JVP and the GIG triad. Positivity $L_{\nu\nu}>0$ as a diagnostic, not a clip.

Native C++/Rust is not needed to fix correctness. Takekawa’s `log_abs_deriv_bessel_k` is the uncentered half-line version of (4). BesselK.jl/Temme-with-AD-safe expansions is a different primal; not timed here.

## Figures

- `figures/method_errors_audit_points.png` — method errors at the seven points
- `figures/seam_v25_z1.png` — $L_\nu$ and $L_{\nu\nu}$ vs $\nu$ at $z=1$. Current nested AD spikes through the Olver threshold; kernel AD and moments stay on the mpmath anchors.

Rerun: `uv run --with mpmath python dev-notes/investigations/2026-09-05/prototype_bessel_derivatives.py`
