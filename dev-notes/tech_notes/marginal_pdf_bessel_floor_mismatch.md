# The `sqrt(ab)`-vs-ratio floor mismatch in VG / NInvG marginal `log_prob`

> Companion: [`vg_em_inverse_moment_singularity.md`](vg_em_inverse_moment_singularity.md)
> covers the sibling degeneracy in the EM E-step conditional moments; this note
> covers the same $a\to0$ / $b\to0$ boundary showing up in the **marginal density
> formula** itself.
> Code: `normix/distributions/{variance_gamma,normal_inverse_gamma}.py`.
> Reported via: `docs/tutorials/core/02_gh_family_tour.md` (NInvG density plotted
> up to `1e53`).

## Problem

`NormalInverseGamma.pdf` (and `VarianceGamma.pdf`) could be wrong by tens of
orders of magnitude at exactly the parameter/observation combinations that are
most common in practice: **NInvG with no skew** (`gamma=0`, the usual symmetric
case) was wrong *everywhere*, not just at one point — every `x` gave a density
inflated by ~$10^{53}$. VG was wrong only exactly at `x=mu`, spiking to ~$10^{10}$
instead of the true finite value.

Both marginals compute their normalising integral over the mixing variable $Y$
via the general GIG identity

$$
\int_0^\infty y^{p-1}\exp\Big(-\tfrac12\big(ay+b/y\big)\Big)\,dy
= 2\Big(\frac ba\Big)^{p/2} K_p\!\big(\sqrt{ab}\big),
$$

evaluated at the **posterior** GIG parameters implied by $x$ (not the prior
subordinator, which NInvG/VG evaluate with their own closed-form
Gamma/InverseGamma constant — see `_from_init_params`-adjacent `log_C` in each
file). For NInvG the posterior $a = \gamma^\top\Lambda\gamma$; for VG the role
of "$b$" is played by $q(x)=(x-\mu)^\top\Lambda(x-\mu)$. Both can be **exactly
zero**: $a=0$ whenever $\gamma=0$ (for *every* $x$, since $a$ doesn't depend on
$x$), and $q=0$ exactly at $x=\mu$.

## Root cause

The old code computed the unfloored square root but a floored ratio:

```python
sqrt_ab = jnp.sqrt(a_gig * b_gig)              # a_gig = 0 exactly -> sqrt_ab = 0
log_bessel = log_kv(p_gig, sqrt_ab)
log_integral = (jnp.log(2.0)
                + 0.5 * p_gig * jnp.log((b_gig + LOG_EPS) / (a_gig + LOG_EPS))
                + log_bessel)
```

`log_kv` internally floors its argument at `jnp.finfo(float64).tiny` ($\approx
2.2\times10^{-308}$), so `log_bessel` became a **constant independent of
`b_gig`** — it lost its $x$-dependence entirely. Meanwhile the ratio term used
`a_gig + LOG_EPS` with `LOG_EPS = 1e-30`. These are two *different* effective
floors on `a`, so the two terms — which must cancel almost exactly for the
$a\to0$ limit to be finite — no longer cancel. The residual mismatch is
$\sim(\text{floor exponent difference})\times|p|\times\ln 10$, tens to hundreds
of nats, i.e. tens of orders of magnitude in the density.

### Why the naive fix ("just floor `sqrt_ab` too") is not quite enough

`GeneralizedInverseGaussian._log_partition_from_theta` already floors the
Bessel argument directly (`sqrt_ab_safe = jnp.maximum(sqrt_ab, LOG_EPS)`), but
only to keep `jax.grad` finite on the *unselected* branch of a `jnp.where` — the
selected value in the degenerate regime comes from an exact analytic
Gamma/InverseGamma branch, not from `psi_bessel`. If NInvG/VG instead floored
only `sqrt_ab` (not `a`/`b` individually) while leaving the ratio's `a`/`b`
unfloored, the two terms would *still* use different effective floors and the
bug would resurface in a different guise.

## Fix

Floor `a` and `b` **before** forming `sqrt(ab)`, using the same floored values
in the ratio, so both terms are functions of the identical floored inputs and
the small-$z$ cancellation goes through as designed:

```python
a_eff = a_gig + LOG_EPS
b_eff = b_gig + LOG_EPS
sqrt_ab = jnp.sqrt(a_eff * b_eff)
log_bessel = log_kv(p_gig, sqrt_ab)
log_integral = jnp.log(2.0) + 0.5 * p_gig * jnp.log(b_eff / a_eff) + log_bessel
```

(VG's formula is the mirror image, with `q` — playing the role of the side that
vanishes — and `2c` in place of `a_gig`, `b_gig`.)

This is a strict generalisation, not a special case for exactly-zero inputs:
checked against a `scipy.special.kve`-based reference across
$a\in\{0,10^{-25},\dots,10^{-2},1\}$ (fixed $b,p$), the fixed formula matches to
$\sim10^{-8}$–$10^{-15}$ absolute error uniformly, including the interior region
where the old (unfixed) formula was already correct. It also fixes a latent
`NaN`-gradient hazard: `d(sqrt(a_gig*b))/d(a_gig) \to \infty` as `a_gig -> 0+`
in the old code, which combined with `d(a_gig)/d(gamma) = 2\Lambda\gamma \to 0`
at `gamma=0` would `0 * inf = NaN` under `jax.grad`; flooring `a_gig` before the
square root keeps that derivative finite everywhere.

Applied in four call sites (joint + `Factor` variant, for each family):

| File | Class | Degenerate variable |
|---|---|---|
| `normal_inverse_gamma.py` | `NormalInverseGamma.log_prob` | `a_gig = gamma^T Lambda gamma` |
| `normal_inverse_gamma.py` | `FactorNormalInverseGamma.log_prob` | `a_gig` (Woodbury form) |
| `variance_gamma.py` | `VarianceGamma.log_prob` | `q = (x-mu)^T Lambda (x-mu)` |
| `variance_gamma.py` | `FactorVarianceGamma.log_prob` | `q` (Woodbury form) |

`NormalInverseGaussian` and `GeneralizedHyperbolic` were **not** touched: their
GIG parameters (`a=lam/mu_ig^2`, `b=lam` for NIG; `a`, `b` directly for GH) are
strictly positive by construction and never sit at this boundary, so the
existing unfloored `sqrt(a*b)` there never triggers the mismatch.

## Verification

- **NInvG, `gamma=0`** is exactly a scaled Student-$t$: $X\sim
  t_{2\alpha}\big(0,\;\sigma^2\beta/\alpha\big)$. Matches `scipy.stats.t` to
  $\sim10^{-15}$ after the fix (was off by $\sim10^{53}$ before).
- **NInvG, multivariate, `gamma=0`** matches `scipy.stats.multivariate_t` with
  scale $(\beta/\alpha)\Sigma$, df $=2\alpha$, to $\sim10^{-15}$.
- **VG at `x=mu`** ($\nu=\alpha-d/2>0$): closed form $\log f(\mu) = \log C +
  \log\Gamma(\nu) - \log 2 - \nu\log c$ (derived by the same small-$z$
  asymptotic, independent of `log_kv`), matches to $\sim10^{-15}$.
- Regression tests: `tests/test_normal_inverse_gamma.py::TestNormalInverseGammaEdgeCases`
  (`test_symmetric_matches_student_t`, `test_symmetric_multivariate_matches_multivariate_t`),
  `tests/test_variance_gamma.py::TestVarianceGammaEdgeCases`
  (`test_mode_density_matches_analytic_limit`, `test_mode_density_continuous`),
  `tests/test_factor_mixture.py` (`test_factor_ninvg_symmetric_matches_full_cov_at_gamma_zero`,
  `test_factor_vg_matches_full_cov_at_mode`).

Pre-existing tests (`TestNormalInverseGammaEdgeCases::test_symmetric`,
`TestVarianceGammaEdgeCases::test_symmetric_case`) already exercised exactly
this configuration but only asserted `isfinite`, which the bug satisfied — a
reminder that finiteness checks don't catch magnitude bugs.

## References

- `docs/theory/gh.md` for the GIG normalising-integral identity and the
  marginal density derivation.
- [`vg_em_inverse_moment_singularity.md`](vg_em_inverse_moment_singularity.md) —
  the sibling $q(x)\to0$ degeneracy in the EM conditional moments (a different
  code path: `GIG.expectation_params()` via `_posterior_gig_params`, not
  `log_kv` called directly from `log_prob`).
