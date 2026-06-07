# VarianceGamma EM diverges to NaN on heavy-tailed data

**Date:** 2026-05-31
**Repro:** `variance_gamma_em_nan.py` (marimo) — run with
`uv run marimo edit dev-notes/investigations/variance_gamma_em_nan.py`.
**Full analysis & fix:** the math (VG vs GH E-step/M-step, exact thresholds, the
`b_post` floor) is in
[`../tech_notes/vg_em_inverse_moment_singularity.md`](../tech_notes/vg_em_inverse_moment_singularity.md).
**Status: fixed** — the E-step now floors `b_post` at `B_POST_FLOOR` (see Mitigations).

## Problem statement

The docs-refactor plan dropped `VarianceGamma` from the `finance/01` index-model
comparison with the note *"its light tails diverge to `nan` on the kurtosis≈20
series."* That explanation is incorrect. This note records the real mechanism.

## What actually happens

Fitting `VarianceGamma` by EM to an equal-weighted S&P 500 index proxy
(n = 1276 train, excess kurtosis ≈ 20):

- The marginal log-likelihood **improves monotonically** for ~16 iterations
  (3.14 → 3.21) and is essentially flat by iteration 16.
- The EM keeps driving the Gamma subordinator shape **α downward**: it crosses
  1.0 near iteration 6 and continues to ≈ 0.69.
- At **iteration 19 the covariance σ becomes `nan`** while α is still finite
  (≈ 0.69), so the covariance M-step is the proximate failure. The next
  iteration is all `nan`.
- This is independent of `regularization` (`none`, `det_sigma_one`) and of
  `m_step_backend` (`cpu`, `jax`).

## Root cause

The VG subordinator is $Y \sim \mathrm{Gamma}(\alpha, \beta)$, whose inverse
moment

$$
E[1/Y] = \frac{\beta}{\alpha - 1}
$$

is finite **only for α > 1** (and $+\infty$ for α ≤ 1). The EM covariance M-step
weights observations by the posterior inverse-variance $E[1/Y \mid x]$. Once the
EM pushes α below 1, the mixing density concentrates mass at $Y \approx 0$,
those weights grow without bound, and the weighted covariance overflows to
`nan`. The marginal moments (`mean`, `cov`) stay finite because they use
$E[Y]=\alpha/\beta$ and $\mathrm{Var}[Y]$, which are finite for all α > 0 — which
is why `cov()` is fine right up to the iteration that blows up.

So this is **not** a "light tails" problem. If anything α < 1 makes the VG
*more* peaked/heavy near the mode. NIG and GH fit the same series without trouble
and reach a comparable log-likelihood (≈ 3.18–3.21); their subordinators
(Inverse Gaussian / GIG) do not have the same α ≤ 1 inverse-moment singularity.

> **Precise mechanism (see the tech note).** The failing object is the
> *posterior* $E[1/Y\mid x]$ from the GIG E-step, not the prior moment
> $\beta/(\alpha-1)$ quoted above. Its divergence threshold is
> $\alpha \le d/2 + 1$ (i.e. $\alpha \le 1.5$ for $d=1$), reached before the
> density itself becomes singular at $\alpha \le d/2$. The root cause is that VG
> is the GH limit $b\to 0$: the posterior carries $b_{\text{post}} = q(x) =
> (x-\mu)^\top\Sigma^{-1}(x-\mu)$, which $\to 0$ at the mode, whereas GH / NIG /
> NInvG keep $b_{\text{post}} = b + q(x) \ge b > 0$.

## Mitigations

- **Practical (today):** stop earlier — the likelihood is flat by iteration ~16,
  so a tighter `tol` or smaller `max_iter` returns the good finite iterate. Or
  use NIG/GH on strongly heavy-tailed data.
- **Library fix (implemented):** the shared E-step floors the posterior GIG
  scale $b_{\text{post}} = b + q(x)$ at `B_POST_FLOOR` $= 10^{-6}$
  (`normix/utils/constants.py`), which bounds $E[1/Y\mid x]$ near the mode. This
  is applied uniformly for all families; it only binds for VG (prior $b=0$) and
  is dormant for GH / NIG / NInvG (prior $b>0$). Rationale, the $\omega$-vs-$b$
  trade-off, and why the classical single-E/single-M structure is kept (no extra
  E-step) are in the tech note.

## Impact on docs

`finance/01` keeps VG out of the index comparison (Normal vs NIG vs GH), but the
reason is the α < 1 inverse-moment instability above, not "light tails". The
plan note has been corrected accordingly.
