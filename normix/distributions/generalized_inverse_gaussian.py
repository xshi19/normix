"""
Generalized Inverse Gaussian (GIG) distribution as an exponential family.

.. math::

    f(x \\mid p, a, b) = \\frac{(a/b)^{p/2}}{2 K_p(\\sqrt{ab})}
    \\, x^{p-1} \\exp\\!\\left(-\\frac{ax + b/x}{2}\\right), \\quad x > 0

**Exponential family structure:**

.. math::

    h(x) = 1, \\quad t(x) = [\\log x,\\; 1/x,\\; x]

.. math::

    \\theta = [p-1,\\; -b/2,\\; -a/2], \\quad \\theta_2 \\le 0,\\; \\theta_3 \\le 0

.. math::

    \\psi(\\theta) = \\log 2 + \\log K_p(\\sqrt{ab}) + \\tfrac{p}{2}\\log(b/a),
    \\quad p = \\theta_1+1,\\; a = -2\\theta_3,\\; b = -2\\theta_2

.. math::

    \\eta = [E[\\log X],\\; E[1/X],\\; E[X]]

**Special cases:**

- :math:`b \\to 0,\\; p > 0`: GIG → :math:`\\mathrm{Gamma}(p,\\; a/2)`
- :math:`a \\to 0,\\; p < 0`: GIG → :math:`\\mathrm{InvGamma}(-p,\\; b/2)`
- :math:`p = -1/2`: GIG → InverseGaussian

**η→θ rescaling** (reduces Fisher condition number):

.. math::

    s = \\sqrt{\\eta_2/\\eta_3}, \\quad
    \\tilde{\\eta} = \\bigl(\\eta_1 + \\tfrac{1}{2}\\log(\\eta_2/\\eta_3),\\;
    \\sqrt{\\eta_2\\eta_3},\\; \\sqrt{\\eta_2\\eta_3}\\bigr)

Solve :math:`\\tilde{\\eta} \\to \\tilde{\\theta}` with symmetric GIG
(:math:`\\tilde{a} = \\tilde{b}`), then unscale.

**Log-Partition Triad Overrides:**

- ``_log_partition_from_theta`` : JAX, uses ``log_kv(backend='jax')``
- ``_grad_log_partition``       : affine image of one ``log_kv_moments`` call
- ``_hessian_log_partition``    : :math:`H = D\\,\\mathrm{cov}\\,D` from the same bundle
- ``_log_partition_cpu``        : numpy + ``log_kv(backend='cpu')``
- ``_grad_log_partition_cpu`` : same affine image, NumPy backend
- ``_hessian_log_partition_cpu``: same covariance, NumPy backend
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from normix.utils.bessel import log_kv, log_kv_moments
from normix.utils.rvs import QuantileTable, build_pinv_table
from normix.exponential_family import ExponentialFamily
from normix.utils.constants import (
    LOG_EPS, TINY, GIG_DEGEN_THRESHOLD, BESSEL_QUAD_LOG_DROP,
    BESSEL_WINDOW_ITERS, BESSEL_WINDOW_HI_MAX,
    THETA_FLOOR, GIG_THETA_PERTURB,
)
from normix.fitting.solvers import (
    solve_bregman, solve_bregman_multistart,
    make_jit_newton_solver,
)


def _gig_degeneracy_flags(p, a, b, xp):
    r"""Small-:math:`z` GIG limit predicate and off-domain flag.

    The one-term Gamma / InverseGamma form is used iff

    .. math::

        z < z_{\max} \quad\text{and}\quad |p|\log(2/z) > C,

    with :math:`z_{\max}=` ``GIG_DEGEN_THRESHOLD`` (bounds the
    :math:`O(z^2)` remainder) and :math:`C=` ``BESSEL_QUAD_LOG_DROP``
    (bounds the :math:`\rho_{|p|}` truncation). Form is
    :math:`\operatorname{sign}(p)`. Combinations outside :math:`\Theta`
    (:math:`b=0,\,p\le 0`; :math:`a=0,\,p\ge 0`) set ``off_theta``.
    """
    p = xp.asarray(p, dtype=xp.float64)
    a = xp.maximum(xp.asarray(a, dtype=xp.float64), 0.0)
    b = xp.maximum(xp.asarray(b, dtype=xp.float64), 0.0)
    z = xp.sqrt(a * b)
    inf = xp.asarray(np.inf, dtype=xp.float64)
    tiny = xp.asarray(np.finfo(np.float64).tiny, dtype=xp.float64)
    log_2_over_z = xp.where(z > 0.0, xp.log(2.0 / xp.maximum(z, tiny)), inf)
    # np.where evaluates both arms: avoid 0 * +∞ → NaN when p=0, z=0.
    safe_log = xp.where(p == 0.0, xp.zeros_like(log_2_over_z), log_2_over_z)
    nats = xp.abs(p) * safe_log
    use_limit = (z < GIG_DEGEN_THRESHOLD) & (nats > BESSEL_QUAD_LOG_DROP)
    off_theta = ((b <= 0.0) & (p <= 0.0)) | ((a <= 0.0) & (p >= 0.0))
    use_gamma = use_limit & (p > 0.0) & (a > 0.0)
    use_invg = use_limit & (p < 0.0) & (b > 0.0)
    return use_gamma, use_invg, off_theta, z


# ---------------------------------------------------------------------------
# Random variate generation helpers (Devroye TDR + PINV wrappers)
# ---------------------------------------------------------------------------
#
# Two GIG sampling methods, neither requiring Bessel evaluation:
#
# 1. ``_gig_rvs_devroye`` — Devroye TDR on :math:`u = w - w_0` in
#    :math:`(p,z,s)` coordinates. Envelope tangents sit at the
#    :math:`e^{-1}` level of :math:`\psi`; ``lax.while_loop`` redraws
#    unaccepted columns (acceptance :math:`\ge e^{-1}` uniformly).
#
# 2. ``build_pinv_table`` + ``rvs_pinv`` — Numerical inverse CDF via
#    :func:`normix.utils.rvs.build_pinv_table` seeded at
#    ``_gig_log_mode(p-1, a, b)``.

_GIG_RVS_TINY64 = jnp.finfo(jnp.float64).tiny
_GIG_TDR_DOUBLE_ITERS = 8
_GIG_TDR_MAX_ROUNDS = 256


def _gig_log_mode(p, a, b) -> jax.Array:
    r"""Log-mode of :math:`W=\log X`, or of :math:`X` when called with :math:`p-1`.

    Interior (:math:`a>0`, :math:`b>0`):

    .. math::

        w_0 = s + \operatorname{asinh}(p/z),
        \qquad z=\sqrt{ab},\; s=\tfrac12\log(b/a).

    At an exact boundary the asinh form is :math:`\infty-\infty`; the
    rationalized :math:`x`-space pair is used instead:

    .. math::

        e^{w_0}
        = \begin{cases}
            (|p|+r)/a & p\ge 0 \\
            b/(|p|+r) & p<0
          \end{cases},
        \qquad r=\sqrt{p^2+ab}.

    For the density of :math:`X` pass :math:`p-1`; for the log-density
    of :math:`W` pass :math:`p`.
    """
    tiny = jnp.asarray(np.finfo(np.float64).tiny)
    p = jnp.asarray(p, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    z = jnp.sqrt(jnp.maximum(a, 0.0) * jnp.maximum(b, 0.0))
    interior = (a > 0.0) & (b > 0.0)
    a_pos = jnp.maximum(a, tiny)
    b_pos = jnp.maximum(b, tiny)
    s = 0.5 * (jnp.log(b_pos) - jnp.log(a_pos))
    w_asinh = s + jnp.arcsinh(p / jnp.maximum(z, tiny))
    r = jnp.hypot(p, z)
    abs_p = jnp.abs(p)
    x0 = jnp.where(
        p >= 0.0,
        (abs_p + r) / jnp.maximum(a, tiny),
        b / jnp.maximum(abs_p + r, tiny),
    )
    w_rat = jnp.log(jnp.maximum(x0, tiny))
    return jnp.where(interior, w_asinh, w_rat)


def _gig_psi(u, p, c_R, c_L):
    r""":math:`\psi(u)=p u - \tfrac{c_R}{2}\operatorname{expm1}(u)
    - \tfrac{c_L}{2}\operatorname{expm1}(-u)`."""
    return p * u - 0.5 * c_R * jnp.expm1(u) - 0.5 * c_L * jnp.expm1(-u)


def _gig_dpsi(u, p, c_R, c_L):
    return p - 0.5 * c_R * jnp.exp(u) + 0.5 * c_L * jnp.exp(-u)


def _gig_tdr_root(p, c_R, c_L, side: float) -> jax.Array:
    r"""Positive root of :math:`\psi(\mathrm{side}\cdot s)=-1` via doubling then bisection."""
    tiny = jnp.asarray(np.finfo(np.float64).tiny)
    hi_max = jnp.asarray(BESSEL_WINDOW_HI_MAX, dtype=jnp.float64)
    coeff = jnp.where(side > 0.0, 0.5 * c_R, 0.5 * c_L)
    hi = jnp.minimum(
        jnp.maximum(jnp.log1p(1.0 / jnp.maximum(coeff, tiny)), 1.0),
        hi_max,
    )

    def psi_at(x):
        return _gig_psi(side * x, p, c_R, c_L)

    def double_body(_i, hi):
        return jnp.minimum(jnp.where(psi_at(hi) > -1.0, hi * 2.0, hi), hi_max)

    hi = jax.lax.fori_loop(0, _GIG_TDR_DOUBLE_ITERS, double_body, hi)
    lo = jnp.zeros((), dtype=jnp.float64)

    def bisect_body(_i, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        wider = psi_at(mid) > -1.0
        return jnp.where(wider, mid, lo), jnp.where(wider, hi, mid)

    _, hi = jax.lax.fori_loop(0, BESSEL_WINDOW_ITERS, bisect_body, (lo, hi))
    return hi


def _gig_tdr_setup(p, a, b):
    r"""Devroye :math:`e^{-1}` TDR envelope in :math:`(p,z,s)` coordinates."""
    tiny = jnp.asarray(np.finfo(np.float64).tiny)
    z = jnp.maximum(jnp.sqrt(a) * jnp.sqrt(b), tiny)
    w0 = _gig_log_mode(p, a, b)
    r = jnp.hypot(p, z)
    abs_p = jnp.abs(p)
    A = r + abs_p
    B = z * (z / jnp.maximum(A, tiny))
    c_R = jnp.where(p >= 0.0, A, B)
    c_L = jnp.where(p >= 0.0, B, A)
    t = _gig_tdr_root(p, c_R, c_L, 1.0)
    sm = _gig_tdr_root(p, c_R, c_L, -1.0)
    zeta = jnp.maximum(-_gig_dpsi(t, p, c_R, c_L), tiny)
    xi = jnp.maximum(_gig_dpsi(-sm, p, c_R, c_L), tiny)
    tp = t - 1.0 / zeta
    sp = sm - 1.0 / xi
    return dict(
        w0=w0, p=p, c_R=c_R, c_L=c_L,
        t=t, s=sm, zeta=zeta, xi=xi, tp=tp, sp=sp, area=t + sm,
    )


def _gig_tdr_propose(key: jax.Array, env: dict, n: int):
    """One envelope proposal of shape ``(n,)``. Returns ``(u, accept)``."""
    k1, k2, k3 = jax.random.split(key, 3)
    U = jax.random.uniform(k1, (n,), dtype=jnp.float64)
    V = jnp.maximum(
        jax.random.uniform(k2, (n,), dtype=jnp.float64), _GIG_RVS_TINY64,
    )
    W = jnp.maximum(
        jax.random.uniform(k3, (n,), dtype=jnp.float64), _GIG_RVS_TINY64,
    )
    tp, sp, zeta, xi, area = (
        env["tp"], env["sp"], env["zeta"], env["xi"], env["area"],
    )
    inv_xi = 1.0 / xi
    width = tp + sp
    pL = inv_xi / area
    pM = width / area
    left = U < pL
    mid = (U >= pL) & (U < pL + pM)
    u = jnp.where(
        left, -sp + jnp.log(V) / xi,
        jnp.where(mid, -sp + V * width, tp - jnp.log(V) / zeta),
    )
    h = jnp.where(
        left, xi * (u + sp),
        jnp.where(mid, 0.0, -zeta * (u - tp)),
    )
    accept = jnp.log(W) <= _gig_psi(u, env["p"], env["c_R"], env["c_L"]) - h
    return u, accept


def _gig_rvs_boundary(key: jax.Array, p, a, b, n: int) -> jax.Array:
    r"""Exact :math:`a=0` or :math:`b=0`: Gamma, InverseGamma, or off-:math:`\Theta` NaN.

    TDR is defined for :math:`a,b>0`. The small-:math:`z` interior limit
    is still sampled by TDR; this path is only the exact boundary.
    """
    alpha_g = jnp.maximum(p, LOG_EPS)
    beta_g = jnp.maximum(a / 2.0, LOG_EPS)
    alpha_ig = jnp.maximum(-p, LOG_EPS)
    beta_ig = jnp.maximum(b / 2.0, LOG_EPS)
    g = jax.random.gamma(key, alpha_g, shape=(n,), dtype=jnp.float64)
    x_gamma = g / beta_g
    g_ig = jax.random.gamma(key, alpha_ig, shape=(n,), dtype=jnp.float64)
    x_invg = beta_ig / g_ig
    use_gamma = (b <= 0.0) & (p > 0.0) & (a > 0.0)
    use_invg = (a <= 0.0) & (p < 0.0) & (b > 0.0)
    nan = jnp.full((n,), jnp.nan, dtype=jnp.float64)
    return jnp.where(use_gamma, x_gamma, jnp.where(use_invg, x_invg, nan))


def _gig_rvs_tdr(key: jax.Array, p, a, b, n: int) -> jax.Array:
    r"""TDR on :math:`w=\log x` for :math:`a,b>0`.

    Envelope tangents at the :math:`e^{-1}` level of the centred log-density
    :math:`\psi`; :func:`jax.lax.while_loop` redraws only unaccepted columns.
    A column that exhausts ``_GIG_TDR_MAX_ROUNDS`` is NaN, never a reject.
    """
    env = _gig_tdr_setup(p, a, b)

    def cond(state):
        _key, done, _u, i = state
        return (~done.all()) & (i < _GIG_TDR_MAX_ROUNDS)

    def body(state):
        key, done, u_out, i = state
        key, sub = jax.random.split(key)
        u, accept = _gig_tdr_propose(sub, env, n)
        u_out = jnp.where(done, u_out, jnp.where(accept, u, u_out))
        return key, done | accept, u_out, i + 1

    u0 = jnp.zeros((n,), dtype=jnp.float64)
    done0 = jnp.zeros((n,), dtype=bool)
    i0 = jnp.asarray(0, dtype=jnp.int32)
    _, done, u, _ = jax.lax.while_loop(cond, body, (key, done0, u0, i0))
    u = jnp.where(done, u, jnp.nan)
    return jnp.exp(env["w0"] + u)


def _gig_rvs_devroye(key: jax.Array, p, a, b, n: int) -> jax.Array:
    r"""Sample *n* :math:`\mathrm{GIG}(p, a, b)` variates via TDR on :math:`w=\log x`.

    Interior :math:`a,b>0` uses Devroye TDR (acceptance at least
    :math:`e^{-1}`). Exact :math:`a=0` or :math:`b=0` dispatches to
    :math:`\mathrm{Gamma}(p,a/2)` / :math:`\mathrm{InvGamma}(-p,b/2)`
    (or NaN off :math:`\Theta`).
    """
    p = jnp.asarray(p, dtype=jnp.float64)
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    boundary = (a <= 0.0) | (b <= 0.0)
    return jax.lax.cond(
        boundary,
        lambda _: _gig_rvs_boundary(key, p, a, b, n),
        lambda _: _gig_rvs_tdr(key, p, a, b, n),
        None,
    )


class GeneralizedInverseGaussian(ExponentialFamily):
    r"""
    Generalized Inverse Gaussian distribution.

    Stored: :math:`p` (shape, any real), :math:`a > 0`, :math:`b > 0`.
    """

    p: jax.Array
    a: jax.Array
    b: jax.Array

    def __init__(self, p, a, b):
        self.p = jnp.asarray(p, dtype=jnp.float64)
        self.a = jnp.asarray(a, dtype=jnp.float64)
        self.b = jnp.asarray(b, dtype=jnp.float64)

    def to_gig(self, *, boundary_eps: float = 0.0) -> "GeneralizedInverseGaussian":
        r"""Identity embedding into the GIG family.

        GIG is already in GIG coordinates, so this returns ``self``. It exists
        so that every subordinator family exposes a uniform ``to_gig()`` for
        the shared prior-to-posterior conjugacy map in the EM E-step (see
        :meth:`~normix.mixtures.joint.JointNormalMixture._posterior_gig_params`).
        ``boundary_eps`` is accepted for API uniformity and ignored.
        """
        del boundary_eps
        return self

    def _divergence_eta(self):
        r"""Degenerate-aware GIG split mirroring the :math:`\psi` branches.

        At the Gamma (:math:`b\to 0`) or InverseGamma (:math:`a\to 0`) boundary
        the possibly-infinite moment uses the closed-form subordinator split;
        in the interior all moments are finite Bessel ratios.
        """
        p, a, b = self.p, self.a, self.b
        use_gamma, use_invg, _, _ = _gig_degeneracy_flags(p, a, b, jnp)

        alpha_g = jnp.maximum(p, LOG_EPS)
        beta_g = jnp.maximum(a / 2.0, LOG_EPS)
        E_log_g = jax.scipy.special.digamma(alpha_g) - jnp.log(beta_g)
        E_X_g = alpha_g / beta_g
        m_g = jnp.where(alpha_g > 1.0, beta_g / (alpha_g - 1.0), jnp.inf)
        eta_g = jnp.array([E_log_g, 0.0, E_X_g])
        v_g = jnp.array([0.0, 1.0, 0.0])

        alpha_ig = jnp.maximum(-p, LOG_EPS)
        beta_ig = jnp.maximum(b / 2.0, LOG_EPS)
        E_log_ig = jnp.log(beta_ig) - jax.scipy.special.digamma(alpha_ig)
        E_inv_ig = alpha_ig / beta_ig
        m_ig = jnp.where(alpha_ig > 1.0, beta_ig / (alpha_ig - 1.0), jnp.inf)
        eta_ig = jnp.array([E_log_ig, E_inv_ig, 0.0])
        v_ig = jnp.array([0.0, 0.0, 1.0])

        eta_int = type(self)._grad_log_partition(self.natural_params())
        m_int = jnp.zeros((), dtype=eta_int.dtype)
        v_int = jnp.zeros_like(eta_int)

        eta = jnp.where(use_gamma, eta_g, jnp.where(use_invg, eta_ig, eta_int))
        m = jnp.where(use_gamma, m_g, jnp.where(use_invg, m_ig, m_int))
        v = jnp.where(use_gamma, v_g, jnp.where(use_invg, v_ig, v_int))
        return eta, m, v

    # ------------------------------------------------------------------
    # Tier 1: Exponential family interface
    # ------------------------------------------------------------------

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        r"""
        :math:`\psi(\theta) = \log 2 + \log K_p(\sqrt{ab}) + (p/2)(\log b - \log a)`.

        Small-:math:`z` limit (see ``_gig_degeneracy_flags``): Gamma
        if :math:`p>0`, InverseGamma if :math:`p<0`. Otherwise the Bessel
        kernel on ``(a_safe, b_safe)`` as in :meth:`_grad_log_partition`.
        Off :math:`\Theta` returns :math:`+\infty`. Clamps on the Gamma /
        InverseGamma branches are unselected-branch guards.
        """
        p = theta[0] + 1.0
        b = jnp.maximum(-2.0 * theta[1], 0.0)
        a = jnp.maximum(-2.0 * theta[2], 0.0)
        use_gamma, use_invg, off_theta, _ = _gig_degeneracy_flags(p, a, b, jnp)

        _, _, _, z_safe, log_sqrt_ba = (
            GeneralizedInverseGaussian._unpack_safe(theta, jnp, LOG_EPS)
        )
        psi_bessel = jnp.log(2.0) + log_kv(p, z_safe) + p * log_sqrt_ba

        alpha_g = jnp.maximum(p, LOG_EPS)
        beta_g = jnp.maximum(a / 2.0, LOG_EPS)
        psi_gamma = (jax.scipy.special.gammaln(alpha_g)
                     - alpha_g * jnp.log(beta_g))

        alpha_ig = jnp.maximum(-p, LOG_EPS)
        beta_ig = jnp.maximum(b / 2.0, LOG_EPS)
        psi_invgamma = (jax.scipy.special.gammaln(alpha_ig)
                        - alpha_ig * jnp.log(beta_ig))

        psi = jnp.where(
            use_gamma, psi_gamma,
            jnp.where(use_invg, psi_invgamma, psi_bessel),
        )
        return jnp.where(off_theta, jnp.inf, psi)

    def natural_params(self) -> jax.Array:
        return jnp.array([self.p - 1.0, -self.b / 2.0, -self.a / 2.0])

    @staticmethod
    def sufficient_statistics(x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.array([jnp.log(x), 1.0 / x, x])

    @staticmethod
    def log_base_measure(x: jax.Array) -> jax.Array:
        return jnp.where(x > 0, jnp.zeros((), jnp.float64), -jnp.inf)

    # ------------------------------------------------------------------
    # Tier 2: η and Fisher as an affine image of one Bessel moment bundle
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_safe(theta, xp, floor: float):
        r"""``p, a_safe, b_safe, z=\sqrt{ab}, \tfrac12\log(b/a)``."""
        theta = xp.asarray(theta, dtype=xp.float64)
        p = theta[0] + 1.0
        b_safe = xp.maximum(-2.0 * theta[1], floor)
        a_safe = xp.maximum(-2.0 * theta[2], floor)
        tiny = xp.asarray(np.finfo(np.float64).tiny, dtype=xp.float64)
        z = xp.sqrt(xp.maximum(a_safe * b_safe, tiny))
        log_sqrt_ba = 0.5 * (xp.log(b_safe) - xp.log(a_safe))
        return p, a_safe, b_safe, z, log_sqrt_ba

    @staticmethod
    def _eta_H_from_moments(m, a_safe, b_safe, log_sqrt_ba, xp):
        r"""GIG :math:`\eta` and :math:`H=\mathrm{Cov}[t(X)]` from one bundle.

        With :math:`u = \log X - \tfrac12\log(b/a)` identified with the
        Bessel coordinate and :math:`s=(x,\mathrm{expm1}(-x),\mathrm{expm1}(x))`,

        .. math::

            \eta = \bigl(u_0+\tfrac12\log(b/a)+E[x],\;
            s_{-}(1+E[s_1]),\; s_{+}(1+E[s_2])\bigr),
            \qquad H = D\,\mathrm{cov}(s)\,D,

        where :math:`s_{\mp}=\exp(\tfrac12\log(a/b)\mp u_0)` and
        :math:`D=\mathrm{diag}(1,s_{-},s_{+})`. No Bessel ratios, no
        :math:`p\pm 1` evaluations.
        """
        u0 = m.u0
        scale_minus = xp.exp(0.5 * (xp.log(a_safe) - xp.log(b_safe)) - u0)
        scale_plus = xp.exp(log_sqrt_ba + u0)
        eta0 = u0 + log_sqrt_ba + m.mean[..., 0]
        eta1 = scale_minus * (1.0 + m.mean[..., 1])
        eta2 = scale_plus * (1.0 + m.mean[..., 2])
        eta = xp.stack([eta0, eta1, eta2], axis=-1)
        scales = xp.stack(
            [xp.ones_like(scale_minus), scale_minus, scale_plus], axis=-1,
        )
        H = m.cov * scales[..., :, None] * scales[..., None, :]
        return eta, H

    @classmethod
    def _grad_log_partition(cls, theta: jax.Array) -> jax.Array:
        r"""
        :math:`\nabla\psi(\theta) = [E[\log X],\; E[1/X],\; E[X]]` from
        one :func:`~normix.utils.bessel.log_kv_moments` call.

        Clamping :math:`a, b` to ``LOG_EPS`` keeps the Bessel kernel in its
        stated domain. The default ``jax.grad`` path fails because
        ``jnp.where`` evaluates all branches and
        :math:`\partial\sqrt{ab}/\partial a \to \infty` as :math:`a \to 0`.
        """
        p, a_safe, b_safe, z, log_sqrt_ba = cls._unpack_safe(
            theta, jnp, LOG_EPS,
        )
        m = log_kv_moments(p, z, backend='jax')
        eta, _ = cls._eta_H_from_moments(m, a_safe, b_safe, log_sqrt_ba, jnp)
        return eta

    @classmethod
    def _hessian_log_partition(cls, theta: jax.Array) -> jax.Array:
        r"""
        :math:`\nabla^2\psi(\theta)=\mathrm{Cov}[t(X)]`.

        One :func:`~normix.utils.bessel.log_kv_moments` call; the GIG
        sufficient statistic is an affine image of
        :math:`(x, e^{-x}-1, e^{x}-1)`. The dense matrix is PSD to
        rounding; its smallest eigenvalue has relative error
        :math:`O(\varepsilon z)` from residual cancellation in ``cov``.
        Valid in the non-degenerate regime
        (:math:`\sqrt{ab} \gg` ``GIG_DEGEN_THRESHOLD``). Newton applies
        relative Tikhonov ``HESSIAN_DAMPING * tr(H_θ)/n`` in θ-space
        before the bound sandwich.
        """
        p, a_safe, b_safe, z, log_sqrt_ba = cls._unpack_safe(
            theta, jnp, LOG_EPS,
        )
        m = log_kv_moments(p, z, backend='jax')
        _, H = cls._eta_H_from_moments(m, a_safe, b_safe, log_sqrt_ba, jnp)
        return H

    # ------------------------------------------------------------------
    # Tier 3: CPU overrides (numpy, same kernel)
    # ------------------------------------------------------------------

    @classmethod
    def _log_partition_cpu(cls, theta) -> float:
        r""":math:`\psi(\theta)` via numpy + ``log_kv(backend='cpu')``. Accepts numpy or JAX arrays."""
        theta = np.asarray(theta, dtype=np.float64)
        p = theta[0] + 1.0
        b = max(-2.0 * theta[1], 0.0)
        a = max(-2.0 * theta[2], 0.0)
        use_gamma, use_invg, off_theta, _ = _gig_degeneracy_flags(p, a, b, np)

        if bool(np.asarray(off_theta)):
            return float(np.inf)
        if bool(np.asarray(use_gamma)):
            from scipy.special import gammaln
            alpha = max(p, TINY)
            beta = max(a / 2.0, TINY)
            return float(gammaln(alpha) - alpha * np.log(beta))
        if bool(np.asarray(use_invg)):
            from scipy.special import gammaln
            alpha = max(-p, TINY)
            beta = max(b / 2.0, TINY)
            return float(gammaln(alpha) - alpha * np.log(beta))

        _, _, _, z_safe, log_sqrt_ba = cls._unpack_safe(theta, np, TINY)
        return float(
            np.log(2.0) + log_kv(p, z_safe, backend='cpu') + p * log_sqrt_ba
        )

    @classmethod
    def _grad_log_partition_cpu(cls, theta) -> np.ndarray:
        r""":math:`\nabla\psi(\theta)` via ``log_kv_moments(backend='cpu')``."""
        p, a_safe, b_safe, z, log_sqrt_ba = cls._unpack_safe(
            theta, np, TINY,
        )
        m = log_kv_moments(p, z, backend='cpu')
        eta, _ = cls._eta_H_from_moments(m, a_safe, b_safe, log_sqrt_ba, np)
        return np.asarray(eta, dtype=np.float64)

    @classmethod
    def _hessian_log_partition_cpu(cls, theta) -> np.ndarray:
        r""":math:`\nabla^2\psi(\theta)` via the same CPU moment bundle."""
        p, a_safe, b_safe, z, log_sqrt_ba = cls._unpack_safe(
            theta, np, TINY,
        )
        m = log_kv_moments(p, z, backend='cpu')
        _, H = cls._eta_H_from_moments(m, a_safe, b_safe, log_sqrt_ba, np)
        return np.asarray(H, dtype=np.float64)

    # ------------------------------------------------------------------
    # Batch CPU expectation parameters (used by JointNormalMixture E-step)
    # ------------------------------------------------------------------

    @staticmethod
    def expectation_params_batch(
        p, a, b, backend: str = 'jax'
    ) -> jax.Array:
        """
        Vectorized η for arrays of (p, a, b), each shape (N,).
        Returns (N, 3) array where columns are [E_log_X, E_inv_X, E_X].

        backend='jax' : vmap over scalar JAX grad
        backend='cpu' : one batched ``log_kv_moments`` call (NumPy kernel)
        """
        if backend == 'cpu':
            return GeneralizedInverseGaussian._expectation_params_batch_cpu(p, a, b)
        p = jnp.asarray(p, dtype=jnp.float64)
        a = jnp.asarray(a, dtype=jnp.float64)
        b = jnp.asarray(b, dtype=jnp.float64)

        def _single(pi, ai, bi):
            return GeneralizedInverseGaussian(p=pi, a=ai, b=bi).expectation_params()

        return jax.vmap(_single)(p, a, b)

    @staticmethod
    def _expectation_params_batch_cpu(p, a, b) -> jax.Array:
        """Vectorized CPU path — one ``log_kv_moments`` call on (N,) arrays."""
        p = np.asarray(p, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        a_safe = np.maximum(a, TINY)
        b_safe = np.maximum(b, TINY)
        tiny = np.finfo(np.float64).tiny
        z = np.sqrt(np.maximum(a_safe * b_safe, tiny))
        log_sqrt_ba = 0.5 * (np.log(b_safe) - np.log(a_safe))
        m = log_kv_moments(p, z, backend='cpu')
        eta, _ = GeneralizedInverseGaussian._eta_H_from_moments(
            m, a_safe, b_safe, log_sqrt_ba, np,
        )
        return jnp.asarray(eta, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Moments and sampling
    # ------------------------------------------------------------------

    def mean(self) -> jax.Array:
        r""":math:`E[X] = \eta_3` from expectation parameters."""
        return self.expectation_params()[2]

    def raw_moment(self, k: jax.Array) -> jax.Array:
        r"""Raw moment :math:`E[X^k] = (b/a)^{k/2}\,K_{p+k}(\sqrt{ab})/K_p(\sqrt{ab})`.

        Requires :math:`a, b > 0`. For the Gamma / InverseGamma boundary
        embeddings use those classes' own :meth:`raw_moment` instead.
        """
        k = jnp.asarray(k, dtype=jnp.float64)
        a = jnp.maximum(self.a, LOG_EPS)
        b = jnp.maximum(self.b, LOG_EPS)
        z = jnp.sqrt(a * b)
        return jnp.exp(
            0.5 * k * (jnp.log(b) - jnp.log(a))
            + log_kv(self.p + k, z)
            - log_kv(self.p, z)
        )

    def raw_moments(self, ks: jax.Array) -> jax.Array:
        r"""Vectorised :meth:`raw_moment` over orders ``ks`` (shared :math:`K_p`)."""
        ks = jnp.asarray(ks, dtype=jnp.float64)
        a = jnp.maximum(self.a, LOG_EPS)
        b = jnp.maximum(self.b, LOG_EPS)
        z = jnp.sqrt(a * b)
        log_kp = log_kv(self.p, z)
        log_k = jax.vmap(lambda k: log_kv(self.p + k, z))(ks)
        return jnp.exp(0.5 * ks * (jnp.log(b) - jnp.log(a)) + log_k - log_kp)

    def var(self) -> jax.Array:
        r""":math:`\mathrm{Var}[X] = E[X^2] - E[X]^2` via three Bessel evaluations.

        Uses the moment formula
        :math:`E[X^r] = (b/a)^{r/2}\,K_{p+r}(\sqrt{ab})/K_p(\sqrt{ab})`
        rather than the full 11-Bessel Fisher Hessian entry ``[2, 2]``.
        """
        m1, m2 = self.raw_moments(jnp.array([1.0, 2.0]))
        return m2 - m1 ** 2

    def mode(self) -> jax.Array:
        r"""Mode :math:`x^\star = \exp(w_0)` with :math:`w_0` from ``_gig_log_mode(p-1, a, b)``.

        Unique positive critical point of the log-density for every
        :math:`p` with :math:`a, b > 0`. The density vanishes
        super-exponentially at 0 when :math:`b > 0`; it does not diverge
        there for :math:`p < 1`. At :math:`b=0,\,p>1` this is the Gamma
        mode :math:`2(p-1)/a`; at :math:`a=0,\,p<0` the InverseGamma mode
        :math:`b/(2(1-p))`.
        """
        return jnp.exp(_gig_log_mode(self.p - 1.0, self.a, self.b))

    def cdf(self, x: jax.Array) -> jax.Array:
        r"""CDF :math:`F(x) = P(X \le x)`.

        Trapezoidal CDF on a :math:`w = \log x` grid built from
        :meth:`log_prob`; seeded at :math:`\log` :meth:`mode`.  In the
        small-:math:`z` Gamma / InverseGamma regimes (see
        ``_gig_degeneracy_flags``) delegates to the limiting CDF.

        JIT-compatible: the degeneracy test uses :func:`jax.lax.cond`
        (no host ``float()`` casts).
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        return self._cdf_or_ppf(x, inverse=False)

    def ppf(self, q: jax.Array) -> jax.Array:
        r"""Quantile (inverse CDF) :math:`F^{-1}(q)` via the PINV table.

        JIT-compatible: see :meth:`cdf`.
        """
        q = jnp.asarray(q, dtype=jnp.float64)
        return self._cdf_or_ppf(q, inverse=True)

    def quantile_table(self) -> QuantileTable:
        r"""Frozen PINV table (non-degenerate regime only).

        Degenerate Gamma / InverseGamma limits bypass the table in
        :meth:`cdf` / :meth:`ppf`; hold this object when evaluating many
        quantiles at fixed non-degenerate parameters.
        """
        log_kernel = lambda w: self.log_prob(jnp.exp(w)) + w
        u_grid, x_grid = build_pinv_table(
            log_kernel, _gig_log_mode(self.p - 1.0, self.a, self.b),
            x_of_w=jnp.exp,
        )
        return QuantileTable(u_grid=u_grid, x_grid=x_grid)

    def _cdf_or_ppf(self, z: jax.Array, *, inverse: bool) -> jax.Array:
        """Shared JIT-safe CDF / PPF with degenerate Gamma / InvGamma limits."""
        p, a, b = self.p, self.a, self.b
        use_gamma, use_invg, _, _ = _gig_degeneracy_flags(p, a, b, jnp)
        use_degen = use_gamma | use_invg

        def _degen(z_):
            from normix.distributions.gamma import Gamma
            from normix.distributions.inverse_gamma import InverseGamma
            g = Gamma(
                alpha=jnp.maximum(p, LOG_EPS),
                beta=jnp.maximum(a / 2.0, LOG_EPS),
            )
            ig = InverseGamma(
                alpha=jnp.maximum(-p, LOG_EPS),
                beta=jnp.maximum(b / 2.0, LOG_EPS),
            )
            if inverse:
                return jnp.where(use_gamma, g.ppf(z_), ig.ppf(z_))
            return jnp.where(use_gamma, g.cdf(z_), ig.cdf(z_))

        def _pinv(z_):
            table = self.quantile_table()
            if inverse:
                return table.ppf(z_)
            return table.cdf(z_)

        return jax.lax.cond(use_degen, _degen, _pinv, z)

    def rvs(
        self, n: int, seed: int = 42, method: str = "devroye",
    ) -> jax.Array:
        r"""Sample *n* observations from :math:`\mathrm{GIG}(p, a, b)`.

        Parameters
        ----------
        n : int
            Sample size.
        seed : int
            Integer seed for JAX PRNG (or ``scipy`` ``random_state`` for ``'scipy'``).
        method : str
            Sampling algorithm:

            * ``'devroye'`` (default) — Transformed density rejection (TDR) on
              :math:`\log(x)`, pure JAX, no Bessel functions.
            * ``'pinv'`` — Numerical inverse CDF (CPU table build + JAX sampling),
              no Bessel. Best for large *n* with fixed parameters.
            * ``'scipy'`` — ``scipy.stats.geninvgauss`` (CPU, original fallback).
        """
        if method == "devroye":
            key = jax.random.PRNGKey(seed)
            return _gig_rvs_devroye(key, self.p, self.a, self.b, n)

        if method == "pinv":
            return self.quantile_table().rvs(n, seed=seed)

        if method == "scipy":
            from scipy import stats
            b_sp = float(np.sqrt(float(self.a) * float(self.b)))
            scale = float(np.sqrt(float(self.b) / float(self.a)))
            samples = stats.geninvgauss.rvs(
                p=float(self.p), b=b_sp, scale=scale,
                size=n, random_state=seed,
            )
            return jnp.asarray(samples, dtype=jnp.float64)

        raise ValueError(f"Unknown rvs method: {method!r}")

    # ------------------------------------------------------------------
    # KL projection onto special-case sub-families
    # ------------------------------------------------------------------

    def to_gamma(self):
        r"""KL projection onto the :class:`Gamma` family.

        Minimises :math:`D_{\mathrm{KL}}(\mathrm{GIG}\,\|\,q)` over
        :math:`q \in \mathrm{Gamma}` by matching the Gamma sufficient
        statistics under the source GIG:
        :math:`E_{q^*}[\log X] = \eta_1,\; E_{q^*}[X] = \eta_3`. Solved via
        :meth:`Gamma.from_expectation`.
        """
        from normix.distributions.gamma import Gamma
        eta = self.expectation_params()
        return Gamma.from_expectation(jnp.array([eta[0], eta[2]]))

    def to_inverse_gamma(self):
        r"""KL projection onto the :class:`InverseGamma` family.

        Matches :math:`E[-1/X] = -\eta_2,\; E[\log X] = \eta_1`.
        """
        from normix.distributions.inverse_gamma import InverseGamma
        eta = self.expectation_params()
        return InverseGamma.from_expectation(jnp.array([-eta[1], eta[0]]))

    def to_inverse_gaussian(self):
        r"""KL projection onto the :class:`InverseGaussian` family.

        Matches :math:`E[X] = \eta_3,\; E[1/X] = \eta_2`. The closed form
        :math:`\lambda = 1/(\eta_2 - 1/\eta_3)` is well-defined whenever
        :math:`\eta_2 > 1/\eta_3` (Jensen, true for every non-degenerate
        GIG); :meth:`InverseGaussian.from_expectation` clamps near
        degenerate inputs.
        """
        from normix.distributions.inverse_gaussian import InverseGaussian
        eta = self.expectation_params()
        return InverseGaussian.from_expectation(jnp.array([eta[2], eta[1]]))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "GeneralizedInverseGaussian":
        theta = jnp.asarray(theta, dtype=jnp.float64)
        p = theta[0] + 1.0
        b = jnp.maximum(-2.0 * theta[1], 0.0)
        a = jnp.maximum(-2.0 * theta[2], 0.0)
        return cls(p=p, a=a, b=b)

    @classmethod
    def from_expectation(
        cls,
        eta: jax.Array,
        *,
        theta0: Optional[jax.Array] = None,
        maxiter: int = 500,
        tol: float = 1e-10,
        backend: str = "jax",
        method: str = "newton",
        verbose: int = 0,
    ) -> "GeneralizedInverseGaussian":
        r"""
        :math:`\eta \to \theta` via :math:`\eta`-rescaling + optimization.

        Rescaling makes the Fisher matrix symmetric (:math:`\tilde{a} = \tilde{b}`),
        reducing condition number by up to :math:`10^{30}` for extreme :math:`a/b` ratios.

        Parameters
        ----------
        theta0 : jax.Array, optional
            Warm-start point (required for JAX solvers; if ``None``, uses
            multi-start CPU solver with Gamma/InvGamma/InvGauss seeds).
        backend : str
            ``'jax'`` (default, JIT-able) or ``'cpu'`` (scipy, more robust).
        method : str
            ``'newton'``, ``'lbfgs'``, or ``'bfgs'``.
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta1, eta2, eta3 = eta[0], eta[1], eta[2]

        s = jnp.sqrt(eta2 / eta3)
        geom = jnp.sqrt(eta2 * eta3)
        eta_scaled = jnp.array([eta1 + 0.5 * jnp.log(eta2 / eta3), geom, geom])

        _GIG_BOUNDS = cls._theta_bounds()

        # Select triad functions for the chosen backend
        if backend == "cpu":
            f = cls._log_partition_cpu
            grad_fn = cls._grad_log_partition_cpu
            hess_fn = cls._hessian_log_partition_cpu
        elif backend == "jax":
            f = cls._log_partition_from_theta
            grad_fn = cls._grad_log_partition
            hess_fn = cls._hessian_log_partition
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        if theta0 is not None:
            theta0 = jnp.asarray(theta0, dtype=jnp.float64)
            theta0_scaled = jnp.array([theta0[0],
                                       theta0[1] * s,
                                       theta0[2] / s])
            theta0_scaled = theta0_scaled.at[1].set(
                jnp.minimum(theta0_scaled[1], THETA_FLOOR))
            theta0_scaled = theta0_scaled.at[2].set(
                jnp.minimum(theta0_scaled[2], THETA_FLOOR))

            # Hot path: stable jitted Newton with cached XLA executable.
            # Avoids the per-call retrace incurred by solve_bregman's
            # fresh Python closures (key bottleneck in GH JAX/JAX EM).
            if backend == "jax" and method == "newton" and verbose <= 0:
                theta_scaled, _, _, _ = _gig_jax_newton_jit(
                    eta_scaled, theta0_scaled,
                    max_steps=int(maxiter), tol=float(tol),
                )
            else:
                solver_kwargs: dict = {
                    "bounds": _GIG_BOUNDS,
                    "max_steps": maxiter,
                    "tol": tol,
                    "grad_fn": grad_fn,
                    "hess_fn": hess_fn,
                }
                # trust-exact doesn't support bounds
                if backend == "cpu" and method == "newton":
                    solver_kwargs["bounds"] = None

                result = solve_bregman(
                    f, eta_scaled, theta0_scaled,
                    backend=backend, method=method, verbose=verbose,
                    **solver_kwargs,
                )
                theta_scaled = result.theta
        else:
            theta0_list = cls._initial_guesses(eta_scaled)
            processed = []
            for t0 in theta0_list:
                t0_np = np.array(t0)
                t0_np[1] = min(t0_np[1], THETA_FLOOR)
                t0_np[2] = min(t0_np[2], THETA_FLOOR)
                processed.append(jnp.asarray(t0_np, dtype=jnp.float64))
            result = solve_bregman_multistart(
                cls._log_partition_cpu, eta_scaled, processed,
                backend="cpu", method="lbfgs",
                bounds=_GIG_BOUNDS, max_steps=maxiter, tol=tol,
                grad_fn=cls._grad_log_partition_cpu,
                verbose=verbose,
            )
            theta_scaled = result.theta

        theta = jnp.array([theta_scaled[0],
                           theta_scaled[1] / s,
                           s * theta_scaled[2]])
        return cls.from_natural(theta)

    @classmethod
    def _theta_bounds(cls):
        # θ₁ unbounded, θ₂ ≤ 0, θ₃ ≤ 0
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf])
        upper = jnp.array([jnp.inf, 0.0, 0.0])
        return (lower, upper)

    @staticmethod
    def _initial_guesses(eta_scaled: jax.Array) -> list:
        """
        Multi-start initial guesses for the scaled GIG problem.

        Uses Gamma, InverseGamma, InverseGaussian special cases.
        """
        from normix.distributions.gamma import Gamma
        from normix.distributions.inverse_gamma import InverseGamma
        from normix.distributions.inverse_gaussian import InverseGaussian

        eta1, eta2, eta3 = (float(eta_scaled[0]),
                            float(eta_scaled[1]),
                            float(eta_scaled[2]))
        starting_points = []
        eps = GIG_THETA_PERTURB

        # 1. Gamma limit (b→0): match η₁ = E[log X], η₃ = E[X]
        try:
            g = Gamma.from_expectation(jnp.array([eta1, eta3]))
            g_theta = g.natural_params()
            starting_points.append(
                np.array([float(g_theta[0]), -eps / 2, float(g_theta[1])]))
        except Exception:
            pass

        # 2. InverseGamma limit (a→0): match η₁ = E[log X], η₂ = E[1/X]
        try:
            ig = InverseGamma.from_expectation(jnp.array([-eta2, eta1]))
            ig_theta = ig.natural_params()
            starting_points.append(
                np.array([float(ig_theta[1]), float(-ig_theta[0]), -eps / 2]))
        except Exception:
            pass

        # 3. InverseGaussian limit (p=-1/2): match η₂ = E[1/X], η₃ = E[X]
        if eta3 > 0 and eta2 > 1.0 / eta3:
            try:
                igauss = InverseGaussian.from_expectation(jnp.array([eta3, eta2]))
                ig_theta = igauss.natural_params()
                starting_points.append(
                    np.array([-1.5, float(ig_theta[1]), float(ig_theta[0])]))
            except Exception:
                pass

        # 4. Perturbed copies
        for sp in list(starting_points):
            for scale in [0.1, 0.5, 2.0, 10.0]:
                perturbed = sp.copy()
                perturbed[1] = min(perturbed[1], -eps * scale / 2)
                perturbed[2] = min(perturbed[2], -eps * scale / 2)
                starting_points.append(perturbed)

        # 5. Fallback
        if not starting_points:
            starting_points.append(np.array([0.0, -0.5, -0.5]))

        return starting_points


# ---------------------------------------------------------------------------
# Stable jitted GIG Newton solver
# ---------------------------------------------------------------------------
#
# Hoisted to module level so that JAX caches the compiled XLA executable
# across all GIG.from_expectation(jax/newton) calls. Without this, every
# warm-started solve inside an EM loop builds fresh Python closures and
# forces JAX to re-trace the same Newton kernel, dominating GH M-step time.

_gig_jax_newton_jit = make_jit_newton_solver(
    f=GeneralizedInverseGaussian._log_partition_from_theta,
    grad_fn=GeneralizedInverseGaussian._grad_log_partition,
    hess_fn=GeneralizedInverseGaussian._hessian_log_partition,
    bounds=GeneralizedInverseGaussian._theta_bounds(),
)


# Convenience alias
GIG = GeneralizedInverseGaussian
