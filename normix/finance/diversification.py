r"""
Diversification analytics: effective number of bets under a torsion.

Variance ENB diagonalizes the return covariance
:math:`\mathrm{Cov}[X] = E[Y]\Sigma + \mathrm{Var}[Y]\,\gamma\gamma^\top`
(:doc:`/theory/enb`); generalized ENB diagonalizes the Hessian of the squared
coherent risk :math:`H_{r^2} = 2\nabla r\,\nabla r^\top + 2 r H_r`
(:doc:`/theory/generalized_enb`). Both reduce to one core: normalize
:math:`d_k v_k^2` over a torsion :math:`T H T^\top = \operatorname{diag}(d)`,
:math:`v = (T^\top)^{-1} w`, and exponentiate the entropy.

Minimum torsion is a diagonalization *strategy*, not a diversification measure.
"""
from __future__ import annotations

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from normix.finance.risk import RiskMeasure
from normix.mixtures.marginal import NormalMixture
from normix.utils.constants import LOG_EPS, TORSION_SPECTRAL_FLOOR


class TorsionDecomposition(eqx.Module):
    r"""Diagonalization :math:`T H T^\top = \operatorname{diag}(d)` of a PSD matrix.

    ``T_inv_T`` is :math:`(T^\top)^{-1}`, the map from portfolio weights
    :math:`w` to torsion-adjusted weights :math:`v`. Both concrete torsions
    supply it in closed form (no solve at evaluation time).
    ``eigenvalues`` is the pre-clamp spectrum used for validity checks.
    ``valid`` is ``False`` when a material negative eigenvalue was present.
    """

    T: Array
    T_inv_T: Array
    d: Array
    eigenvalues: Array
    valid: Array


class Torsion(eqx.Module):
    """Strategy: how to diagonalize a PSD matrix into uncorrelated bets."""

    @abc.abstractmethod
    def decompose(self, H: Array) -> TorsionDecomposition:
        r"""Return :math:`(T, (T^\top)^{-1}, d)` with :math:`THT^\top = \operatorname{diag}(d)`.

        Roundoff-scale negative eigenvalues are projected to zero; material
        indefiniteness sets ``valid=False`` (callers NaN the ENB).
        """


class MinimumTorsion(Torsion):
    r"""Constrained minimum torsion (Meucci 2014): :math:`T = C^{-1/2}\operatorname{diag}(s)^{-1}`.

    Closed form at :math:`D = I`. The iterative algorithm (unconstrained
    :math:`D`) is deferred; it would land as a sibling
    ``IterativeMinimumTorsion`` without touching callers.
    """

    def decompose(self, H: Array) -> TorsionDecomposition:
        H = 0.5 * (H + H.T)
        h = jnp.diag(H)
        h_scale = jnp.maximum(jnp.max(h), LOG_EPS)
        h = jnp.maximum(h, TORSION_SPECTRAL_FLOOR * h_scale)
        s = jnp.sqrt(h)
        C = H / jnp.outer(s, s)
        C = 0.5 * (C + C.T)
        S, U = jnp.linalg.eigh(C)
        S_scale = jnp.maximum(jnp.max(jnp.abs(S)), LOG_EPS)
        valid = jnp.all(S >= -TORSION_SPECTRAL_FLOOR * S_scale)
        S_psd = jnp.maximum(S, 0.0)
        S_inv = jnp.maximum(
            S_psd, TORSION_SPECTRAL_FLOOR * jnp.maximum(jnp.max(S_psd), LOG_EPS),
        )
        C_half = (U * jnp.sqrt(S_psd)) @ U.T
        C_inv_half = (U * jax.lax.rsqrt(S_inv)) @ U.T
        T = C_inv_half / s[None, :]
        T_inv_T = C_half * s[None, :]
        return TorsionDecomposition(
            T=T,
            T_inv_T=T_inv_T,
            d=jnp.ones_like(h),
            eigenvalues=S,
            valid=valid,
        )


class PCATorsion(Torsion):
    r"""Principal-components torsion (Meucci 2010): :math:`T = E^\top`, :math:`d = \Lambda`.

    Rows ordered by descending eigenvalue. Eigenvector sign/order is unstable
    under near-ties — documented; :class:`MinimumTorsion` is the default.
    """

    def decompose(self, H: Array) -> TorsionDecomposition:
        H = 0.5 * (H + H.T)
        lam, E = jnp.linalg.eigh(H)
        # Ascending from eigh -> reverse to descending.
        lam = lam[::-1]
        E = E[:, ::-1]
        scale = jnp.maximum(jnp.max(jnp.abs(lam)), LOG_EPS)
        valid = jnp.all(lam >= -TORSION_SPECTRAL_FLOOR * scale)
        d = jnp.maximum(lam, 0.0)
        T = E.T
        return TorsionDecomposition(
            T=T,
            T_inv_T=T,  # orthogonal: (T^T)^{-1} = E^T = T
            d=d,
            eigenvalues=lam,
            valid=valid,
        )


class ENBResult(eqx.Module):
    r"""Effective number of bets and its decomposition at one portfolio.

    ``p`` is normalized by :math:`\sum_k d_k v_k^2` (a simplex up to floating
    point). ``risk`` is the 1-homogeneous risk whose square was diagonalized:
    portfolio volatility :math:`\sqrt{w^\top\Sigma_X w}` for
    :class:`VarianceENB`, :math:`\rho(w^\top X)` for :class:`GeneralizedENB`.
    ``enb`` is NaN when the local matrix is materially indefinite, the
    portfolio risk is non-positive, or the total contribution vanishes.
    """

    enb: Array
    p: Array
    risk: Array
    v: Array
    d: Array
    T: Array
    eigenvalues: Array


def _enb_core(
    dec: TorsionDecomposition,
    w: Array,
    risk: Array,
    *,
    risk_positive: Array,
) -> ENBResult:
    r"""Shared ENB kernel from a torsion decomposition and weights."""
    w = jnp.asarray(w, dtype=jnp.float64)
    v = dec.T_inv_T @ w
    c = dec.d * v * v
    total = jnp.sum(c)
    p = c / jnp.maximum(total, LOG_EPS)
    entropy = -jnp.sum(p * jnp.log(jnp.maximum(p, LOG_EPS)))
    N = jnp.exp(entropy)
    valid = dec.valid & risk_positive & (total > LOG_EPS)
    enb = jnp.where(valid, N, jnp.nan)
    p_out = jnp.where(valid, p, jnp.full_like(p, jnp.nan))
    return ENBResult(
        enb=enb,
        p=p_out,
        risk=risk,
        v=v,
        d=dec.d,
        T=dec.T,
        eigenvalues=dec.eigenvalues,
    )


class VarianceENB(eqx.Module):
    r"""Variance-based effective number of bets (:doc:`/theory/enb`).

    Diagonalizes the *return covariance* ``model.cov()``
    :math:`= E[Y]\Sigma + \mathrm{Var}[Y]\,\gamma\gamma^\top` — not the
    dispersion :math:`\Sigma`. Deterministic: no subordinator sample.
    Requires a finite ``Var[Y]`` (same moment caveat as
    :meth:`~normix.mixtures.marginal.NormalMixture.skewness`).
    """

    torsion: Torsion = MinimumTorsion()

    @eqx.filter_jit
    def evaluate(self, model: NormalMixture, w: Array) -> ENBResult:
        r"""ENB of :math:`w^\top X` under the fitted model. Y-free by design.

        vmap-able over a weight grid:
        ``jax.vmap(enb.evaluate, in_axes=(None, 0))``.
        """
        return self.evaluate_covariance(model.cov(), w)

    @eqx.filter_jit
    def evaluate_covariance(self, Sigma_X: Array, w: Array) -> ENBResult:
        r"""ENB from an explicit covariance (empirical, factor-model, stress).

        Precondition: ``Sigma_X`` PSD with nonzero diagonal, ``w != 0``.
        """
        Sigma_X = jnp.asarray(Sigma_X, dtype=jnp.float64)
        w = jnp.asarray(w, dtype=jnp.float64)
        dec = self.torsion.decompose(Sigma_X)
        # Scale-invariant p: diagonalizing Sigma_X (not 2 Sigma_X) is equivalent.
        v = dec.T_inv_T @ w
        total = jnp.sum(dec.d * v * v)
        risk = jnp.sqrt(jnp.maximum(total, 0.0))
        return _enb_core(dec, w, risk, risk_positive=total > LOG_EPS)


class GeneralizedENB(eqx.Module):
    r"""ENB of a squared coherent risk measure (:doc:`/theory/generalized_enb`).

    For 1-homogeneous :math:`\rho`, diagonalizes
    :math:`H_{r^2}(w) = 2\nabla r\,\nabla r^\top + 2 r H_r` where
    :math:`(r, \nabla r, H_r)` come from **one** call to
    ``risk.value_grad_hess_w(model, w, Y)``. Meaningful only for
    :math:`r(w) > 0` (else :math:`r^2` is not convex at ``w`` and ``enb``
    is NaN).
    """

    risk: RiskMeasure
    torsion: Torsion = MinimumTorsion()

    @eqx.filter_jit
    def evaluate(self, model: NormalMixture, w: Array, Y: Array) -> ENBResult:
        r"""ENB at ``w`` under subordinator sample ``Y`` (common random numbers).

        ``Y`` is mandatory — draw once via
        ``model.joint.subordinator().rvs(n, seed)`` and share across finance
        evaluations. Not promised vmap-able over ``w``: the fused bundle seeds
        its CMC bracket from the PINV ``ppf``.
        """
        w = jnp.asarray(w, dtype=jnp.float64)
        r, g, H_r = self.risk.value_grad_hess_w(model, w, Y)
        H2 = 2.0 * jnp.outer(g, g) + 2.0 * r * H_r
        dec = self.torsion.decompose(H2)
        return _enb_core(dec, w, r, risk_positive=r > 0.0)
