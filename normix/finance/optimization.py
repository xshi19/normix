r"""
Mean-risk portfolio optimization for normal-mixture models.

For a multivariate normal mixture :math:`X \stackrel{d}{=} \mu + \gamma Y
+ \sqrt{Y} Z` with :math:`Z \sim \mathcal{N}(0, \Sigma)`, the mean-risk
problem

.. math::

    \min_w \rho(w^\top X) \quad \text{s.t.} \quad
    w^\top e = 1, \quad E[w^\top X] \ge m

reduces — for any coherent risk measure :math:`\rho` — to a two-dimensional
problem in the *reduced coordinates* :math:`\tilde\mu = w^\top\mu` and
:math:`\tilde\gamma = w^\top\gamma`. The minimum-dispersion weights that
realise a given :math:`(\tilde\mu, \tilde\gamma)` are

.. math::

    w^*(\tilde\mu, \tilde\gamma) = \Sigma^{-1}[\mu\;\gamma\;e]\,A^{-1}
    [\tilde\mu\;\tilde\gamma\;1]^\top, \qquad
    A = [\mu\;\gamma\;e]^\top \Sigma^{-1} [\mu\;\gamma\;e],

and the realised dispersion is
:math:`g(\tilde\mu, \tilde\gamma) = [\tilde\mu\;\tilde\gamma\;1] A^{-1}
[\tilde\mu\;\tilde\gamma\;1]^\top`. The map
:math:`(\tilde\mu, \tilde\gamma) \mapsto \rho` is the **efficient surface**;
its lower envelope under the return constraint is the **efficient frontier**.

The inverse formulae hold when :math:`A` is nonsingular, that is when
:math:`r = \operatorname{rank}[\mu\;\gamma\;e] = 3`. If :math:`r < 3`, the
attainable pairs form an affine set :math:`\mathcal{S}` of dimension
:math:`r-1`. On :math:`\mathcal{S}` the same formulae hold with the
pseudoinverse :math:`A^{+}` in place of :math:`A^{-1}`. A target outside
:math:`\mathcal{S}` is not a portfolio.

See :doc:`../theory/mean_risk_optimization` for the derivation.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.linalg import solve_triangular

from normix.finance.risk import RiskMeasure
from normix.mixtures.marginal import NormalMixture
from normix.utils.constants import MEAN_RISK_REACH_ATOL

# Reciprocal golden ratio for golden-section search.
_INV_PHI = (5.0 ** 0.5 - 1.0) / 2.0

_COORDINATE_NAMES = ("mu_tilde", "gamma_tilde", "budget")


class EfficientSurface(eqx.Module):
    r"""Efficient surface :math:`(\tilde\mu, \tilde\gamma) \mapsto \rho` on a grid.

    ``risk[i, j]`` is the risk of the minimum-dispersion portfolio with
    reduced coordinates ``(mu_tilde[i], gamma_tilde[j])``;
    ``expected_return[i, j] = mu_tilde[i] + gamma_tilde[j] * E[Y]``.
    """

    mu_tilde: Array
    gamma_tilde: Array
    risk: Array
    expected_return: Array


class EfficientFrontier(eqx.Module):
    r"""Mean-risk efficient frontier: minimum risk per target expected return.

    Each entry is the solution of the reduced problem on the constraint line
    :math:`\tilde\mu + \tilde\gamma E[Y] = m`, including the realised
    portfolio ``weights`` of shape ``(K, d)``.
    """

    expected_return: Array
    risk: Array
    mu_tilde: Array
    gamma_tilde: Array
    weights: Array


def _golden_section_min(f, lo: Array, hi: Array, n_iter: int) -> Array:
    r"""Vectorized golden-section minimiser of a unimodal ``f: (K,) -> (K,)``.

    Shrinks each bracket ``[lo_k, hi_k]`` independently; returns the
    per-component minimiser. Used to minimise risk along return-constraint
    lines (the risk is convex in the reduced coordinates).
    """
    def body(_, bracket):
        a, b = bracket
        width = (b - a) * _INV_PHI
        c = b - width
        d = a + width
        take_left = f(c) < f(d)
        return jnp.where(take_left, a, c), jnp.where(take_left, d, b)

    a, b = jax.lax.fori_loop(0, n_iter, body, (lo, hi))
    return 0.5 * (a + b)


def _reduced_factors(
    model: NormalMixture,
) -> tuple[Array, Array, Array, Array, int]:
    r"""Whitened SVD of :math:`M = [\mu\;\gamma\;e]`.

    Returns ``(A, S, Vh, LT_U, rank)``. With :math:`\Sigma = L L^\top`,

    .. math::

        B = L^{-1} M = U \operatorname{diag}(S) V^\top,

    ``Vh`` stores :math:`V^\top` and ``LT_U`` stores :math:`L^{-\top} U`.
    The Gram matrix :math:`A = B^\top B = M^\top \Sigma^{-1} M` is
    symmetric positive semidefinite and :math:`3\times 3`. It is singular
    when :math:`\operatorname{rank}(M) < 3`; it is not inverted here.

    ``rank`` is the number of singular values strictly above
    :math:`\max(d, 3)\, \varepsilon\, \sigma_{\max}`. Here :math:`\varepsilon`
    is machine epsilon for the dtype of :math:`B`, and :math:`\sigma_{\max}`
    is its largest singular value. This is the default relative tolerance
    of a rank-revealing SVD (NumPy ``linalg.matrix_rank`` on a
    :math:`d \times 3` matrix). A backward-stable SVD perturbs singular
    values by about :math:`\varepsilon \|B\|` times a small factor of the
    dimensions, so a singular value below the tolerance is collinear at
    rounding scale.

    The test is applied to :math:`\sigma(B)`, not to the eigenvalues
    :math:`\sigma(B)^2` of :math:`A`. Forming :math:`A` and factoring it
    squares the condition number. A Cholesky factor of :math:`A` is then
    undefined, or non-finite, once a singular value of :math:`B` approaches
    :math:`\sqrt{\varepsilon}`.
    """
    mu = model.mu
    gamma = model.gamma
    e = jnp.ones_like(mu)
    M = jnp.stack([mu, gamma, e], axis=1)                  # (d, 3)
    B = solve_triangular(model.L_Sigma, M, lower=True)    # L^{-1} M
    U, S, Vh = jnp.linalg.svd(B, full_matrices=False)
    LT_U = solve_triangular(model.L_Sigma.T, U, lower=False)
    A = B.T @ B

    singular = np.asarray(S)
    sigma_max = float(singular[0])
    if not np.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError(
            "coordinate matrix M = [mu, gamma, e] has numerical rank 0")
    eps = float(np.finfo(singular.dtype).eps)
    d = int(mu.shape[0])
    tol = max(d, 3) * eps * sigma_max
    rank = int(np.count_nonzero(singular > tol))
    if rank < 1:
        raise ValueError(
            "coordinate matrix M = [mu, gamma, e] has numerical rank 0")
    return A, S, Vh, LT_U, rank


def _raise_from_host_residual(values: np.ndarray) -> None:
    """Raise if any entry of :math:`M^\\top w - c` exceeds the reach gap."""
    bad = [
        name for name, value in zip(_COORDINATE_NAMES, values, strict=True)
        if not np.isfinite(value) or abs(float(value)) > MEAN_RISK_REACH_ATOL
    ]
    if not bad:
        return
    noun = "coordinate" if len(bad) == 1 else "coordinates"
    shown = np.array2string(np.asarray(values), precision=6)
    raise ValueError(
        f"unattainable reduced {noun} {', '.join(bad)}; "
        f"residual of M^T w - c is {shown}")


class MeanRiskProblem(eqx.Module):
    r"""Mean-risk optimization in reduced :math:`(\tilde\mu, \tilde\gamma)` coordinates.

    Bundles a fitted :class:`~normix.mixtures.marginal.NormalMixture` and a
    :class:`~normix.finance.risk.RiskMeasure`. All heavy evaluations share a
    fixed subordinator sample ``Y`` (common random numbers); draw it once via
    ``model.joint.subordinator().rvs(n, seed)``.

    The whitened factorisation of :math:`M = [\mu\;\gamma\;e]` is computed
    once at construction. The Gram matrix :math:`A = M^\top\Sigma^{-1}M`
    is stored for the minimum-dispersion anchor; it is positive semidefinite
    and need not be invertible. Weights use the truncated pseudoinverse of
    the whitened factor and are returned only for attainable targets.
    """

    model: NormalMixture
    risk: RiskMeasure
    A: Array
    S: Array
    Vh: Array
    LT_U: Array
    rank: int = eqx.field(static=True)

    def __init__(self, model: NormalMixture, risk: RiskMeasure):
        A, S, Vh, LT_U, rank = _reduced_factors(model)
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'risk', risk)
        object.__setattr__(self, 'A', A)
        object.__setattr__(self, 'S', S)
        object.__setattr__(self, 'Vh', Vh)
        object.__setattr__(self, 'LT_U', LT_U)
        object.__setattr__(self, 'rank', rank)

    # ------------------------------------------------------------------
    # Reduced-coordinate algebra
    # ------------------------------------------------------------------

    def E_Y(self) -> Array:
        r""":math:`E[Y]` — subordinator mean."""
        return self.model.joint.subordinator().mean()

    def _target(self, mu_tilde: Array, gamma_tilde: Array) -> Array:
        return jnp.stack([
            jnp.asarray(mu_tilde, dtype=jnp.float64),
            jnp.asarray(gamma_tilde, dtype=jnp.float64),
            jnp.ones((), dtype=jnp.float64),
        ])

    def _solve(self, c: Array) -> tuple[Array, Array]:
        r"""Truncated factors :math:`w = L^{-\top} U_r S_r^{-1} V_r^\top c`
        and :math:`g = \|S_r^{-1} V_r^\top c\|^2`.
        """
        r = self.rank
        alpha = (self.Vh[:r] @ c) / self.S[:r]
        w = self.LT_U[:, :r] @ alpha
        g = jnp.dot(alpha, alpha)
        return w, g

    def _constraint_residual(self, C: Array) -> Array:
        r"""Residual :math:`M^\top w - c = P_r c - c` for one target or a batch."""
        r = self.rank
        Vh_r = self.Vh[:r]
        return (C @ Vh_r.T) @ Vh_r - C

    def _raise_if_unreachable(self, c: Array) -> None:
        resid = self._constraint_residual(c)
        try:
            values = np.array([float(resid[0]), float(resid[1]), float(resid[2])])
        except jax.errors.ConcretizationTypeError:
            return
        _raise_from_host_residual(values)

    def _raise_if_any_unreachable(self, C: Array) -> None:
        resid = np.asarray(self._constraint_residual(C))
        flat = resid.reshape(-1, 3)
        worst = int(np.argmax(np.max(np.abs(flat), axis=1)))
        _raise_from_host_residual(flat[worst])

    def weights(self, mu_tilde: Array, gamma_tilde: Array) -> Array:
        r"""Minimum-dispersion weights realising :math:`(\tilde\mu, \tilde\gamma)`.

        Raises
        ------
        ValueError
            If :math:`(\tilde\mu, \tilde\gamma)` is outside the reachable
            set. The message names each coordinate of
            :math:`M^\top w - c` whose absolute residual exceeds
            ``MEAN_RISK_REACH_ATOL``.
        """
        c = self._target(mu_tilde, gamma_tilde)
        w, _g = self._solve(c)
        self._raise_if_unreachable(c)
        return w

    def dispersion(self, mu_tilde: Array, gamma_tilde: Array) -> Array:
        r"""Realised dispersion :math:`g(\tilde\mu, \tilde\gamma) = w^{*\top}\Sigma w^*`.

        Raises
        ------
        ValueError
            If :math:`(\tilde\mu, \tilde\gamma)` is outside the reachable set.
        """
        c = self._target(mu_tilde, gamma_tilde)
        _w, g = self._solve(c)
        self._raise_if_unreachable(c)
        return g

    def expected_return(self, mu_tilde: Array, gamma_tilde: Array) -> Array:
        r""":math:`m = \tilde\mu + \tilde\gamma\,E[Y]`."""
        return mu_tilde + gamma_tilde * self.E_Y()

    def min_variance_point(self) -> tuple[Array, Array]:
        r"""Reduced coordinates :math:`(\tilde\mu, \tilde\gamma)` of the global
        minimum-variance portfolio :math:`w = \Sigma^{-1}e / (e^\top\Sigma^{-1}e)`.

        A convenient anchor for choosing efficient-surface grid ranges.
        """
        denom = self.A[2, 2]
        return self.A[0, 2] / denom, self.A[1, 2] / denom

    def projection_at(self, mu_tilde: Array, gamma_tilde: Array):
        r"""Univariate portfolio return at :math:`(\tilde\mu, \tilde\gamma)`.

        Projects the minimum-dispersion :meth:`weights`; the result is a
        ``Univariate*`` instance with location :math:`\tilde\mu`, skewness
        :math:`\tilde\gamma`, variance :math:`g(\tilde\mu, \tilde\gamma)`.
        """
        return self.model.project(self.weights(mu_tilde, gamma_tilde))

    def _line_frame(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Affine frame :math:`c(t) = c_0 + t\, dc` of a rank-2 reachable set.

        :math:`c_0` has third coordinate 1 and :math:`dc` has third
        coordinate 0. ``dc`` is nonzero.
        """
        V = np.asarray(self.Vh[:2], dtype=np.float64).T
        budget = V[2]
        scale = float(budget @ budget)
        if scale <= float(np.finfo(np.float64).eps):
            raise ValueError(
                "budget coordinate is orthogonal to the row space of M")
        alpha0 = budget / scale
        direction = np.array([-budget[1], budget[0]], dtype=np.float64)
        direction /= np.linalg.norm(direction)
        return V @ alpha0, V @ direction

    def _search_return_line(
        self,
        returns: Array,
        Y: Array,
        gamma_bounds: tuple[float, float],
        n_iter: int,
    ) -> tuple[Array, Array]:
        r"""Golden-section search of :math:`\tilde\mu = m - \tilde\gamma E[Y]`."""
        E_Y = self.E_Y()
        lo = jnp.full_like(returns, gamma_bounds[0])
        hi = jnp.full_like(returns, gamma_bounds[1])

        def risk_of_gamma(gamma_vec: Array) -> Array:
            return self._surface_risk(returns - gamma_vec * E_Y, gamma_vec, Y)

        gamma_star = _golden_section_min(risk_of_gamma, lo, hi, n_iter)
        return returns - gamma_star * E_Y, gamma_star

    def _degenerate_targets(
        self,
        returns: Array,
        Y: Array,
        gamma_bounds: tuple[float, float],
        n_iter: int,
    ) -> tuple[Array, Array]:
        """Coordinates on :math:`S` realising each target return, or an error."""
        returns_np = np.asarray(returns, dtype=np.float64)
        EY = float(self.E_Y())
        lo = float(gamma_bounds[0])
        hi = float(gamma_bounds[1])
        if self.rank == 1:
            return self._point_targets(returns, returns_np, EY, lo, hi)
        return self._line_targets(returns, returns_np, Y, EY, lo, hi, gamma_bounds, n_iter)

    def _point_targets(
        self,
        returns: Array,
        returns_np: np.ndarray,
        EY: float,
        lo: float,
        hi: float,
    ) -> tuple[Array, Array]:
        v = np.asarray(self.Vh[0], dtype=np.float64)
        if abs(float(v[2])) <= float(np.finfo(np.float64).eps):
            raise ValueError(
                "budget coordinate is orthogonal to the row space of M")
        c = v / v[2]
        m_star = float(c[0] + c[1] * EY)
        if np.any(np.abs(returns_np - m_star) > MEAN_RISK_REACH_ATOL):
            raise ValueError(
                "expected return is not attainable; the reachable set is the "
                f"point (mu_tilde, gamma_tilde) = ({c[0]}, {c[1]}), "
                f"with expected return {m_star}")
        if not lo - MEAN_RISK_REACH_ATOL <= float(c[1]) <= hi + MEAN_RISK_REACH_ATOL:
            raise ValueError(
                "unattainable reduced coordinate gamma_tilde; the only "
                f"portfolio has gamma_tilde={c[1]} outside gamma_bounds {(lo, hi)}")
        return jnp.full_like(returns, c[0]), jnp.full_like(returns, c[1])

    def _line_targets(
        self,
        returns: Array,
        returns_np: np.ndarray,
        Y: Array,
        EY: float,
        lo: float,
        hi: float,
        gamma_bounds: tuple[float, float],
        n_iter: int,
    ) -> tuple[Array, Array]:
        c0, dc = self._line_frame()
        slope = float(dc[0] + dc[1] * EY)
        m0 = float(c0[0] + c0[1] * EY)
        if abs(slope) <= MEAN_RISK_REACH_ATOL:
            if np.any(np.abs(returns_np - m0) > MEAN_RISK_REACH_ATOL):
                raise ValueError(
                    "expected return is not attainable; on the reachable set "
                    f"the return is constant and equal to {m0}")
            # The return constraint contains S, so the line search stays on S.
            return self._search_return_line(returns, Y, gamma_bounds, n_iter)
        t = (returns_np - m0) / slope
        gamma_np = c0[1] + t * dc[1]
        if np.any(gamma_np < lo - MEAN_RISK_REACH_ATOL) or np.any(
            gamma_np > hi + MEAN_RISK_REACH_ATOL
        ):
            raise ValueError(
                "unattainable reduced coordinate gamma_tilde; the portfolio "
                "on the reachable set has "
                f"gamma_tilde={np.array2string(gamma_np, precision=6)} "
                f"outside gamma_bounds {(lo, hi)}")
        mu_np = c0[0] + t * dc[0]
        return jnp.asarray(mu_np), jnp.asarray(gamma_np)

    # ------------------------------------------------------------------
    # Risk on the efficient surface
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def _surface_risk(self, mu_flat: Array, gamma_flat: Array, Y: Array) -> Array:
        r"""Risk over flattened reduced coordinates (vectorized, JIT-able)."""
        C = jnp.stack([mu_flat, gamma_flat, jnp.ones_like(mu_flat)], axis=-1)
        r = self.rank
        alpha = (C @ self.Vh[:r].T) / self.S[:r]
        g = jnp.sum(alpha * alpha, axis=-1)
        sigma = jnp.sqrt(g)
        return jax.vmap(self.risk.value_reduced, in_axes=(0, 0, 0, None))(
            mu_flat, gamma_flat, sigma, Y)

    def risk_at(self, mu_tilde: Array, gamma_tilde: Array, Y: Array) -> Array:
        r"""Efficient-surface risk at a single :math:`(\tilde\mu, \tilde\gamma)`."""
        sigma = jnp.sqrt(self.dispersion(mu_tilde, gamma_tilde))
        return self.risk.value_reduced(
            jnp.asarray(mu_tilde, dtype=jnp.float64),
            jnp.asarray(gamma_tilde, dtype=jnp.float64), sigma, Y)

    def efficient_surface(
        self, mu_tilde: Array, gamma_tilde: Array, Y: Array,
    ) -> EfficientSurface:
        r"""Evaluate the efficient surface over the grid ``mu_tilde × gamma_tilde``.

        ``mu_tilde`` and ``gamma_tilde`` are 1-D arrays; the returned
        ``risk`` has shape ``(len(mu_tilde), len(gamma_tilde))``. Memory
        scales as ``len(mu_tilde) * len(gamma_tilde) * len(Y)``.
        Every grid node must lie in the reachable set.

        Raises
        ------
        ValueError
            If any node lies outside the reachable set.
        """
        mu_tilde = jnp.asarray(mu_tilde, dtype=jnp.float64)
        gamma_tilde = jnp.asarray(gamma_tilde, dtype=jnp.float64)
        MU, GA = jnp.meshgrid(mu_tilde, gamma_tilde, indexing='ij')
        C = jnp.stack([MU.ravel(), GA.ravel(), jnp.ones(MU.size)], axis=-1)
        self._raise_if_any_unreachable(C)
        risk = self._surface_risk(MU.ravel(), GA.ravel(), Y).reshape(MU.shape)
        return EfficientSurface(
            mu_tilde=mu_tilde, gamma_tilde=gamma_tilde,
            risk=risk, expected_return=MU + GA * self.E_Y(),
        )

    def efficient_frontier(
        self,
        returns: Array,
        Y: Array,
        gamma_bounds: tuple[float, float],
        n_iter: int = 48,
    ) -> EfficientFrontier:
        r"""Minimum risk for each target expected return in ``returns``.

        For every target :math:`m`, minimises the risk along the part of
        :math:`\tilde\mu + \tilde\gamma E[Y] = m` that lies in the reachable
        set and in ``gamma_bounds``. When the reachable set is a plane this
        is golden-section search in :math:`\tilde\gamma`. When it is a line,
        the return constraint selects one point of the line, or the whole
        line if the return is constant there. A constant return that misses
        the line, and any target off the reachable set, is infeasible.
        When the reachable set is a point, that portfolio is returned for
        its own expected return and every other return is infeasible.

        Raises
        ------
        ValueError
            If a target return is not attainable on the reachable set, or
            the feasible :math:`\tilde\gamma` lies outside ``gamma_bounds``.
        """
        returns = jnp.asarray(returns, dtype=jnp.float64)
        if self.rank == 3:
            mu_star, gamma_star = self._search_return_line(
                returns, Y, gamma_bounds, n_iter)
        else:
            mu_star, gamma_star = self._degenerate_targets(
                returns, Y, gamma_bounds, n_iter)
        C = jnp.stack([mu_star, gamma_star, jnp.ones_like(mu_star)], axis=-1)
        self._raise_if_any_unreachable(C)
        risk_star = self._surface_risk(mu_star, gamma_star, Y)
        weights = jax.vmap(self.weights)(mu_star, gamma_star)
        return EfficientFrontier(
            expected_return=returns, risk=risk_star,
            mu_tilde=mu_star, gamma_tilde=gamma_star, weights=weights,
        )
