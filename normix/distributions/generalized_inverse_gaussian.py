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
- ``_grad_log_partition``       : analytical Bessel ratios (5 :math:`K_\\nu` calls)
- ``_hessian_log_partition``    : analytical 7-Bessel Hessian in :math:`\\theta`-space
- ``_log_partition_cpu``        : numpy + ``log_kv(backend='cpu')``
- ``_grad_log_partition_cpu``   : analytical Bessel ratios via ``scipy.kve``
- ``_hessian_log_partition_cpu``: central FD on ``_log_partition_cpu``
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from normix.utils.bessel import log_kv
from normix.exponential_family import ExponentialFamily
from normix.utils.constants import (
    LOG_EPS, TINY, BESSEL_EPS_V, GIG_DEGEN_THRESHOLD,
    THETA_FLOOR, FD_EPS_FISHER, GIG_THETA_PERTURB,
)
from normix.fitting.solvers import (
    bregman_objective, solve_bregman, solve_bregman_multistart,
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

    # ------------------------------------------------------------------
    # Tier 1: Exponential family interface
    # ------------------------------------------------------------------

    @staticmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        r"""
        :math:`\psi(\theta) = \log 2 + \log K_p(\sqrt{ab}) + (p/2)(\log b - \log a)`.

        Degenerate limit (:math:`\sqrt{ab} < \text{threshold}`): delegate to
        Gamma/InverseGamma. All branches use safe clamped values so no NaN
        gradients from non-selected ``jnp.where`` branches.
        """
        p = theta[0] + 1.0
        b = jnp.maximum(-2.0 * theta[1], 0.0)
        a = jnp.maximum(-2.0 * theta[2], 0.0)
        sqrt_ab = jnp.sqrt(a * b)

        # Safe sqrt_ab to prevent log_kv(p, 0) blow-up
        sqrt_ab_safe = jnp.maximum(sqrt_ab, LOG_EPS)

        # General Bessel case
        psi_bessel = (jnp.log(2.0) + log_kv(p, sqrt_ab_safe)
                      + 0.5 * p * (jnp.log(b + LOG_EPS) - jnp.log(a + LOG_EPS)))

        # Gamma limit (b→0, p>0): Gamma(p, a/2)
        alpha_g = jnp.maximum(p, LOG_EPS)
        beta_g = jnp.maximum(a / 2.0, LOG_EPS)
        psi_gamma = (jax.scipy.special.gammaln(alpha_g)
                     - alpha_g * jnp.log(beta_g))

        # InverseGamma limit (a→0, p<0): InvGamma(-p, b/2)
        alpha_ig = jnp.maximum(-p, LOG_EPS)
        beta_ig = jnp.maximum(b / 2.0, LOG_EPS)
        psi_invgamma = (jax.scipy.special.gammaln(alpha_ig)
                        - alpha_ig * jnp.log(beta_ig))

        use_degen = sqrt_ab < GIG_DEGEN_THRESHOLD
        use_gamma = use_degen & (b <= a)   # b≈0: Gamma limit
        use_invg = use_degen & (a < b)     # a≈0: InvGamma limit

        return jnp.where(use_gamma, psi_gamma,
               jnp.where(use_invg, psi_invgamma,
                         psi_bessel))

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
    # Tier 2: Analytical gradient + Hessian (Bessel ratios in θ-space)
    # ------------------------------------------------------------------

    @classmethod
    def _grad_log_partition(cls, theta: jax.Array) -> jax.Array:
        r"""
        :math:`\nabla\psi(\theta) = [E[\log X],\; E[1/X],\; E[X]]` via analytical Bessel ratios.

        Uses 5 :math:`\log K_\nu` evaluations (at orders :math:`p, p\pm 1, p\pm\varepsilon`)
        and the identities:

        .. math::

            E[1/X] = \sqrt{a/b}\cdot K_{p-1}(\sqrt{ab}) / K_p(\sqrt{ab}), \quad
            E[X]   = \sqrt{b/a}\cdot K_{p+1}(\sqrt{ab}) / K_p(\sqrt{ab})

        .. math::

            E[\log X] = \partial_p \log K_p(\sqrt{ab}) + \tfrac{1}{2}\log(b/a)

        Clamping :math:`a, b` to ``LOG_EPS`` ensures Bessel small-:math:`z` asymptotics
        handle the degenerate Gamma/InvGamma limits. The default ``jax.grad`` path
        fails because ``jnp.where`` evaluates all branches and
        :math:`\partial\sqrt{ab}/\partial a \to \infty` as :math:`a \to 0`.
        """
        p = theta[0] + 1.0
        b_safe = jnp.maximum(-2.0 * theta[1], LOG_EPS)
        a_safe = jnp.maximum(-2.0 * theta[2], LOG_EPS)
        sqrt_ab = jnp.sqrt(a_safe * b_safe)
        log_sqrt_ba = 0.5 * (jnp.log(b_safe) - jnp.log(a_safe))

        orders = jnp.array([p, p - 1.0, p + 1.0,
                            p - BESSEL_EPS_V, p + BESSEL_EPS_V])
        evals = jax.vmap(log_kv)(orders, jnp.full(5, sqrt_ab))
        L, L_m1, L_p1, L_vm, L_vp = evals

        E_inv_X = jnp.exp(L_m1 - L - log_sqrt_ba)
        E_X     = jnp.exp(L_p1 - L + log_sqrt_ba)
        E_log_X = (L_vp - L_vm) / (2.0 * BESSEL_EPS_V) + log_sqrt_ba

        return jnp.array([E_log_X, E_inv_X, E_X])

    @classmethod
    def _hessian_log_partition(cls, theta: jax.Array) -> jax.Array:
        r"""
        :math:`\nabla^2\psi(\theta)` — analytical 7-Bessel Hessian in :math:`\theta`-space.

        Uses exact Bessel recurrences for :math:`z`-derivatives and finite differences
        for :math:`\nu`-derivatives. 7 :math:`\log K_\nu` evaluations total.

        The mixed derivative :math:`\partial^2\log K_\nu/\partial\nu\partial z` is
        approximated via integer-shift FD reusing evaluations at :math:`p\pm 1, p\pm 2`.

        Note: :math:`H_\theta` may have small negative eigenvalues from this
        approximation; the Newton solver applies ``HESSIAN_DAMPING`` before
        solving so convergence is not affected.

        Valid in the non-degenerate regime (:math:`\sqrt{ab} \gg` ``GIG_DEGEN_THRESHOLD``).
        """
        p = theta[0] + 1.0
        b = jnp.maximum(-2.0 * theta[1], LOG_EPS)
        a = jnp.maximum(-2.0 * theta[2], LOG_EPS)
        z = jnp.sqrt(jnp.maximum(a * b, jnp.finfo(jnp.float64).tiny))

        # 7 Bessel evaluations for all needed derivatives
        orders = jnp.array([p, p - 1.0, p + 1.0, p - 2.0, p + 2.0,
                            p - BESSEL_EPS_V, p + BESSEL_EPS_V])
        evals = jax.vmap(log_kv)(orders, jnp.full(7, z))
        L, L_m1, L_p1, L_m2, L_p2, L_vm, L_vp = evals

        r_m1 = jnp.exp(L_m1 - L)
        r_p1 = jnp.exp(L_p1 - L)
        r_m2 = jnp.exp(L_m2 - L)
        r_p2 = jnp.exp(L_p2 - L)

        L_z  = -0.5 * (r_m1 + r_p1)
        L_zz = 0.25 * (r_m2 + 2.0 + r_p2) - 0.25 * (r_m1 + r_p1) ** 2
        L_vv = (L_vp - 2.0 * L + L_vm) / (BESSEL_EPS_V ** 2)

        # Integer-shift FD for L_vz (reuses p±1, p±2 evaluations)
        L_z_p1 = -0.5 * (jnp.exp(L - L_p1) + jnp.exp(L_p2 - L_p1))
        L_z_m1 = -0.5 * (jnp.exp(L_m2 - L_m1) + jnp.exp(L - L_m1))
        L_vz = (L_z_p1 - L_z_m1) / 2.0

        a_over_z = a / z
        b_over_z = b / z

        H11 = L_vv
        H12 = -L_vz * a_over_z - 1.0 / b
        H13 = -L_vz * b_over_z + 1.0 / a
        H22 = a_over_z ** 2 * L_zz - a_over_z ** 2 / z * L_z - 2.0 * p / b ** 2
        H23 = L_zz + L_z / z
        H33 = b_over_z ** 2 * L_zz - b_over_z ** 2 / z * L_z + 2.0 * p / a ** 2

        return jnp.array([[H11, H12, H13],
                           [H12, H22, H23],
                           [H13, H23, H33]])

    # ------------------------------------------------------------------
    # Tier 3: CPU overrides (numpy + scipy Bessel, no JAX dispatch)
    # ------------------------------------------------------------------

    @classmethod
    def _log_partition_cpu(cls, theta) -> float:
        r""":math:`\psi(\theta)` via numpy + ``log_kv(backend='cpu')``. Accepts numpy or JAX arrays."""
        theta = np.asarray(theta, dtype=np.float64)
        p = theta[0] + 1.0
        b = max(-2.0 * theta[1], 0.0)
        a = max(-2.0 * theta[2], 0.0)
        sqrt_ab = np.sqrt(a * b)

        if sqrt_ab < GIG_DEGEN_THRESHOLD:
            from scipy.special import gammaln
            if b <= a:
                alpha = max(p, TINY)
                beta = max(a / 2.0, TINY)
                return float(gammaln(alpha) - alpha * np.log(beta))
            else:
                alpha = max(-p, TINY)
                beta = max(b / 2.0, TINY)
                return float(gammaln(alpha) - alpha * np.log(beta))

        sqrt_ab_safe = max(sqrt_ab, TINY)
        return float(
            np.log(2.0) + log_kv(p, sqrt_ab_safe, backend='cpu')
            + 0.5 * p * (np.log(max(b, TINY)) - np.log(max(a, TINY)))
        )

    @classmethod
    def _grad_log_partition_cpu(cls, theta) -> np.ndarray:
        r""":math:`\nabla\psi(\theta) = [E[\log X],\; E[1/X],\; E[X]]` via ``scipy.kve``. Pure CPU."""
        theta = np.asarray(theta, dtype=np.float64)
        p = theta[0] + 1.0
        b_safe = max(-2.0 * theta[1], TINY)
        a_safe = max(-2.0 * theta[2], TINY)
        sqrt_ab = np.sqrt(a_safe * b_safe)
        log_sqrt_ba = 0.5 * (np.log(b_safe) - np.log(a_safe))

        log_kp    = float(log_kv(p,         sqrt_ab, backend='cpu'))
        log_kp_m1 = float(log_kv(p - 1.0,  sqrt_ab, backend='cpu'))
        log_kp_p1 = float(log_kv(p + 1.0,  sqrt_ab, backend='cpu'))

        E_inv_X = np.exp(log_kp_m1 - log_kp - log_sqrt_ba)
        E_X     = np.exp(log_kp_p1 - log_kp + log_sqrt_ba)

        log_kp_pe = float(log_kv(p + BESSEL_EPS_V, sqrt_ab, backend='cpu'))
        log_kp_me = float(log_kv(p - BESSEL_EPS_V, sqrt_ab, backend='cpu'))
        E_log_X = (log_kp_pe - log_kp_me) / (2.0 * BESSEL_EPS_V) + log_sqrt_ba

        return np.array([E_log_X, E_inv_X, E_X])

    @classmethod
    def _hessian_log_partition_cpu(cls, theta) -> np.ndarray:
        r""":math:`\nabla^2\psi(\theta)` via central finite differences on ``_log_partition_cpu``."""
        theta = np.asarray(theta, dtype=np.float64)
        n = len(theta)
        H = np.zeros((n, n))
        eps = FD_EPS_FISHER
        f0 = cls._log_partition_cpu(theta)
        for i in range(n):
            for j in range(n):
                if i == j:
                    th_p = theta.copy(); th_p[i] += eps
                    th_m = theta.copy(); th_m[i] -= eps
                    fp = cls._log_partition_cpu(th_p)
                    fm = cls._log_partition_cpu(th_m)
                    H[i, i] = (fp - 2.0 * f0 + fm) / eps ** 2
                elif j > i:
                    th_pp = theta.copy(); th_pp[i] += eps; th_pp[j] += eps
                    th_pm = theta.copy(); th_pm[i] += eps; th_pm[j] -= eps
                    th_mp = theta.copy(); th_mp[i] -= eps; th_mp[j] += eps
                    th_mm = theta.copy(); th_mm[i] -= eps; th_mm[j] -= eps
                    fpp = cls._log_partition_cpu(th_pp)
                    fpm = cls._log_partition_cpu(th_pm)
                    fmp = cls._log_partition_cpu(th_mp)
                    fmm = cls._log_partition_cpu(th_mm)
                    H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * eps ** 2)
        return H

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
        backend='cpu' : vectorized scipy.kve (6 C-level array calls)
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
        """Vectorized CPU path — 6 scipy.kve calls on (N,) arrays."""
        p = np.asarray(p, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        a_safe = np.maximum(a, TINY)
        b_safe = np.maximum(b, TINY)
        sqrt_ab = np.sqrt(a_safe * b_safe)
        log_sqrt_ba = 0.5 * (np.log(b_safe) - np.log(a_safe))

        log_kp    = log_kv(p,         sqrt_ab, backend='cpu')
        log_kp_m1 = log_kv(p - 1.0,  sqrt_ab, backend='cpu')
        log_kp_p1 = log_kv(p + 1.0,  sqrt_ab, backend='cpu')

        E_inv_X = np.exp(log_kp_m1 - log_kp - log_sqrt_ba)
        E_X     = np.exp(log_kp_p1 - log_kp + log_sqrt_ba)

        log_kp_pe = log_kv(p + BESSEL_EPS_V, sqrt_ab, backend='cpu')
        log_kp_me = log_kv(p - BESSEL_EPS_V, sqrt_ab, backend='cpu')
        E_log_X = (log_kp_pe - log_kp_me) / (2.0 * BESSEL_EPS_V) + log_sqrt_ba

        return jnp.column_stack([
            jnp.asarray(E_log_X),
            jnp.asarray(E_inv_X),
            jnp.asarray(E_X),
        ])

    # ------------------------------------------------------------------
    # Moments and sampling
    # ------------------------------------------------------------------

    def mean(self) -> jax.Array:
        r""":math:`E[X] = \eta_3` from expectation parameters."""
        return self.expectation_params()[2]

    def var(self) -> jax.Array:
        r""":math:`\mathrm{Var}[X] = \partial^2\psi/\partial\theta_3^2` = Fisher information [2,2]."""
        return self.fisher_information()[2, 2]

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
        from normix.distributions._gig_rvs import (
            gig_rvs_devroye, gig_build_pinv_table, gig_rvs_pinv,
        )

        if method == "devroye":
            key = jax.random.PRNGKey(seed)
            return gig_rvs_devroye(key, self.p, self.a, self.b, n)

        if method == "pinv":
            key = jax.random.PRNGKey(seed)
            u_grid, x_grid = gig_build_pinv_table(self.p, self.a, self.b)
            return gig_rvs_pinv(key, u_grid, x_grid, n)

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
        import numpy as np
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


# Convenience alias
GIG = GeneralizedInverseGaussian
