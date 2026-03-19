"""
Generalized Inverse Gaussian (GIG) distribution as an exponential family.

PDF: f(x|p,a,b) = (a/b)^{p/2} / (2 K_p(√(ab))) · x^{p-1} · exp(-(ax+b/x)/2)

Exponential family:
  h(x)  = 1
  t(x)  = [log x, 1/x, x]
  θ     = [p-1, -b/2, -a/2]     (θ₂ ≤ 0, θ₃ ≤ 0)
  ψ(θ)  = log 2 + log K_p(√(ab)) + (p/2) log(b/a)
         where p = θ₁+1, a = -2θ₃, b = -2θ₂
  η     = [E[log X], E[1/X], E[X]]

Special cases:
  b→0, p>0:  GIG → Gamma(p, a/2)
  a→0, p<0:  GIG → InverseGamma(-p, b/2)
  p=-1/2:    GIG → InverseGaussian

η→θ optimization uses η-rescaling to reduce Fisher condition number:
  s = √(η₂/η₃),  η̃ = (η₁+½log(η₂/η₃), √(η₂η₃), √(η₂η₃))
  Solve η̃→θ̃ with symmetric GIG (ã=b̃), then unscale.
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
    HESSIAN_DAMPING, THETA_FLOOR, FD_EPS_FISHER,
)
from normix.fitting.solvers import (
    bregman_objective, solve_newton_scan, solve_lbfgs,
    solve_scipy_multistart, solve_cpu_lbfgs,
)

jax.config.update("jax_enable_x64", True)


class GeneralizedInverseGaussian(ExponentialFamily):
    """
    Generalized Inverse Gaussian distribution.

    Stored: p (shape, any real), a > 0, b > 0.
    """

    p: jax.Array
    a: jax.Array
    b: jax.Array

    def __init__(self, p, a, b):
        self.p = jnp.asarray(p, dtype=jnp.float64)
        self.a = jnp.asarray(a, dtype=jnp.float64)
        self.b = jnp.asarray(b, dtype=jnp.float64)

    # ------------------------------------------------------------------
    # Exponential family interface
    # ------------------------------------------------------------------

    def _log_partition_from_theta(self, theta: jax.Array) -> jax.Array:
        """
        ψ(θ) = log 2 + log K_p(√(ab)) + (p/2)(log b − log a)

        Degenerate limit (√(ab) < threshold): delegate to Gamma/InverseGamma.
        All branches use safe clamped values so no NaN gradients from
        non-selected jnp.where branches.
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
        # Safe clamp: alpha must be > 0 for gammaln
        alpha_g = jnp.maximum(p, LOG_EPS)
        beta_g = jnp.maximum(a / 2.0, LOG_EPS)
        psi_gamma = (jax.scipy.special.gammaln(alpha_g)
                     - alpha_g * jnp.log(beta_g))

        # InverseGamma limit (a→0, p<0): InvGamma(-p, b/2)
        # Safe clamp: alpha must be > 0 for gammaln
        alpha_ig = jnp.maximum(-p, LOG_EPS)
        beta_ig = jnp.maximum(b / 2.0, LOG_EPS)
        psi_invgamma = (jax.scipy.special.gammaln(alpha_ig)
                        - alpha_ig * jnp.log(beta_ig))

        # Switch based on which limit applies
        use_degen = sqrt_ab < GIG_DEGEN_THRESHOLD
        use_gamma = use_degen & (b <= a)   # b≈0: Gamma limit
        use_invg = use_degen & (a < b)     # a≈0: InvGamma limit

        return jnp.where(use_gamma, psi_gamma,
               jnp.where(use_invg, psi_invgamma,
                         psi_bessel))

    def natural_params(self) -> jax.Array:
        return jnp.array([self.p - 1.0, -self.b / 2.0, -self.a / 2.0])

    def sufficient_statistics(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.array([jnp.log(x), 1.0 / x, x])

    def log_base_measure(self, x: jax.Array) -> jax.Array:
        return jnp.where(x > 0, jnp.zeros((), jnp.float64), -jnp.inf)

    # ------------------------------------------------------------------
    # expectation_params with backend selection
    # ------------------------------------------------------------------

    def expectation_params(self, backend: str = 'jax') -> jax.Array:
        """
        η = [E[log X], E[1/X], E[X]].

        backend='jax' : jax.grad of log partition (JIT-able, default)
        backend='cpu' : analytical Bessel ratios via scipy.kve (fast)
        """
        if backend == 'cpu':
            return self._expectation_params_cpu()
        return jax.grad(self._log_partition_from_theta)(self.natural_params())

    def _expectation_params_cpu(self) -> jax.Array:
        """Analytical GIG expectations using scipy Bessel (scalar inputs)."""
        p = float(self.p)
        a = float(self.a)
        b = float(self.b)

        a_safe = max(a, TINY)
        b_safe = max(b, TINY)
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

        return jnp.array([E_log_X, E_inv_X, E_X])

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

    def mean(self) -> jax.Array:
        """E[X] = η₃ from expectation parameters."""
        return self.expectation_params()[2]

    def var(self) -> jax.Array:
        """Var[X] = ∂²ψ/∂θ₃² = Fisher information [2,2]."""
        return self.fisher_information()[2, 2]

    def rvs(self, n: int, seed: int = 42) -> "np.ndarray":
        import numpy as np
        from scipy import stats
        b_sp = float(np.sqrt(float(self.a) * float(self.b)))
        scale = float(np.sqrt(float(self.b) / float(self.a)))
        return stats.geninvgauss.rvs(p=float(self.p), b=b_sp, scale=scale,
                                     size=n, random_state=seed)

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
        solver: str = "newton",
        maxiter: int = 500,
        tol: float = 1e-10,
        scan_length: int = 20,
    ) -> "GeneralizedInverseGaussian":
        """
        η → θ via η-rescaling + optimization.

        Rescaling makes the Fisher matrix symmetric (ã = b̃), reducing
        condition number by up to 10^30 for extreme a/b ratios.

        Parameters
        ----------
        theta0 : warm-start point (required for JAX solvers)
        solver : warm-start solver when theta0 is provided:
            'newton'            — JAX Newton, autodiff Hessian via jax.hessian
            'newton_analytical' — JAX Newton, analytical Hessian (7 log_kv calls)
            'lbfgs'             — JAXopt L-BFGS (gradient-only, ~5 log_kv/step)
            'cpu'               — scipy L-BFGS with CPU Bessel ratios (no JAX dispatch)
        scan_length : fixed Newton iteration count for the two Newton solvers
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta1, eta2, eta3 = eta[0], eta[1], eta[2]

        # η-rescaling: s = √(η₂/η₃)
        s = jnp.sqrt(eta2 / eta3)
        geom = jnp.sqrt(eta2 * eta3)
        eta_scaled = jnp.array([eta1 + 0.5 * jnp.log(eta2 / eta3), geom, geom])

        if theta0 is not None:
            theta0 = jnp.asarray(theta0, dtype=jnp.float64)
            theta0_scaled = jnp.array([theta0[0],
                                       theta0[1] * s,
                                       theta0[2] / s])
            theta0_scaled = theta0_scaled.at[1].set(
                jnp.minimum(theta0_scaled[1], THETA_FLOOR))
            theta0_scaled = theta0_scaled.at[2].set(
                jnp.minimum(theta0_scaled[2], THETA_FLOOR))

            if solver == "newton":
                theta_scaled = solve_newton_scan(
                    eta_scaled, theta0_scaled, _log_partition_gig_static,
                    constrained_indices=(1, 2), tol=tol, scan_length=scan_length,
                )
            elif solver == "newton_analytical":
                theta_scaled = solve_newton_scan(
                    eta_scaled, theta0_scaled, _log_partition_gig_static,
                    constrained_indices=(1, 2), tol=tol, scan_length=scan_length,
                    grad_hess_fn=_analytical_grad_hess_phi,
                )
            elif solver == "lbfgs":
                theta_scaled = solve_lbfgs(
                    eta_scaled, theta0_scaled, _log_partition_gig_static,
                    constrained_indices=(1, 2), maxiter=maxiter, tol=tol,
                )
            elif solver == "cpu":
                theta_scaled = solve_cpu_lbfgs(
                    eta_scaled, theta0_scaled, _cpu_objective_and_grad,
                    bounds=[(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)],
                    maxiter=500, tol=tol,
                )
            else:
                raise ValueError(f"Unknown solver: {solver!r}. "
                                 "Choose 'newton', 'newton_analytical', 'lbfgs', or 'cpu'.")
        else:
            # Cold-start: scipy multi-start for robustness
            theta0_list = _initial_guesses(eta_scaled)
            theta_scaled = solve_scipy_multistart(
                eta_scaled, theta0_list, _log_partition_gig_static,
                bounds=[(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)],
                theta_floor={1: THETA_FLOOR, 2: THETA_FLOOR},
                maxiter=maxiter, tol=tol,
            )

        # Unscale: θ₂ = θ̃₂/s,  θ₃ = s·θ̃₃
        theta = jnp.array([theta_scaled[0],
                           theta_scaled[1] / s,
                           s * theta_scaled[2]])
        return cls.from_natural(theta)

    @classmethod
    def fit_mle(
        cls,
        X: jax.Array,
        *,
        theta0=None,
        maxiter: int = 500,
        tol: float = 1e-10,
    ) -> "GeneralizedInverseGaussian":
        X = jnp.asarray(X, dtype=jnp.float64)
        eta_hat = jnp.array([jnp.mean(jnp.log(X)),
                              jnp.mean(1.0 / X),
                              jnp.mean(X)])
        return cls.from_expectation(eta_hat, theta0=theta0, maxiter=maxiter, tol=tol)

    @classmethod
    def _dummy_instance(cls) -> "GeneralizedInverseGaussian":
        return cls(p=jnp.ones(()), a=jnp.ones(()), b=jnp.ones(()))

    def fisher_information(self) -> jax.Array:
        """
        Numerical Fisher information matrix I(θ) = ∇²ψ(θ) via finite differences.

        The analytical hessian requires second-order derivatives through log_kv
        w.r.t. the order v, which involves differentiating the pure_callback.
        We use central FD instead for robustness.
        """
        import numpy as np
        theta = np.array(self.natural_params())
        n = len(theta)
        H = np.zeros((n, n))
        eps = FD_EPS_FISHER
        for i in range(n):
            for j in range(n):
                if i == j:
                    th_p = theta.copy(); th_p[i] += eps
                    th_m = theta.copy(); th_m[i] -= eps
                    f0 = float(self._log_partition_from_theta(jnp.array(theta)))
                    fp = float(self._log_partition_from_theta(jnp.array(th_p)))
                    fm = float(self._log_partition_from_theta(jnp.array(th_m)))
                    H[i, i] = (fp - 2*f0 + fm) / eps**2
                elif j > i:
                    th_pp = theta.copy(); th_pp[i] += eps; th_pp[j] += eps
                    th_pm = theta.copy(); th_pm[i] += eps; th_pm[j] -= eps
                    th_mp = theta.copy(); th_mp[i] -= eps; th_mp[j] += eps
                    th_mm = theta.copy(); th_mm[i] -= eps; th_mm[j] -= eps
                    fpp = float(self._log_partition_from_theta(jnp.array(th_pp)))
                    fpm = float(self._log_partition_from_theta(jnp.array(th_pm)))
                    fmp = float(self._log_partition_from_theta(jnp.array(th_mp)))
                    fmm = float(self._log_partition_from_theta(jnp.array(th_mm)))
                    H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
        return jnp.array(H)

    @classmethod
    def _theta_bounds(cls):
        # θ₁ unbounded, θ₂ ≤ 0, θ₃ ≤ 0
        lower = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf])
        upper = jnp.array([jnp.inf, 0.0, 0.0])
        return (lower, upper)


# ---------------------------------------------------------------------------
# GIG-specific optimization helpers
# ---------------------------------------------------------------------------

def _log_partition_gig_static(theta: jax.Array) -> jax.Array:
    """Stateless ψ_GIG(θ) — used as the log-partition function in solvers."""
    dummy = GeneralizedInverseGaussian(p=jnp.ones(()), a=jnp.ones(()), b=jnp.ones(()))
    return dummy._log_partition_from_theta(theta)


# ---------------------------------------------------------------------------
# GIG analytical gradient and Hessian (7 log_kv calls per step)
# ---------------------------------------------------------------------------

def _gig_bessel_quantities(p: jax.Array, z: jax.Array):
    """Compute log K_v and its first/second derivatives at order p, argument z.

    Uses 7 batched log_kv evaluations:
        [p, p−1, p+1, p−2, p+2, p−ε, p+ε]

    Returns (L, L_z, L_zz, L_v, L_vv, L_vz) where:
      L    = log K_p(z)
      L_z  = ∂L/∂z  = −(K_{p−1}+K_{p+1})/(2K_p)            (exact recurrence)
      L_zz = ∂²L/∂z² = ¼(r_{p−2}+2+r_{p+2}) − ¼(r_{p−1}+r_{p+1})²  (exact)
      L_v  = ∂L/∂v  ≈ (L(p+ε) − L(p−ε)) / (2ε)              (FD)
      L_vv = ∂²L/∂v² ≈ (L(p+ε) − 2L + L(p−ε)) / ε²          (FD)
      L_vz = ∂²L/∂v∂z ≈ (L_z(p+1) − L_z(p−1)) / 2           (integer shift FD)
    """
    z_safe = jnp.maximum(z, jnp.finfo(jnp.float64).tiny)
    orders = jnp.array([p, p - 1.0, p + 1.0, p - 2.0, p + 2.0,
                        p - BESSEL_EPS_V, p + BESSEL_EPS_V])
    evals = jax.vmap(log_kv)(orders, jnp.full(7, z_safe))
    L, L_m1, L_p1, L_m2, L_p2, L_vm, L_vp = evals

    r_m1 = jnp.exp(L_m1 - L)
    r_p1 = jnp.exp(L_p1 - L)
    r_m2 = jnp.exp(L_m2 - L)
    r_p2 = jnp.exp(L_p2 - L)

    L_z  = -0.5 * (r_m1 + r_p1)
    L_zz = 0.25 * (r_m2 + 2.0 + r_p2) - 0.25 * (r_m1 + r_p1) ** 2

    L_v  = (L_vp - L_vm) / (2.0 * BESSEL_EPS_V)
    L_vv = (L_vp - 2.0 * L + L_vm) / (BESSEL_EPS_V ** 2)

    L_z_p1 = -0.5 * (jnp.exp(L - L_p1) + jnp.exp(L_p2 - L_p1))
    L_z_m1 = -0.5 * (jnp.exp(L_m2 - L_m1) + jnp.exp(L - L_m1))
    L_vz = (L_z_p1 - L_z_m1) / 2.0

    return L, L_z, L_zz, L_v, L_vv, L_vz


def _analytical_grad_hess_phi(phi: jax.Array, eta: jax.Array):
    """Gradient and Hessian of f(φ) = ψ(θ(φ)) − θ·η in the reparametrised space.

    Reparametrisation: θ₁ = φ₁, θ₂ = −exp(φ₂), θ₃ = −exp(φ₃)

    Uses analytical Fisher information ∇²_θ ψ via exact Bessel recurrences
    for z-derivatives and FD for p (order of K_v).

    Only 7 log_kv evaluations per call, vs ~25 for jax.hessian.
    """
    theta = jnp.array([phi[0], -jnp.exp(phi[1]), -jnp.exp(phi[2])])
    p = theta[0] + 1.0
    b = -2.0 * theta[1]
    a = -2.0 * theta[2]
    z = jnp.sqrt(jnp.maximum(a * b, jnp.finfo(jnp.float64).tiny))

    _, L_z, L_zz, L_v, L_vv, L_vz = _gig_bessel_quantities(p, z)

    a_over_z = a / z
    b_over_z = b / z

    eta_tilde_1 = L_v + 0.5 * (jnp.log(b) - jnp.log(a))
    eta_tilde_2 = -L_z * a_over_z - p / b
    eta_tilde_3 = -L_z * b_over_z + p / a

    g_theta = jnp.array([eta_tilde_1 - eta[0],
                          eta_tilde_2 - eta[1],
                          eta_tilde_3 - eta[2]])

    j = jnp.array([1.0, theta[1], theta[2]])
    g_phi = g_theta * j

    H11 = L_vv
    H12 = -L_vz * a_over_z - 1.0 / b
    H13 = -L_vz * b_over_z + 1.0 / a
    H22 = a_over_z ** 2 * L_zz - a_over_z ** 2 / z * L_z - 2.0 * p / b ** 2
    H23 = L_zz + L_z / z
    H33 = b_over_z ** 2 * L_zz - b_over_z ** 2 / z * L_z + 2.0 * p / a ** 2

    H_theta = jnp.array([[H11, H12, H13],
                          [H12, H22, H23],
                          [H13, H23, H33]])

    H_phi = H_theta * jnp.outer(j, j)
    H_phi = H_phi.at[1, 1].add(g_theta[1] * theta[1])
    H_phi = H_phi.at[2, 2].add(g_theta[2] * theta[2])

    return g_phi, H_phi


# ---------------------------------------------------------------------------
# CPU-side objective+gradient for GIG (scipy Bessel, no JAX dispatch)
# ---------------------------------------------------------------------------

def _cpu_objective_and_grad(theta_np, eta_np):
    """Objective and gradient via CPU Bessel ratios (scipy.kve).

    Used by ``solve_cpu_lbfgs`` in the general solver module.
    """
    p = theta_np[0] + 1.0
    b_safe = max(-2.0 * theta_np[1], TINY)
    a_safe = max(-2.0 * theta_np[2], TINY)
    sqrt_ab = np.sqrt(a_safe * b_safe)

    lkv = float(log_kv(p, sqrt_ab, backend='cpu'))
    psi = (np.log(2.0) + lkv
           + 0.5 * p * (np.log(b_safe) - np.log(a_safe)))
    obj = float(psi - np.dot(theta_np, eta_np))

    gig_tmp = GeneralizedInverseGaussian(p=p, a=a_safe, b=b_safe)
    eta_hat = np.asarray(gig_tmp._expectation_params_cpu())
    grad = eta_hat - eta_np
    return obj, grad


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
    eps = 1e-4

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
