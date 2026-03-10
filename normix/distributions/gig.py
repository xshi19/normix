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
import jaxopt

from normix._bessel import log_kv
from normix.exponential_family import ExponentialFamily

jax.config.update("jax_enable_x64", True)

# When √(ab) < threshold, delegate to Gamma or InverseGamma limits
_DEGEN_THRESHOLD = 1e-10
_EPS = 1e-30


class GIG(ExponentialFamily):
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
        sqrt_ab_safe = jnp.maximum(sqrt_ab, _EPS)

        # General Bessel case
        psi_bessel = (jnp.log(2.0) + log_kv(p, sqrt_ab_safe)
                      + 0.5 * p * (jnp.log(b + _EPS) - jnp.log(a + _EPS)))

        # Gamma limit (b→0, p>0): Gamma(p, a/2)
        # Safe clamp: alpha must be > 0 for gammaln
        alpha_g = jnp.maximum(p, _EPS)
        beta_g = jnp.maximum(a / 2.0, _EPS)
        psi_gamma = (jax.scipy.special.gammaln(alpha_g)
                     - alpha_g * jnp.log(beta_g))

        # InverseGamma limit (a→0, p<0): InvGamma(-p, b/2)
        # Safe clamp: alpha must be > 0 for gammaln
        alpha_ig = jnp.maximum(-p, _EPS)
        beta_ig = jnp.maximum(b / 2.0, _EPS)
        psi_invgamma = (jax.scipy.special.gammaln(alpha_ig)
                        - alpha_ig * jnp.log(beta_ig))

        # Switch based on which limit applies
        use_degen = sqrt_ab < _DEGEN_THRESHOLD
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
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "GIG":
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
    ) -> "GIG":
        """
        η → θ via η-rescaling + jaxopt.LBFGSB.

        Rescaling makes the Fisher matrix symmetric (ã = b̃), reducing
        condition number by up to 10^30 for extreme a/b ratios.
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        eta1, eta2, eta3 = eta[0], eta[1], eta[2]

        # η-rescaling: s = √(η₂/η₃)
        s = jnp.sqrt(eta2 / eta3)
        geom = jnp.sqrt(eta2 * eta3)
        eta_scaled = jnp.array([eta1 + 0.5 * jnp.log(eta2 / eta3), geom, geom])

        # Multi-start: initial guesses from special cases
        theta0_list = _initial_guesses(eta_scaled)

        # Solve the scaled (symmetric) problem
        theta_scaled = _solve_eta_to_theta(eta_scaled, theta0_list, maxiter, tol)

        # Unscale: θ₂ = θ̃₂/s,  θ₃ = s·θ̃₃
        theta = jnp.array([theta_scaled[0],
                           theta_scaled[1] / s,
                           s * theta_scaled[2]])
        return cls.from_natural(theta)

    @classmethod
    def fit_mle(cls, X: jax.Array) -> "GIG":
        X = jnp.asarray(X, dtype=jnp.float64)
        eta_hat = jnp.array([jnp.mean(jnp.log(X)),
                              jnp.mean(1.0 / X),
                              jnp.mean(X)])
        return cls.from_expectation(eta_hat)

    @classmethod
    def _dummy_instance(cls) -> "GIG":
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
        eps = 1e-4
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
# Optimization helpers
# ---------------------------------------------------------------------------

def _log_partition_gig_static(theta: jax.Array) -> jax.Array:
    """Stateless ψ_GIG(θ) — used directly in the objective."""
    dummy = GIG(p=jnp.ones(()), a=jnp.ones(()), b=jnp.ones(()))
    return dummy._log_partition_from_theta(theta)


def _objective(theta: jax.Array, eta: jax.Array) -> jax.Array:
    return _log_partition_gig_static(theta) - jnp.dot(theta, eta)


def _solve_eta_to_theta(
    eta: jax.Array,
    theta0_list: list,
    maxiter: int,
    tol: float,
) -> jax.Array:
    """
    Multi-start LBFGSB: try each theta0, return best solution.
    Runs in pure Python (not JIT'd) since multi-start is control-flow heavy.
    """
    import numpy as np
    eta_np = np.array(eta)

    def objective_np(theta_np):
        theta = jnp.asarray(theta_np, dtype=jnp.float64)
        val = _objective(theta, jnp.asarray(eta_np, dtype=jnp.float64))
        return float(val)

    from scipy.optimize import minimize

    lower = np.array([-np.inf, -np.inf, -np.inf])
    upper = np.array([np.inf, 0.0, 0.0])
    bounds = list(zip(lower, upper))

    best_theta = None
    best_val = np.inf

    for t0 in theta0_list:
        t0_np = np.array(t0)
        t0_np[1] = min(t0_np[1], -1e-8)
        t0_np[2] = min(t0_np[2], -1e-8)
        try:
            res = minimize(
                fun=objective_np,
                x0=t0_np,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': tol**2, 'gtol': tol},
            )
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_theta = res.x
        except Exception:
            pass

    if best_theta is None:
        best_theta = np.array(theta0_list[0])
        best_theta[1] = min(best_theta[1], -1e-8)
        best_theta[2] = min(best_theta[2], -1e-8)

    return jnp.asarray(best_theta, dtype=jnp.float64)


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
GeneralizedInverseGaussian = GIG
