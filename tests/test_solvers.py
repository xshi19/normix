"""
Tests for normix/fitting/solvers.py.

Tests the new solve_bregman / solve_bregman_multistart API across:
  - A trivial quadratic f(θ) = ½‖θ‖² (known closed-form solution θ* = η)
  - Gamma distribution (closed-form ψ, verify against from_expectation)
  - GIG distribution (all solver variants and multi-start)
  - Backward-compatible wrappers
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from normix.fitting.solvers import (
    BregmanResult,
    bregman_objective,
    solve_bregman,
    solve_bregman_multistart,
    _setup_reparam,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quadratic_f(theta: jax.Array) -> jax.Array:
    """f(θ) = ½‖θ‖²  →  ∇f(θ) = θ  →  θ* = η."""
    return 0.5 * jnp.dot(theta, theta)


def quadratic_grad(theta_np: np.ndarray) -> np.ndarray:
    """CPU gradient of quadratic_f."""
    return theta_np.copy()


# ---------------------------------------------------------------------------
# BregmanResult
# ---------------------------------------------------------------------------

class TestBregmanResult:

    def test_fields(self):
        r = BregmanResult(
            theta=jnp.array([1.0, 2.0]),
            fun=0.5, grad_norm=1e-12, num_steps=10, converged=True,
        )
        assert r.converged
        assert r.num_steps == 10
        np.testing.assert_allclose(r.theta, [1.0, 2.0])

    def test_immutable(self):
        r = BregmanResult(theta=jnp.zeros(2), fun=0.0,
                          grad_norm=0.0, num_steps=0, converged=True)
        with pytest.raises(Exception):
            r.fun = 1.0  # frozen dataclass


# ---------------------------------------------------------------------------
# bregman_objective
# ---------------------------------------------------------------------------

class TestBregmanObjective:

    def test_quadratic(self):
        theta = jnp.array([1.0, 2.0])
        eta = jnp.array([0.5, 1.5])
        val = bregman_objective(theta, eta, quadratic_f)
        expected = 0.5 * (1.0**2 + 2.0**2) - (1.0 * 0.5 + 2.0 * 1.5)
        np.testing.assert_allclose(float(val), expected, rtol=1e-10)

    def test_minimum_at_eta(self):
        eta = jnp.array([1.0, -1.0])
        val_at_min = bregman_objective(eta, eta, quadratic_f)
        np.testing.assert_allclose(float(val_at_min), -0.5 * float(jnp.dot(eta, eta)),
                                   rtol=1e-10)


# ---------------------------------------------------------------------------
# _setup_reparam
# ---------------------------------------------------------------------------

class TestSetupReparam:

    def test_no_bounds(self):
        theta = jnp.array([1.0, -2.0])
        phi0, to_theta, to_phi = _setup_reparam(theta, None)
        np.testing.assert_allclose(phi0, theta)
        np.testing.assert_allclose(to_theta(phi0), theta)
        np.testing.assert_allclose(to_phi(theta), theta)

    def test_neg_exp_bound(self):
        # (-inf, 0): theta = -exp(phi)
        theta = jnp.array([-0.5, -2.0])
        bounds = [(-np.inf, 0.0), (-np.inf, 0.0)]
        phi0, to_theta, to_phi = _setup_reparam(theta, bounds)
        # round-trip
        theta_rt = to_theta(phi0)
        np.testing.assert_allclose(theta_rt, theta, rtol=1e-10)
        # all reconstructed values should be negative
        assert float(theta_rt[0]) < 0
        assert float(theta_rt[1]) < 0

    def test_pos_exp_bound(self):
        theta = jnp.array([2.0, 3.0])
        bounds = [(0.0, np.inf), (0.0, np.inf)]
        phi0, to_theta, to_phi = _setup_reparam(theta, bounds)
        theta_rt = to_theta(phi0)
        np.testing.assert_allclose(theta_rt, theta, rtol=1e-10)
        assert float(theta_rt[0]) > 0

    def test_mixed_bounds(self):
        theta = jnp.array([1.0, -0.5, -0.3])
        bounds = [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]
        phi0, to_theta, to_phi = _setup_reparam(theta, bounds)
        theta_rt = to_theta(phi0)
        np.testing.assert_allclose(theta_rt, theta, rtol=1e-10)

    def test_bounded_interval(self):
        theta = jnp.array([0.3])
        bounds = [(0.0, 1.0)]
        phi0, to_theta, to_phi = _setup_reparam(theta, bounds)
        theta_rt = to_theta(phi0)
        np.testing.assert_allclose(theta_rt, theta, rtol=1e-6)
        # enforce constraint
        assert 0.0 < float(theta_rt[0]) < 1.0


# ---------------------------------------------------------------------------
# Quadratic tests: known closed-form solution θ* = η
# ---------------------------------------------------------------------------

ETA_QUAD = jnp.array([1.0, -2.0, 0.5])
THETA0_QUAD = jnp.zeros(3)


class TestSolveBregmanQuadratic:
    """For f(θ)=½‖θ‖², min_θ [f(θ)−θ·η] has θ*=η analytically."""

    @pytest.mark.parametrize("method", ["newton", "lbfgs", "bfgs"])
    def test_jax_unconstrained(self, method):
        r = solve_bregman(quadratic_f, ETA_QUAD, THETA0_QUAD,
                          backend="jax", method=method, max_steps=100, tol=1e-8)
        assert isinstance(r, BregmanResult)
        np.testing.assert_allclose(r.theta, ETA_QUAD, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("method", ["lbfgs", "bfgs"])
    def test_cpu_hybrid(self, method):
        r = solve_bregman(quadratic_f, ETA_QUAD, THETA0_QUAD,
                          backend="cpu", method=method, max_steps=200, tol=1e-8)
        np.testing.assert_allclose(r.theta, ETA_QUAD, rtol=1e-5, atol=1e-7)

    def test_cpu_with_grad_fn(self):
        eta = jnp.array([1.0, -2.0])
        theta0 = jnp.zeros(2)
        r = solve_bregman(
            lambda t: 0.5 * jnp.dot(t, t),
            eta, theta0,
            backend="cpu", method="lbfgs", max_steps=200, tol=1e-8,
            grad_fn=lambda t: t.copy(),  # ∇f = θ  for quadratic
        )
        np.testing.assert_allclose(r.theta, eta, rtol=1e-5, atol=1e-7)

    def test_with_bounds_lbfgs(self):
        # η = [−0.5] should be found even with bound (−∞, 0).
        # Start close to the solution in reparameterised space (phi=log(-theta)):
        # theta0=-0.4 → phi0=log(0.4)≈-0.92, optimum at phi*=log(0.5)≈-0.69.
        eta = jnp.array([-0.5])
        theta0 = jnp.array([-0.4])
        bounds = [(-np.inf, 0.0)]
        r = solve_bregman(
            lambda t: 0.5 * jnp.dot(t, t),
            eta, theta0,
            backend="jax", method="lbfgs", bounds=bounds, max_steps=200, tol=1e-8,
        )
        np.testing.assert_allclose(r.theta, eta, rtol=1e-5, atol=1e-7)
        # constraint satisfied
        assert float(r.theta[0]) < 0

    def test_result_fun_near_minimum(self):
        r = solve_bregman(quadratic_f, ETA_QUAD, THETA0_QUAD,
                          backend="jax", method="lbfgs", max_steps=200, tol=1e-10)
        # At θ*=η: f(θ*)−θ*·η = ½‖η‖² − η·η = −½‖η‖²
        expected_fun = -0.5 * float(jnp.dot(ETA_QUAD, ETA_QUAD))
        np.testing.assert_allclose(r.fun, expected_fun, rtol=1e-5)


# ---------------------------------------------------------------------------
# Gamma distribution tests
# ---------------------------------------------------------------------------

class TestSolveBregmanGamma:
    """Gamma has closed-form from_expectation; verify solvers match."""

    def setup_method(self):
        from normix import Gamma
        self.alpha, self.beta = 3.0, 2.0
        g = Gamma(alpha=self.alpha, beta=self.beta)
        self.eta = g.expectation_params()
        self.theta_true = g.natural_params()
        self.f = g._log_partition_from_theta

    @pytest.mark.parametrize("method", ["newton", "lbfgs", "bfgs"])
    def test_jax_backend(self, method):
        # theta[1] must be strictly negative (not 0) due to log(-theta[1]) reparam
        theta0 = jnp.array([0.0, -1.0])
        bounds = [(-np.inf, np.inf), (-np.inf, 0.0)]
        r = solve_bregman(
            self.f, self.eta, theta0,
            backend="jax", method=method, bounds=bounds, max_steps=200, tol=1e-9,
        )
        np.testing.assert_allclose(r.theta, self.theta_true, rtol=1e-5)

    @pytest.mark.parametrize("method", ["lbfgs", "bfgs"])
    def test_cpu_backend(self, method):
        theta0 = jnp.array([0.0, -1.0])
        bounds = [(-np.inf, np.inf), (-np.inf, 0.0)]
        r = solve_bregman(
            self.f, self.eta, theta0,
            backend="cpu", method=method, bounds=bounds, max_steps=300, tol=1e-9,
        )
        np.testing.assert_allclose(r.theta, self.theta_true, rtol=1e-5)


# ---------------------------------------------------------------------------
# GIG distribution tests
# ---------------------------------------------------------------------------

class TestSolveBregmanGIG:
    """GIG is the hard case — tests all solver variants."""

    def setup_method(self):
        from normix import GIG
        self.p, self.a, self.b = 0.5, 1.0, 1.0
        gig = GIG(p=self.p, a=self.a, b=self.b)
        self.eta = gig.expectation_params()
        self.theta_true = gig.natural_params()
        # warm theta0 that's close to truth
        self.theta0 = self.theta_true + jnp.array([0.1, -0.1, -0.1])

    def _check(self, result):
        """Verify GIG round-trip."""
        from normix import GIG
        gig2 = GIG.from_natural(result.theta)
        np.testing.assert_allclose(float(gig2.p), self.p, rtol=1e-4)
        np.testing.assert_allclose(float(gig2.a), self.a, rtol=1e-4)
        np.testing.assert_allclose(float(gig2.b), self.b, rtol=1e-4)

    @pytest.mark.parametrize("backend,method,kwargs", [
        ("jax", "newton", {}),
        ("jax", "newton", {"analytical_hessian": True}),
        ("jax", "lbfgs", {}),
        ("jax", "bfgs", {}),
        ("cpu", "lbfgs", {}),
    ])
    def test_from_expectation_warm_start(self, backend, method, kwargs):
        from normix import GIG
        gig2 = GIG.from_expectation(
            self.eta, theta0=self.theta_true,
            backend=backend, method=method, maxiter=200, **kwargs,
        )
        np.testing.assert_allclose(float(gig2.p), self.p, rtol=1e-4)
        np.testing.assert_allclose(float(gig2.a), self.a, rtol=1e-4)
        np.testing.assert_allclose(float(gig2.b), self.b, rtol=1e-4)

    def test_from_expectation_cold_start(self):
        from normix import GIG
        gig2 = GIG.from_expectation(self.eta)
        np.testing.assert_allclose(float(gig2.p), self.p, rtol=1e-3)

    def test_solve_bregman_jax_newton(self):
        from normix import GIG
        bounds = [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]
        r = solve_bregman(
            GIG._log_partition_from_theta, self.eta, self.theta0,
            backend="jax", method="newton",
            bounds=bounds, max_steps=30, tol=1e-9,
        )
        self._check(r)

    def test_solve_bregman_cpu_hybrid(self):
        from normix import GIG
        bounds = [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]
        r = solve_bregman(
            GIG._log_partition_from_theta, self.eta, self.theta0,
            backend="cpu", method="lbfgs",
            bounds=bounds, max_steps=200, tol=1e-9,
        )
        self._check(r)

    def test_solve_bregman_cpu_pure_grad(self):
        from normix.distributions.generalized_inverse_gaussian import (
            _log_partition_gig_cpu, _gig_cpu_grad,
        )
        bounds = [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]
        r = solve_bregman(
            _log_partition_gig_cpu, self.eta, self.theta0,
            backend="cpu", method="lbfgs",
            bounds=bounds, max_steps=200, tol=1e-9,
            grad_fn=_gig_cpu_grad,
        )
        self._check(r)

    def test_result_converged(self):
        from normix import GIG
        bounds = [(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)]
        r = solve_bregman(
            GIG._log_partition_from_theta, self.eta, self.theta_true,
            backend="jax", method="lbfgs",
            bounds=bounds, max_steps=500, tol=1e-9,
        )
        assert r.converged or r.grad_norm < 1e-6


# ---------------------------------------------------------------------------
# Multi-start tests
# ---------------------------------------------------------------------------

class TestSolveBregmanMultistart:

    def test_jax_newton_vmap(self):
        """vmap multi-start on quadratic — all starts should converge."""
        eta = jnp.array([1.0, -0.5])
        # 4 starting points stacked
        theta0_batch = jnp.array([
            [0.0, 0.0],
            [2.0, -1.0],
            [-1.0, 0.5],
            [0.5, -0.3],
        ])
        r = solve_bregman_multistart(
            quadratic_f, eta, theta0_batch,
            backend="jax", method="newton",
            max_steps=50, tol=1e-8,
        )
        np.testing.assert_allclose(r.theta, eta, rtol=1e-5, atol=1e-7)

    def test_cpu_loop(self):
        """CPU for-loop multi-start on quadratic."""
        eta = jnp.array([1.0, -0.5])
        theta0_list = [jnp.array([0.0, 0.0]), jnp.array([2.0, -1.0])]
        r = solve_bregman_multistart(
            quadratic_f, eta, theta0_list,
            backend="cpu", method="lbfgs",
            max_steps=200, tol=1e-8,
        )
        np.testing.assert_allclose(r.theta, eta, rtol=1e-5, atol=1e-7)

    def test_gig_multistart_cold(self):
        """GIG cold-start multistart via CPU — recovered eta_hat should match eta."""
        from normix import GIG
        from normix.distributions.generalized_inverse_gaussian import (
            _log_partition_gig_cpu, _gig_cpu_grad, _initial_guesses,
        )
        gig = GIG(p=0.5, a=1.0, b=1.0)
        eta = gig.expectation_params()
        s = jnp.sqrt(eta[1] / eta[2])
        geom = jnp.sqrt(eta[1] * eta[2])
        eta_scaled = jnp.array([eta[0] + 0.5 * jnp.log(eta[1] / eta[2]), geom, geom])

        theta0_list = _initial_guesses(eta_scaled)
        processed = [jnp.asarray(t, dtype=jnp.float64) for t in theta0_list]
        r = solve_bregman_multistart(
            _log_partition_gig_cpu, eta_scaled, processed,
            backend="cpu", method="lbfgs",
            bounds=[(-np.inf, np.inf), (-np.inf, 0.0), (-np.inf, 0.0)],
            max_steps=300, tol=1e-9,
            grad_fn=_gig_cpu_grad,
        )
        assert r.grad_norm < 1e-5

    def test_best_selected(self):
        """Verify the multistart returns the start with lowest objective."""
        eta = jnp.array([0.5])
        # Give one good start and one bad start
        good = jnp.array([0.4])
        bad = jnp.array([10.0])
        theta0_list = [bad, good]
        r = solve_bregman_multistart(
            lambda t: 0.5 * jnp.dot(t, t),
            eta, theta0_list,
            backend="cpu", method="lbfgs",
            max_steps=200, tol=1e-8,
        )
        np.testing.assert_allclose(r.theta, eta, rtol=1e-5, atol=1e-7)



# ---------------------------------------------------------------------------
# ExponentialFamily.from_expectation uses solve_bregman
# ---------------------------------------------------------------------------

class TestExponentialFamilyFromExpectation:
    """After removing jaxopt from base class, verify from_expectation still works."""

    def test_gamma_roundtrip(self):
        from normix import Gamma
        g = Gamma(alpha=3.0, beta=2.0)
        eta = g.expectation_params()
        g2 = Gamma.from_expectation(eta)
        np.testing.assert_allclose(float(g2.alpha), 3.0, rtol=1e-5)
        np.testing.assert_allclose(float(g2.beta), 2.0, rtol=1e-5)

    def test_inverse_gaussian_roundtrip(self):
        from normix import InverseGaussian
        ig = InverseGaussian(mu=2.0, lam=4.0)
        eta = ig.expectation_params()
        ig2 = InverseGaussian.from_expectation(eta)
        np.testing.assert_allclose(float(ig2.mu), 2.0, rtol=1e-6)
        np.testing.assert_allclose(float(ig2.lam), 4.0, rtol=1e-6)

    def test_gig_backend_and_method_args(self):
        """GIG from_expectation uses same backend/method args as base class."""
        from normix import GIG
        gig = GIG(p=0.5, a=1.0, b=1.0)
        eta = gig.expectation_params()
        gig2 = GIG.from_expectation(
            eta, backend="jax", method="lbfgs", theta0=gig.natural_params(),
        )
        np.testing.assert_allclose(float(gig2.p), 0.5, rtol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
