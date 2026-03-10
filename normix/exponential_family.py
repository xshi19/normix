"""
ExponentialFamily base class for JAX normix.

Subclasses implement four abstract methods:
  _log_partition_from_theta(theta) → scalar
  natural_params()                 → 1-D array θ
  sufficient_statistics(x)         → 1-D array t(x)
  log_base_measure(x)              → scalar

Everything else is derived automatically:
  log_partition()       = ψ(θ)
  expectation_params()  = ∇ψ(θ)   via jax.grad
  fisher_information()  = ∇²ψ(θ)  via jax.hessian
  log_prob(x)           = log h(x) + t(x)·θ − ψ(θ)

Constructors:
  from_natural(theta)     → subclass instance
  from_expectation(eta)   → solved via jaxopt.LBFGSB
  fit_mle(X)              → from_expectation(mean t(X))
"""
from __future__ import annotations

import abc
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxopt

jax.config.update("jax_enable_x64", True)


class ExponentialFamily(eqx.Module):
    """
    Abstract base class for exponential family distributions.

    Concrete subclasses must implement:
        _log_partition_from_theta, natural_params,
        sufficient_statistics, log_base_measure
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement these
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _log_partition_from_theta(self, theta: jax.Array) -> jax.Array:
        """ψ(θ) — single source of truth for the log-partition function."""

    @abc.abstractmethod
    def natural_params(self) -> jax.Array:
        """θ from stored classical parameters."""

    @abc.abstractmethod
    def sufficient_statistics(self, x: jax.Array) -> jax.Array:
        """t(x) for a *single* unbatched observation."""

    @abc.abstractmethod
    def log_base_measure(self, x: jax.Array) -> jax.Array:
        """log h(x) for a *single* unbatched observation."""

    # ------------------------------------------------------------------
    # Derived quantities — automatic via JAX autodiff
    # ------------------------------------------------------------------

    def log_partition(self) -> jax.Array:
        """ψ(θ) at current parameters."""
        return self._log_partition_from_theta(self.natural_params())

    def expectation_params(self) -> jax.Array:
        """η = ∇ψ(θ) via jax.grad."""
        return jax.grad(self._log_partition_from_theta)(self.natural_params())

    def fisher_information(self) -> jax.Array:
        """I(θ) = ∇²ψ(θ) via jax.hessian."""
        return jax.hessian(self._log_partition_from_theta)(self.natural_params())

    def log_prob(self, x: jax.Array) -> jax.Array:
        """log p(x|θ) = log h(x) + θᵀt(x) − ψ(θ), single observation."""
        theta = self.natural_params()
        return (self.log_base_measure(x)
                + jnp.dot(self.sufficient_statistics(x), theta)
                - self._log_partition_from_theta(theta))

    # ------------------------------------------------------------------
    # Constructors — classmethod stubs; subclasses override as needed
    # ------------------------------------------------------------------

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "ExponentialFamily":
        """Construct from natural parameters θ. Subclasses must override."""
        raise NotImplementedError(f"{cls.__name__}.from_natural not implemented")

    @classmethod
    def from_expectation(
        cls,
        eta: jax.Array,
        *,
        theta0: Optional[jax.Array] = None,
        maxiter: int = 500,
        tol: float = 1e-10,
    ) -> "ExponentialFamily":
        """
        Construct from expectation parameters η by solving ∇ψ(θ) = η.

        Uses jaxopt.LBFGSB to minimise ψ(θ) − θ·η (the conjugate dual).
        Subclasses can override for closed-form inverses.
        """
        eta = jnp.asarray(eta, dtype=jnp.float64)
        bounds = cls._theta_bounds()

        def objective(theta: jax.Array) -> jax.Array:
            # minimise ψ(θ) − θ·η  (convex, minimum at ∇ψ=η)
            dummy = cls.from_natural(theta)
            return dummy._log_partition_from_theta(theta) - jnp.dot(theta, eta)

        if theta0 is None:
            theta0 = cls._init_theta_from_eta(eta)

        if bounds is not None:
            solver = jaxopt.LBFGSB(fun=objective, maxiter=maxiter, tol=tol)
            result = solver.run(theta0, bounds=bounds)
        else:
            solver = jaxopt.LBFGS(fun=objective, maxiter=maxiter, tol=tol)
            result = solver.run(theta0)

        return cls.from_natural(result.params)

    @classmethod
    def fit_mle(cls, X: jax.Array) -> "ExponentialFamily":
        """
        MLE via exponential family identity: η̂ = mean_i t(xᵢ).

        Batches over X using jax.vmap, then calls from_expectation(η̂).
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        # vmap sufficient_statistics over the batch dimension
        dummy = cls._dummy_instance()
        t_fn = lambda x: dummy.sufficient_statistics(x)
        stats = jax.vmap(t_fn)(X)   # (n, dim_t)
        eta_hat = jnp.mean(stats, axis=0)
        return cls.from_expectation(eta_hat)

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def _theta_bounds(cls):
        """
        Bounds for θ used in LBFGSB.  Return None for unconstrained.
        Subclasses override to provide (lower, upper) tuple of arrays.
        """
        return None

    @classmethod
    def _init_theta_from_eta(cls, eta: jax.Array) -> jax.Array:
        """Initial θ guess from η.  Default: zero vector. Override in subclasses."""
        return jnp.zeros_like(eta)

    @classmethod
    def _dummy_instance(cls) -> "ExponentialFamily":
        """
        Return a dummy instance for calling instance methods (e.g. sufficient_statistics)
        without concrete parameters.  Subclasses may override if needed.
        """
        raise NotImplementedError(
            f"{cls.__name__}._dummy_instance — override or use fit_mle with a real instance"
        )
