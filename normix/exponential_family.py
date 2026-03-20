"""
ExponentialFamily base class for JAX normix.

Subclasses implement four abstract methods:
  _log_partition_from_theta(theta) → scalar
  natural_params()                 → 1-D array θ
  sufficient_statistics(x)         → 1-D array t(x)
  log_base_measure(x)              → scalar

The Log-Partition Triad
-----------------------
Each subclass gets six derived classmethods, grouped as three pairs:

    JAX (JIT-able)                   CPU (numpy/scipy)
    ──────────────                   ─────────────────
    _log_partition_from_theta        _log_partition_cpu
    _grad_log_partition              _grad_log_partition_cpu
    _hessian_log_partition           _hessian_log_partition_cpu

Tier 1 (_log_partition_from_theta) is abstract; all others have defaults.
Tier 2 (JAX grad/hessian) defaults to jax.grad / jax.hessian; subclasses
  override with analytical formulas when available.
Tier 3 (CPU) defaults to wrapping the JAX versions; Bessel-dependent
  distributions (GIG) override with native numpy/scipy implementations.

Everything else is derived automatically:
  log_partition()       = ψ(θ)
  expectation_params()  = ∇ψ(θ)   via _grad_log_partition
  fisher_information()  = ∇²ψ(θ)  via _hessian_log_partition
  log_prob(x)           = log h(x) + t(x)·θ − ψ(θ)

Constructors:
  from_natural(theta)     → subclass instance
  from_expectation(eta)   → solved via solve_bregman
  fit_mle(X)              → from_expectation(mean t(X))
"""
from __future__ import annotations

import abc
from typing import Optional

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class ExponentialFamily(eqx.Module):
    """
    Abstract base class for exponential family distributions.

    Concrete subclasses must implement:
        _log_partition_from_theta, natural_params,
        sufficient_statistics, log_base_measure
    """

    # ------------------------------------------------------------------
    # Tier 1: Abstract (subclass MUST implement)
    # ------------------------------------------------------------------

    @staticmethod
    @abc.abstractmethod
    def _log_partition_from_theta(theta: jax.Array) -> jax.Array:
        """ψ(θ) — single source of truth for the log-partition function."""

    @abc.abstractmethod
    def natural_params(self) -> jax.Array:
        """θ from stored classical parameters."""

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(x: jax.Array) -> jax.Array:
        """t(x) for a *single* unbatched observation."""

    @staticmethod
    @abc.abstractmethod
    def log_base_measure(x: jax.Array) -> jax.Array:
        """log h(x) for a *single* unbatched observation."""

    # ------------------------------------------------------------------
    # Tier 2: JAX grad & Hessian (override with analytical formulas)
    # ------------------------------------------------------------------

    @classmethod
    def _grad_log_partition(cls, theta: jax.Array) -> jax.Array:
        """∇ψ(θ). Default: jax.grad."""
        return jax.grad(cls._log_partition_from_theta)(theta)

    @classmethod
    def _hessian_log_partition(cls, theta: jax.Array) -> jax.Array:
        """∇²ψ(θ) = I(θ). Default: jax.hessian."""
        return jax.hessian(cls._log_partition_from_theta)(theta)

    # ------------------------------------------------------------------
    # Tier 3: CPU versions (override with native numpy/scipy for Bessel)
    # ------------------------------------------------------------------

    @classmethod
    def _log_partition_cpu(cls, theta) -> float:
        """ψ(θ) on CPU. Default: wraps JAX version."""
        return float(cls._log_partition_from_theta(
            jnp.asarray(theta, dtype=jnp.float64)))

    @classmethod
    def _grad_log_partition_cpu(cls, theta) -> np.ndarray:
        """∇ψ(θ) on CPU. Default: wraps JAX grad."""
        return np.asarray(cls._grad_log_partition(
            jnp.asarray(theta, dtype=jnp.float64)))

    @classmethod
    def _hessian_log_partition_cpu(cls, theta) -> np.ndarray:
        """∇²ψ(θ) on CPU. Default: wraps JAX hessian."""
        return np.asarray(cls._hessian_log_partition(
            jnp.asarray(theta, dtype=jnp.float64)))

    # ------------------------------------------------------------------
    # Derived quantities — via triad
    # ------------------------------------------------------------------

    def log_partition(self) -> jax.Array:
        """ψ(θ) at current parameters."""
        return type(self)._log_partition_from_theta(self.natural_params())

    def expectation_params(self, backend: str = 'jax') -> jax.Array:
        """η = ∇ψ(θ).

        Parameters
        ----------
        backend : 'jax' (default, JIT-able) or 'cpu' (numpy/scipy)
        """
        theta = self.natural_params()
        if backend == 'cpu':
            return jnp.asarray(
                type(self)._grad_log_partition_cpu(np.asarray(theta)))
        return type(self)._grad_log_partition(theta)

    def fisher_information(self, backend: str = 'jax') -> jax.Array:
        """I(θ) = ∇²ψ(θ).

        Parameters
        ----------
        backend : 'jax' (default, JIT-able) or 'cpu' (numpy/scipy)
        """
        theta = self.natural_params()
        if backend == 'cpu':
            return jnp.asarray(
                type(self)._hessian_log_partition_cpu(np.asarray(theta)))
        return type(self)._hessian_log_partition(theta)

    def log_prob(self, x: jax.Array) -> jax.Array:
        """log p(x|θ) = log h(x) + θᵀt(x) − ψ(θ), single observation."""
        cls = type(self)
        theta = self.natural_params()
        return (cls.log_base_measure(x)
                + jnp.dot(cls.sufficient_statistics(x), theta)
                - cls._log_partition_from_theta(theta))

    def pdf(self, x: jax.Array) -> jax.Array:
        """p(x|θ), single observation. Batch via jax.vmap."""
        return jnp.exp(self.log_prob(x))

    def mean(self) -> jax.Array:
        """E[X]. Subclasses should override with analytical formulas."""
        raise NotImplementedError(f"{type(self).__name__}.mean not implemented")

    def var(self) -> jax.Array:
        """Var[X]. Subclasses should override with analytical formulas."""
        raise NotImplementedError(f"{type(self).__name__}.var not implemented")

    def std(self) -> jax.Array:
        """Std[X] = √Var[X]."""
        return jnp.sqrt(self.var())

    def cdf(self, x: jax.Array) -> jax.Array:
        """CDF F(x). Subclasses should override with analytical formulas."""
        raise NotImplementedError(f"{type(self).__name__}.cdf not implemented")

    def rvs(self, n: int, seed: int = 42) -> "np.ndarray":
        """Sample n observations. Uses numpy/scipy (not JIT-able)."""
        raise NotImplementedError(f"{type(self).__name__}.rvs not implemented")

    # ------------------------------------------------------------------
    # Constructors — classmethod stubs; subclasses override as needed
    # ------------------------------------------------------------------

    @classmethod
    def from_natural(cls, theta: jax.Array) -> "ExponentialFamily":
        """Construct from natural parameters θ. Subclasses must override."""
        raise NotImplementedError(f"{cls.__name__}.from_natural not implemented")

    @classmethod
    def bregman_divergence(
        cls,
        theta: jax.Array,
        eta: jax.Array,
    ) -> jax.Array:
        """Bregman divergence ψ(θ) − θ·η (conjugate dual).

        Minimising over θ yields ∇ψ(θ*) = η, i.e. the natural parameters
        corresponding to expectation parameters η.
        """
        return cls._log_partition_from_theta(theta) - jnp.dot(theta, eta)

    @classmethod
    def from_expectation(
        cls,
        eta: jax.Array,
        *,
        theta0: Optional[jax.Array] = None,
        maxiter: int = 500,
        tol: float = 1e-10,
        backend: str = "jax",
        method: str = "lbfgs",
    ) -> "ExponentialFamily":
        """
        Construct from expectation parameters η by solving ∇ψ(θ) = η.

        Minimises the Bregman divergence ψ(θ) − θ·η via solve_bregman.
        Subclasses can override for closed-form inverses.

        Parameters
        ----------
        backend : 'jax' (default, JIT-able) or 'cpu' (scipy, more robust)
        method  : 'lbfgs' (default), 'bfgs', or 'newton'
        """
        from normix.fitting.solvers import solve_bregman

        eta = jnp.asarray(eta, dtype=jnp.float64)
        if theta0 is None:
            theta0 = cls._init_theta_from_eta(eta)

        jax_bounds = cls._theta_bounds()
        if jax_bounds is not None:
            lower, upper = jax_bounds
            bounds = [(float(lo), float(hi)) for lo, hi in zip(lower, upper)]
        else:
            bounds = None

        if backend == 'cpu':
            f = cls._log_partition_cpu
            grad_fn = cls._grad_log_partition_cpu
            hess_fn = cls._hessian_log_partition_cpu
        else:
            f = cls._log_partition_from_theta
            grad_fn = cls._grad_log_partition
            hess_fn = cls._hessian_log_partition

        result = solve_bregman(
            f, eta, theta0,
            backend=backend, method=method,
            bounds=bounds, max_steps=maxiter, tol=tol,
            grad_fn=grad_fn, hess_fn=hess_fn,
        )
        return cls.from_natural(result.theta)

    @classmethod
    def fit_mle(
        cls,
        X: jax.Array,
        *,
        theta0: Optional[jax.Array] = None,
        maxiter: int = 500,
        tol: float = 1e-10,
    ) -> "ExponentialFamily":
        """
        MLE via exponential family identity: η̂ = mean_i t(xᵢ).

        Batches over X using jax.vmap, then calls from_expectation(η̂).

        Parameters
        ----------
        X : (n, ...) array of observations
        theta0 : optional initial natural parameters θ₀ for the η→θ solver
        maxiter : maximum iterations for the η→θ solver
        tol : convergence tolerance for the η→θ solver
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        stats = jax.vmap(cls.sufficient_statistics)(X)   # (n, dim_t)
        eta_hat = jnp.mean(stats, axis=0)
        return cls.from_expectation(eta_hat, theta0=theta0, maxiter=maxiter, tol=tol)

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
