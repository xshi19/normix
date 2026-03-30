"""
ExponentialFamily base class for JAX normix.

Subclasses implement four abstract methods:

- ``_log_partition_from_theta(theta)`` → scalar :math:`\\psi(\\theta)`
- ``natural_params()`` → 1-D array :math:`\\theta`
- ``sufficient_statistics(x)`` → 1-D array :math:`t(x)`
- ``log_base_measure(x)`` → scalar :math:`\\log h(x)`

**Log-Partition Triad**

Each subclass gets six derived classmethods, grouped as three pairs::

    JAX (JIT-able)                 CPU (numpy/scipy)
    ─────────────────────────────  ─────────────────────────────
    _log_partition_from_theta      _log_partition_cpu
    _grad_log_partition            _grad_log_partition_cpu
    _hessian_log_partition         _hessian_log_partition_cpu

Tier 1 (``_log_partition_from_theta``) is abstract; all others have defaults.
Tier 2 defaults to ``jax.grad`` / ``jax.hessian``; subclasses override with
analytical formulas when available.
Tier 3 defaults to wrapping Tier 2; GIG overrides with native numpy/scipy.

**Derived automatically:**

.. math::

    \\eta = \\nabla\\psi(\\theta), \\quad
    I(\\theta) = \\nabla^2\\psi(\\theta), \\quad
    \\log p(x \\mid \\theta) = \\log h(x) + \\theta^\\top t(x) - \\psi(\\theta)
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
        r""":math:`\psi(\theta)` — single source of truth for the log-partition function."""

    @abc.abstractmethod
    def natural_params(self) -> jax.Array:
        r""":math:`\theta` from stored classical parameters."""

    @staticmethod
    @abc.abstractmethod
    def sufficient_statistics(x: jax.Array) -> jax.Array:
        r""":math:`t(x)` for a *single* unbatched observation."""

    @staticmethod
    @abc.abstractmethod
    def log_base_measure(x: jax.Array) -> jax.Array:
        r""":math:`\log h(x)` for a *single* unbatched observation."""

    # ------------------------------------------------------------------
    # Tier 2: JAX grad & Hessian (override with analytical formulas)
    # ------------------------------------------------------------------

    @classmethod
    def _grad_log_partition(cls, theta: jax.Array) -> jax.Array:
        r""":math:`\nabla\psi(\theta)`. Default: ``jax.grad``."""
        return jax.grad(cls._log_partition_from_theta)(theta)

    @classmethod
    def _hessian_log_partition(cls, theta: jax.Array) -> jax.Array:
        r""":math:`\nabla^2\psi(\theta) = I(\theta)`. Default: ``jax.hessian``."""
        return jax.hessian(cls._log_partition_from_theta)(theta)

    # ------------------------------------------------------------------
    # Tier 3: CPU versions (override with native numpy/scipy for Bessel)
    # ------------------------------------------------------------------

    @classmethod
    def _log_partition_cpu(cls, theta) -> float:
        r""":math:`\psi(\theta)` on CPU. Default: wraps JAX version."""
        return float(cls._log_partition_from_theta(
            jnp.asarray(theta, dtype=jnp.float64)))

    @classmethod
    def _grad_log_partition_cpu(cls, theta) -> np.ndarray:
        r""":math:`\nabla\psi(\theta)` on CPU. Default: wraps JAX grad."""
        return np.asarray(cls._grad_log_partition(
            jnp.asarray(theta, dtype=jnp.float64)))

    @classmethod
    def _hessian_log_partition_cpu(cls, theta) -> np.ndarray:
        r""":math:`\nabla^2\psi(\theta)` on CPU. Default: wraps JAX hessian."""
        return np.asarray(cls._hessian_log_partition(
            jnp.asarray(theta, dtype=jnp.float64)))

    # ------------------------------------------------------------------
    # Derived quantities — via triad
    # ------------------------------------------------------------------

    def log_partition(self) -> jax.Array:
        r""":math:`\psi(\theta)` at current parameters."""
        return type(self)._log_partition_from_theta(self.natural_params())

    def expectation_params(self, backend: str = 'jax') -> jax.Array:
        r""":math:`\eta = \nabla\psi(\theta)`.

        Parameters
        ----------
        backend : str
            ``'jax'`` (default, JIT-able) or ``'cpu'`` (numpy/scipy).
        """
        theta = self.natural_params()
        if backend == 'cpu':
            return jnp.asarray(
                type(self)._grad_log_partition_cpu(np.asarray(theta)))
        return type(self)._grad_log_partition(theta)

    def fisher_information(self, backend: str = 'jax') -> jax.Array:
        r""":math:`I(\theta) = \nabla^2\psi(\theta)`.

        Parameters
        ----------
        backend : str
            ``'jax'`` (default, JIT-able) or ``'cpu'`` (numpy/scipy).
        """
        theta = self.natural_params()
        if backend == 'cpu':
            return jnp.asarray(
                type(self)._hessian_log_partition_cpu(np.asarray(theta)))
        return type(self)._hessian_log_partition(theta)

    def log_prob(self, x: jax.Array) -> jax.Array:
        r""":math:`\log p(x\mid\theta) = \log h(x) + \theta^\top t(x) - \psi(\theta)`, single observation."""
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
        r""":math:`\mathrm{Std}[X] = \sqrt{\mathrm{Var}[X]}`."""
        return jnp.sqrt(self.var())

    def cdf(self, x: jax.Array) -> jax.Array:
        """CDF F(x). Subclasses should override with analytical formulas."""
        raise NotImplementedError(f"{type(self).__name__}.cdf not implemented")

    def rvs(self, n: int, seed: int = 42) -> jax.Array:
        """Sample n observations via JAX PRNG (JIT-able)."""
        raise NotImplementedError(f"{type(self).__name__}.rvs not implemented")

    # ------------------------------------------------------------------
    # Divergences (Tier 2 — subclasses may override)
    # ------------------------------------------------------------------

    def squared_hellinger(self, other: "ExponentialFamily") -> jax.Array:
        r"""Squared Hellinger distance :math:`H^2(p, q)`.

        Default uses the general exponential-family formula via :math:`\psi`.
        Subclasses may override for numerically improved variants.
        """
        from normix.divergences import squared_hellinger_from_psi
        cls = type(self)
        return squared_hellinger_from_psi(
            cls._log_partition_from_theta,
            self.natural_params(),
            other.natural_params(),
        )

    def kl_divergence(self, other: "ExponentialFamily") -> jax.Array:
        r"""KL divergence :math:`D_{\mathrm{KL}}(\mathrm{self} \| \mathrm{other})`.

        Default uses the Bregman-divergence formula via :math:`\psi` and :math:`\nabla\psi`.
        Subclasses may override.
        """
        from normix.divergences import kl_divergence_from_psi
        cls = type(self)
        return kl_divergence_from_psi(
            cls._log_partition_from_theta,
            cls._grad_log_partition,
            self.natural_params(),
            other.natural_params(),
        )

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
        r"""Bregman divergence :math:`\psi(\theta) - \theta\cdot\eta` (conjugate dual).

        Minimising over :math:`\theta` yields :math:`\nabla\psi(\theta^*) = \eta`,
        i.e. the natural parameters corresponding to expectation parameters :math:`\eta`.
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
        verbose: int = 0,
    ) -> "ExponentialFamily":
        r"""
        Construct from expectation parameters :math:`\eta` by solving :math:`\nabla\psi(\theta) = \eta`.

        Minimises the Bregman divergence :math:`\psi(\theta) - \theta\cdot\eta`
        via ``solve_bregman``. Subclasses can override for closed-form inverses.

        Parameters
        ----------
        backend : str
            ``'jax'`` (default, JIT-able) or ``'cpu'`` (scipy, more robust).
        method : str
            ``'lbfgs'`` (default), ``'bfgs'``, or ``'newton'``.
        verbose : int
            0 = silent, >= 1 = print solver summary.
        """
        from normix.fitting.solvers import solve_bregman

        eta = jnp.asarray(eta, dtype=jnp.float64)
        if theta0 is None:
            theta0 = jnp.zeros_like(eta)

        bounds = cls._theta_bounds()

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
            verbose=verbose,
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
        verbose: int = 0,
    ) -> "ExponentialFamily":
        r"""
        MLE via exponential family identity: :math:`\hat\eta = \frac{1}{n}\sum_i t(x_i)`.

        Batches over ``X`` using ``jax.vmap``, then calls ``from_expectation``\ (:math:`\hat\eta`).

        Parameters
        ----------
        X : jax.Array
            ``(n, ...)`` array of observations.
        theta0 : jax.Array, optional
            Initial natural parameters :math:`\theta_0` for the :math:`\eta\to\theta` solver.
        maxiter : int
            Maximum iterations for the :math:`\eta\to\theta` solver.
        tol : float
            Convergence tolerance for the :math:`\eta\to\theta` solver.
        verbose : int
            0 = silent, >= 1 = print solver summary.
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        stats = jax.vmap(cls.sufficient_statistics)(X)   # (n, dim_t)
        eta_hat = jnp.mean(stats, axis=0)
        return cls.from_expectation(
            eta_hat, theta0=theta0, maxiter=maxiter, tol=tol, verbose=verbose)

    def fit(
        self,
        X: jax.Array,
        *,
        maxiter: int = 500,
        tol: float = 1e-10,
        verbose: int = 0,
        **kwargs,
    ) -> "ExponentialFamily":
        r"""Fit using self as initialization (warm start).

        Computes :math:`\hat\eta = \frac{1}{n}\sum_i t(x_i)` and solves
        ``from_expectation``\ (:math:`\hat\eta`) using ``self.natural_params()``
        as the initial :math:`\theta_0`.

        Parameters
        ----------
        X : jax.Array
            ``(n, ...)`` array of observations.
        maxiter : int
            Maximum iterations for the :math:`\eta\to\theta` solver.
        tol : float
            Convergence tolerance for the :math:`\eta\to\theta` solver.
        verbose : int
            0 = silent, >= 1 = print solver summary.
        """
        cls = type(self)
        X = jnp.asarray(X, dtype=jnp.float64)
        stats = jax.vmap(cls.sufficient_statistics)(X)
        eta_hat = jnp.mean(stats, axis=0)
        return cls.from_expectation(
            eta_hat, theta0=self.natural_params(),
            maxiter=maxiter, tol=tol, verbose=verbose, **kwargs)

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    @classmethod
    def default_init(cls, X: jax.Array) -> "ExponentialFamily":
        r"""Moment-based initialisation from data.

        Computes :math:`\hat\eta = \frac{1}{n}\sum_i t(x_i)` and inverts to get
        an initial model. For distributions with closed-form ``from_expectation``
        (Gamma, InverseGamma, InverseGaussian), this gives the MLE directly.
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        stats = jax.vmap(cls.sufficient_statistics)(X)
        eta_hat = jnp.mean(stats, axis=0)
        return cls.from_expectation(eta_hat)

    @classmethod
    def _theta_bounds(cls):
        """
        Bounds for θ used in LBFGSB.  Return None for unconstrained.
        Subclasses override to provide (lower, upper) tuple of arrays.
        """
        return None
