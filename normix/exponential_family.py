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
from typing import Callable, Optional

import numpy as np

import equinox as eqx
import jax
import jax.numpy as jnp

from normix.utils.constants import RENYI_TAYLOR_EPS


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

    @classmethod
    def _on_positive_support(cls) -> bool:
        """True when the sample space is :math:`x > 0`."""
        return False

    def _log_density_at_zero(self) -> jax.Array:
        r"""One-sided limit of :math:`\log p(x)` as :math:`x \to 0^+`.

        The default is :math:`-\infty`. Families whose density diverges or
        stays finite at the origin override this.
        """
        return jnp.asarray(-jnp.inf, dtype=jnp.float64)

    def log_prob(self, x: jax.Array) -> jax.Array:
        r""":math:`\log p(x\mid\theta) = \log h(x) + \theta^\top t(x) - \psi(\theta)`, single observation.

        On a positive support, :math:`t(x)` is evaluated at
        :math:`x_{\mathrm{safe}} = x` if :math:`x > 0` and at :math:`1`
        otherwise, then the value is replaced by the endpoint limit at
        :math:`x = 0` and by :math:`-\infty` for :math:`x < 0`. Masking
        the input and the value is required: an output-only ``where``
        leaves NaN parameter gradients.
        """
        cls = type(self)
        theta = self.natural_params()
        psi = cls._log_partition_from_theta(theta)

        def _interior(x_: jax.Array) -> jax.Array:
            return (cls.log_base_measure(x_)
                    + jnp.dot(cls.sufficient_statistics(x_), theta)
                    - psi)

        if not cls._on_positive_support():
            return _interior(x)

        x = jnp.asarray(x, dtype=jnp.float64)
        on = x > 0
        x_safe = jnp.where(on, x, jnp.ones_like(x))
        off = jnp.where(x == 0, self._log_density_at_zero(), -jnp.inf)
        return jnp.where(on, _interior(x_safe), off)

    def _cdf_on_positive_line(self, x: jax.Array, cdf_at: Callable) -> jax.Array:
        r"""CDF on :math:`(0, \infty)`: 0 for :math:`x \le 0`, with the integrand masked."""
        x = jnp.asarray(x, dtype=jnp.float64)
        on = x > 0
        x_safe = jnp.where(on, x, jnp.ones_like(x))
        return jnp.where(on, cdf_at(x_safe), jnp.zeros((), dtype=jnp.float64))

    def pdf(self, x: jax.Array) -> jax.Array:
        """p(x|θ), single observation. Batch via jax.vmap."""
        return jnp.exp(self.log_prob(x))

    # ------------------------------------------------------------------
    # Information-theoretic quantities (entropy, varentropy, Rényi)
    # ------------------------------------------------------------------

    def _log_base_measure_constant(self) -> jax.Array:
        r"""Constant value of :math:`\log h(x)` on the support.

        :meth:`log_density_power`, :meth:`entropy`, :meth:`varentropy`, and
        :meth:`renyi` assume the base measure :math:`h(x)` is **constant on
        the support** — true for :class:`~normix.distributions.gamma.Gamma`,
        :class:`~normix.distributions.inverse_gamma.InverseGamma`,
        :class:`~normix.distributions.generalized_inverse_gaussian.GeneralizedInverseGaussian`,
        :class:`~normix.distributions.normal.MultivariateNormal` (all
        :math:`0`), and every
        :class:`~normix.mixtures.joint.JointNormalMixture`
        (:math:`-\tfrac{d}{2}\log 2\pi`). Distributions whose
        :math:`\log h(x)` varies with :math:`x` (e.g.
        :class:`~normix.distributions.inverse_gaussian.InverseGaussian`)
        override the information-theoretic methods instead. Returns
        :math:`0` by default.
        """
        return jnp.asarray(0.0, dtype=jnp.float64)

    def log_density_power(self, alpha: jax.Array) -> jax.Array:
        r"""Log density-power integral :math:`R(\alpha) = \log\int p(x)^\alpha\,\mu(dx)`.

        For an exponential family whose base measure is constant on the support,

        .. math::

            R(\alpha) = (\alpha - 1)\,b_0 + \psi(\alpha\theta) - \alpha\,\psi(\theta),
            \qquad b_0 = \log h,

        because :math:`p_\theta(x)^\alpha \propto \exp\{\alpha\theta^\top t(x)\}`
        stays in the family with natural parameter :math:`\alpha\theta`. This is
        the cumulant generator of the information content
        :math:`-\log p(X)`: entropy is :math:`H = -R'(1)`, varentropy is
        :math:`V_H = R''(1)`, and the Rényi entropy of order :math:`\alpha` is
        :math:`R(\alpha)/(1-\alpha)`.

        The identity equals :math:`\log\int p^\alpha` only for
        :math:`\alpha\theta \in \Theta`. Subclasses whose :math:`\psi` admits an
        analytic continuation off :math:`\Theta` (including Gamma, InverseGamma,
        and their joints) return
        :math:`+\infty` when the powered density is not integrable, so
        :meth:`renyi` inherits the sign of :math:`1-\alpha`.
        """
        cls = type(self)
        alpha = jnp.asarray(alpha, dtype=jnp.float64)
        theta = self.natural_params()
        b0 = self._log_base_measure_constant()
        return ((alpha - 1.0) * b0
                + cls._log_partition_from_theta(alpha * theta)
                - alpha * cls._log_partition_from_theta(theta))

    def entropy(self) -> jax.Array:
        r"""Differential entropy :math:`H = \mathbb{E}[-\log p(X)] = -R'(1)`.

        Obtained by differentiating :meth:`log_density_power`; for constant
        base measure this equals :math:`\psi(\theta) - \theta^\top\eta - b_0`.
        """
        one = jnp.asarray(1.0, dtype=jnp.float64)
        return -jax.grad(self.log_density_power)(one)

    def varentropy(self) -> jax.Array:
        r"""Varentropy :math:`V_H = \mathrm{Var}[-\log p(X)] = R''(1)`.

        Obtained by differentiating :meth:`log_density_power` twice. For
        constant base measure this is the Fisher quadratic form
        :math:`\theta^\top I(\theta)\,\theta`. The autodiff route is used
        deliberately: it flows through the accurate ``log_kv`` custom JVP.
        """
        one = jnp.asarray(1.0, dtype=jnp.float64)
        return jax.grad(jax.grad(self.log_density_power))(one)

    def renyi(self, alpha: jax.Array) -> jax.Array:
        r"""Rényi entropy :math:`H_\alpha = (1-\alpha)^{-1}\log\int p(x)^\alpha\,\mu(dx)`.

        Defined for order :math:`\alpha > 0`. At :math:`\alpha = 1` the value
        is the removable-singularity limit, the Shannon :meth:`entropy`; the
        first-order expansion within the integrability domain
        :math:`H_\alpha = H - \tfrac{1}{2}V_H(\alpha-1)
        + \mathcal{O}((\alpha-1)^2)` links it to the varentropy.

        The Taylor branch supplies the limiting order derivative
        :math:`H_\alpha'(1)=-V_H/2`. Ordinary autodiff differentiates both
        distribution parameters and order, including mixed derivatives.
        A :func:`jax.lax.cond` evaluates varentropy only near one for scalar
        calls. The unused denominator is safe under ``vmap``.
        """
        alpha = jnp.asarray(alpha, dtype=jnp.float64)
        delta = alpha - 1.0
        power = self.log_density_power(alpha)
        near = (jnp.abs(delta) < RENYI_TAYLOR_EPS) & jnp.isfinite(power)

        def _taylor(_: None) -> jax.Array:
            return self.entropy() - 0.5 * self.varentropy() * delta

        def _exact(_: None) -> jax.Array:
            return power / jnp.where(near, 1.0, -delta)

        return jax.lax.cond(near, _taylor, _exact, None)

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

    def _divergence_lift(self) -> "ExponentialFamily":
        r"""Return ``self`` in its divergence gauge.

        The gauge is the ambient family whose :math:`\psi` is faithful on the
        convex hull of lifted members (GIG for subordinators, JointGH for
        joints). Default: identity.
        """
        return self

    def _divergence_eta(self) -> tuple:
        r"""Exact :math:`(\eta_{\mathrm{fin}}, m, v)` in gauge coordinates.

        Represents :math:`\eta = \eta_{\mathrm{fin}} + m\,v` where :math:`m\ge 0`
        is the one possibly-divergent scalar moment (may be ``inf``) and
        :math:`v` is its structural direction in the gauge :math:`t`-layout.
        Default: finite :math:`\eta = \nabla\psi(\theta)`, :math:`m = 0`.
        """
        eta = self.expectation_params()
        return eta, jnp.zeros((), dtype=eta.dtype), jnp.zeros_like(eta)

    def _divergence_pair(
        self, other: "ExponentialFamily"
    ) -> tuple["ExponentialFamily", "ExponentialFamily"]:
        """Lift both operands and check gauge / dimension compatibility."""
        lp = self._divergence_lift()
        lq = other._divergence_lift()
        if type(lp) is not type(lq):
            raise TypeError(
                f"Incompatible divergence gauges: "
                f"{type(self).__name__} lifts to {type(lp).__name__}, "
                f"{type(other).__name__} lifts to {type(lq).__name__}"
            )
        if getattr(lp, "d", None) is not None and getattr(lq, "d", None) is not None:
            if lp.d != lq.d:
                raise ValueError(
                    f"Dimension mismatch for divergences: "
                    f"{type(self).__name__} has d={lp.d}, "
                    f"{type(other).__name__} has d={lq.d}"
                )
        return lp, lq

    def squared_hellinger(self, other: "ExponentialFamily") -> jax.Array:
        r"""Squared Hellinger distance :math:`H^2(p, q)`.

        Both operands are lifted into their divergence gauge before evaluating
        the general exponential-family formula via the gauge :math:`\psi`.
        """
        from normix.divergences import squared_hellinger_from_psi
        lp, lq = self._divergence_pair(other)
        return squared_hellinger_from_psi(
            type(lp)._log_partition_from_theta,
            lp.natural_params(),
            lq.natural_params(),
        )

    def kl_divergence(self, other: "ExponentialFamily") -> jax.Array:
        r"""KL divergence :math:`D_{\mathrm{KL}}(\mathrm{self} \| \mathrm{other})`.

        Uses the gauge :math:`\psi` after lifting both operands, with
        :math:`\eta_p` from the source family's closed-form tail split
        (``_divergence_eta``) rather than :math:`\nabla\psi` at a
        boundary :math:`\theta`.
        """
        from normix.divergences import kl_divergence_from_eta
        lp, lq = self._divergence_pair(other)
        eta_fin, m, v = self._divergence_eta()
        return kl_divergence_from_eta(
            type(lp)._log_partition_from_theta,
            lp.natural_params(),
            lq.natural_params(),
            eta_fin,
            m,
            v,
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
    def _implicit_natural_params(cls, eta: jax.Array, theta: jax.Array) -> jax.Array:
        r"""Attach the interior inverse derivative :math:`H(\theta)^{-1}d\eta`.

        ``theta`` is the result of a solve on stopped inputs. Diagonal
        equilibration removes units before the positive-definiteness check
        and Cholesky solve. Unidentifiable, nonfinite, or bound-active
        solutions have NaN derivatives; no damping changes the inverse.
        """
        @jax.custom_jvp
        def inverse(target, solved):
            return solved

        @inverse.defjvp
        def inverse_jvp(primals, tangents):
            target, solved = primals
            d_eta, _ = tangents
            theta = inverse(target, solved)
            hessian = cls._hessian_log_partition(theta)
            diagonal = jnp.diag(hessian)
            valid = (jnp.all(jnp.isfinite(theta)) & jnp.all(jnp.isfinite(hessian))
                     & jnp.all(diagonal > 0.0))
            bounds = cls._theta_bounds()
            if bounds is not None:
                valid = valid & jnp.all((theta > bounds[0]) & (theta < bounds[1]))
            scale = jnp.sqrt(jnp.where(diagonal > 0.0, diagonal, 1.0))
            scaled = hessian / scale[:, None] / scale[None, :]
            scaled = jnp.where(valid, (scaled + scaled.T) / 2.0, jnp.eye(theta.size))
            eigenvalues = jnp.linalg.eigvalsh(scaled)
            cutoff = theta.size * jnp.finfo(theta.dtype).eps * eigenvalues[-1]
            valid = valid & (eigenvalues[0] > cutoff)
            factor = jnp.linalg.cholesky(jnp.where(valid, scaled, jnp.eye(theta.size)))
            tangent = jax.scipy.linalg.cho_solve((factor, True), d_eta / scale) / scale
            return theta, tangent * jnp.where(valid, 1.0, jnp.nan)

        return inverse(eta, jax.lax.stop_gradient(theta))

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
        The JAX path differentiates the solved stationarity condition:
        :math:`d\theta=H(\theta)^{-1}d\eta`, in forward and reverse mode.
        Initial guesses and optimizer iterations are not differentiated.
        Singular, numerically rank-deficient, nonfinite, or bound-active
        solutions have NaN derivatives. The CPU backend is eager-only
        and does not support JAX transformations.

        Parameters
        ----------
        backend : str
            ``'jax'`` (default, JIT-able) or ``'cpu'`` (scipy, more robust).
        method : str
            ``'lbfgs'`` (default), ``'bfgs'``, or ``'newton'``.
        verbose : int
            0 = silent, >= 1 = print solver summary.

        Raises
        ------
        RuntimeError
            The solve did not converge. Under ``jit`` / ``lax.scan`` the
            returned natural parameters are NaN instead of a silent model.
        """
        from normix.fitting.solvers import solve_bregman, _require_solved_theta

        eta_target = jnp.asarray(eta, dtype=jnp.float64)
        eta = jax.lax.stop_gradient(eta_target) if backend == "jax" else eta_target
        if theta0 is None:
            theta0 = jnp.zeros_like(eta)

        theta0 = jax.lax.stop_gradient(theta0)
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
        theta = _require_solved_theta(
            result.theta, result.converged, grad_norm=result.grad_norm,
            traced=isinstance(eta_target, jax.core.Tracer),
        )
        if backend == "jax":
            theta = cls._implicit_natural_params(eta_target, theta)
        return cls.from_natural(theta)

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

        Raises
        ------
        ValueError
            On a positive support, if a concrete ``X`` is empty, contains a
            non-finite or non-positive value, has a single observation, or
            is constant. Constant data has no finite MLE: the Jensen gap
            :math:`\log\bar x - \overline{\log x}` is 0. Under ``jit`` /
            ``vmap`` the sample is a tracer and this check is skipped, so
            a traced call keeps the previous contract.
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        cls._validate_positive_sample(X)
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
    def _validate_positive_sample(cls, X: jax.Array) -> None:
        """Reject samples for which a positive-support MLE does not exist.

        Host-side only. A traced ``X`` (``jit``, ``vmap``) skips the
        check so those transforms still run; eager calls raise.
        Multivariate families leave this as a no-op: their support is
        :math:`\\mathbb{R}^d`, and a constant sample still has a
        (singular) normal MLE.
        """
        if not cls._on_positive_support():
            return
        if isinstance(X, jax.core.Tracer):
            return
        arr = np.asarray(X)
        if arr.ndim == 0 or arr.size == 0 or arr.shape[0] == 0:
            raise ValueError(f"{cls.__name__}.fit_mle received an empty sample")
        if not np.isfinite(arr).all():
            raise ValueError(
                f"{cls.__name__}.fit_mle requires every observation to be finite")
        if np.any(arr <= 0):
            raise ValueError(
                f"{cls.__name__}.fit_mle requires observations on the "
                f"positive support (x > 0)")
        if arr.shape[0] < 2:
            raise ValueError(
                f"{cls.__name__}.fit_mle requires n >= 2; a single "
                f"observation has no finite MLE")
        flat = np.reshape(arr, (arr.shape[0], -1))
        if np.all(np.max(flat, axis=0) == np.min(flat, axis=0)):
            raise ValueError(
                f"{cls.__name__}.fit_mle: constant data has no finite MLE "
                f"(Jensen gap log(mean) - mean(log) is 0)")

    @classmethod
    def _theta_bounds(cls):
        """
        Bounds for θ used in LBFGSB.  Return None for unconstrained.
        Subclasses override to provide (lower, upper) tuple of arrays.
        """
        return None
