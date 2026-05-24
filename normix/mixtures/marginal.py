"""
Marginal mixture base classes.

:class:`MarginalMixture` is the abstract interface fitters and the
divergences module depend on. :class:`NormalMixture` is the
full-covariance implementation, owning a
:class:`~normix.mixtures.joint.JointNormalMixture`. A factor-analysis
implementation lives in the sibling class :class:`FactorNormalMixture`
(see ``docs/design/mixtures.md`` § 6).

NormalMixture provides:

- ``log_prob(x)`` — closed-form marginal log-density
- ``e_step(X)`` — :func:`jax.vmap` over conditional expectations
- ``m_step(X, expectations)`` — returns new :class:`NormalMixture`
- ``fit(X, ...)`` — convenience EM fitting with multi-start
"""
from __future__ import annotations

import abc
from typing import Any, Dict

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp


from normix.utils.constants import SIGMA_INIT_REG
from normix.utils.rvs import build_pinv_table


class MarginalMixture(eqx.Module):
    r"""Abstract interface for marginal mixtures used by the EM fitters.

    Concrete subclasses pick the storage of the Gaussian dispersion (full
    Cholesky in :class:`NormalMixture`, low-rank-plus-diagonal in
    :class:`FactorNormalMixture`) and the type of the EM expectation
    pytree (``NormalMixtureEta`` vs. ``FactorMixtureStats``).

    The fitter depends only on this contract; it does not know which
    storage form a model uses.
    """

    # ------------------------------------------------------------------
    # Distribution surface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def log_prob(self, x: jax.Array) -> jax.Array:
        """Marginal :math:`\\log f(x)` for a single observation."""

    def pdf(self, x: jax.Array) -> jax.Array:
        """Marginal :math:`f(x)` for a single observation."""
        return jnp.exp(self.log_prob(x))

    def marginal_log_likelihood(self, X: jax.Array) -> jax.Array:
        """Mean log-likelihood over a dataset."""
        X = jnp.asarray(X, dtype=jnp.float64)
        return jnp.mean(jax.vmap(self.log_prob)(X))

    # ------------------------------------------------------------------
    # EM hooks (stats type chosen by subclass)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def e_step(self, X: jax.Array, *, backend: str = 'jax') -> Any:
        """E-step: aggregated expectation parameters for the batch."""

    @abc.abstractmethod
    def m_step(self, eta: Any, **kwargs) -> "MarginalMixture":
        """Full M-step: updates all parameters; returns a new model."""

    @abc.abstractmethod
    def m_step_normal(self, eta: Any) -> "MarginalMixture":
        """M-step for normal parameters only (MCECM cycle 1)."""

    @abc.abstractmethod
    def m_step_subordinator(self, eta: Any, **kwargs) -> "MarginalMixture":
        """M-step for subordinator parameters only (MCECM cycle 2)."""

    @abc.abstractmethod
    def compute_eta_from_model(self) -> Any:
        """Reconstruct the expectation pytree from the model's own parameters."""

    @abc.abstractmethod
    def em_convergence_params(self) -> Any:
        r"""Pytree whose leaf-wise change measures EM convergence.

        Subordinator parameters are intentionally excluded (their solver
        has its own tolerance, and including them inflates iteration
        counts). For full-covariance models this is
        ``(mu, gamma, L_Sigma)``; for factor-analysis models it is
        ``(mu, gamma, F F^T + D)`` to sidestep the rotational gauge.
        """

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def fit(
        self,
        X: jax.Array,
        *,
        algorithm: str = 'em',
        verbose: int = 0,
        max_iter: int = 200,
        tol: float = 1e-3,
        regularization: str = 'none',
        e_step_backend: str = 'jax',
        m_step_backend: str = 'cpu',
        m_step_method: str = 'newton',
    ) -> "EMResult":
        """Fit using ``self`` as initialisation. Returns
        :class:`~normix.fitting.em.EMResult`."""
        from normix.fitting.em import BatchEMFitter
        fitter = BatchEMFitter(
            algorithm=algorithm,
            verbose=verbose, max_iter=max_iter, tol=tol,
            regularization=regularization,
            e_step_backend=e_step_backend, m_step_backend=m_step_backend,
            m_step_method=m_step_method,
        )
        return fitter.fit(self, X)


class NormalMixture(MarginalMixture):
    r"""
    Marginal :math:`f(x) = \int_0^\infty f(x,y)\,dy` for a normal
    variance-mean mixture.

    Not an exponential family. Owns a
    :class:`~normix.mixtures.joint.JointNormalMixture` (which is). The
    classical parameters :math:`(\mu, \gamma, \Sigma, \text{subordinator})`
    are forwarded as read-only properties; use :meth:`replace` to obtain
    a new model with updated parameters (modules are immutable).
    """

    _joint: "JointNormalMixture"  # type: ignore[name-defined]

    _NORMAL_KEYS = ('mu', 'gamma', 'L_Sigma')

    def __init__(self, joint):
        from normix.mixtures.joint import JointNormalMixture
        self._joint = joint

    @property
    def joint(self) -> "JointNormalMixture":
        return self._joint

    @property
    def d(self) -> int:
        return self._joint.d

    # ------------------------------------------------------------------
    # Forwarding views of joint parameters
    # ------------------------------------------------------------------

    @property
    def mu(self) -> jax.Array:
        r""":math:`\mu` — location parameter (forwarded from the joint)."""
        return self._joint.mu

    @property
    def gamma(self) -> jax.Array:
        r""":math:`\gamma` — skewness parameter (forwarded from the joint)."""
        return self._joint.gamma

    @property
    def L_Sigma(self) -> jax.Array:
        r"""Lower Cholesky factor of :math:`\Sigma` (forwarded from the joint)."""
        return self._joint.L_Sigma

    def sigma(self) -> jax.Array:
        r"""Dispersion :math:`\Sigma = L_\Sigma L_\Sigma^\top` (forwarded from the joint).

        Distinct from :meth:`cov`, which returns the *marginal* covariance
        :math:`E[Y]\,\Sigma + \mathrm{Var}[Y]\,\gamma\gamma^\top`.
        """
        return self._joint.sigma()

    def log_det_sigma(self) -> jax.Array:
        r""":math:`\log|\Sigma|` (forwarded from the joint)."""
        return self._joint.log_det_sigma()

    @classmethod
    @abc.abstractmethod
    def _joint_class(cls) -> type:
        """Return the :class:`JointNormalMixture` subclass paired with this marginal."""

    @classmethod
    @abc.abstractmethod
    def _subordinator_keys(cls) -> tuple:
        """Stored subordinator field names on the joint, e.g. ``('alpha', 'beta')``.

        Used by :meth:`replace` to dispatch updates between the normal
        block (:attr:`_NORMAL_KEYS`) and the subordinator block.
        """

    def mean(self) -> jax.Array:
        r""":math:`E[X] = \mu + \gamma E[Y]`."""
        j = self._joint
        E_Y = j.subordinator().mean()
        return j.mu + j.gamma * E_Y

    def cov(self) -> jax.Array:
        r""":math:`\mathrm{Cov}[X] = E[Y]\,\Sigma + \mathrm{Var}[Y]\,\gamma\gamma^\top`."""
        j = self._joint
        E_Y = j.subordinator().mean()
        Var_Y = j.subordinator().var()
        return E_Y * j.sigma() + Var_Y * jnp.outer(j.gamma, j.gamma)

    def rvs(self, n: int, seed: int = 42) -> jax.Array:
        """Sample X from the marginal distribution."""
        X, _ = self._joint.rvs(n, seed)
        return X

    # ------------------------------------------------------------------
    # Divergences (delegate to joint)
    # ------------------------------------------------------------------

    def squared_hellinger(self, other: "NormalMixture") -> jax.Array:
        r"""Squared Hellinger distance via joint distributions (upper bound on marginal)."""
        return self._joint.squared_hellinger(other._joint)

    def kl_divergence(self, other: "NormalMixture") -> jax.Array:
        r"""KL divergence via joint distributions."""
        return self._joint.kl_divergence(other._joint)

    # ------------------------------------------------------------------
    # Eta from model (for incremental EM initialisation)
    # ------------------------------------------------------------------

    def compute_eta_from_model(self) -> "NormalMixtureEta":
        r"""Reconstruct :math:`\eta` from the model's own parameters.

        Uses the *marginal* expectations of the joint sufficient statistics:

        .. math::

            \eta_4 = \mu + \gamma\,E[Y], \quad
            \eta_5 = \mu\,E[1/Y] + \gamma, \quad
            \eta_6 = \Sigma + \mu\mu^\top E[1/Y] + \gamma\gamma^\top E[Y]
                     + \mu\gamma^\top + \gamma\mu^\top
        """
        from normix.fitting.eta import NormalMixtureEta

        E_log_Y, E_inv_Y, E_Y = self._subordinator_expectations()
        j = self._joint
        mu, gamma = j.mu, j.gamma
        sigma = j.sigma()

        return NormalMixtureEta(
            E_log_Y=E_log_Y,
            E_inv_Y=E_inv_Y,
            E_Y=E_Y,
            E_X=mu + gamma * E_Y,
            E_X_inv_Y=mu * E_inv_Y + gamma,
            E_XXT_inv_Y=(sigma
                         + jnp.outer(mu, mu) * E_inv_Y
                         + jnp.outer(gamma, gamma) * E_Y
                         + jnp.outer(mu, gamma)
                         + jnp.outer(gamma, mu)),
        )

    @abc.abstractmethod
    def _subordinator_expectations(self):
        r"""Return ``(E[log Y], E[1/Y], E[Y])`` under the subordinator prior."""

    # ------------------------------------------------------------------
    # EM E-step
    # ------------------------------------------------------------------

    def e_step(self, X: jax.Array, backend: str = 'jax') -> "NormalMixtureEta":
        r"""
        Full E-step: subordinator conditionals + batch aggregation.

        Returns a :class:`~normix.fitting.eta.NormalMixtureEta` with the
        six aggregated expectation parameters.

        Parameters
        ----------
        X : (n, d) data array
        backend : str
            ``'jax'`` (default): ``jax.vmap`` over ``conditional_expectations``.
            ``'cpu'``: quad forms in JAX + GIG Bessel on CPU.
        """
        sub_exp = self._e_step_subordinator(X, backend=backend)
        return self._aggregate_eta(X, sub_exp)

    def _e_step_subordinator(
        self, X: jax.Array, backend: str = 'jax',
    ) -> Dict[str, jax.Array]:
        r"""Per-observation subordinator conditional expectations.

        Returns dict ``{E_log_Y: (n,), E_inv_Y: (n,), E_Y: (n,)}``.
        """
        if backend == 'cpu':
            return self._e_step_subordinator_cpu(X)
        return jax.vmap(self._joint.conditional_expectations)(X)

    def _e_step_subordinator_cpu(self, X: jax.Array) -> Dict[str, jax.Array]:
        """CPU path: quad forms in JAX (vmapped) + GIG Bessel via scipy."""
        from normix.distributions.generalized_inverse_gaussian import GIG

        j = self._joint
        if not hasattr(j, '_posterior_gig_params'):
            raise NotImplementedError(
                f"{type(j).__name__} does not implement _posterior_gig_params. "
                "backend='cpu' requires this method. Use backend='jax' instead."
            )

        X = jnp.asarray(X, dtype=jnp.float64)

        def _quad_scalars(x):
            z, w, z2, w2, zw = j._quad_forms(x)
            return z2, w2

        z2_all, w2_all = jax.vmap(_quad_scalars)(X)

        p_post, a_post, b_post = j._posterior_gig_params(z2_all, w2_all)

        eta = GIG.expectation_params_batch(p_post, a_post, b_post, backend='cpu')

        return {
            'E_log_Y': eta[:, 0],
            'E_inv_Y': eta[:, 1],
            'E_Y':     eta[:, 2],
        }

    @staticmethod
    def _aggregate_eta(X: jax.Array, sub_exp: Dict[str, jax.Array]) -> "NormalMixtureEta":
        """Average per-observation expectations into a NormalMixtureEta."""
        from normix.fitting.eta import NormalMixtureEta

        X = jnp.asarray(X, dtype=jnp.float64)
        E_inv_Y = sub_exp['E_inv_Y']

        return NormalMixtureEta(
            E_log_Y=jnp.mean(sub_exp['E_log_Y']),
            E_inv_Y=jnp.mean(E_inv_Y),
            E_Y=jnp.mean(sub_exp['E_Y']),
            E_X=jnp.mean(X, axis=0),
            E_X_inv_Y=jnp.mean(X * E_inv_Y[:, None], axis=0),
            E_XXT_inv_Y=jnp.mean(
                jnp.einsum('ni,nj,n->nij', X, X, E_inv_Y), axis=0),
        )

    # ------------------------------------------------------------------
    # η → model: closed-form M-step as exponential-family inversion
    # ------------------------------------------------------------------

    @classmethod
    def from_expectation(
        cls, eta: "NormalMixtureEta", **kwargs,
    ) -> "NormalMixture":
        r"""Construct from expectation parameters :math:`\eta`.

        Wraps :meth:`JointNormalMixture.from_expectation`, which performs
        the exact closed-form M-step on the normal block and the
        subordinator's ``from_expectation`` on the subordinator block.

        This is the canonical η→model map: any prior or shrinkage target
        :math:`\eta_0` can be inspected as a concrete model via
        ``cls.from_expectation(eta_0).sigma()`` etc.

        Parameters
        ----------
        eta : NormalMixtureEta
            Aggregated expectation parameters.
        **kwargs
            Forwarded to :meth:`JointNormalMixture.from_expectation`
            (e.g. ``backend``, ``method``, ``maxiter``, ``theta0``).
        """
        return cls(cls._joint_class().from_expectation(eta, **kwargs))

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------

    def m_step(
        self,
        eta: "NormalMixtureEta",
        **kwargs,
    ) -> "NormalMixture":
        r"""Full M-step: update normal params + subordinator from :math:`\eta`.

        Equivalent to ``type(self).from_expectation(eta, **kwargs)``;
        ``self`` is only used to dispatch on the subclass. Subclasses
        with iterative subordinator solvers (e.g.
        :class:`~normix.distributions.generalized_hyperbolic.GeneralizedHyperbolic`)
        may override to inject a warm-start :math:`\theta_0`.
        """
        return type(self).from_expectation(eta, **kwargs)

    def m_step_normal(
        self,
        eta: "NormalMixtureEta",
    ) -> "NormalMixture":
        r"""M-step for normal parameters only (MCECM Cycle 1).

        Updates :math:`\mu, \gamma, \Sigma`; subordinator unchanged.
        """
        from normix.mixtures.joint import JointNormalMixture

        mu_new, gamma_new, L_new = JointNormalMixture._mstep_normal_params(eta)
        return self._replace_normal_params(mu_new, gamma_new, L_new)

    def m_step_subordinator(
        self,
        eta: "NormalMixtureEta",
        **kwargs,
    ) -> "NormalMixture":
        r"""M-step for the subordinator only (MCECM Cycle 2).

        Reads the subordinator-relevant fields of ``eta``; normal
        parameters are read from ``self._joint`` and copied unchanged.
        Subclasses with iterative solvers may override to add warm-start
        or sanity-check fallbacks.
        """
        j = self._joint
        joint_cls = type(j)
        sub = joint_cls._subordinator_from_eta(
            eta, theta0=j.subordinator().natural_params(), **kwargs)
        new_joint = joint_cls._from_normal_and_subordinator(
            j.mu, j.gamma, j.L_Sigma, sub)
        return type(self)(new_joint)

    # ------------------------------------------------------------------
    # M-step helpers
    # ------------------------------------------------------------------

    def _replace_normal_params(self, mu, gamma, L) -> "NormalMixture":
        """Return a copy with updated (mu, gamma, L_Sigma), subordinator unchanged."""
        new_joint = eqx.tree_at(
            lambda j: (j.mu, j.gamma, j.L_Sigma),
            self._joint,
            (mu, gamma, L),
        )
        return type(self)(new_joint)

    # ------------------------------------------------------------------
    # Update API: replace(**) returns a new instance
    # ------------------------------------------------------------------

    def replace(self, **updates) -> "NormalMixture":
        r"""Return a new model with selected parameters replaced.

        Accepts any subset of:

        - normal parameters: ``mu``, ``gamma``, ``L_Sigma``;
        - dispersion alias ``sigma`` — converted to ``L_Sigma`` via
          Cholesky (mutually exclusive with ``L_Sigma``);
        - subordinator parameters declared by :meth:`_subordinator_keys`
          (e.g. ``alpha, beta`` for VG / NInvG, ``mu_ig, lam`` for NIG,
          ``p, a, b`` for GH).

        The actual storage lives in :attr:`joint`; this method does an
        immutable update via :func:`equinox.tree_at`.

        Examples
        --------
        >>> vg2 = vg.replace(mu=new_mu)                    # change μ
        >>> vg3 = vg.replace(alpha=2.5, beta=0.5)          # change subordinator
        >>> vg4 = vg.replace(sigma=sigma2 * jnp.eye(d))    # set Σ via covariance
        """
        if 'sigma' in updates:
            if 'L_Sigma' in updates:
                raise ValueError(
                    "specify either `sigma` or `L_Sigma`, not both")
            updates['L_Sigma'] = jnp.linalg.cholesky(
                jnp.asarray(updates.pop('sigma'), dtype=jnp.float64))

        sub_keys = type(self)._subordinator_keys()
        valid = set(self._NORMAL_KEYS) | set(sub_keys)
        unknown = set(updates) - valid
        if unknown:
            raise ValueError(
                f"unknown parameter(s) {sorted(unknown)} for "
                f"{type(self).__name__}; valid keys are {sorted(valid)}")

        if not updates:
            return self

        keys = tuple(updates.keys())
        new_values = tuple(
            jnp.asarray(updates[k], dtype=jnp.float64) for k in keys)

        new_joint = eqx.tree_at(
            lambda j: tuple(getattr(j, k) for k in keys),
            self._joint,
            new_values,
        )
        return type(self)(new_joint)

    # ------------------------------------------------------------------
    # Regularisation
    # ------------------------------------------------------------------

    def _rescale(self, scale: jax.Array) -> "NormalMixture":
        r"""Apply :math:`Y \to s\,Y` reparameterisation.

        Pushes the scale ``s`` through every parameter so the joint
        distribution is unchanged: :math:`\Sigma \to \Sigma/s`,
        :math:`\gamma \to \gamma/s`, and the subordinator-side rescale
        is delegated to :meth:`_build_rescaled` (e.g.
        :math:`(a, b) \to (a/s, b\cdot s)` for GIG).
        """
        j = self._joint
        L_new = j.L_Sigma / jnp.sqrt(scale)
        gamma_new = j.gamma / scale
        return self._build_rescaled(j.mu, gamma_new, L_new, scale)

    def regularize_det_sigma(
        self, target_log_det: float = 0.0,
    ) -> "NormalMixture":
        r"""Rescale to enforce :math:`\log|\Sigma| = \mathrm{target\_log\_det}`.

        Picks :math:`s = \exp((\log|\Sigma| - \tau)/d)` and applies
        :meth:`_rescale`. The default ``target_log_det = 0`` recovers the
        :math:`|\Sigma| = 1` convention; passing the log-determinant of
        an initial reference Σ implements the ``det_sigma_x`` family.
        """
        j = self._joint
        log_scale = (j.log_det_sigma() - target_log_det) / j.d
        return self._rescale(jnp.exp(log_scale))

    def regularize_det_sigma_one(self) -> "NormalMixture":
        r"""Enforce :math:`|\Sigma| = 1`. Alias for
        :meth:`regularize_det_sigma` with ``target_log_det = 0``.
        """
        return self.regularize_det_sigma(0.0)

    def regularize_a_eq_b(self) -> "NormalMixture":
        r"""Rescale subordinator so that :math:`a = b = \sqrt{ab}` for
        GIG-parameterised families.

        Default implementation is a no-op; override in subclasses with
        both :math:`a, b > 0` (currently GH and NIG; VG and NInvG have
        a degenerate ``a=0`` or ``b=0`` and the default no-op is the
        right behaviour).
        """
        return self

    @abc.abstractmethod
    def _build_rescaled(
        self,
        mu: jax.Array,
        gamma_new: jax.Array,
        L_new: jax.Array,
        scale: jax.Array,
    ) -> "NormalMixture":
        """Construct a new model with rescaled subordinator parameters."""

    # ------------------------------------------------------------------
    # Convergence hook
    # ------------------------------------------------------------------

    def em_convergence_params(self):
        r"""Pytree whose leaf-wise change measures EM convergence.

        Returns ``(mu, gamma, L_Sigma)``. Subordinator parameters
        (``p, a, b``) are excluded — their solver has its own tolerance
        and including them inflates iteration counts.
        """
        j = self._joint
        return (j.mu, j.gamma, j.L_Sigma)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @classmethod
    def default_init(cls, X: jax.Array) -> "NormalMixture":
        """Moment-based initialisation from data.

        Returns a model with:
          mu    = sample mean
          gamma = zeros
          Sigma = empirical covariance (regularized)
          subordinator = distribution-specific defaults

        Useful as a starting point for ``model.fit(X)``.
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        n, d = X.shape
        mu = jnp.mean(X, axis=0)
        X_centered = X - mu
        sigma_emp = (X_centered.T @ X_centered) / n + SIGMA_INIT_REG * jnp.eye(d)
        gamma = jnp.zeros(d, dtype=jnp.float64)
        return cls._from_init_params(mu, gamma, sigma_emp)

    @classmethod
    @abc.abstractmethod
    def _from_init_params(
        cls, mu: jax.Array, gamma: jax.Array, sigma: jax.Array,
    ) -> "NormalMixture":
        """Construct a model with default subordinator parameters for initialisation."""


# ============================================================================
# Univariate marginal-mixture mixin (d=1 surface: scalar API + cdf/ppf)
# ============================================================================


class _UnivariateNormalMixtureMixin:
    r"""Common 1-D surface for ``Univariate*`` marginal-mixture classes.

    Subclasses use multiple inheritance with their multivariate parent
    (e.g. ``class UnivariateVarianceGamma(_UnivariateNormalMixtureMixin,
    VarianceGamma)``). Provides:

    - d=1 validation in ``__init__``;
    - ``cdf`` / ``ppf`` via :func:`build_pinv_table` seeded at the marginal mean;
    - scalar ``mean``, ``var``, ``std``, ``log_prob``, ``pdf``;
    - ``(n,)``-shaped ``rvs``.

    EM machinery (``e_step``, ``m_step``, ``from_expectation``,
    ``replace``, ``fit``, regularisation) is inherited unchanged because
    the underlying joint is still a 1-D :class:`JointNormalMixture`.

    The PINV table is seeded at :meth:`mean` (no closed-form mode for the
    Bessel-mixture marginals).  The seed is only used as a bisection
    starting point, so a rough estimate is sufficient.
    """

    def __init__(self, joint):
        if int(joint.mu.shape[0]) != 1:
            raise ValueError(
                f"{type(self).__name__} requires d=1, got d={int(joint.mu.shape[0])}")
        super().__init__(joint)

    def _pinv_grids(self) -> tuple[jax.Array, jax.Array]:
        log_kernel = lambda w: self.log_prob(jnp.atleast_1d(w))
        return build_pinv_table(log_kernel, self.mean())

    def cdf(self, x: jax.Array) -> jax.Array:
        r"""CDF :math:`F(x)` via PINV lookup."""
        u_grid, x_grid = self._pinv_grids()
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.interp(x, x_grid, u_grid, left=0.0, right=1.0)

    def ppf(self, q: jax.Array) -> jax.Array:
        r"""Quantile function :math:`F^{-1}(q)` via PINV lookup."""
        u_grid, x_grid = self._pinv_grids()
        q = jnp.asarray(q, dtype=jnp.float64)
        return jnp.interp(q, u_grid, x_grid)

    def mean(self) -> jax.Array:
        r"""Scalar :math:`E[X]` (unwraps the parent's ``(1,)`` return)."""
        return super().mean()[0]

    def var(self) -> jax.Array:
        r"""Scalar :math:`\mathrm{Var}[X] = \mathrm{Cov}[X][0,0]`."""
        return self.cov()[0, 0]

    def std(self) -> jax.Array:
        r"""Scalar standard deviation."""
        return jnp.sqrt(self.var())

    def rvs(self, n: int, seed: int = 42) -> jax.Array:
        r"""Sample ``n`` scalars (shape ``(n,)``)."""
        return super().rvs(n, seed)[:, 0]

    def log_prob(self, x: jax.Array) -> jax.Array:
        r"""Marginal :math:`\log f(x)` for a scalar (or ``(1,)``) input."""
        x = jnp.asarray(x, dtype=jnp.float64)
        return super().log_prob(jnp.atleast_1d(x))

    def pdf(self, x: jax.Array) -> jax.Array:
        r"""Marginal density :math:`f(x)`."""
        return jnp.exp(self.log_prob(x))
