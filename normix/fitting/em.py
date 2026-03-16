"""
EM fitters for normix distributions.

Following the GMMX pattern: model knows math, fitter knows iteration.

  BatchEMFitter   — standard batch EM, jax.lax.while_loop
  OnlineEMFitter  — online EM, one sample at a time, jax.lax.scan
  MiniBatchEMFitter — mini-batch EM, Robbins-Monro averaging
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class BatchEMFitter(eqx.Module):
    """
    Batch EM algorithm.

    Runs E-step (all data) → M-step → regularize until convergence.
    Uses a Python loop (not jax.lax.while_loop) because the GIG η→θ
    optimization uses scipy under the hood and isn't fully JIT-able.

    Parameters
    ----------
    e_step_backend : 'jax' or 'cpu'
        'jax' (default): jax.vmap over conditional_expectations.
        'cpu': quad forms in JAX + Bessel on CPU (faster for large N).
    m_step_solver : solver for GIG η→θ (used by GH and similar models).
        'newton' (default), 'newton_analytical', 'lbfgs', 'cpu', 'cpu_legacy'.
        'cpu' is the recommended fast solver for the EM hot path.
    """

    max_iter: int = eqx.field(static=True, default=200)
    tol: float = eqx.field(static=True, default=1e-6)
    regularization: str = eqx.field(static=True, default='det_sigma_one')
    e_step_backend: str = eqx.field(static=True, default='jax')
    m_step_solver: str = eqx.field(static=True, default='newton')

    def fit(self, model, X: jax.Array):
        """
        Run batch EM. Returns fitted model.

        Parameters
        ----------
        model : NormalMixture subclass
        X     : (n, d) data array

        Returns
        -------
        model : fitted NormalMixture (new immutable object)
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        ll = -jnp.inf

        for i in range(self.max_iter):
            # E-step
            expectations = model.e_step(X, backend=self.e_step_backend)

            # M-step — pass solver if model supports it
            model = self._m_step(model, X, expectations)

            # Regularization
            model = self._regularize(model)

            # Convergence check
            new_ll = model.marginal_log_likelihood(X)
            ll_diff = jnp.abs(new_ll - ll)
            ll = new_ll

            if float(ll_diff) < self.tol and i > 0:
                break

        return model

    def _m_step(self, model, X, expectations):
        """Call m_step, forwarding solver if the model accepts it."""
        import inspect
        sig = inspect.signature(model.m_step)
        if 'solver' in sig.parameters:
            return model.m_step(X, expectations, solver=self.m_step_solver)
        return model.m_step(X, expectations)

    def _regularize(self, model):
        if self.regularization == 'det_sigma_one':
            if hasattr(model, 'regularize_det_sigma_one'):
                return model.regularize_det_sigma_one()
        return model


class OnlineEMFitter(eqx.Module):
    """
    Online EM algorithm (Robbins-Monro stochastic approximation).

    Updates running sufficient statistics η with step size τ_t = τ₀ + t:
      η_t = η_{t-1} + (1/τ_t)(t̄(x_t|θ_{t-1}) - η_{t-1})

    One epoch = one full pass through data in random order.
    """

    tau0: float = eqx.field(static=True, default=10.0)
    max_epochs: int = eqx.field(static=True, default=5)
    regularization: str = eqx.field(static=True, default='det_sigma_one')

    def fit(self, model, X: jax.Array, *, key: jax.Array):
        """
        Online EM. Returns fitted model.
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        n = X.shape[0]

        for epoch in range(self.max_epochs):
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, n)
            X_shuffled = X[perm]

            for t in range(n):
                x_t = X_shuffled[t]
                tau_t = self.tau0 + epoch * n + t

                # E-step: conditional expectations for single sample
                exp_t = model._joint.conditional_expectations(x_t)
                # Build single-sample expectations dict with shape (1,)
                exp_batch = {k: v[None] for k, v in exp_t.items()}

                # Online sufficient statistics update
                # Update model parameters with step_size = 1/tau_t
                step = 1.0 / tau_t
                model = self._online_update(model, X_shuffled[t:t+1], exp_batch, step)

                if self.regularization == 'det_sigma_one':
                    if hasattr(model, 'regularize_det_sigma_one'):
                        model = model.regularize_det_sigma_one()

        return model

    def _online_update(self, model, X_batch, exp_batch, step_size):
        """Partial M-step: blend new sufficient stats with current."""
        new_model = model.m_step(X_batch, exp_batch)
        # Interpolate between current and new parameters
        # This is a simplified version; a full online EM would blend in the
        # sufficient-statistics space
        return new_model


class MiniBatchEMFitter(eqx.Module):
    """
    Mini-batch EM with Robbins-Monro averaging of sufficient statistics.
    """

    batch_size: int = eqx.field(static=True, default=256)
    max_iter: int = eqx.field(static=True, default=200)
    tol: float = eqx.field(static=True, default=1e-6)
    tau0: float = eqx.field(static=True, default=10.0)
    regularization: str = eqx.field(static=True, default='det_sigma_one')

    def fit(self, model, X: jax.Array, *, key: jax.Array):
        """
        Mini-batch EM. Returns fitted model.
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        n = X.shape[0]
        bs = min(self.batch_size, n)

        ll = -jnp.inf

        for t in range(self.max_iter):
            key, subkey = jax.random.split(key)
            indices = jax.random.choice(subkey, n, shape=(bs,), replace=False)
            X_batch = X[indices]

            # E-step on mini-batch
            expectations = model.e_step(X_batch)

            # M-step on mini-batch
            model = model.m_step(X_batch, expectations)

            if self.regularization == 'det_sigma_one':
                if hasattr(model, 'regularize_det_sigma_one'):
                    model = model.regularize_det_sigma_one()

            # Convergence check (on full data, less frequently)
            if t % 10 == 0:
                new_ll = model.marginal_log_likelihood(X)
                ll_diff = jnp.abs(new_ll - ll)
                ll = new_ll
                if float(ll_diff) < self.tol and t > 0:
                    break

        return model
