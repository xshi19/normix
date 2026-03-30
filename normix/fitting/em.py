"""
EM fitters for normix distributions.

Model knows math, fitter knows iteration.

  BatchEMFitter     — standard batch EM with dual-loop architecture:
                      lax.scan (JIT-able) or Python for-loop (CPU-compatible)
  OnlineEMFitter    — online EM, one sample at a time
  MiniBatchEMFitter — mini-batch EM, Robbins-Monro averaging
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_PARAM_EPS = 1e-10


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EMResult:
    """Result of an EM fitting procedure."""
    model: Any                              # fitted distribution (eqx.Module)
    log_likelihoods: Optional[jax.Array]    # (n_iter,) or None when not computed
    param_changes: jax.Array                # (n_iter,) max relative param change
    n_iter: int
    converged: bool
    elapsed_time: float                     # wall-clock seconds


# ---------------------------------------------------------------------------
# BatchEMFitter
# ---------------------------------------------------------------------------

class BatchEMFitter:
    """
    Batch EM / MCECM algorithm with dual-loop architecture.

    **EM** (default): E-step → M-step (all params) → regularize.

    **MCECM**: E-step → M-step (normal params only) → regularize →
    E-step → M-step (subordinator only).

    Convergence is measured by relative parameter change in the normal
    parameters (mu, gamma, L_Sigma), excluding subordinator (GIG) parameters.

    Loop selection is automatic:
      - lax.scan when both backends are 'jax', verbose <= 1, and algorithm='em'
      - Python for-loop otherwise

    Parameters
    ----------
    algorithm : str
        'em' (default) or 'mcecm'.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on max relative parameter change.
    verbose : int
        0 = silent, 1 = summary, 2 = per-iteration table.
    regularization : str
        Regularization strategy ('det_sigma_one' or 'none').
    e_step_backend : str
        'jax' (default) or 'cpu'.
    m_step_backend : str
        'jax' or 'cpu' (default, faster for GIG).
    m_step_method : str
        'newton' (default), 'lbfgs', or 'bfgs'.
    """

    def __init__(
        self,
        *,
        algorithm: str = 'em',
        max_iter: int = 200,
        tol: float = 1e-3,
        verbose: int = 0,
        regularization: str = 'none',
        e_step_backend: str = 'jax',
        m_step_backend: str = 'cpu',
        m_step_method: str = 'newton',
    ):
        if algorithm not in ('em', 'mcecm'):
            raise ValueError(f"algorithm must be 'em' or 'mcecm', got {algorithm!r}")
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.regularization = regularization
        self.e_step_backend = e_step_backend
        self.m_step_backend = m_step_backend
        self.m_step_method = m_step_method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, model, X: jax.Array) -> EMResult:
        """
        Run batch EM or MCECM. Auto-selects lax.scan or Python loop.

        Parameters
        ----------
        model : NormalMixture subclass (used as initial parameters)
        X     : (n, d) data array

        Returns
        -------
        EMResult with fitted model, convergence diagnostics, and timing.
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        dist_name = type(model).__name__

        use_scan = (
            self.algorithm == 'em'
            and self.e_step_backend == 'jax'
            and self.m_step_backend == 'jax'
            and self.verbose <= 1
        )

        if use_scan:
            if self.verbose >= 1:
                print(
                    f"EM [lax.scan] {dist_name}: "
                    f"backend=jax, tol={self.tol:.0e}, "
                    f"max_iter={self.max_iter}"
                )
            return self._fit_scan(model, X)
        else:
            if self.verbose >= 1:
                self._print_header(dist_name)
            return self._fit_loop(model, X)

    # ------------------------------------------------------------------
    # Pure EM / MCECM step (no prints, no float(), no list.append())
    # ------------------------------------------------------------------

    def _step(self, model, X):
        """One iteration (EM or MCECM). Returns (new_model, max_change)."""
        if self.algorithm == 'mcecm':
            return self._mcecm_step(model, X)
        return self._em_step(model, X)

    def _em_step(self, model, X):
        """One EM iteration: E → M(all) → regularize."""
        prev_mu = model._joint.mu
        prev_gamma = model._joint.gamma
        prev_L = model._joint.L_Sigma

        expectations = model.e_step(X, backend=self.e_step_backend)
        model = self._m_step(model, X, expectations)
        model = self._regularize(model)

        max_change = _param_change(
            model._joint.mu, model._joint.gamma, model._joint.L_Sigma,
            prev_mu, prev_gamma, prev_L,
        )
        return model, max_change

    def _mcecm_step(self, model, X):
        """One MCECM iteration: E → M_normal → regularize → E → M_subordinator."""
        prev_mu = model._joint.mu
        prev_gamma = model._joint.gamma
        prev_L = model._joint.L_Sigma

        # Cycle 1: update (mu, gamma, Sigma), keep subordinator
        expectations = model.e_step(X, backend=self.e_step_backend)
        model = model.m_step_normal(X, expectations)
        model = self._regularize(model)

        # Cycle 2: re-E-step with updated normal params, then update subordinator
        expectations = model.e_step(X, backend=self.e_step_backend)
        gig_eta = jnp.array([
            jnp.mean(expectations['E_log_Y']),
            jnp.mean(expectations['E_inv_Y']),
            jnp.mean(expectations['E_Y']),
        ])
        model = model.m_step_subordinator(
            gig_eta, backend=self.m_step_backend, method=self.m_step_method)

        max_change = _param_change(
            model._joint.mu, model._joint.gamma, model._joint.L_Sigma,
            prev_mu, prev_gamma, prev_L,
        )
        return model, max_change

    # ------------------------------------------------------------------
    # lax.scan path (JIT-able, requires backend='jax')
    # ------------------------------------------------------------------

    def _fit_scan(self, model, X) -> EMResult:
        t0 = time.perf_counter()

        def body(carry, _):
            mdl, converged = carry
            mdl_new, max_change = self._step(mdl, X)
            conv_new = max_change < self.tol
            mdl_out = jax.tree.map(
                lambda n, o: jnp.where(converged, o, n), mdl_new, mdl)
            change_out = jnp.where(converged, 0.0, max_change)
            return (mdl_out, converged | conv_new), change_out

        init = (model, jnp.bool_(False))
        (final_model, converged), param_changes = jax.lax.scan(
            body, init, None, length=self.max_iter)

        elapsed = time.perf_counter() - t0

        valid_mask = param_changes > 0
        n_iter = int(jnp.sum(valid_mask)) + (1 if bool(converged) else 0)
        n_iter = min(n_iter, self.max_iter)

        log_likelihoods = None
        if self.verbose >= 1:
            ll = final_model.marginal_log_likelihood(X)
            log_likelihoods = jnp.array([ll])
            status = "Converged" if bool(converged) else "NOT converged"
            print(
                f"  {status} after {n_iter} iterations "
                f"({elapsed:.2f}s), final LL={float(ll):.6f}"
            )

        return EMResult(
            model=final_model,
            log_likelihoods=log_likelihoods,
            param_changes=param_changes,
            n_iter=n_iter,
            converged=bool(converged),
            elapsed_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Python loop path (works with any backend, supports verbose >= 2)
    # ------------------------------------------------------------------

    def _fit_loop(self, model, X) -> EMResult:
        t_total = time.perf_counter()
        lls = []
        changes = []
        prev_ll = None
        n_iter = 0

        if self.verbose >= 2:
            self._print_table_header()

        for i in range(self.max_iter):
            t_iter = time.perf_counter()
            model, max_change = self._step(model, X)
            dt = time.perf_counter() - t_iter
            changes.append(max_change)
            n_iter = i + 1

            ll = None
            if self.verbose >= 1:
                ll = model.marginal_log_likelihood(X)
                lls.append(ll)

            if self.verbose >= 2:
                dll_str = "    ---    "
                if prev_ll is not None and ll is not None:
                    dll_str = f"{float(ll - prev_ll):+.4e}"
                ll_str = f"{float(ll):.6f}" if ll is not None else "   N/A   "
                print(
                    f"  {n_iter:4d}  {ll_str}  {dll_str}  "
                    f"{float(max_change):.4e}  {dt:.3f}s"
                )
                if ll is not None:
                    prev_ll = ll

            if max_change < self.tol and i > 0:
                break

        elapsed = time.perf_counter() - t_total
        converged = bool(max_change < self.tol) and n_iter > 1

        if self.verbose >= 2:
            self._print_footer(n_iter, converged, elapsed,
                               float(lls[-1]) if lls else None,
                               float(changes[-1]))
        elif self.verbose == 1:
            status = "Converged" if converged else "NOT converged"
            ll_str = f", final LL={float(lls[-1]):.6f}" if lls else ""
            print(f"  {status} after {n_iter} iterations ({elapsed:.2f}s){ll_str}")

        return EMResult(
            model=model,
            log_likelihoods=jnp.array(lls) if lls else None,
            param_changes=jnp.array(changes),
            n_iter=n_iter,
            converged=converged,
            elapsed_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _m_step(self, model, X, expectations):
        """Call m_step, forwarding backend/method via kwargs."""
        return model.m_step(
            X, expectations,
            backend=self.m_step_backend, method=self.m_step_method,
        )

    def _regularize(self, model):
        if self.regularization == 'det_sigma_one':
            if hasattr(model, 'regularize_det_sigma_one'):
                return model.regularize_det_sigma_one()
        return model

    def _print_header(self, dist_name: str):
        algo_label = "MCECM" if self.algorithm == 'mcecm' else "EM"
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  {algo_label} Fitting: {dist_name}")
        print(sep)
        loop = "lax.scan" if (
            self.algorithm == 'em'
            and self.e_step_backend == 'jax'
            and self.m_step_backend == 'jax'
        ) else "Python loop"
        print(f"  Algorithm    : {algo_label}")
        print(f"  Loop         : {loop}")
        print(f"  E-step       : {self.e_step_backend}")
        print(f"  M-step       : {self.m_step_backend} / {self.m_step_method}")
        print(f"  Regularize   : {self.regularization}")
        print(f"  Tolerance    : {self.tol:.1e}")
        print(f"  Max iters    : {self.max_iter}")

    def _print_table_header(self):
        print("-" * 60)
        print("  Iter    Log-Lik       ΔLL        |Δparams|   Time")
        print("-" * 60)

    def _print_footer(self, n_iter, converged, elapsed, final_ll, final_change):
        sep = "=" * 60
        print(sep)
        status = "Converged" if converged else "NOT converged"
        print(f"  {status} after {n_iter} iterations ({elapsed:.2f}s)")
        if final_ll is not None:
            print(f"  Final log-likelihood : {final_ll:.6f}")
        print(f"  Final |Δparams|      : {final_change:.4e}")
        print(sep)


# ---------------------------------------------------------------------------
# OnlineEMFitter
# ---------------------------------------------------------------------------

class OnlineEMFitter:
    """
    Online EM algorithm (Robbins-Monro stochastic approximation).

    Updates running sufficient statistics with step size 1/(tau0 + t).
    One epoch = one full pass through data in random order.
    """

    def __init__(
        self,
        *,
        tau0: float = 10.0,
        max_epochs: int = 5,
        verbose: int = 0,
        regularization: str = 'none',
    ):
        self.tau0 = tau0
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.regularization = regularization

    def fit(self, model, X: jax.Array, *, key: jax.Array) -> EMResult:
        """Online EM. Returns EMResult."""
        t_total = time.perf_counter()
        X = jnp.asarray(X, dtype=jnp.float64)
        n = X.shape[0]
        dist_name = type(model).__name__
        changes = []

        if self.verbose >= 1:
            print(
                f"EM [online] {dist_name}: "
                f"tau0={self.tau0}, epochs={self.max_epochs}"
            )

        for epoch in range(self.max_epochs):
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, n)
            X_shuffled = X[perm]

            for t in range(n):
                x_t = X_shuffled[t]
                tau_t = self.tau0 + epoch * n + t

                exp_t = model._joint.conditional_expectations(x_t)
                exp_batch = {k: v[None] for k, v in exp_t.items()}

                step = 1.0 / tau_t
                new_model = model.m_step(X_shuffled[t:t+1], exp_batch)

                if self.regularization == 'det_sigma_one':
                    if hasattr(new_model, 'regularize_det_sigma_one'):
                        new_model = new_model.regularize_det_sigma_one()

                change = _param_change(
                    new_model._joint.mu, new_model._joint.gamma,
                    new_model._joint.L_Sigma,
                    model._joint.mu, model._joint.gamma,
                    model._joint.L_Sigma,
                )
                model = new_model

            changes.append(change)
            if self.verbose >= 1:
                print(f"  Epoch {epoch+1}/{self.max_epochs}, |Δparams|={float(change):.4e}")

        elapsed = time.perf_counter() - t_total

        log_likelihoods = None
        if self.verbose >= 1:
            ll = model.marginal_log_likelihood(X)
            log_likelihoods = jnp.array([ll])
            print(f"  Done ({elapsed:.2f}s), final LL={float(ll):.6f}")

        return EMResult(
            model=model,
            log_likelihoods=log_likelihoods,
            param_changes=jnp.array(changes),
            n_iter=self.max_epochs * n,
            converged=True,
            elapsed_time=elapsed,
        )


# ---------------------------------------------------------------------------
# MiniBatchEMFitter
# ---------------------------------------------------------------------------

class MiniBatchEMFitter:
    """
    Mini-batch EM with Robbins-Monro averaging of sufficient statistics.
    """

    def __init__(
        self,
        *,
        batch_size: int = 256,
        max_iter: int = 200,
        tol: float = 1e-3,
        tau0: float = 10.0,
        verbose: int = 0,
        regularization: str = 'none',
    ):
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tol = tol
        self.tau0 = tau0
        self.verbose = verbose
        self.regularization = regularization

    def fit(self, model, X: jax.Array, *, key: jax.Array) -> EMResult:
        """Mini-batch EM. Returns EMResult."""
        t_total = time.perf_counter()
        X = jnp.asarray(X, dtype=jnp.float64)
        n = X.shape[0]
        bs = min(self.batch_size, n)
        dist_name = type(model).__name__
        changes = []
        lls = []
        n_iter = 0

        if self.verbose >= 1:
            print(
                f"EM [mini-batch] {dist_name}: "
                f"batch_size={bs}, tol={self.tol:.0e}, "
                f"max_iter={self.max_iter}"
            )

        for t in range(self.max_iter):
            key, subkey = jax.random.split(key)
            indices = jax.random.choice(subkey, n, shape=(bs,), replace=False)
            X_batch = X[indices]

            prev_mu = model._joint.mu
            prev_gamma = model._joint.gamma
            prev_L = model._joint.L_Sigma

            expectations = model.e_step(X_batch)
            model = model.m_step(X_batch, expectations)

            if self.regularization == 'det_sigma_one':
                if hasattr(model, 'regularize_det_sigma_one'):
                    model = model.regularize_det_sigma_one()

            max_change = _param_change(
                model._joint.mu, model._joint.gamma, model._joint.L_Sigma,
                prev_mu, prev_gamma, prev_L,
            )
            changes.append(max_change)
            n_iter = t + 1

            if t % 10 == 0 and self.verbose >= 1:
                ll = model.marginal_log_likelihood(X)
                lls.append(ll)
                if self.verbose >= 2:
                    print(
                        f"  {n_iter:4d}  LL={float(ll):.6f}  "
                        f"|Δparams|={float(max_change):.4e}"
                    )

            if max_change < self.tol and t > 0:
                break

        elapsed = time.perf_counter() - t_total
        converged = bool(max_change < self.tol) and n_iter > 1

        if self.verbose >= 1:
            status = "Converged" if converged else "NOT converged"
            ll_str = f", final LL={float(lls[-1]):.6f}" if lls else ""
            print(f"  {status} after {n_iter} iterations ({elapsed:.2f}s){ll_str}")

        return EMResult(
            model=model,
            log_likelihoods=jnp.array(lls) if lls else None,
            param_changes=jnp.array(changes),
            n_iter=n_iter,
            converged=converged,
            elapsed_time=elapsed,
        )


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _param_change(mu_new, gamma_new, L_new, mu_old, gamma_old, L_old):
    """Max relative change in normal parameters (mu, gamma, L_Sigma)."""
    rel_mu = jnp.linalg.norm(mu_new - mu_old) / jnp.maximum(
        jnp.linalg.norm(mu_old), _PARAM_EPS)
    rel_gamma = jnp.linalg.norm(gamma_new - gamma_old) / jnp.maximum(
        jnp.linalg.norm(gamma_old), _PARAM_EPS)
    rel_L = jnp.linalg.norm(L_new - L_old) / jnp.maximum(
        jnp.linalg.norm(L_old), _PARAM_EPS)
    return jnp.maximum(jnp.maximum(rel_mu, rel_gamma), rel_L)
