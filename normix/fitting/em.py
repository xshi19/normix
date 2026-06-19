"""
EM fitters for normix distributions.

Model knows math, fitter knows iteration.

  BatchEMFitter        — standard batch EM with dual-loop architecture:
                         lax.scan (JIT-able) or Python for-loop (CPU-compatible)
  IncrementalEMFitter  — online / mini-batch EM with pluggable eta update rules
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp

from normix.utils.constants import PARAM_CHANGE_EPS


def _materialize_incremental_subkeys(key: jax.Array, max_steps: int) -> jax.Array:
    """Stack PRNG keys for each incremental step via a sequential ``split`` chain.

    Matches RNG usage of iterating ``jax.random.split`` in a Python loop, so a
    :func:`~jax.lax.scan` replay uses the same subkeys batching over steps.
    """
    if max_steps <= 0:
        return jnp.empty((0, *key.shape), dtype=key.dtype)
    rows: list[jax.Array] = []
    k = key
    for _ in range(max_steps):
        k, sk = jax.random.split(k)
        rows.append(sk)
    return jnp.stack(rows)


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
    converged: Optional[bool]               # None for IncrementalEMFitter (no convergence criterion)
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
      - lax.scan when both backends are 'jax', verbose <= 1, algorithm='em',
        and no eta_update rule
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
        Strategy applied after each M-step:

        * ``'none'`` — no regularization.
        * ``'det_sigma_one'`` — enforce :math:`|\\Sigma| = 1` (the
          original GH convention; equivalent to
          ``regularize_det_sigma(target_log_det=0)``).
        * ``'det_sigma_x'`` — enforce :math:`|\\Sigma| = |\\Sigma_0|`
          where :math:`\\Sigma_0` is the dispersion of the *initial*
          model passed to :meth:`fit`. Useful when comparing GH /
          FactorGH parameters against VG / NIG / NInvG, which leave
          :math:`|\\Sigma|` at the empirical scale.
        * ``'a_eq_b'`` — rescale the GIG subordinator so
          :math:`a = b = \\sqrt{ab}`. Trivial no-op for VG / NInvG /
          MultivariateNormal.
    e_step_backend : str
        'jax' (default) or 'cpu'.
    m_step_backend : str
        'jax' or 'cpu' (default, faster for GIG).
    m_step_method : str
        'newton' (default), 'lbfgs', or 'bfgs'.
    eta_update : EtaUpdateRule or None
        Optional eta combination rule (e.g. ``Shrinkage(IdentityUpdate(),
        eta0, tau)``). When set, the E-step output is transformed before
        the M-step.
    """

    _REGULARIZATIONS = ('none', 'det_sigma_one', 'det_sigma_x', 'a_eq_b')

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
        eta_update=None,
    ):
        if algorithm not in ('em', 'mcecm'):
            raise ValueError(f"algorithm must be 'em' or 'mcecm', got {algorithm!r}")
        if regularization not in self._REGULARIZATIONS:
            raise ValueError(
                f"regularization must be one of {self._REGULARIZATIONS}, "
                f"got {regularization!r}")
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.regularization = regularization
        self.e_step_backend = e_step_backend
        self.m_step_backend = m_step_backend
        self.m_step_method = m_step_method
        self.eta_update = eta_update
        # Set by ``fit`` for ``regularization='det_sigma_x'``: the
        # log-determinant of the initial model's Σ, captured once.
        self._target_log_det: jax.Array | None = None

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

        # Capture initial log|Σ| once so 'det_sigma_x' targets the
        # initial reference scale throughout EM.
        if self.regularization == 'det_sigma_x':
            self._target_log_det = jnp.asarray(
                model.log_det_sigma(), dtype=jnp.float64)

        use_scan = (
            self.algorithm == 'em'
            and self.e_step_backend == 'jax'
            and self.m_step_backend == 'jax'
            and self.verbose <= 1
            and self.eta_update is None
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

    def _step(self, model, X, eta_state=None):
        """One iteration (EM or MCECM). Returns (new_model, max_change, eta_state)."""
        if self.algorithm == 'mcecm':
            return self._mcecm_step(model, X, eta_state)
        return self._em_step(model, X, eta_state)

    def _em_step(self, model, X, eta_state=None):
        """One EM iteration: E → (eta_update) → M(all) → regularize.

        Returns ``(model, max_change, eta_state)``.
        """
        prev_params = model.em_convergence_params()

        eta = model.e_step(X, backend=self.e_step_backend)

        if eta_state is not None:
            eta_prev, rule_state, step = eta_state
            eta, rule_state = self.eta_update(
                eta_prev, eta, step, X.shape[0], rule_state)
            eta_state = (eta, rule_state, step + 1)

        model = model.m_step(
            eta, backend=self.m_step_backend, method=self.m_step_method)
        model = self._regularize(model)

        max_change = _param_change(model.em_convergence_params(), prev_params)
        return model, max_change, eta_state

    def _mcecm_step(self, model, X, eta_state=None):
        """One MCECM iteration: E → M_normal → regularize → E → M_subordinator.

        Returns ``(model, max_change, eta_state)``.
        """
        prev_params = model.em_convergence_params()

        eta = model.e_step(X, backend=self.e_step_backend)

        if eta_state is not None:
            eta_prev, rule_state, step = eta_state
            eta, rule_state = self.eta_update(
                eta_prev, eta, step, X.shape[0], rule_state)
            eta_state = (eta, rule_state, step + 1)

        model = model.m_step_normal(eta)
        model = self._regularize(model)

        eta = model.e_step(X, backend=self.e_step_backend)
        model = model.m_step_subordinator(
            eta, backend=self.m_step_backend, method=self.m_step_method)

        max_change = _param_change(model.em_convergence_params(), prev_params)
        return model, max_change, eta_state

    # ------------------------------------------------------------------
    # lax.scan path (JIT-able, requires backend='jax')
    # ------------------------------------------------------------------

    def _fit_scan(self, model, X) -> EMResult:
        t0 = time.perf_counter()

        def body(carry, _):
            mdl, converged, n = carry
            mdl_new, max_change, _ = self._step(mdl, X)
            conv_new = max_change < self.tol
            mdl_out = jax.tree.map(
                lambda n, o: jnp.where(converged, o, n), mdl_new, mdl)
            change_out = jnp.where(converged, 0.0, max_change)
            n_out = n + jnp.where(converged, 0, 1)
            return (mdl_out, converged | conv_new, n_out), change_out

        init = (model, jnp.bool_(False), jnp.int32(0))
        (final_model, converged, n_iter), param_changes = jax.lax.scan(
            body, init, None, length=self.max_iter)

        elapsed = time.perf_counter() - t0

        log_likelihoods = None
        if self.verbose >= 1:
            ll = final_model.marginal_log_likelihood(X)
            log_likelihoods = jnp.array([ll])
            status = "Converged" if bool(converged) else "NOT converged"
            print(
                f"  {status} after {int(n_iter)} iterations "
                f"({elapsed:.2f}s), final LL={float(ll):.6f}"
            )

        return EMResult(
            model=final_model,
            log_likelihoods=log_likelihoods,
            param_changes=param_changes,
            n_iter=n_iter,
            converged=converged,
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

        eta_state = None
        if self.eta_update is not None:
            eta_prev = model.compute_eta_from_model()
            rule_state = self.eta_update.initial_state()
            eta_state = (eta_prev, rule_state, 0)

        if self.verbose >= 2:
            self._print_table_header()

        for i in range(self.max_iter):
            t_iter = time.perf_counter()
            model, max_change, eta_state = self._step(model, X, eta_state)
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

    def _regularize(self, model):
        if self.regularization == 'none':
            return model
        if self.regularization == 'det_sigma_one':
            if hasattr(model, 'regularize_det_sigma'):
                return model.regularize_det_sigma(0.0)
            return model
        if self.regularization == 'det_sigma_x':
            if hasattr(model, 'regularize_det_sigma'):
                return model.regularize_det_sigma(self._target_log_det)
            return model
        if self.regularization == 'a_eq_b':
            if hasattr(model, 'regularize_a_eq_b'):
                return model.regularize_a_eq_b()
            return model
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
# IncrementalEMFitter
# ---------------------------------------------------------------------------

class IncrementalEMFitter:
    """
    Incremental EM with pluggable eta update rules.

    Replaces ``OnlineEMFitter`` and ``MiniBatchEMFitter``.  Processes data
    in random mini-batches, applies an :class:`~normix.fitting.eta_rules.EtaUpdateRule`
    to combine the running :math:`\\eta` with each batch estimate, then
    M-steps on the combined :math:`\\eta`.

    Parameters
    ----------
    eta_update : EtaUpdateRule
        How to combine running η with each batch estimate.
    batch_size : int
        Observations per batch.
    max_steps : int
        Number of batches to process (total budget).
    inner_iter : int
        1 = online (default); >1 = fine-tuning on each batch.
    verbose : int
        0 = silent; 1 = periodic summary.

        Scan path (**JAX backends only**):

        ``verbose`` must be ``0`` so diagnostics do not rely on Python
        side effects each step.

    regularization : str
        Same options as :class:`BatchEMFitter` —
        ``'none'`` | ``'det_sigma_one'`` | ``'det_sigma_x'`` |
        ``'a_eq_b'``.
    e_step_backend, m_step_backend, m_step_method : str
        Passed through to ``e_step`` / ``m_step``.
    """

    _REGULARIZATIONS = BatchEMFitter._REGULARIZATIONS

    def __init__(
        self,
        *,
        eta_update=None,
        batch_size: int = 256,
        max_steps: int = 200,
        inner_iter: int = 1,
        verbose: int = 0,
        regularization: str = 'none',
        e_step_backend: str = 'jax',
        m_step_backend: str = 'cpu',
        m_step_method: str = 'newton',
    ):
        if eta_update is None:
            from normix.fitting.eta_rules import RobbinsMonroUpdate
            eta_update = RobbinsMonroUpdate(tau0=10.0)
        if regularization not in self._REGULARIZATIONS:
            raise ValueError(
                f"regularization must be one of {self._REGULARIZATIONS}, "
                f"got {regularization!r}")
        self.eta_update = eta_update
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.inner_iter = inner_iter
        self.verbose = verbose
        self.regularization = regularization
        self.e_step_backend = e_step_backend
        self.m_step_backend = m_step_backend
        self.m_step_method = m_step_method
        self._target_log_det: jax.Array | None = None

    def fit(self, model, X: jax.Array, *, key: jax.Array) -> EMResult:
        """Run incremental EM. Returns :class:`EMResult`."""
        X = jnp.asarray(X, dtype=jnp.float64)
        n = int(X.shape[0])
        bs = min(self.batch_size, n)
        dist_name = type(model).__name__

        if self.regularization == 'det_sigma_x':
            self._target_log_det = jnp.asarray(
                model.log_det_sigma(), dtype=jnp.float64)

        step_keys = _materialize_incremental_subkeys(key, self.max_steps)
        use_scan = (
            self.verbose == 0
            and self.e_step_backend == 'jax'
            and self.m_step_backend == 'jax'
        )

        if use_scan:
            return self._fit_incremental_scan(model, X, n, bs, step_keys)

        return self._fit_incremental_python(
            model, X, n, bs, step_keys, dist_name)

    def _fit_incremental_scan(
        self,
        model,
        X: jax.Array,
        n_obs: int,
        batch_sz: int,
        step_keys: jax.Array,
    ) -> EMResult:
        t_total = time.perf_counter()

        eb = self.e_step_backend
        mb = self.m_step_backend
        mm = self.m_step_method

        eta_init = model.compute_eta_from_model()
        rs_init = self.eta_update.initial_state()

        bs_i = jnp.asarray(batch_sz, dtype=jnp.int32)

        if self.max_steps == 0:
            elapsed = time.perf_counter() - t_total
            return EMResult(
                model=model,
                log_likelihoods=None,
                param_changes=jnp.empty((0,), dtype=jnp.float64),
                n_iter=0,
                converged=None,
                elapsed_time=elapsed,
            )

        def outer_body(carry, inputs):
            model_c, eta_run, rule_state = carry
            sk, step = inputs
            indices = jax.random.choice(
                sk, n_obs, shape=(batch_sz,), replace=False)
            X_batch = X[indices]

            prev_params = model_c.em_convergence_params()

            if self.inner_iter > 1:
                model_mid = jax.lax.fori_loop(
                    0,
                    self.inner_iter,
                    lambda _, mj: self._incremental_inner_step_jax(
                        mj, X_batch, mb, mm),
                    model_c,
                )
                eta_new = model_mid.compute_eta_from_model()
            else:
                model_mid = model_c
                eta_new = model_mid.e_step(X_batch, backend=eb)

            eta_run, rule_state = self.eta_update(
                eta_run, eta_new, step, bs_i, rule_state)
            model_next = model_mid.m_step(
                eta_run, backend=mb, method=mm)
            model_next = self._regularize(model_next)

            max_change = _param_change(
                model_next.em_convergence_params(), prev_params)
            return (model_next, eta_run, rule_state), max_change

        scan_xs = (
            step_keys,
            jnp.arange(self.max_steps, dtype=jnp.int32),
        )

        init = (model, eta_init, rs_init)
        (final_model, _, _), param_changes = jax.lax.scan(
            outer_body, init, scan_xs)

        elapsed = time.perf_counter() - t_total

        return EMResult(
            model=final_model,
            log_likelihoods=None,
            param_changes=param_changes,
            n_iter=self.max_steps,
            converged=None,
            elapsed_time=elapsed,
        )

    def _fit_incremental_python(
        self,
        model,
        X: jax.Array,
        n: int,
        bs: int,
        step_keys: jax.Array,
        dist_name: str,
    ) -> EMResult:
        t_total = time.perf_counter()
        changes = []
        lls = []

        eb = self.e_step_backend
        mb = self.m_step_backend
        mm = self.m_step_method

        if self.verbose >= 1:
            rule_name = type(self.eta_update).__name__
            print(
                f"EM [incremental] {dist_name}: "
                f"rule={rule_name}, batch_size={bs}, "
                f"max_steps={self.max_steps}, inner_iter={self.inner_iter}"
            )

        eta_prev = model.compute_eta_from_model()
        rule_state = self.eta_update.initial_state()

        for step in range(self.max_steps):
            sk = step_keys[step]
            indices = jax.random.choice(sk, n, shape=(bs,), replace=False)
            X_batch = X[indices]

            prev_params = model.em_convergence_params()

            if self.inner_iter > 1:
                for _ in range(self.inner_iter):
                    eta = model.e_step(X_batch, backend=eb)
                    model = model.m_step(
                        eta, backend=mb, method=mm)
                    model = self._regularize(model)
                eta_new = model.compute_eta_from_model()
            else:
                eta_new = model.e_step(X_batch, backend=eb)

            eta_prev, rule_state = self.eta_update(
                eta_prev, eta_new, step, bs, rule_state)
            model = model.m_step(
                eta_prev, backend=mb, method=mm)
            model = self._regularize(model)

            max_change = _param_change(
                model.em_convergence_params(), prev_params)
            changes.append(max_change)

            if self.verbose >= 1 and (step + 1) % max(1, self.max_steps // 10) == 0:
                ll = model.marginal_log_likelihood(X)
                lls.append(ll)
                print(
                    f"  step {step+1:4d}/{self.max_steps}  "
                    f"LL={float(ll):.6f}  |Δparams|={float(max_change):.4e}"
                )

        elapsed = time.perf_counter() - t_total

        log_likelihoods = None
        if self.verbose >= 1:
            ll = model.marginal_log_likelihood(X)
            log_likelihoods = jnp.array(lls + [ll]) if lls else jnp.array([ll])
            print(f"  Done ({elapsed:.2f}s), final LL={float(ll):.6f}")

        return EMResult(
            model=model,
            log_likelihoods=log_likelihoods,
            param_changes=jnp.array(changes),
            n_iter=self.max_steps,
            converged=None,
            elapsed_time=elapsed,
        )

    def _incremental_inner_step_jax(
        self, model_inner, X_batch: jax.Array, mb: str, mm: str,
    ):
        eta_inner = model_inner.e_step(X_batch, backend=self.e_step_backend)
        mi2 = model_inner.m_step(
            eta_inner, backend=mb, method=mm)
        return self._regularize(mi2)

    # Reuse the dispatch from BatchEMFitter so behaviour stays in sync.
    _regularize = BatchEMFitter._regularize


def _param_change(new_params, old_params) -> jax.Array:
    """Max relative L2 change across leaves of a model's convergence pytree.

    Both pytrees come from :meth:`MarginalMixture.em_convergence_params`
    (called before and after the iteration). Per-leaf relative change is
    ``||new - old|| / max(||old||, eps)``; the overall measure is the
    maximum across leaves.
    """
    leaves_new = jax.tree.leaves(new_params)
    leaves_old = jax.tree.leaves(old_params)
    rels = jnp.stack([
        jnp.linalg.norm(n - o) / jnp.maximum(jnp.linalg.norm(o), PARAM_CHANGE_EPS)
        for n, o in zip(leaves_new, leaves_old)
    ])
    return jnp.max(rels)
