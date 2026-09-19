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
    """Result of an EM fitting procedure.

    ``converged`` and ``diverged`` are separate because batch EM has three
    outcomes: tolerance met (``converged=True``), a non-finite iterate was
    reverted (``diverged=True``), or ``max_iter`` was exhausted with neither
    (both ``False``). They are not opposites — ``converged=False`` covers both
    divergence and a finite but not-yet-converged stop.
    """
    model: Any                              # fitted distribution (eqx.Module)
    log_likelihoods: Optional[jax.Array]    # (n_iter,) conjugacy mean LL, or None
    param_changes: jax.Array                # (n_iter,) hybrid-RMS diagnostic
    n_iter: int
    converged: Optional[bool]               # None for IncrementalEMFitter (no convergence criterion)
    elapsed_time: float                     # wall-clock seconds
    diverged: bool = False                  # True when a non-finite iterate was reverted


# ---------------------------------------------------------------------------
# BatchEMFitter
# ---------------------------------------------------------------------------

class BatchEMFitter:
    """
    Batch EM / MCECM algorithm with dual-loop architecture.

    **EM** (default): E-step → M-step (all params) → regularize.

    **MCECM**: E-step → M-step (normal params only) → regularize →
    E-step → M-step (subordinator only).

    Convergence is the Aitken remaining gap of the mean log-likelihood
    :math:`\\ell_n` (nats per observation) from the E-step conjugacy
    identity; stop when :math:`\\ell_\\infty - \\ell_{t+1} <` ``tol``.
    Three consecutive :math:`\\ell` values are required, so the earliest
    stop is ``n_iter=3``. Hybrid-scale RMS on
    :meth:`~normix.mixtures.marginal.MarginalMixture.em_convergence_params`
    is recorded in :attr:`EMResult.param_changes` as a diagnostic, not
    the stop. Covariance regularisations leave the density unchanged, so
    they share this criterion. Under scalar-:math:`\\tau` MAP shrinkage
    the tracked sequence is the penalised objective.

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
        Convergence tolerance on the Aitken remaining gap of
        :math:`\\ell_n` in nats per observation. ``tol=1e-3`` means the
        extrapolated remaining improvement is less than :math:`10^{-3}`
        nats/obs.
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
    track_ll : bool
        When ``True``, record the per-iteration conjugacy mean
        log-likelihood (the Aitken sequence) in
        :attr:`EMResult.log_likelihoods` without requiring
        ``verbose >= 1``. These are :math:`\\ell_n(\\theta_t)` at the
        E-step parameters, not a second Bessel pass through
        :meth:`~normix.mixtures.marginal.MarginalMixture.marginal_log_likelihood`.
    m_step_kwargs : dict or None
        Extra keyword arguments forwarded verbatim to every ``m_step`` /
        ``m_step_subordinator`` call (in addition to ``backend`` and
        ``method``). Used to thread estimand controls such as the VG
        ``alpha_min`` shape bound down to the subordinator's
        ``from_expectation``. Values must be static (e.g. Python floats) to
        stay compatible with the ``lax.scan`` path. ``None`` = no extras.
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
        track_ll: bool = False,
        m_step_kwargs: Optional[dict] = None,
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
        self.track_ll = track_ll
        self.m_step_kwargs = dict(m_step_kwargs) if m_step_kwargs else {}
        # Test hook: force a non-finite max_change at this 1-based step.
        # Used by ``tests/test_em_regression.py`` (``TestEMDivergenceGuard``).
        self._force_nonfinite_at_step: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, model, X: jax.Array) -> EMResult:
        """
        Run batch EM or MCECM. Auto-selects lax.scan or Python loop.

        Parameters
        ----------
        model : NormalMixture subclass (used as initial parameters)
        X : jax.Array
            Data array, shape ``(n, d)``.

        Returns
        -------
        EMResult with fitted model, convergence diagnostics, and timing.
        """
        X = jnp.asarray(X, dtype=jnp.float64)
        dist_name = type(model).__name__

        target_log_det = None
        if self.regularization == 'det_sigma_x':
            target_log_det = jnp.asarray(
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
            return self._fit_scan(model, X, target_log_det)
        else:
            if self.verbose >= 1:
                self._print_header(dist_name)
            return self._fit_loop(model, X, target_log_det)

    # ------------------------------------------------------------------
    # Pure EM / MCECM step (no prints, no float(), no list.append())
    # ------------------------------------------------------------------

    def _step(self, model, X, eta_state=None, *, target_log_det=None):
        """One iteration (EM or MCECM).

        Returns ``(new_model, max_change, eta_state, objective)``.
        ``objective`` is :math:`\\ell_n(\\theta_t)` from the E-step (MCECM:
        first E-step of the cycle), plus the MAP penalty when
        :meth:`_penalized_objective` applies.
        """
        if self.algorithm == 'mcecm':
            return self._mcecm_step(model, X, eta_state, target_log_det=target_log_det)
        return self._em_step(model, X, eta_state, target_log_det=target_log_det)

    def _em_step(self, model, X, eta_state=None, *, target_log_det=None):
        """One EM iteration: E → (eta_update) → M(all) → regularize.

        Returns ``(model, max_change, eta_state, objective)``.
        """
        prev_params = model.em_convergence_params()

        estep = model.e_step(X, backend=self.e_step_backend)
        eta, log_lik = estep.eta, estep.log_lik
        objective = self._penalized_objective(model, log_lik)

        if eta_state is not None:
            eta_prev, rule_state, step = eta_state
            eta, rule_state = self.eta_update(
                eta_prev, eta, step, X.shape[0], rule_state)
            eta_state = (eta, rule_state, step + 1)

        model = model.m_step(
            eta, backend=self.m_step_backend, method=self.m_step_method,
            **self.m_step_kwargs)
        model = self._regularize(model, target_log_det)

        max_change = _param_change(model.em_convergence_params(), prev_params)
        return model, max_change, eta_state, objective

    def _mcecm_step(self, model, X, eta_state=None, *, target_log_det=None):
        """One MCECM iteration: E → M_normal → regularize → E → M_subordinator.

        Returns ``(model, max_change, eta_state, objective)``. The
        objective is :math:`\\ell_n` from the first E-step of the cycle.
        """
        prev_params = model.em_convergence_params()

        estep = model.e_step(X, backend=self.e_step_backend)
        eta, log_lik = estep.eta, estep.log_lik
        objective = self._penalized_objective(model, log_lik)

        if eta_state is not None:
            eta_prev, rule_state, step = eta_state
            eta, rule_state = self.eta_update(
                eta_prev, eta, step, X.shape[0], rule_state)
            eta_state = (eta, rule_state, step + 1)

        model = model.m_step_normal(eta)
        model = self._regularize(model, target_log_det)

        eta = model.e_step(X, backend=self.e_step_backend).eta
        model = model.m_step_subordinator(
            eta, backend=self.m_step_backend, method=self.m_step_method,
            **self.m_step_kwargs)

        max_change = _param_change(model.em_convergence_params(), prev_params)
        return model, max_change, eta_state, objective

    def _map_penalty_applies(self, model) -> bool:
        r"""True when :meth:`_penalized_objective` adds the MAP term.

        Scalar-:math:`\tau` :class:`~normix.fitting.eta_rules.Shrinkage`
        wrapping :class:`~normix.fitting.eta_rules.IdentityUpdate` on a
        model that exposes joint :math:`(\theta,\psi)` (a
        :class:`~normix.mixtures.marginal.NormalMixture`) and a
        :class:`~normix.fitting.eta.NormalMixtureEta` target. Factor
        mixtures have no joint :math:`\theta`, so the penalty is skipped.
        """
        rule = self.eta_update
        if rule is None:
            return False
        from normix.fitting.eta import NormalMixtureEta
        from normix.fitting.eta_rules import IdentityUpdate, Shrinkage
        if not isinstance(rule, Shrinkage):
            return False
        if not isinstance(rule.base, IdentityUpdate):
            return False
        if type(rule.tau) is type(rule.eta0):
            return False
        joint = getattr(model, '_joint', None)
        return joint is not None and isinstance(rule.eta0, NormalMixtureEta)

    def _penalized_objective(self, model, log_lik: jax.Array) -> jax.Array:
        r"""MAP objective :math:`\ell_n + \tau(\theta\cdot\eta_0 - \psi)` when it is maximised.

        See :meth:`_map_penalty_applies`. Otherwise returns :math:`\ell_n`.
        """
        if not self._map_penalty_applies(model):
            return log_lik
        rule = self.eta_update
        joint = model._joint
        eta0 = rule.eta0
        eta0_flat = jnp.concatenate([
            jnp.stack([eta0.E_log_Y, eta0.E_inv_Y, eta0.E_Y]),
            jnp.ravel(eta0.E_X),
            jnp.ravel(eta0.E_X_inv_Y),
            jnp.ravel(eta0.E_XXT_inv_Y),
        ])
        theta = joint.natural_params()
        psi = joint.log_partition()
        return log_lik + rule.tau * (jnp.dot(theta, eta0_flat) - psi)

    def _objective_should_increase(self, model) -> bool:
        """True when a negative :math:`\\Delta` is an EM bug, not an η-update artefact."""
        if self.eta_update is None:
            return True
        from normix.fitting.eta_rules import IdentityUpdate
        if isinstance(self.eta_update, IdentityUpdate):
            return True
        return self._map_penalty_applies(model)

    # ------------------------------------------------------------------
    # lax.scan path (JIT-able, requires backend='jax')
    # ------------------------------------------------------------------

    def _fit_scan(self, model, X, target_log_det) -> EMResult:
        """JIT-able EM loop via :func:`~jax.lax.scan`.

        The scan carry keeps ``converged`` and ``diverged`` as separate booleans
        rather than a single status code: convergence accepts the new model,
        divergence reverts to the previous one, and ``done = converged | diverged``
        freezes both paths. Two flags also map directly onto :class:`EMResult`
        and compose cleanly with ``jnp.where`` / ``|`` inside the scanned body.
        """
        t0 = time.perf_counter()
        track_ll = self.track_ll

        def body(carry, _):
            mdl, converged, diverged, n, ll_prev, ll_prev2 = carry
            mdl_new, max_change, _, obj = self._step(
                mdl, X, target_log_det=target_log_det)
            step_no = n + 1
            force_at = self._force_nonfinite_at_step
            if force_at is not None:
                max_change = jnp.where(step_no == force_at, jnp.nan, max_change)
            finite = jnp.isfinite(max_change) & jnp.isfinite(obj)
            done = converged | diverged
            remaining = _aitken_remaining(obj, ll_prev, ll_prev2)
            ready = step_no >= 3
            # Freeze both flags after a prior stop so scan padding cannot
            # flip converged↔diverged on an already-accepted / reverted model.
            conv_new = ~done & finite & ready & (remaining < self.tol)
            diverged_new = diverged | (~finite & ~done)
            keep_old = done | ~finite
            mdl_out = jax.tree.map(
                lambda new, old: jnp.where(keep_old, old, new), mdl_new, mdl)
            change_out = jnp.where(done, 0.0, max_change)
            n_out = n + jnp.where(done, 0, 1)
            ll_prev2_out = jnp.where(keep_old, ll_prev2, ll_prev)
            ll_prev_out = jnp.where(keep_old, ll_prev, obj)
            obj_out = jnp.where(done, ll_prev, obj)
            carry_out = (
                mdl_out, converged | conv_new, diverged_new, n_out,
                ll_prev_out, ll_prev2_out,
            )
            if track_ll:
                return carry_out, (change_out, obj_out)
            return carry_out, change_out

        init = (
            model, jnp.bool_(False), jnp.bool_(False), jnp.int32(0),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        if track_ll:
            (
                (final_model, converged, diverged, n_iter, last_ll, _),
                (param_changes, lls),
            ) = jax.lax.scan(body, init, None, length=self.max_iter)
            log_likelihoods = lls
        else:
            (final_model, converged, diverged, n_iter, last_ll, _), param_changes = (
                jax.lax.scan(body, init, None, length=self.max_iter))
            log_likelihoods = None

        n_keep = int(n_iter)
        param_changes = param_changes[:n_keep]
        if log_likelihoods is not None:
            log_likelihoods = log_likelihoods[:n_keep]

        elapsed = time.perf_counter() - t0

        if self.verbose >= 1:
            if log_likelihoods is None:
                log_likelihoods = jnp.array([last_ll])
            ll = float(last_ll)
            if diverged:
                status = "Diverged (non-finite iterate; kept last finite model)"
            elif converged:
                status = "Converged"
            else:
                status = "NOT converged"
            print(
                f"  {status} after {int(n_iter)} iterations "
                f"({elapsed:.2f}s), E-step LL={ll:.6f}"
            )

        return EMResult(
            model=final_model,
            log_likelihoods=log_likelihoods,
            param_changes=param_changes,
            n_iter=int(n_iter),
            converged=bool(converged),
            elapsed_time=elapsed,
            diverged=bool(diverged),
        )

    # ------------------------------------------------------------------
    # Python loop path (works with any backend, supports verbose >= 2)
    # ------------------------------------------------------------------

    def _fit_loop(self, model, X, target_log_det) -> EMResult:
        t_total = time.perf_counter()
        lls = []
        changes = []
        prev_ll = None
        ll_prev = None
        ll_prev2 = None
        n_iter = 0
        diverged = False
        remaining = jnp.array(jnp.inf)
        max_change = jnp.array(jnp.inf)

        eta_state = None
        if self.eta_update is not None:
            eta_prev = model.compute_eta_from_model()
            rule_state = self.eta_update.initial_state()
            eta_state = (eta_prev, rule_state, 0)

        if self.verbose >= 2:
            self._print_table_header()

        for i in range(self.max_iter):
            t_iter = time.perf_counter()
            prev_model = model
            model, max_change, eta_state, obj = self._step(
                model, X, eta_state, target_log_det=target_log_det)
            force_at = self._force_nonfinite_at_step
            if force_at is not None and (i + 1) == force_at:
                max_change = jnp.array(jnp.nan)
            dt = time.perf_counter() - t_iter
            changes.append(max_change)
            n_iter = i + 1

            if not (bool(jnp.isfinite(max_change)) and bool(jnp.isfinite(obj))):
                model = prev_model
                diverged = True
                if self.verbose >= 1:
                    print(
                        "  EM diverged (non-finite parameter change); "
                        "keeping last finite iterate."
                    )
                break

            remaining = jnp.array(jnp.inf)
            if n_iter >= 3:
                remaining = _aitken_remaining(obj, ll_prev, ll_prev2)

            if self.track_ll or self.verbose >= 1:
                lls.append(obj)

            if (
                self.verbose >= 1
                and self._objective_should_increase(prev_model)
                and ll_prev is not None
                and float(obj - ll_prev) < 0.0
            ):
                print(
                    "  EM objective decreased "
                    f"(Δ={float(obj - ll_prev):+.4e} nats/obs); "
                    "Aitken remaining is not a stop."
                )

            if self.verbose >= 2:
                dll_str = "    ---    "
                if prev_ll is not None:
                    dll_str = f"{float(obj - prev_ll):+.4e}"
                rem_str = (
                    f"{float(remaining):.4e}" if n_iter >= 3 else "   n/a   "
                )
                print(
                    f"  {n_iter:4d}  {float(obj):.6f}  {dll_str}  "
                    f"{rem_str}  {float(max_change):.4e}  {dt:.3f}s"
                )
                prev_ll = obj

            ll_prev2 = ll_prev
            ll_prev = obj

            if n_iter >= 3 and bool(remaining < self.tol):
                break

        elapsed = time.perf_counter() - t_total
        converged = (
            not diverged
            and n_iter >= 3
            and bool(remaining < self.tol)
        )

        if self.verbose >= 2:
            self._print_footer(n_iter, converged, elapsed,
                               float(lls[-1]) if lls else None,
                               float(changes[-1]),
                               float(remaining) if n_iter >= 3 else None)
        elif self.verbose == 1:
            if diverged:
                status = "Diverged (non-finite iterate; kept last finite model)"
            else:
                status = "Converged" if converged else "NOT converged"
            ll_str = f", E-step LL={float(lls[-1]):.6f}" if lls else ""
            print(f"  {status} after {n_iter} iterations ({elapsed:.2f}s){ll_str}")

        return EMResult(
            model=model,
            log_likelihoods=jnp.array(lls) if lls else None,
            param_changes=jnp.array(changes),
            n_iter=n_iter,
            converged=converged,
            elapsed_time=elapsed,
            diverged=diverged,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _regularize(self, model, target_log_det=None):
        if self.regularization == 'none':
            return model
        if self.regularization == 'det_sigma_one':
            if hasattr(model, 'regularize_det_sigma'):
                return model.regularize_det_sigma(0.0)
            return model
        if self.regularization == 'det_sigma_x':
            if hasattr(model, 'regularize_det_sigma'):
                return model.regularize_det_sigma(target_log_det)
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
        print(f"  Tolerance    : {self.tol:.1e} nats/obs (Aitken remaining)")
        print(f"  Max iters    : {self.max_iter}")

    def _print_table_header(self):
        print("-" * 60)
        print("  Iter    Log-Lik       ΔLL        remaining   |Δparams|   Time")
        print("-" * 60)

    def _print_footer(self, n_iter, converged, elapsed, final_ll, final_change,
                      remaining=None):
        sep = "=" * 60
        print(sep)
        status = "Converged" if converged else "NOT converged"
        print(f"  {status} after {n_iter} iterations ({elapsed:.2f}s)")
        if final_ll is not None:
            print(f"  Final log-likelihood : {final_ll:.6f}")
        if remaining is not None:
            print(f"  Aitken remaining     : {remaining:.4e}")
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

    def fit(self, model, X: jax.Array, *, key: jax.Array) -> EMResult:
        """Run incremental EM. Returns :class:`EMResult`."""
        X = jnp.asarray(X, dtype=jnp.float64)
        n = int(X.shape[0])
        bs = min(self.batch_size, n)
        dist_name = type(model).__name__

        target_log_det = None
        if self.regularization == 'det_sigma_x':
            target_log_det = jnp.asarray(
                model.log_det_sigma(), dtype=jnp.float64)

        step_keys = _materialize_incremental_subkeys(key, self.max_steps)
        use_scan = (
            self.verbose == 0
            and self.e_step_backend == 'jax'
            and self.m_step_backend == 'jax'
        )

        if use_scan:
            return self._fit_incremental_scan(
                model, X, n, bs, step_keys, target_log_det)

        return self._fit_incremental_python(
            model, X, n, bs, step_keys, dist_name, target_log_det)

    def _fit_incremental_scan(
        self,
        model,
        X: jax.Array,
        n_obs: int,
        batch_sz: int,
        step_keys: jax.Array,
        target_log_det,
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
                        mj, X_batch, mb, mm, target_log_det),
                    model_c,
                )
                eta_new = model_mid.compute_eta_from_model()
            else:
                model_mid = model_c
                eta_new = model_mid.e_step(X_batch, backend=eb).eta

            eta_run, rule_state = self.eta_update(
                eta_run, eta_new, step, bs_i, rule_state)
            model_next = model_mid.m_step(
                eta_run, backend=mb, method=mm)
            model_next = self._regularize(model_next, target_log_det)

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
        target_log_det,
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
                    eta = model.e_step(X_batch, backend=eb).eta
                    model = model.m_step(
                        eta, backend=mb, method=mm)
                    model = self._regularize(model, target_log_det)
                eta_new = model.compute_eta_from_model()
            else:
                eta_new = model.e_step(X_batch, backend=eb).eta

            eta_prev, rule_state = self.eta_update(
                eta_prev, eta_new, step, bs, rule_state)
            model = model.m_step(
                eta_prev, backend=mb, method=mm)
            model = self._regularize(model, target_log_det)

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
        self,
        model_inner,
        X_batch: jax.Array,
        mb: str,
        mm: str,
        target_log_det,
    ):
        eta_inner = model_inner.e_step(X_batch, backend=self.e_step_backend).eta
        mi2 = model_inner.m_step(
            eta_inner, backend=mb, method=mm)
        return self._regularize(mi2, target_log_det)

    # Reuse the dispatch from BatchEMFitter so behaviour stays in sync.
    _regularize = BatchEMFitter._regularize


def _aitken_remaining(
    ll: jax.Array, ll_prev: jax.Array, ll_prev2: jax.Array,
) -> jax.Array:
    r"""Aitken remaining gap :math:`\ell_\infty - \ell_{t+1}`.

    With :math:`\Delta_1 = \ell_t - \ell_{t-1}`,
    :math:`\Delta_2 = \ell_{t+1} - \ell_t`, :math:`a = \Delta_2/\Delta_1`,

    .. math::

        \ell_\infty - \ell_{t+1} = \frac{a}{1-a}\,\Delta_2

    when :math:`\Delta_1 > 0` and :math:`0 \le a < 1`. Both increments
    at a few ulps of :math:`\ell` yield remaining 0 (stalled). Otherwise
    ``inf`` (do not stop).
    """
    delta1 = ll_prev - ll_prev2
    delta2 = ll - ll_prev
    eps = jnp.finfo(jnp.float64).eps
    scale = 1.0 + jnp.maximum(
        jnp.abs(ll), jnp.maximum(jnp.abs(ll_prev), jnp.abs(ll_prev2)))
    stalled = (
        (jnp.abs(delta1) <= eps * scale)
        & (jnp.abs(delta2) <= eps * scale)
    )
    safe_delta1 = jnp.where(delta1 == 0.0, jnp.ones_like(delta1), delta1)
    a = delta2 / safe_delta1
    valid = (delta1 > 0.0) & (a >= 0.0) & (a < 1.0)
    remaining = delta2 * a / jnp.maximum(1.0 - a, eps)
    return jnp.where(
        stalled, jnp.zeros_like(ll),
        jnp.where(valid, remaining, jnp.full_like(ll, jnp.inf)),
    )


def _rms(x: jax.Array) -> jax.Array:
    """Root-mean-square of a leaf: ``||x||_2 / sqrt(m)`` with ``m = x.size``."""
    m = jnp.maximum(x.size, 1)
    return jnp.linalg.norm(x) / jnp.sqrt(m.astype(x.dtype))


def _param_change(new_params, old_params) -> jax.Array:
    """Max hybrid-scale RMS change across leaves of a model's convergence pytree.

    Both pytrees come from :meth:`MarginalMixture.em_convergence_params`
    (called before and after the iteration). Per-leaf change is
    ``rms(new - old) / (1 + rms(old))`` with
    ``rms(v) = ||v||_2 / sqrt(m)``; the overall measure is the maximum
    across leaves. Stored in :attr:`EMResult.param_changes` as a
    diagnostic; stopping uses :func:`_aitken_remaining`.

    RMS (rather than raw L2) keeps a fixed ``tol`` roughly dimension-free:
    the same per-coordinate drift yields the same leaf score for
    :math:`d=1` and large :math:`d`, and prevents the :math:`d^2`-sized
    :math:`L_\\Sigma` leaf from dominating :math:`\\mu` / :math:`\\gamma`.
    The additive ``1`` keeps the criterion well-behaved when a leaf is
    near zero (common for :math:`\\mu` in centred returns); a pure
    relative ``||Δ|| / ||old||`` then inflates tiny absolute drifts along
    the :math:`(\\mu, \\gamma)` ridge.
    """
    leaves_new = jax.tree.leaves(new_params)
    leaves_old = jax.tree.leaves(old_params)
    rels = jnp.stack([
        _rms(n - o) / (1.0 + _rms(o))
        for n, o in zip(leaves_new, leaves_old)
    ])
    return jnp.max(rels)
