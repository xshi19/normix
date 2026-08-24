"""
Head-to-head: exponential-family MLE / EM vs likelihood gradient descent.

Compares the GPJax / FlowJAX recipe (Adam or L-BFGS on a softplus-reparametrised
negative log-likelihood) against normix's current fitters:

* GIG / Gamma: ``fit_mle`` = mean sufficient statistics + Bregman η→θ
* GH: batch EM (CPU E-step, ``det_sigma_one``)

No extra dependencies: Adam is hand-rolled; L-BFGS uses jaxopt (already a
runtime dep). scipy L-BFGS-B is the box-constrained NLL baseline (GIG only).

Usage:
    uv run python benchmarks/bench_gradient_fitting.py
    uv run python benchmarks/bench_gradient_fitting.py --section gig,grad
    uv run python benchmarks/bench_gradient_fitting.py --quick --save
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import normix  # noqa: F401  — suppress jaxopt deprecation, enable x64
from benchmarks.utils import save_result, hdr, sep

from normix.distributions.gamma import Gamma
from normix.distributions.generalized_inverse_gaussian import (
    GeneralizedInverseGaussian as GIG,
)
from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic as GH
from normix.fitting.em import BatchEMFitter
from normix.utils.bessel import log_kv
from normix.utils.constants import BESSEL_EPS_V

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------------------------------------------------------------------------
# Softplus reparametrisation (the GPJax / FlowJAX / paramax pattern)
# ---------------------------------------------------------------------------

def inv_softplus(x: jax.Array) -> jax.Array:
    """Inverse of ``jax.nn.softplus``; stable for large ``x``."""
    x = jnp.maximum(x, 1e-12)
    return jnp.where(x > 20.0, x, jnp.log(jnp.expm1(x)))


# ---------------------------------------------------------------------------
# Hand-rolled Adam (optax is not a core dep)
# ---------------------------------------------------------------------------

def adam_minimize(loss_fn, phi0, *, n_steps: int, lr: float):
    """JIT ``lax.scan`` Adam on a scalar ``loss_fn(phi) -> float``.

    Returns ``(phi_final, loss_trace)`` with ``loss_trace.shape == (n_steps,)``.
    """
    b1, b2, eps = 0.9, 0.999, 1e-8
    grad_fn = jax.grad(loss_fn)

    def body(carry, _):
        phi, m, v, t = carry
        g = grad_fn(phi)
        t = t + 1.0
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * jnp.square(g)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        phi = phi - lr * mhat / (jnp.sqrt(vhat) + eps)
        return (phi, m, v, t), loss_fn(phi)

    init = (
        phi0,
        jnp.zeros_like(phi0),
        jnp.zeros_like(phi0),
        jnp.array(0.0, dtype=jnp.float64),
    )
    (phi, _, _, _), losses = jax.lax.scan(body, init, None, length=n_steps)
    return phi, losses


def lbfgs_minimize(loss_fn, phi0, *, maxiter: int, tol: float = 1e-8):
    """Unconstrained jaxopt L-BFGS. Returns ``(phi, n_iter, error)``."""
    import jaxopt
    solver = jaxopt.LBFGS(
        fun=loss_fn, maxiter=maxiter, tol=tol, implicit_diff=False, jit=True,
    )
    result = solver.run(phi0)
    n_iter = int(getattr(result.state, "iter_num", maxiter))
    err = float(getattr(result.state, "error", np.nan))
    return result.params, n_iter, err


def scipy_lbfgsb(loss_and_grad, phi0_np, bounds, *, maxiter: int):
    """Box-constrained scipy L-BFGS-B on a numpy ``(f, g)`` callback."""
    from scipy.optimize import minimize
    t0 = time.perf_counter()
    res = minimize(
        lambda z: loss_and_grad(z)[0],
        np.asarray(phi0_np, dtype=np.float64),
        jac=lambda z: loss_and_grad(z)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "gtol": 1e-8},
    )
    elapsed = time.perf_counter() - t0
    return res.x, float(res.fun), int(res.nit), bool(res.success), elapsed


def _finite(x) -> bool:
    arr = np.asarray(x, dtype=np.float64)
    return bool(np.all(np.isfinite(arr)))


# ===========================================================================
# Gamma control (no Bessel)
# ===========================================================================

def gamma_pack(alpha, beta):
    return jnp.array([inv_softplus(alpha), inv_softplus(beta)])


def gamma_unpack(phi):
    return jax.nn.softplus(phi[0]), jax.nn.softplus(phi[1])


def run_gamma(n: int, n_adam: int, n_lbfgs: int) -> dict:
    true = Gamma(alpha=jnp.array(2.0), beta=jnp.array(1.5))
    X = true.rvs(n, seed=0)
    nll_true = float(-jnp.mean(jax.vmap(true.log_prob)(X)))

    # Perturbed init: (α, β) = (1.0, 3.0)
    phi0 = gamma_pack(jnp.array(1.0), jnp.array(3.0))

    def nll_phi(phi):
        a, b = gamma_unpack(phi)
        dist = Gamma(alpha=a, beta=b)
        return -jnp.mean(jax.vmap(dist.log_prob)(X))

    nll_phi = jax.jit(nll_phi)

    t0 = time.perf_counter()
    mle = Gamma.fit_mle(X)
    t_mle = time.perf_counter() - t0
    nll_mle = float(-jnp.mean(jax.vmap(mle.log_prob)(X)))

    t0 = time.perf_counter()
    phi_ad, _ = jax.jit(lambda p: adam_minimize(nll_phi, p, n_steps=n_adam, lr=5e-3))(phi0)
    phi_ad.block_until_ready()
    t_adam = time.perf_counter() - t0
    a_ad, b_ad = gamma_unpack(phi_ad)

    t0 = time.perf_counter()
    phi_lb, n_it, err = lbfgs_minimize(nll_phi, phi0, maxiter=n_lbfgs)
    phi_lb.block_until_ready()
    t_lbfgs = time.perf_counter() - t0
    a_lb, b_lb = gamma_unpack(phi_lb)

    def row(name, alpha, beta, nll, elapsed, extra=None):
        rec = {
            "method": name,
            "alpha": float(alpha),
            "beta": float(beta),
            "d_alpha": abs(float(alpha) - 2.0),
            "d_beta": abs(float(beta) - 1.5),
            "nll": float(nll),
            "dnll": float(nll) - nll_true,
            "elapsed_s": float(elapsed),
            "finite": _finite([alpha, beta, nll]),
        }
        if extra:
            rec.update(extra)
        return rec

    return {
        "n": n,
        "nll_true": nll_true,
        "methods": [
            row("fit_mle", mle.alpha, mle.beta, nll_mle, t_mle),
            row("adam+softplus", a_ad, b_ad, float(nll_phi(phi_ad)), t_adam,
                {"n_steps": n_adam, "lr": 5e-3}),
            row("lbfgs+softplus", a_lb, b_lb, float(nll_phi(phi_lb)), t_lbfgs,
                {"n_iter": n_it, "opt_error": err}),
        ],
    }


# ===========================================================================
# GIG
# ===========================================================================

def gig_pack(p, a, b):
    return jnp.array([p, inv_softplus(a), inv_softplus(b)])


def gig_unpack(phi):
    return phi[0], jax.nn.softplus(phi[1]), jax.nn.softplus(phi[2])


def gig_param_err(p, a, b, p0, a0, b0):
    """Relative hybrid-scale error on (p, a, b)."""
    dp = abs(p - p0) / (1.0 + abs(p0))
    da = abs(a - a0) / (1.0 + abs(a0))
    db = abs(b - b0) / (1.0 + abs(b0))
    return max(dp, da, db)


GIG_CASES = [
    ("interior",            1.0,  2.0,  1.0),
    ("invgauss p=-1/2",    -0.5,  2.0,  1.0),
    ("asymmetric a>>b",     0.5, 10.0,  0.1),
    ("asymmetric a<<b",    -1.0,  0.1, 10.0),
    ("near-Gamma b=1e-4",   2.0,  2.0,  1e-4),
    ("large sqrt(ab)=100",  1.0,  100.0, 100.0),
    ("large a=1e4, b=1e-3", 1.0,  1e4,  1e-3),
]


def run_gig_case(label, p0, a0, b0, n, n_adam, n_lbfgs, seed=0) -> dict:
    true = GIG(p=jnp.array(p0), a=jnp.array(a0), b=jnp.array(b0))
    X = np.asarray(true.rvs(n, seed=seed))
    Xj = jnp.asarray(X)
    nll_true = float(-jnp.mean(jax.vmap(true.log_prob)(Xj)))

    # Same perturbed init for every iterative NLL method.
    # Softplus domain: keep a, b positive. p is shifted additively.
    p_init = p0 + 0.7
    a_init = max(a0 * 2.5, 0.05)
    b_init = max(b0 * 0.4, 0.05)
    phi0 = gig_pack(jnp.array(p_init), jnp.array(a_init), jnp.array(b_init))
    # Classical (unconstrained-via-bounds) start for scipy L-BFGS-B:
    theta_class = np.array([p_init, a_init, b_init], dtype=np.float64)

    def nll_phi(phi):
        p, a, b = gig_unpack(phi)
        dist = GIG(p=p, a=a, b=b)
        return -jnp.mean(jax.vmap(dist.log_prob)(Xj))

    nll_phi = jax.jit(nll_phi)

    def nll_class(z):
        dist = GIG(p=z[0], a=z[1], b=z[2])
        return -jnp.mean(jax.vmap(dist.log_prob)(Xj))

    nll_class_vg = jax.jit(jax.value_and_grad(nll_class))

    def scipy_fg(z_np):
        z = jnp.asarray(np.ravel(z_np), dtype=jnp.float64)
        val, g = nll_class_vg(z)
        return float(np.asarray(val)), np.asarray(g, dtype=np.float64).ravel()

    methods = []

    # --- A. fit_mle (moment matching + Bregman) ---
    t0 = time.perf_counter()
    try:
        mle = GIG.fit_mle(Xj)
        t = time.perf_counter() - t0
        nll = float(-jnp.mean(jax.vmap(mle.log_prob)(Xj)))
        methods.append({
            "method": "fit_mle",
            "p": float(mle.p), "a": float(mle.a), "b": float(mle.b),
            "param_err": gig_param_err(float(mle.p), float(mle.a), float(mle.b), p0, a0, b0),
            "nll": nll, "dnll": nll - nll_true, "elapsed_s": t,
            "finite": _finite([mle.p, mle.a, mle.b, nll]),
        })
    except Exception as e:
        methods.append({"method": "fit_mle", "error": str(e)[:200],
                        "elapsed_s": time.perf_counter() - t0, "finite": False})

    # --- B. Adam + softplus ---
    t0 = time.perf_counter()
    try:
        phi_ad, losses = jax.jit(
            lambda p: adam_minimize(nll_phi, p, n_steps=n_adam, lr=1e-2)
        )(phi0)
        phi_ad.block_until_ready()
        t = time.perf_counter() - t0
        p, a, b = gig_unpack(phi_ad)
        nll = float(nll_phi(phi_ad))
        methods.append({
            "method": "adam+softplus",
            "p": float(p), "a": float(a), "b": float(b),
            "param_err": gig_param_err(float(p), float(a), float(b), p0, a0, b0),
            "nll": nll, "dnll": nll - nll_true, "elapsed_s": t,
            "n_steps": n_adam, "lr": 1e-2,
            "nll_final_scan": float(losses[-1]),
            "finite": _finite([p, a, b, nll]),
        })
    except Exception as e:
        methods.append({"method": "adam+softplus", "error": str(e)[:200],
                        "elapsed_s": time.perf_counter() - t0, "finite": False})

    # --- C. L-BFGS + softplus ---
    t0 = time.perf_counter()
    try:
        phi_lb, n_it, err = lbfgs_minimize(nll_phi, phi0, maxiter=n_lbfgs)
        if hasattr(phi_lb, "block_until_ready"):
            phi_lb.block_until_ready()
        t = time.perf_counter() - t0
        p, a, b = gig_unpack(phi_lb)
        nll = float(nll_phi(phi_lb))
        methods.append({
            "method": "lbfgs+softplus",
            "p": float(p), "a": float(a), "b": float(b),
            "param_err": gig_param_err(float(p), float(a), float(b), p0, a0, b0),
            "nll": nll, "dnll": nll - nll_true, "elapsed_s": t,
            "n_iter": n_it, "opt_error": err,
            "finite": _finite([p, a, b, nll]),
        })
    except Exception as e:
        methods.append({"method": "lbfgs+softplus", "error": str(e)[:200],
                        "elapsed_s": time.perf_counter() - t0, "finite": False})

    # --- D. scipy L-BFGS-B on classical (p, a>0, b>0), no softplus ---
    bounds = [(None, None), (1e-12, None), (1e-12, None)]
    try:
        z, nll, nit, ok, t = scipy_lbfgsb(
            scipy_fg, theta_class, bounds, maxiter=n_lbfgs)
        methods.append({
            "method": "lbfgsb-box NLL",
            "p": float(z[0]), "a": float(z[1]), "b": float(z[2]),
            "param_err": gig_param_err(float(z[0]), float(z[1]), float(z[2]), p0, a0, b0),
            "nll": nll, "dnll": nll - nll_true, "elapsed_s": t,
            "n_iter": nit, "success": ok,
            "finite": _finite([z[0], z[1], z[2], nll]),
        })
    except Exception as e:
        methods.append({"method": "lbfgsb-box NLL", "error": str(e)[:200],
                        "finite": False})

    return {
        "label": label,
        "true": {"p": p0, "a": a0, "b": b0},
        "n": n,
        "nll_true": nll_true,
        "init": {"p": p_init, "a": a_init, "b": b_init},
        "methods": methods,
    }


# ===========================================================================
# Gradient-quality diagnostic (independent of fitting)
# ===========================================================================

def _gig_nll_jax(p, a, b, X):
    return -jnp.mean(jax.vmap(GIG(p=p, a=a, b=b).log_prob)(X))


def _gig_nll_cpu(p, a, b, X_np):
    """NLL via the CPU log-partition (scipy kve), not the JAX custom_jvp path."""
    gig = GIG(p=jnp.array(p), a=jnp.array(a), b=jnp.array(b))
    theta = np.asarray(gig.natural_params())
    t = np.stack([np.log(X_np), 1.0 / X_np, X_np], axis=1)
    psi = float(GIG._log_partition_cpu(theta))
    return float(-np.mean(t @ theta - psi))


def run_grad_diag(n: int = 2000) -> dict:
    """jax.grad(NLL) wrt p vs central FD of the CPU NLL, plus log_kv ∂ν."""
    rows = []
    grid = [
        (1.0, 2.0, 1.0),
        (5.0, 1.0, 1.0),
        (1.0, 100.0, 100.0),
        (2.0, 2.0, 1e-4),
        (1.0, 1e4, 1e-3),
        (-0.5, 2.0, 1.0),
    ]
    for p0, a0, b0 in grid:
        true = GIG(p=jnp.array(p0), a=jnp.array(a0), b=jnp.array(b0))
        X = np.asarray(true.rvs(n, seed=1))
        Xj = jnp.asarray(X)
        p, a, b = jnp.array(p0), jnp.array(a0), jnp.array(b0)

        d_nll_dp = float(jax.grad(lambda pp: _gig_nll_jax(pp, a, b, Xj))(p))
        eps = 1e-6
        nll_plus = _gig_nll_cpu(p0 + eps, a0, b0, X)
        nll_minus = _gig_nll_cpu(p0 - eps, a0, b0, X)
        fd_cpu = (nll_plus - nll_minus) / (2.0 * eps)

        z = float(np.sqrt(a0 * b0))
        dlogkv_dv = float(jax.grad(lambda v: log_kv(v, jnp.array(z)))(jnp.array(p0)))
        # Independent FD on CPU kve in log-space
        from scipy.special import kve
        def logkve(v, zz):
            return np.log(np.maximum(kve(v, zz), 1e-300)) - zz  # kve = K e^{+z}
        # Actually kve(v,z) = exp(z) K_v(z), so log K = log kve - z
        fd_kv = (logkve(p0 + BESSEL_EPS_V, z) - logkve(p0 - BESSEL_EPS_V, z)) / (
            2.0 * BESSEL_EPS_V
        )

        rows.append({
            "p": p0, "a": a0, "b": b0, "sqrt_ab": z,
            "dNLL_dp_jax": d_nll_dp,
            "dNLL_dp_fd_cpu": fd_cpu,
            "dNLL_rel_err": abs(d_nll_dp - fd_cpu) / (1.0 + abs(fd_cpu)),
            "dlogkv_dv_jax": dlogkv_dv,
            "dlogkv_dv_fd_kve": float(fd_kv),
            "dlogkv_rel_err": abs(dlogkv_dv - fd_kv) / (1.0 + abs(fd_kv)),
        })
    return {"n": n, "eps_nll": 1e-6, "eps_kv": BESSEL_EPS_V, "rows": rows}


# ===========================================================================
# GH (d=2 latent-variable)
# ===========================================================================

def gh_pack(model: GH) -> jax.Array:
    j = model._joint
    d = int(j.d)
    L = j.L_Sigma
    chunks = [j.mu, j.gamma]
    for i in range(d):
        for k in range(d):
            if i == k:
                chunks.append(inv_softplus(L[i, i]).reshape(()))
            elif i > k:
                chunks.append(L[i, k].reshape(()))
    chunks += [j.p.reshape(()), inv_softplus(j.a).reshape(()),
               inv_softplus(j.b).reshape(())]
    return jnp.hstack([jnp.atleast_1d(c) for c in chunks])


def gh_unpack(phi: jax.Array, d: int) -> GH:
    i = 0
    mu = phi[i:i + d]; i += d
    gamma = phi[i:i + d]; i += d
    L = jnp.zeros((d, d), dtype=jnp.float64)
    for r in range(d):
        for c in range(d):
            if r == c:
                L = L.at[r, c].set(jax.nn.softplus(phi[i])); i += 1
            elif r > c:
                L = L.at[r, c].set(phi[i]); i += 1
    p = phi[i]; i += 1
    a = jax.nn.softplus(phi[i]); i += 1
    b = jax.nn.softplus(phi[i])
    sigma = L @ L.T
    # SPD guard: tiny jitter if a scan step went wild
    sigma = sigma + 1e-10 * jnp.eye(d)
    return GH.from_classical(mu=mu, gamma=gamma, sigma=sigma, p=p, a=a, b=b)


GH_CASES = [
    ("interior d=2", dict(
        mu=np.zeros(2), gamma=np.array([0.3, -0.1]),
        sigma=np.array([[1.0, 0.3], [0.3, 0.8]]),
        p=1.0, a=2.0, b=2.0,
    )),
    ("near-VG b=1e-3", dict(
        mu=np.zeros(2), gamma=np.array([0.2, 0.2]),
        sigma=np.eye(2),
        p=2.0, a=2.0, b=1e-3,
    )),
    ("asymmetric a>>b", dict(
        mu=np.array([0.1, -0.1]), gamma=np.array([0.4, 0.0]),
        sigma=np.eye(2),
        p=0.5, a=10.0, b=0.1,
    )),
]


def run_gh_case(label, params, n, n_em, n_adam, n_lbfgs, seed=0) -> dict:
    true = GH.from_classical(**{k: (jnp.asarray(v) if k in ("mu", "gamma", "sigma")
                                    else v) for k, v in params.items()})
    X = jnp.asarray(true.rvs(n, seed=seed))
    nll_true = float(-true.marginal_log_likelihood(X))
    d = 2

    # Shared naive init: moment-based (p=a=b=1), same starting point for all.
    init = GH._from_init_params(
        mu=jnp.mean(X, axis=0),
        gamma=jnp.zeros(d),
        sigma=jnp.cov(X, rowvar=False) + 0.1 * jnp.eye(d),
    )
    nll_init = float(-init.marginal_log_likelihood(X))
    phi0 = gh_pack(init)

    def nll_phi(phi):
        return -gh_unpack(phi, d).marginal_log_likelihood(X)

    methods = []

    t0 = time.perf_counter()
    try:
        em = BatchEMFitter(
            max_iter=n_em, tol=1e-5, verbose=0,
            e_step_backend="cpu", regularization="det_sigma_one",
        ).fit(init, X)
        t = time.perf_counter() - t0
        model = em.model
        nll = float(-model.marginal_log_likelihood(X))
        j = model._joint
        methods.append({
            "method": "EM",
            "p": float(j.p), "a": float(j.a), "b": float(j.b),
            "nll": nll, "dnll": nll - nll_true, "elapsed_s": t,
            "n_iter": em.n_iter, "converged": bool(em.converged),
            "diverged": bool(em.diverged),
            "finite": _finite([nll, j.p, j.a, j.b]),
        })
    except Exception as e:
        methods.append({"method": "EM", "error": str(e)[:240],
                        "elapsed_s": time.perf_counter() - t0, "finite": False})

    t0 = time.perf_counter()
    try:
        # JIT the scan; first call includes compile.
        run = jax.jit(lambda p: adam_minimize(nll_phi, p, n_steps=n_adam, lr=1e-3))
        phi_ad, losses = run(phi0)
        phi_ad.block_until_ready()
        t = time.perf_counter() - t0
        nll = float(nll_phi(phi_ad))
        fitted = gh_unpack(phi_ad, d)._joint
        methods.append({
            "method": "adam+softplus",
            "p": float(fitted.p), "a": float(fitted.a), "b": float(fitted.b),
            "nll": nll, "dnll": nll - nll_true, "elapsed_s": t,
            "n_steps": n_adam, "lr": 1e-3,
            "nll_scan_last": float(losses[-1]),
            "finite": _finite([nll, fitted.p, fitted.a, fitted.b]),
        })
    except Exception as e:
        methods.append({"method": "adam+softplus", "error": str(e)[:240],
                        "elapsed_s": time.perf_counter() - t0, "finite": False})

    t0 = time.perf_counter()
    try:
        phi_lb, n_it, err = lbfgs_minimize(nll_phi, phi0, maxiter=n_lbfgs, tol=1e-6)
        t = time.perf_counter() - t0
        nll = float(nll_phi(phi_lb))
        fitted = gh_unpack(phi_lb, d)._joint
        methods.append({
            "method": "lbfgs+softplus",
            "p": float(fitted.p), "a": float(fitted.a), "b": float(fitted.b),
            "nll": nll, "dnll": nll - nll_true, "elapsed_s": t,
            "n_iter": n_it, "opt_error": err,
            "finite": _finite([nll, fitted.p, fitted.a, fitted.b]),
        })
    except Exception as e:
        methods.append({"method": "lbfgs+softplus", "error": str(e)[:240],
                        "elapsed_s": time.perf_counter() - t0, "finite": False})

    return {
        "label": label,
        "true": {k: (np.asarray(v).tolist() if hasattr(v, "tolist") else v)
                 for k, v in params.items()},
        "n": n,
        "nll_true": nll_true,
        "nll_init": nll_init,
        "methods": methods,
    }


# ===========================================================================
# Printing
# ===========================================================================

def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "NaN"
    if abs(xf) >= 1e4 or (abs(xf) < 1e-3 and xf != 0.0):
        return f"{xf:.2e}"
    return f"{xf:.{nd}f}"


def print_gamma(res):
    hdr("Gamma control (no Bessel)")
    print(f"  true (α, β) = (2.0, 1.5)   n = {res['n']}   "
          f"NLL* = {_fmt(res['nll_true'])}")
    print(f"  {'method':<18} {'α':>10} {'β':>10} {'Δα':>10} {'Δβ':>10} "
          f"{'ΔNLL':>10} {'time':>8}")
    for m in res["methods"]:
        print(f"  {m['method']:<18} {_fmt(m.get('alpha')):>10} "
              f"{_fmt(m.get('beta')):>10} {_fmt(m.get('d_alpha')):>10} "
              f"{_fmt(m.get('d_beta')):>10} {_fmt(m.get('dnll')):>10} "
              f"{_fmt(m.get('elapsed_s'), 3)+'s':>8}")


def print_gig(cases):
    hdr("GIG: fit_mle vs NLL gradient methods")
    for c in cases:
        t = c["true"]
        print(f"\n  [{c['label']}]  true (p, a, b) = "
              f"({t['p']}, {t['a']}, {t['b']})   "
              f"NLL* = {_fmt(c['nll_true'])}")
        print(f"  {'method':<18} {'p':>10} {'a':>10} {'b':>10} "
              f"{'param_err':>10} {'ΔNLL':>10} {'time':>8} {'ok':>5}")
        for m in c["methods"]:
            if "error" in m:
                print(f"  {m['method']:<18}  ERROR: {m['error'][:70]}")
                continue
            print(f"  {m['method']:<18} {_fmt(m.get('p')):>10} "
                  f"{_fmt(m.get('a')):>10} {_fmt(m.get('b')):>10} "
                  f"{_fmt(m.get('param_err')):>10} {_fmt(m.get('dnll')):>10} "
                  f"{_fmt(m.get('elapsed_s'), 3)+'s':>8} "
                  f"{str(m.get('finite')):>5}")


def print_grad(res):
    hdr("Gradient quality: jax.grad(NLL) vs CPU finite difference")
    print(f"  n = {res['n']}   FD ε_NLL = {res['eps_nll']}   "
          f"BESSEL_EPS_V = {res['eps_kv']}")
    print(f"  {'(p,a,b)':<28} {'∂NLL/∂p jax':>14} {'∂NLL/∂p cpu':>14} "
          f"{'rel':>10} {'∂logK/∂ν rel':>12}")
    for r in res["rows"]:
        lab = f"({r['p']:g}, {r['a']:g}, {r['b']:g})"
        print(f"  {lab:<28} {_fmt(r['dNLL_dp_jax'], 5):>14} "
              f"{_fmt(r['dNLL_dp_fd_cpu'], 5):>14} "
              f"{_fmt(r['dNLL_rel_err'], 2):>10} "
              f"{_fmt(r['dlogkv_rel_err'], 2):>12}")


def print_gh(cases):
    hdr("GH d=2: EM vs NLL gradient methods (shared moment init)")
    for c in cases:
        print(f"\n  [{c['label']}]  n = {c['n']}   "
              f"NLL* = {_fmt(c['nll_true'])}   "
              f"NLL_init = {_fmt(c['nll_init'])}")
        print(f"  {'method':<18} {'p':>10} {'a':>10} {'b':>10} "
              f"{'ΔNLL':>10} {'time':>8} {'ok':>5}")
        for m in c["methods"]:
            if "error" in m:
                print(f"  {m['method']:<18}  ERROR: {m['error'][:80]}")
                continue
            print(f"  {m['method']:<18} {_fmt(m.get('p')):>10} "
                  f"{_fmt(m.get('a')):>10} {_fmt(m.get('b')):>10} "
                  f"{_fmt(m.get('dnll')):>10} "
                  f"{_fmt(m.get('elapsed_s'), 3)+'s':>8} "
                  f"{str(m.get('finite')):>5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section", type=str, default="gamma,gig,grad,gh",
        help="Comma-separated: gamma,gig,grad,gh")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer samples / steps (smoke)")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    sections = [s.strip() for s in args.section.split(",")]

    if args.quick:
        n_g, n_adam, n_lbfgs = 400, 200, 30
        n_gh, n_em, n_adam_gh, n_lbfgs_gh = 200, 8, 40, 15
        gig_cases = GIG_CASES[:3]
        gh_cases = GH_CASES[:1]
    else:
        n_g, n_adam, n_lbfgs = 2000, 1500, 150
        n_gh, n_em, n_adam_gh, n_lbfgs_gh = 600, 25, 200, 40
        gig_cases = GIG_CASES
        gh_cases = GH_CASES

    payload = {"quick": args.quick, "sections": sections}

    if "gamma" in sections:
        payload["gamma"] = run_gamma(n_g, n_adam, n_lbfgs)
        print_gamma(payload["gamma"])

    if "gig" in sections:
        rows = []
        for lab, p, a, b in gig_cases:
            print(f"\n  running GIG {lab} ...", flush=True)
            rows.append(run_gig_case(lab, p, a, b, n_g, n_adam, n_lbfgs))
        payload["gig"] = rows
        print_gig(rows)

    if "grad" in sections:
        payload["grad"] = run_grad_diag(n=n_g)
        print_grad(payload["grad"])

    if "gh" in sections:
        rows = []
        for lab, params in gh_cases:
            print(f"\n  running GH {lab} ...", flush=True)
            rows.append(run_gh_case(
                lab, params, n_gh, n_em, n_adam_gh, n_lbfgs_gh))
        payload["gh"] = rows
        print_gh(rows)

    sep()
    if args.save:
        path = save_result("gradient_fitting", payload)
        print(f"Saved {path}")
    print("done.")


if __name__ == "__main__":
    main()
