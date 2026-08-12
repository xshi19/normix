"""Helpers for the subordinator-tracking empirical study.

Public-API only — nothing here belongs in the ``normix`` package.
Notation matches ``dev-notes/research/subordinator_tracking_portfolio.md``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.linalg import solve_triangular

from normix import NormalInverseGaussian
from normix.distributions.inverse_gaussian import InverseGaussian
from normix.fitting.eta import affine_combine
from normix.fitting.shrinkage_targets import eta0_from_model

jax.config.update("jax_enable_x64", True)

# Daily-equity γ is O(10^{-2}); the package default tol=1e-3 stops after one
# EM step (rms(Δγ)/(1+rms(γ)) ≈ 10^{-3}). Tighter tol is a study choice,
# not a package change.
NIG_FIT_KW: dict[str, Any] = dict(
    max_iter=200,
    tol=1e-5,
    regularization="a_eq_b",
    e_step_backend="cpu",
    m_step_backend="cpu",
    verbose=0,
)

QTILDE_FLOOR = 1e-18


# ---------------------------------------------------------------------------
# Paths / cache
# ---------------------------------------------------------------------------

def repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in (here.parent, *here.parents):
        if (cand / "pyproject.toml").is_file() and (cand / "normix").is_dir():
            return cand
    return Path.cwd()


def cache_dir() -> Path:
    p = Path(__file__).resolve().parent / "_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figure_dir() -> Path:
    p = cache_dir() / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _npz_path(name: str) -> Path:
    path = cache_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_or_compute(name: str, fn: Callable[[], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Load a ``.npz`` cache, or compute, save, and return it."""
    path = _npz_path(name)
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    data = fn()
    np.savez(path, **data)
    return data


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sp500() -> pd.DataFrame:
    path = repo_root() / "data" / "sp500_returns.csv"
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def data_hygiene(panel: pd.DataFrame) -> dict[str, Any]:
    abs_big = panel.abs() > 0.5
    n_big = int(abs_big.to_numpy().sum())
    locs = []
    if n_big:
        hits = abs_big.stack()
        for (dt, tic), flag in hits.items():
            if flag:
                locs.append((str(pd.Timestamp(dt).date()), str(tic), float(panel.loc[dt, tic])))
    zero_var = [c for c in panel.columns if float(panel[c].std()) == 0.0]
    return {
        "n_obs": int(panel.shape[0]),
        "n_names": int(panel.shape[1]),
        "n_abs_gt_0.5": n_big,
        "abs_gt_0.5": locs,
        "zero_var": zero_var,
        "n_nan": int(panel.isna().to_numpy().sum()),
    }


def nested_universe(tickers: list[str], sizes: list[int], seed: int) -> dict[int, list[str]]:
    """Nested random subsets: the size-``d`` set is a prefix of a permutation."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tickers))
    out: dict[int, list[str]] = {}
    for d in sizes:
        if d > len(tickers):
            raise ValueError(f"d={d} exceeds n_tickers={len(tickers)}")
        out[int(d)] = [tickers[i] for i in order[:d]]
    return out


# ---------------------------------------------------------------------------
# NIG fit / serialize
# ---------------------------------------------------------------------------

def fit_nig(X: jax.Array, **kwargs) -> Any:
    kw = {**NIG_FIT_KW, **kwargs}
    X = jnp.asarray(X, dtype=jnp.float64)
    init = NormalInverseGaussian.default_init(X)
    return init.fit(X, **kw)


def dump_nig(path: Path, model: NormalInverseGaussian, **meta: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mu=np.asarray(model.mu, dtype=np.float64),
        gamma=np.asarray(model.gamma, dtype=np.float64),
        sigma=np.asarray(model.sigma(), dtype=np.float64),
        mu_ig=np.asarray(model.mu_ig, dtype=np.float64),
        lam=np.asarray(model.lam, dtype=np.float64),
    )
    if meta:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def load_nig(path: Path) -> NormalInverseGaussian:
    with np.load(path) as z:
        return NormalInverseGaussian.from_classical(
            mu=z["mu"], gamma=z["gamma"], sigma=z["sigma"],
            mu_ig=float(z["mu_ig"]), lam=float(z["lam"]),
        )


def scale_gamma(model: NormalInverseGaussian, c: float) -> NormalInverseGaussian:
    return model.replace(gamma=jnp.asarray(c, dtype=jnp.float64) * model.gamma)


def restrict_nig(model: NormalInverseGaussian, d: int) -> NormalInverseGaussian:
    """Leading-``d`` coordinate submodel (same subordinator)."""
    sig = np.asarray(model.sigma())
    return NormalInverseGaussian.from_classical(
        mu=np.asarray(model.mu)[:d],
        gamma=np.asarray(model.gamma)[:d],
        sigma=sig[:d, :d],
        mu_ig=float(model.mu_ig),
        lam=float(model.lam),
    )


def load_or_fit_generator(
    panel: pd.DataFrame,
    tickers: list[str],
    *,
    name: str = "generator_nig_d50_seed0.npz",
) -> tuple[NormalInverseGaussian, dict[str, Any]]:
    path = cache_dir() / name
    meta_path = path.with_suffix(".json")
    if path.exists():
        model = load_nig(path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return model, meta
    X = jnp.asarray(panel[tickers].to_numpy(), dtype=np.float64)
    t0 = time.perf_counter()
    result = fit_nig(X, verbose=1)
    meta = {
        "tickers": list(tickers),
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "elapsed_sec": float(result.elapsed_time),
        "wall_sec": time.perf_counter() - t0,
        "n_obs": int(X.shape[0]),
        "d": int(X.shape[1]),
        "fit_kw": {k: v for k, v in NIG_FIT_KW.items() if k != "verbose"},
    }
    dump_nig(path, result.model, **meta)
    return result.model, meta


# ---------------------------------------------------------------------------
# Tracker / SNR
# ---------------------------------------------------------------------------

def _whiten(L: jax.Array, v: jax.Array) -> jax.Array:
    return solve_triangular(L, v, lower=True)


def tracker_stats(model) -> dict[str, Any]:
    """Gauge-aware SNR diagnostics from a fitted (or true) mixture."""
    L = model.L_Sigma
    gamma = model.gamma
    z = _whiten(L, gamma)
    q_tilde = float(z @ z)
    sub = model.subordinator()
    e = float(sub.mean())
    v = float(sub.var())
    # InverseGaussian: μ₃ = 3 v²/e (avoids a Bessel call per fitted model).
    if isinstance(sub, InverseGaussian) and e != 0.0:
        mu3 = 3.0 * v ** 2 / e
    else:
        m1, m2, m3 = np.asarray(sub.raw_moments(jnp.array([1.0, 2.0, 3.0])))
        mu3 = float(m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3)
    kappa_lev = q_tilde * e
    kappa = q_tilde * v / e if e != 0.0 else float("nan")
    t_dagger = (2.0 * v / e - mu3 / v) if (e != 0.0 and v != 0.0) else float("nan")
    inv_sigma_gamma = np.asarray(solve_triangular(L.T, z, lower=False))
    if q_tilde > QTILDE_FLOOR:
        w_star = inv_sigma_gamma / q_tilde
    else:
        w_star = np.full_like(inv_sigma_gamma, np.nan)
    return dict(
        q_tilde=q_tilde,
        e=e,
        v=v,
        mu3=mu3,
        kappa=kappa,
        kappa_lev=kappa_lev,
        t_dagger=t_dagger,
        cv2=v / e ** 2 if e != 0.0 else float("nan"),
        inv_sigma_gamma=inv_sigma_gamma,
        w_star=w_star,
        corr_theory=float(np.sqrt(kappa / (1.0 + kappa))) if kappa > 0 else 0.0,
    )


def nig_fast_stats(model) -> dict[str, Any]:
    """NIG-only SNR stats using closed-form IG moments (no Bessel)."""
    L = model.L_Sigma
    z = _whiten(L, model.gamma)
    q_tilde = float(z @ z)
    mu_ig = float(model.mu_ig)
    lam = float(model.lam)
    e = mu_ig
    v = mu_ig ** 3 / lam
    inv_sigma_gamma = np.asarray(solve_triangular(L.T, z, lower=False))
    w_star = inv_sigma_gamma / q_tilde if q_tilde > QTILDE_FLOOR else np.full_like(inv_sigma_gamma, np.nan)
    kappa = q_tilde * v / e if e != 0.0 else float("nan")
    return dict(
        q_tilde=q_tilde, e=e, v=v, kappa=kappa,
        kappa_lev=q_tilde * e,
        inv_sigma_gamma=inv_sigma_gamma, w_star=w_star,
    )


def tracker_only(model, X: jax.Array) -> jax.Array:
    """Linear tracker ``Ŷ = s(X)/q̃``; NaN when ``q̃=0``. No Bessel."""
    X = jnp.asarray(X, dtype=jnp.float64)
    L, mu, gamma = model.L_Sigma, model.mu, model.gamma
    Z = jax.vmap(lambda x: _whiten(L, x - mu))(X)
    z_g = _whiten(L, gamma)
    q_tilde = z_g @ z_g
    s = Z @ z_g
    return jnp.where(q_tilde > QTILDE_FLOOR, s / q_tilde, jnp.nan)


def y_estimators(model, X: jax.Array) -> dict[str, jax.Array]:
    """Tracker, linear Bayes, and posterior mean of ``Y`` given ``X``.

    Uses the model's parameters (true or fitted). At ``q̃ = 0`` the tracker
    is NaN and linear Bayes collapses to the prior mean ``e``.
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    st = tracker_stats(model)
    L, mu, gamma = model.L_Sigma, model.mu, model.gamma
    Z = jax.vmap(lambda x: _whiten(L, x - mu))(X)
    z_g = _whiten(L, gamma)
    q_tilde = z_g @ z_g
    s = Z @ z_g
    Y_hat = jnp.where(q_tilde > QTILDE_FLOOR, s / q_tilde, jnp.nan)
    e = jnp.float64(st["e"])
    kappa = jnp.float64(st["kappa"] if np.isfinite(st["kappa"]) else 0.0)
    Y_lin = e + (kappa / (1.0 + kappa)) * (jnp.nan_to_num(Y_hat, nan=float(e)) - e)
    Y_post = jax.vmap(lambda x: model.joint.conditional_expectations(x)["E_Y"])(X)
    q = jnp.sum(Z ** 2, axis=1)
    Y_hat_sq = jnp.square(jnp.nan_to_num(Y_hat))
    q_perp = q - q_tilde * Y_hat_sq
    return dict(Y_hat=Y_hat, Y_lin=Y_lin, Y_post=Y_post, q=q, q_perp=q_perp, s=s)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15 or not np.isfinite(na + nb):
        return float("nan")
    return float(a @ b) / (na * nb)


def mse(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((a[m] - b[m]) ** 2))


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def sign_flip(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Day-wise sign flip of demeaned returns (kills odd joint moments)."""
    X = np.asarray(X, dtype=np.float64)
    xbar = X.mean(axis=0)
    signs = rng.choice(np.array([-1.0, 1.0]), size=X.shape[0])
    return xbar + signs[:, None] * (X - xbar)


def block_bootstrap_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


# ---------------------------------------------------------------------------
# Online EM
# ---------------------------------------------------------------------------

def ewma_weight(half_life: float) -> float:
    return float(1.0 - 2.0 ** (-1.0 / half_life))


def ewma_neff(half_life: float) -> float:
    w = ewma_weight(half_life)
    return float((2.0 - w) / w)


def ewma_smooth(x: np.ndarray, half_life: float) -> np.ndarray:
    w = ewma_weight(half_life)
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    out[0] = x[0]
    a = 1.0 - w
    for t in range(1, x.shape[0]):
        out[t] = a * out[t - 1] + w * x[t]
    return out


def unit_gross(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    s = np.abs(w).sum()
    return w / s if s > 0 else w


def turnover(w_prev: np.ndarray, w_new: np.ndarray) -> float:
    return float(0.5 * np.abs(unit_gross(w_new) - unit_gross(w_prev)).sum())


@eqx.filter_jit
def _nig_online_step(model, eta, x, w, tau, eta0):
    """One Cappé–Moulines EWMA (+ optional shrinkage) step; no re-gauging."""
    eta_hat = model.e_step(x[None, :], backend="jax")
    eta_base = affine_combine(eta, eta_hat, 1.0 - w, w)
    factor_a = tau / (1.0 + tau)
    factor_b = 1.0 / (1.0 + tau)
    eta_new = affine_combine(eta0, eta_base, factor_a, factor_b)
    model_new = model.m_step(eta_new, backend="jax")
    return model_new, eta_new


def online_em_path(
    model0,
    X: jax.Array,
    *,
    half_life: float,
    tau: float = 0.0,
    eta0=None,
) -> dict[str, np.ndarray]:
    """Chronological EWMA online EM. ``tau=0`` is pure EWMA.

    Records in-sample (``model_t``) and filtered (``model_{t-1}``) trackers.
    Does **not** call ``regularize_a_eq_b`` (gauge-mixing; see the plan §12).
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    n = int(X.shape[0])
    w = jnp.float64(ewma_weight(half_life))
    tau_arr = jnp.float64(tau)
    eta = model0.compute_eta_from_model()
    if eta0 is None:
        eta0 = eta0_from_model(model0) if tau > 0 else eta
    model = model0

    q_t = np.empty(n)
    kappa_t = np.empty(n)
    e_t = np.empty(n)
    Y_in = np.empty(n)
    Y_filt = np.empty(n)
    cos_prev = np.empty(n)
    to_t = np.empty(n)
    invg_t = np.empty((n, int(X.shape[1])))
    w_star_prev = None
    st = nig_fast_stats(model)

    for t in range(n):
        x = X[t]
        st_prev = st
        Y_filt[t] = _tracker_one(st_prev, np.asarray(model.mu), np.asarray(x))
        model, eta = _nig_online_step(model, eta, x, w, tau_arr, eta0)
        st = nig_fast_stats(model)
        q_t[t] = st["q_tilde"]
        kappa_t[t] = st["kappa"]
        e_t[t] = st["e"]
        invg_t[t] = st["inv_sigma_gamma"]
        Y_in[t] = _tracker_one(st, np.asarray(model.mu), np.asarray(x))
        if w_star_prev is None:
            to_t[t] = 0.0
            cos_prev[t] = 1.0
        else:
            to_t[t] = turnover(w_star_prev, st["w_star"])
            cos_prev[t] = cosine(w_star_prev, st["w_star"])
        w_star_prev = st["w_star"]

    return dict(
        q_tilde=q_t, kappa=kappa_t, e=e_t,
        Y_hat_in=Y_in, Y_hat_filt=Y_filt,
        turnover=to_t, cos_lag1=cos_prev,
        inv_sigma_gamma=invg_t,
    )


def _tracker_one(st: dict[str, Any], mu: np.ndarray, x: np.ndarray) -> float:
    q = st["q_tilde"]
    if not np.isfinite(q) or q <= QTILDE_FLOOR:
        return float("nan")
    return float(st["inv_sigma_gamma"] @ (x - mu) / q)


# ---------------------------------------------------------------------------
# Time-varying synthetic generator
# ---------------------------------------------------------------------------

def sigma_plane_gamma(gamma: np.ndarray, L: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotate ``γ`` in the Σ-metric so ``q̃`` is invariant.

    ``theta`` shape ``(T,)``; returns ``gamma_t`` shape ``(T, d)``.
    """
    L = np.asarray(L, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    z = np.linalg.solve(L, gamma)
    nz = np.linalg.norm(z)
    e = np.zeros_like(z)
    e[0] = 1.0
    if abs(z @ e) > 0.9 * nz * np.linalg.norm(e):
        e = np.zeros_like(z)
        e[min(1, z.size - 1)] = 1.0
    e = e - z * ((z @ e) / (z @ z))
    e = e / np.linalg.norm(e) * nz
    c, s = np.cos(theta), np.sin(theta)
    z_t = c[:, None] * z[None, :] + s[:, None] * e[None, :]
    return z_t @ L.T


def make_tv_path(
    model: NormalInverseGaussian,
    T: int,
    *,
    jump_at: int,
    jump_scale: float,
    theta_max: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """``X_t = μ + γ_t Y_t + √Y_t Z_t`` with a slow γ-rotation and a Y-scale jump."""
    L = np.asarray(model.L_Sigma)
    mu = np.asarray(model.mu)
    gamma0 = np.asarray(model.gamma)
    d = mu.shape[0]
    theta = np.linspace(0.0, theta_max, T)
    gamma_t = sigma_plane_gamma(gamma0, L, theta)

    mu_ig = float(model.mu_ig)
    lam = float(model.lam)
    n1, n2 = int(jump_at), int(T - jump_at)
    Y1 = np.asarray(InverseGaussian(mu=mu_ig, lam=lam).rvs(n1, seed=seed))
    Y2 = np.asarray(InverseGaussian(mu=mu_ig * jump_scale, lam=lam * jump_scale).rvs(
        n2, seed=seed + 1))
    Y = np.concatenate([Y1, Y2])
    e_t = np.concatenate([np.full(n1, mu_ig), np.full(n2, mu_ig * jump_scale)])
    # IG: v = μ³/λ; after Y → sY, (μ,λ) → (sμ, sλ) so v → s² v.
    v0 = mu_ig ** 3 / lam
    v_t = np.concatenate([np.full(n1, v0), np.full(n2, (jump_scale ** 2) * v0)])

    key = jax.random.PRNGKey(seed + 2)
    Z = np.asarray(jax.random.normal(key, shape=(T, d), dtype=jnp.float64))
    X = mu[None, :] + gamma_t * Y[:, None] + np.sqrt(Y)[:, None] * (Z @ L.T)

    # q̃ is invariant under the Σ-metric rotation; recompute per row as a check.
    z0 = np.linalg.solve(L, gamma0)
    q_tilde = float(z0 @ z0)
    kappa_t = q_tilde * v_t / e_t
    invg_t = np.linalg.solve(L @ L.T, gamma_t.T).T  # (T, d)
    return dict(
        X=X, Y=Y, gamma_t=gamma_t, e_t=e_t, v_t=v_t,
        kappa_t=kappa_t, q_tilde=np.full(T, q_tilde),
        inv_sigma_gamma_t=invg_t, jump_at=np.array(jump_at),
        jump_scale=np.array(jump_scale), theta_max=np.array(theta_max),
    )


# ---------------------------------------------------------------------------
# Volatility proxies (Phase 1/3; used in the online rehearsal vs true Y)
# ---------------------------------------------------------------------------

def realized_var_ewma(r: np.ndarray, half_life: float) -> np.ndarray:
    return ewma_smooth(np.asarray(r, dtype=np.float64) ** 2, half_life)


def cross_sectional_dispersion(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sig = X.std(axis=0, ddof=1)
    sig = np.where(sig > 0, sig, 1.0)
    m = X.mean(axis=1, keepdims=True)
    z = (X - m) / sig[None, :]
    return np.mean(z ** 2, axis=1)


# ---------------------------------------------------------------------------
# Phase 0 trials
# ---------------------------------------------------------------------------

def mse_laws_trial(model, n: int, seed: int) -> dict[str, float]:
    """One Monte Carlo draw of tracker / Bayes / posterior MSE against true Y."""
    X, Y = model.joint.rvs(n, seed)
    Y = np.asarray(Y)
    est = y_estimators(model, X)
    st = tracker_stats(model)
    Y_hat = np.asarray(est["Y_hat"])
    Y_lin = np.asarray(est["Y_lin"])
    Y_post = np.asarray(est["Y_post"])
    finite_hat = np.isfinite(Y_hat)
    out = dict(
        q_tilde=st["q_tilde"],
        kappa=st["kappa"],
        e=st["e"],
        v=st["v"],
        mse_hat=mse(Y_hat, Y),
        mse_lin=mse(Y_lin, Y),
        mse_post=mse(Y_post, Y),
        mse_hat_theory=st["e"] / st["q_tilde"] if st["q_tilde"] > QTILDE_FLOOR else float("nan"),
        mse_lin_theory=st["v"] / (1.0 + st["kappa"]) if np.isfinite(st["kappa"]) else st["v"],
        mse_post_bound=st["v"] / (1.0 + st["kappa"]) if np.isfinite(st["kappa"]) else st["v"],
        corr_hat=pearson(Y_hat, Y),
        corr_lin=pearson(Y_lin, Y),
        corr_post=pearson(Y_post, Y),
        corr_hat_theory=st["corr_theory"],
        var_Y=float(np.var(Y)),
        var_Yhat=float(np.var(Y_hat[finite_hat])) if finite_hat.any() else float("nan"),
        var_Yhat_theory=(st["v"] + st["e"] / st["q_tilde"]) if st["q_tilde"] > QTILDE_FLOOR else float("nan"),
    )
    return out


def refit_trial(true_model, n: int, seed: int) -> dict[str, float]:
    """Sample from ``true_model``, cold-start EM, compare ``q̃`` and direction."""
    X, Y = true_model.joint.rvs(n, seed)
    result = fit_nig(X)
    fitted = result.model
    st_true = tracker_stats(true_model)
    st_hat = tracker_stats(fitted)
    # Tracker only (triangular solves). Skip the posterior here: vmapping
    # Bessel conditionals on a new model each trial recompiles and OOMs.
    Y_hat = np.asarray(tracker_only(fitted, X))
    return dict(
        q_tilde_true=st_true["q_tilde"],
        q_tilde_hat=st_hat["q_tilde"],
        kappa_true=st_true["kappa"],
        kappa_hat=st_hat["kappa"],
        cosine=cosine(st_true["inv_sigma_gamma"], st_hat["inv_sigma_gamma"]),
        corr_hat_trueY=pearson(Y_hat, np.asarray(Y)),
        corr_post_trueY=float("nan"),
        n_iter=float(result.n_iter),
        converged=float(bool(result.converged)),
        elapsed=float(result.elapsed_time),
    )
