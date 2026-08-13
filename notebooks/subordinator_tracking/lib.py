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

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.stats import norm as gauss, spearmanr

from normix import GeneralizedHyperbolic, NormalInverseGaussian, VarianceGamma
from normix.distributions.inverse_gaussian import InverseGaussian
from normix.fitting.eta import NormalMixtureEta, affine_combine
from normix.fitting.shrinkage_targets import eta0_from_model

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

# GH: nested continuation from a fitted NIG (exact interior embedding),
# then free (p, a, b). Cheaper and a cleaner family-sensitivity test than
# GH.default_init, which itself fits NIG/VG/NInvG for five EM steps each.
GH_FIT_KW: dict[str, Any] = dict(
    max_iter=80,
    tol=1e-5,
    regularization="a_eq_b",
    e_step_backend="cpu",
    m_step_backend="cpu",
    verbose=0,
)

# VG: a_eq_b is a no-op (degenerate GIG). alpha_min='density' keeps the
# marginal density bounded (α > d/2).
VG_FIT_KW: dict[str, Any] = dict(
    max_iter=80,
    tol=1e-5,
    e_step_backend="cpu",
    m_step_backend="cpu",
    verbose=0,
    alpha_min="density",
)

QTILDE_FLOOR = 1e-18
SIZES = [5, 10, 25, 50, 100, 200, 468]
PRIMARY_D = 50
PRIMARY_SEED = 0


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


def _q_perp_one(st: dict[str, Any], L: np.ndarray, mu: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    yhat = _tracker_one(st, mu, x)
    z = np.linalg.solve(np.asarray(L, dtype=np.float64), np.asarray(x - mu, dtype=np.float64))
    q = float(z @ z)
    qt = st["q_tilde"]
    if not np.isfinite(yhat) or not np.isfinite(qt):
        return yhat, float("nan")
    return yhat, q - qt * yhat * yhat


def online_em_path(
    model0,
    X: jax.Array,
    *,
    half_life: float | None = None,
    tau: float = 0.0,
    eta0=None,
    sample_weighted: bool = False,
    n0: int = 0,
) -> dict[str, np.ndarray]:
    """Chronological online EM. EWMA if ``half_life`` is set; else $1/(n_0+t)$.

    Records in-sample (``model_t``) and filtered (``model_{t-1}``) trackers
    and $q_\\perp$. Does **not** call ``regularize_a_eq_b``.
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    n = int(X.shape[0])
    tau_arr = jnp.float64(tau)
    eta = model0.compute_eta_from_model()
    if eta0 is None:
        eta0 = eta0_from_model(model0) if tau > 0 else eta
    model = model0
    w_ewma = jnp.float64(ewma_weight(half_life)) if (half_life is not None and not sample_weighted) else None

    q_t = np.empty(n)
    kappa_t = np.empty(n)
    e_t = np.empty(n)
    Y_in = np.empty(n)
    Y_filt = np.empty(n)
    qp_in = np.empty(n)
    qp_filt = np.empty(n)
    P_filt = np.empty(n)
    cos_prev = np.empty(n)
    to_t = np.empty(n)
    invg_t = np.empty((n, int(X.shape[1])))
    w_star_prev = None
    st = nig_fast_stats(model)

    for t in range(n):
        x = X[t]
        mu_prev = np.asarray(model.mu)
        L_prev = np.asarray(model.L_Sigma)
        Y_filt[t], qp_filt[t] = _q_perp_one(st, L_prev, mu_prev, np.asarray(x))
        P_filt[t] = float(st["w_star"] @ np.asarray(x)) if np.all(np.isfinite(st["w_star"])) else np.nan
        if sample_weighted:
            w = jnp.float64(1.0 / (n0 + t + 1))
        else:
            w = w_ewma
        model, eta = _nig_online_step(model, eta, x, w, tau_arr, eta0)
        st = nig_fast_stats(model)
        q_t[t] = st["q_tilde"]
        kappa_t[t] = st["kappa"]
        e_t[t] = st["e"]
        invg_t[t] = st["inv_sigma_gamma"]
        Y_in[t], qp_in[t] = _q_perp_one(st, np.asarray(model.L_Sigma), np.asarray(model.mu), np.asarray(x))
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
        q_perp_in=qp_in, q_perp_filt=qp_filt,
        P_filt=P_filt,
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


# ---------------------------------------------------------------------------
# GH / VG fit / serialize
# ---------------------------------------------------------------------------

def dump_gh(path: Path, model: GeneralizedHyperbolic, **meta: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mu=np.asarray(model.mu, dtype=np.float64),
        gamma=np.asarray(model.gamma, dtype=np.float64),
        sigma=np.asarray(model.sigma(), dtype=np.float64),
        p=np.asarray(model.p, dtype=np.float64),
        a=np.asarray(model.a, dtype=np.float64),
        b=np.asarray(model.b, dtype=np.float64),
    )
    if meta:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def load_gh(path: Path) -> GeneralizedHyperbolic:
    with np.load(path) as z:
        return GeneralizedHyperbolic.from_classical(
            mu=z["mu"], gamma=z["gamma"], sigma=z["sigma"],
            p=float(z["p"]), a=float(z["a"]), b=float(z["b"]),
        )


def dump_vg(path: Path, model: VarianceGamma, **meta: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mu=np.asarray(model.mu, dtype=np.float64),
        gamma=np.asarray(model.gamma, dtype=np.float64),
        sigma=np.asarray(model.sigma(), dtype=np.float64),
        alpha=np.asarray(model.alpha, dtype=np.float64),
        beta=np.asarray(model.beta, dtype=np.float64),
    )
    if meta:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def load_vg(path: Path) -> VarianceGamma:
    with np.load(path) as z:
        return VarianceGamma.from_classical(
            mu=z["mu"], gamma=z["gamma"], sigma=z["sigma"],
            alpha=float(z["alpha"]), beta=float(z["beta"]),
        )


def fit_gh(X: jax.Array, init: GeneralizedHyperbolic | None = None, **kwargs) -> Any:
    kw = {**GH_FIT_KW, **kwargs}
    X = jnp.asarray(X, dtype=jnp.float64)
    if init is None:
        init = GeneralizedHyperbolic.default_init(X)
    return init.fit(X, **kw)


def fit_vg(X: jax.Array, **kwargs) -> Any:
    kw = {**VG_FIT_KW, **kwargs}
    X = jnp.asarray(X, dtype=jnp.float64)
    init = VarianceGamma.default_init(X)
    return init.fit(X, **kw)


def load_or_fit_nig_named(
    panel: pd.DataFrame,
    tickers: list[str],
    name: str,
) -> tuple[NormalInverseGaussian, dict[str, Any]]:
    path = cache_dir() / "fits" / name
    meta_path = path.with_suffix(".json")
    if path.exists():
        model = load_nig(path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return model, meta
    X = jnp.asarray(panel[tickers].to_numpy(), dtype=np.float64)
    t0 = time.perf_counter()
    result = fit_nig(X)
    meta = {
        "tickers": list(tickers),
        "family": "nig",
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "elapsed_sec": float(result.elapsed_time),
        "wall_sec": time.perf_counter() - t0,
        "n_obs": int(X.shape[0]),
        "d": int(X.shape[1]),
    }
    dump_nig(path, result.model, **meta)
    return result.model, meta


def load_or_fit_gh_from_nig(
    nig: NormalInverseGaussian,
    X: jax.Array,
    name: str,
    tickers: list[str] | None = None,
) -> tuple[GeneralizedHyperbolic, dict[str, Any]]:
    """Continue EM in GH from the NIG embedding (nested-family warm start)."""
    path = cache_dir() / "fits" / name
    meta_path = path.with_suffix(".json")
    if path.exists():
        model = load_gh(path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return model, meta
    X = jnp.asarray(X, dtype=jnp.float64)
    init = nig.to_generalized_hyperbolic()
    t0 = time.perf_counter()
    result = fit_gh(X, init=init)
    meta = {
        "tickers": list(tickers) if tickers is not None else [],
        "family": "gh",
        "init": "nig_embedding",
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "elapsed_sec": float(result.elapsed_time),
        "wall_sec": time.perf_counter() - t0,
        "n_obs": int(X.shape[0]),
        "d": int(X.shape[1]),
    }
    dump_gh(path, result.model, **meta)
    return result.model, meta


def load_or_fit_vg_named(
    panel: pd.DataFrame,
    tickers: list[str],
    name: str,
) -> tuple[VarianceGamma, dict[str, Any]]:
    path = cache_dir() / "fits" / name
    meta_path = path.with_suffix(".json")
    if path.exists():
        model = load_vg(path)
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        return model, meta
    X = jnp.asarray(panel[tickers].to_numpy(), dtype=np.float64)
    t0 = time.perf_counter()
    result = fit_vg(X)
    meta = {
        "tickers": list(tickers),
        "family": "vg",
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "elapsed_sec": float(result.elapsed_time),
        "wall_sec": time.perf_counter() - t0,
        "n_obs": int(X.shape[0]),
        "d": int(X.shape[1]),
    }
    dump_vg(path, result.model, **meta)
    return result.model, meta


# ---------------------------------------------------------------------------
# SNR rows / moments / portfolios
# ---------------------------------------------------------------------------

def snr_row(model, *, family: str, d: int, seed: int = 0) -> dict[str, Any]:
    st = tracker_stats(model)
    q = st["q_tilde"]
    k = st["kappa"]
    return dict(
        family=family,
        d=int(d),
        seed=int(seed),
        q_tilde=q,
        e=st["e"],
        v=st["v"],
        cv2=st["cv2"],
        kappa_lev=st["kappa_lev"],
        kappa=k,
        corr_theory=st["corr_theory"],
        mse_rel= (1.0 / k) if (np.isfinite(k) and k > 0) else float("inf"),
        t_dagger=st["t_dagger"],
        inv_q= (1.0 / q) if q > QTILDE_FLOOR else float("inf"),
        t_star_le_inv_q=bool(st["t_dagger"] <= (1.0 / q) if q > QTILDE_FLOOR else True),
        n_iter=np.nan,
        converged=np.nan,
    )


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    r = spearmanr(a[m], b[m])
    return float(r.statistic)


def sample_central_moments(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    x = x - x.mean()
    m2 = float(np.mean(x ** 2))
    m3 = float(np.mean(x ** 3))
    m4 = float(np.mean(x ** 4))
    skew = m3 / m2 ** 1.5 if m2 > 0 else float("nan")
    kurt = m4 / m2 ** 2 - 3.0 if m2 > 0 else float("nan")
    return dict(var=m2, skew=skew, kurt_excess=kurt)


def sample_skew(w: np.ndarray, X: np.ndarray) -> float:
    p = np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64).ravel()
    return sample_central_moments(p)["skew"]


def acf(x: np.ndarray, lags: int = 21) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    x = x - x.mean()
    var = float(np.mean(x ** 2))
    out = np.empty(lags + 1)
    out[0] = 1.0
    if var <= 0:
        out[1:] = np.nan
        return out
    for k in range(1, lags + 1):
        out[k] = float(np.mean(x[k:] * x[:-k]) / var)
    return out


def rolling_sumsq(x: np.ndarray, window: int = 21) -> np.ndarray:
    """Centered rolling sum of squares (realized variance proxy)."""
    s = pd.Series(np.asarray(x, dtype=np.float64) ** 2)
    return s.rolling(window, center=True, min_periods=max(5, window // 3)).sum().to_numpy()


def t_of_weights(w: np.ndarray, Sigma: np.ndarray, gamma: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64).ravel()
    g = float(w @ np.asarray(gamma, dtype=np.float64).ravel())
    s2 = float(w @ (np.asarray(Sigma, dtype=np.float64) @ w))
    if abs(g) < 1e-18:
        return float("inf")
    return s2 / (g * g)


def min_var_weights(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    ones = np.ones(L.shape[0])
    z = np.linalg.solve(L, ones)
    w = np.linalg.solve(L.T, z)
    s = w.sum()
    return w / s if s != 0 else w


def pc1_weights(Sigma: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(np.asarray(Sigma, dtype=np.float64))
    w = evecs[:, int(np.argmax(evals))]
    return w / np.abs(w).sum()


def model_skew_at_t(t: float, e: float, v: float, mu3: float) -> float:
    den = (v + t * e) ** 1.5
    if den <= 0 or not np.isfinite(den):
        return float("nan")
    return (mu3 + 3.0 * t * v) / den


def gamma_market_split(gamma: np.ndarray) -> dict[str, float]:
    """γ = g 1 + δ with 1ᵀδ = 0."""
    gvec = np.asarray(gamma, dtype=np.float64).ravel()
    g = float(gvec.mean())
    delta = gvec - g
    return dict(g=g, delta_l2=float(np.dot(delta, delta)), delta=delta)


def gaussian_thin_bound(c: np.ndarray, q_tilde: float) -> np.ndarray:
    """Proposition 2: P(Ŷ ≤ −c) ≤ 2 Φ(−2 √(c q̃))."""
    c = np.asarray(c, dtype=np.float64)
    z = -2.0 * np.sqrt(np.maximum(c, 0.0) * q_tilde)
    return 2.0 * gauss.cdf(z)


def empirical_left_tail(yhat: np.ndarray, c: np.ndarray) -> np.ndarray:
    yhat = np.asarray(yhat, dtype=np.float64)
    yhat = yhat[np.isfinite(yhat)]
    c = np.asarray(c, dtype=np.float64)
    return np.array([float(np.mean(yhat <= -cc)) for cc in c])


def weight_anatomy(w: np.ndarray, tickers: list[str], n_top: int = 10) -> dict[str, Any]:
    w = np.asarray(w, dtype=np.float64).ravel()
    ug = unit_gross(w)
    order = np.argsort(-np.abs(ug))
    top = [(tickers[i], float(ug[i])) for i in order[:n_top]]
    return dict(
        n=int(w.size),
        gross=float(np.abs(w).sum()),
        net=float(w.sum()),
        long_share=float(np.clip(ug, 0, None).sum()),
        n_long=int((ug > 0).sum()),
        n_short=int((ug < 0).sum()),
        herfindahl=float(np.sum(ug ** 2)),
        top=top,
        unit_gross=ug,
    )


# ---------------------------------------------------------------------------
# Sample-skew maximisation (autodiff + multi-start)
# ---------------------------------------------------------------------------

def maximize_sample_skew(
    X: np.ndarray,
    *,
    n_starts: int = 20,
    seed: int = 0,
    w0_extra: list[np.ndarray] | None = None,
    maxiter: int = 200,
) -> dict[str, Any]:
    """Maximise sample skewness of wᵀX over directions (unit sphere)."""
    Xj = jnp.asarray(X, dtype=jnp.float64)
    d = int(Xj.shape[1])

    def _obj(z):
        w = z / jnp.linalg.norm(z)
        p = Xj @ w
        p = p - jnp.mean(p)
        m2 = jnp.mean(p * p)
        m3 = jnp.mean(p * p * p)
        return -m3 / (m2 ** 1.5 + 1e-18)

    val_and_grad = jax.jit(jax.value_and_grad(_obj))

    def fun(z):
        v, g = val_and_grad(jnp.asarray(z, dtype=jnp.float64))
        return float(v), np.asarray(g, dtype=np.float64)

    rng = np.random.default_rng(seed)
    starts = [rng.normal(size=d) for _ in range(n_starts)]
    if w0_extra:
        starts.extend([np.asarray(w, dtype=np.float64).ravel() for w in w0_extra])

    best_fun = np.inf
    best_w = starts[0]
    n_ok = 0
    for z0 in starts:
        z0 = np.asarray(z0, dtype=np.float64)
        nrm = np.linalg.norm(z0)
        if nrm < 1e-15:
            continue
        z0 = z0 / nrm
        res = minimize(fun, z0, method="L-BFGS-B", jac=True, options={"maxiter": maxiter})
        if not np.isfinite(res.fun):
            continue
        n_ok += 1
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_w = np.asarray(res.x, dtype=np.float64)

    w = best_w / np.linalg.norm(best_w)
    skew = -best_fun if np.isfinite(best_fun) else float("nan")
    return dict(w=w, skew=float(skew), n_ok=n_ok, n_starts=len(starts))


# ---------------------------------------------------------------------------
# Sign-flip null / block bootstrap (NIG, cached per trial)
# ---------------------------------------------------------------------------

def sign_flip_null_nig(
    X: np.ndarray,
    *,
    B: int,
    seed: int,
    cache_stem: str,
) -> dict[str, np.ndarray]:
    """Refit NIG on day-wise sign-flips; return the null of q̃ and κ."""

    def _run():
        rec = {k: [] for k in ("rep", "q_tilde", "kappa", "n_iter", "converged", "elapsed")}
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        for b in range(B):
            name = f"{cache_stem}/b{b}.npz"

            def _fn(b=b, rng_seed=int(rng.integers(0, 2 ** 31 - 1))):
                Xb = sign_flip(X, np.random.default_rng(rng_seed))
                result = fit_nig(jnp.asarray(Xb))
                st = nig_fast_stats(result.model)
                return dict(
                    q_tilde=np.asarray(st["q_tilde"]),
                    kappa=np.asarray(st["kappa"]),
                    n_iter=np.asarray(result.n_iter, dtype=np.float64),
                    converged=np.asarray(float(bool(result.converged))),
                    elapsed=np.asarray(result.elapsed_time),
                )

            row = load_or_compute(name, _fn)
            rec["rep"].append(b)
            for k in ("q_tilde", "kappa", "n_iter", "converged", "elapsed"):
                rec[k].append(np.asarray(row[k]).reshape(()))
            if (b + 1) % 10 == 0:
                jax.clear_caches()
                print(f"    sign-flip {b+1}/{B}  ({time.perf_counter()-t0:.1f}s)")
        return {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}

    return load_or_compute(f"{cache_stem}/summary.npz", _run)


def signflip_pvalue(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=np.float64)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(obs):
        return float("nan")
    return float((1.0 + np.sum(null >= obs)) / (null.size + 1.0))


def block_bootstrap_nig(
    X: np.ndarray,
    *,
    B: int,
    block: int,
    seed: int,
    cache_stem: str,
    w_star_ref: np.ndarray,
) -> dict[str, np.ndarray]:
    """Moving-block bootstrap of NIG fits; cosine cone and κ distribution."""

    def _run():
        rec = {k: [] for k in (
            "rep", "q_tilde", "kappa", "cosine", "n_iter", "converged", "elapsed",
        )}
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        for b in range(B):
            name = f"{cache_stem}/b{b}.npz"

            def _fn(b=b, rng_seed=int(rng.integers(0, 2 ** 31 - 1))):
                idx = block_bootstrap_indices(X.shape[0], block, np.random.default_rng(rng_seed))
                result = fit_nig(jnp.asarray(X[idx]))
                st = nig_fast_stats(result.model)
                return dict(
                    q_tilde=np.asarray(st["q_tilde"]),
                    kappa=np.asarray(st["kappa"]),
                    cosine=np.asarray(cosine(w_star_ref, st["w_star"])),
                    n_iter=np.asarray(result.n_iter, dtype=np.float64),
                    converged=np.asarray(float(bool(result.converged))),
                    elapsed=np.asarray(result.elapsed_time),
                )

            row = load_or_compute(name, _fn)
            rec["rep"].append(b)
            for k in ("q_tilde", "kappa", "cosine", "n_iter", "converged", "elapsed"):
                rec[k].append(np.asarray(row[k]).reshape(()))
            if (b + 1) % 10 == 0:
                jax.clear_caches()
                print(f"    block-boot {b+1}/{B}  ({time.perf_counter()-t0:.1f}s)")
        return {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}

    return load_or_compute(f"{cache_stem}/summary.npz", _run)


# ---------------------------------------------------------------------------
# Eigen-attribution of q̃ (Phase 2)
# ---------------------------------------------------------------------------

def qtilde_eigen_attribution(model) -> dict[str, np.ndarray]:
    """q̃ = Σ_k (u_kᵀγ)² / λ_k along eigenvectors of Σ."""
    Sigma = np.asarray(model.sigma(), dtype=np.float64)
    gamma = np.asarray(model.gamma, dtype=np.float64).ravel()
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 1e-18)
    scores = (evecs.T @ gamma) ** 2 / evals
    order = np.argsort(-evals)
    return dict(
        evals=evals[order],
        contrib=scores[order],
        share=scores[order] / scores.sum() if scores.sum() > 0 else scores[order],
    )


def sigma_only_tau(model, tau_sigma: float) -> NormalMixtureEta:
    """Per-field τ that shrinks only E[XXᵀ/Y] (the Σ block)."""
    d = int(np.asarray(model.mu).shape[0])
    z = jnp.float64(0.0)
    return NormalMixtureEta(
        E_inv_Y=z, E_Y=z, E_log_Y=z,
        E_X=jnp.zeros(d), E_X_inv_Y=jnp.zeros(d),
        E_XXT_inv_Y=jnp.full((d, d), jnp.float64(tau_sigma)),
    )
