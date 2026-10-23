#!/usr/bin/env python3
"""Research prototypes for log K_ν order derivatives. Does not modify the package.

Compares the current custom-JVP finite-difference path against:
  - native AD of each regime kernel (no order FD)
  - centered whole-line moment quadrature (DLMF 10.32.9)
  - analytic small-z leading derivatives
  - Takekawa-style moments on the existing half-line GL rule

Also re-checks the 2026-09-05 audit numbers and a few non-Bessel headlines.

    uv run python prototype_bessel_derivatives.py
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.special import exp1, logsumexp as sp_logsumexp, roots_legendre

from normix.utils.bessel import (
    _GL_NODES_NP,
    _GL_WEIGHTS_NP,
    _HANKEL_THRESHOLD,
    _OLVER_V_THRESHOLD,
    _hankel_log_kv,
    _log_kv_scalar,
    _olver_log_kv,
    _quadrature_log_kv,
    _smallz_log_kv,
    log_kv,
)
from normix.utils.constants import BESSEL_EPS_V, BESSEL_SMALLZ_THRESHOLD, LOG_EPS

HERE = pathlib.Path(__file__).resolve().parent
AUDIT_JSON = HERE / "bessel-derivative-results.json"
OUT_JSON = HERE / "prototype-bessel-results.json"
FIGDIR = HERE / "figures"

H = float(BESSEL_EPS_V)
AUDIT_POINTS = [
    (0.0, 1.0),
    (0.5, 1.0),
    (25.0, 0.1),
    (25.0, 1.0),
    (25.0, 1e-6),
    (1.0, 1e4),
    (5.0, 1e6),
]


# ---------------------------------------------------------------------------
# Current path and regime
# ---------------------------------------------------------------------------

def regime(v: float, z: float) -> str:
    v_abs = abs(v)
    if z > max(_HANKEL_THRESHOLD, v_abs * v_abs / 4.0):
        return "hankel"
    if v_abs > _OLVER_V_THRESHOLD:
        return "olver"
    if (z < BESSEL_SMALLZ_THRESHOLD) and (v_abs > 0.5):
        return "smallz"
    return "quadrature"


def fd_stencil_regimes(v: float, z: float) -> dict:
    return {
        "center": regime(v, z),
        "v+h": regime(v + H, z),
        "v-h": regime(v - H, z),
        "v+2h": regime(v + 2 * H, z),
        "v-2h": regime(v - 2 * H, z),
        "crosses_seam": len({regime(v + H, z), regime(v - H, z), regime(v, z)}) > 1,
    }


# ---------------------------------------------------------------------------
# Kernel AD: differentiate the approximation, not an FD of log_kv
# ---------------------------------------------------------------------------

_k_d1 = jax.jit(jax.grad(_log_kv_scalar, argnums=0))
_k_d2 = jax.jit(jax.grad(jax.grad(_log_kv_scalar, argnums=0), argnums=0))
_h_d1 = jax.jit(jax.grad(_hankel_log_kv, argnums=0))
_h_d2 = jax.jit(jax.grad(jax.grad(_hankel_log_kv, argnums=0), argnums=0))
_o_d1 = jax.jit(jax.grad(_olver_log_kv, argnums=0))
_o_d2 = jax.jit(jax.grad(jax.grad(_olver_log_kv, argnums=0), argnums=0))
_q_d1 = jax.jit(jax.grad(_quadrature_log_kv, argnums=0))
_q_d2 = jax.jit(jax.grad(jax.grad(_quadrature_log_kv, argnums=0), argnums=0))
_s_d1 = jax.jit(jax.grad(_smallz_log_kv, argnums=0))
_s_d2 = jax.jit(jax.grad(jax.grad(_smallz_log_kv, argnums=0), argnums=0))
_cur_d1 = jax.jit(jax.grad(log_kv, argnums=0))
_cur_d2 = jax.jit(jax.grad(jax.grad(log_kv, argnums=0), argnums=0))


def _pair(fn, v, z):
    vv = jnp.asarray(v)
    zz = jnp.asarray(z)
    return float(fn(vv, zz)), float(jax.grad(fn, argnums=0)(vv, zz))


# ---------------------------------------------------------------------------
# Analytic small-z leading term (ν>0)
# ---------------------------------------------------------------------------

def smallz_analytic(v: float, z: float) -> dict:
    from jax.scipy.special import digamma, gammaln
    from jax.scipy.special import polygamma
    va = abs(v)
    log_k = float(gammaln(va) + va * jnp.log(2.0 / z) - jnp.log(2.0))
    d1 = float(digamma(va) + jnp.log(2.0 / z)) * (1.0 if v >= 0 else -1.0)
    d2 = float(polygamma(1, va))
    return {"log_k": log_k, "d_order": d1, "d2_order": d2}


# ---------------------------------------------------------------------------
# Centered whole-line moment quadrature (numpy, adaptive interval)
# ---------------------------------------------------------------------------

def _x_minus_sinh(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x2 = x * x
    series = (
        -x * x2 / 6.0
        * (1.0 + x2 / 20.0 + x2**2 / 840.0 + x2**3 / 60480.0 + x2**4 / 6_652_800.0)
    )
    return np.where(np.abs(x) < 0.4, series, x - np.sinh(x))


def _cosh_m1(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sinh(0.5 * np.asarray(x, dtype=np.float64)) ** 2


def _logw_x(v: float, kappa: float, x: np.ndarray) -> np.ndarray:
    return v * _x_minus_sinh(x) - kappa * _cosh_m1(x)


def _expand_tail(v: float, kappa: float, side: float, log_drop: float) -> float:
    x = side
    for _ in range(60):
        if float(_logw_x(v, kappa, np.array([x]))[0]) <= -log_drop:
            return x
        x *= 2.0
        if abs(x) > 200.0:
            return x
    return x


def moment_bundle(v: float, z: float, n: int = 128, log_drop: float = 50.0) -> dict:
    """L, L_ν, L_νν, L_z from one centered quadrature in y = √κ (u − u₀)."""
    z = max(float(z), np.finfo(np.float64).tiny)
    kappa = math.sqrt(v * v + z * z)
    scale = math.sqrt(kappa)
    u0 = math.asinh(v / z)

    def logw_y(y: float) -> float:
        return float(_logw_x(v, kappa, np.array([y / scale]))[0])

    left_y, right_y = -0.5, 0.5
    while logw_y(left_y) > -log_drop and left_y > -200.0:
        left_y *= 2.0
    while logw_y(right_y) > -log_drop and right_y < 200.0:
        right_y *= 2.0

    nodes, weights = roots_legendre(n)
    half = 0.5 * (right_y - left_y)
    mid = 0.5 * (right_y + left_y)
    y = mid + half * nodes
    x = y / scale
    lw = _logw_x(v, kappa, x)
    log_mass = sp_logsumexp(lw + np.log(weights) + math.log(abs(half)) - math.log(scale))
    w = np.exp(lw + np.log(weights) + math.log(abs(half)) - math.log(scale) - log_mass)
    mean_x = float(np.dot(w, x))
    var_x = float(np.dot(w, (x - mean_x) ** 2))
    cosh_u0 = kappa / z
    sinh_u0 = v / z
    cosh_u = cosh_u0 * np.cosh(x) + sinh_u0 * np.sinh(x)
    e_c = float(np.dot(w, cosh_u))
    return {
        "log_k": v * u0 - kappa + log_mass - math.log(2.0),
        "d_order": u0 + mean_x,
        "d2_order": var_x,
        "d_z": -e_c,
        "interval_y": [left_y, right_y],
        "n": n,
        "u0": u0,
        "kappa": kappa,
    }


def gauss_hermite_bundle(v: float, z: float, n: int = 64) -> dict:
    """Probabilists' Gauss–Hermite on the centered exponent."""
    from scipy.special import roots_hermitenorm
    z = max(float(z), np.finfo(np.float64).tiny)
    kappa = math.sqrt(v * v + z * z)
    scale = math.sqrt(kappa)
    u0 = math.asinh(v / z)
    y, wgh = roots_hermitenorm(n)
    x = y / scale
    lw = _logw_x(v, kappa, x)
    log_term = lw + 0.5 * y * y - math.log(scale)
    log_mass = sp_logsumexp(log_term + np.log(np.maximum(wgh, np.finfo(np.float64).tiny)))
    w = np.exp(log_term + np.log(np.maximum(wgh, np.finfo(np.float64).tiny)) - log_mass)
    mean_x = float(np.dot(w, x))
    var_x = float(np.dot(w, (x - mean_x) ** 2))
    return {
        "log_k": v * u0 - kappa + log_mass - math.log(2.0),
        "d_order": u0 + mean_x,
        "d2_order": var_x,
        "n": n,
    }


def half_line_moments(v: float, z: float) -> dict:
    """Moments of the current 64-node Takekawa GL rule (uncentered)."""
    v_abs = abs(v)
    z_safe = max(z, LOG_EPS)
    P = 50.0
    T = max(math.log(2.0 * P / z_safe), 1.0)
    for _ in range(6):
        T = max(math.log(2.0 * (P + v_abs * T) / z_safe), 1.0)
    T = T + 2.0
    t = T * (_GL_NODES_NP + 1.0) / 2.0
    # log f = -z cosh t + log cosh(v t)
    ax = np.abs(v_abs * t)
    log_cosh_vt = ax + np.log1p(np.exp(-2.0 * ax)) - math.log(2.0)
    log_f = -z * np.cosh(t) + log_cosh_vt
    log_w = np.log(_GL_WEIGHTS_NP) + log_f + math.log(T / 2.0)
    log_k = float(sp_logsumexp(log_w))
    w = np.exp(log_w - log_k)
    s = t * np.tanh(v_abs * t)
    sech2 = 1.0 / np.cosh(v_abs * t) ** 2
    e_s = float(np.dot(w, s))
    var_s = float(np.dot(w, (s - e_s) ** 2))
    e_t2_sech2 = float(np.dot(w, t * t * sech2))
    e_c = float(np.dot(w, np.cosh(t)))
    sign = 1.0 if v >= 0 else -1.0
    return {
        "log_k": log_k,
        "d_order": sign * e_s,
        "d2_order": var_s + e_t2_sech2,
        "d_z": -e_c,
        "T": T,
    }


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def load_audit():
    payload = json.loads(AUDIT_JSON.read_text())
    refs = {}
    for row in payload["cases"]:
        refs[(row["v"], row["z"])] = {
            "log_k": float(row["reference"]["log_k"]),
            "d_order": float(row["reference"]["d_order"]),
            "d2_order": float(row["reference"]["d2_order"]),
            "normix": row["normix"],
        }
    return payload, refs


def mp_spotcheck(v: float, z: float, dps: int = 40) -> list[float]:
    import mpmath as mp
    with mp.workdps(dps):
        nu, arg = mp.mpf(v), mp.mpf(z)
        f = lambda order: mp.log(mp.besselk(order, arg))
        return [float(f(nu)), float(mp.diff(f, nu, 1)), float(mp.diff(f, nu, 2))]


def scaled_err(got: float, ref: float) -> float:
    return abs(got - ref) / max(1.0, abs(ref))


def half_integer_l_v(z: float) -> float:
    """DLMF 10.38.7: ∂_ν log K_ν(z) at ν=1/2 equals e^{2z} E_1(2z)."""
    return math.exp(2.0 * z) * float(exp1(2.0 * z))


# ---------------------------------------------------------------------------
# Other audit headlines (cheap)
# ---------------------------------------------------------------------------

def verify_other_claims() -> dict:
    from normix import GIG, Gamma, InverseGamma

    out = {}
    cpu = float(log_kv(1000.0, 500.0, backend="cpu"))
    jax_val = float(log_kv(1000.0, 500.0))
    try:
        mp_ref = mp_spotcheck(1000.0, 500.0, dps=25)[0]
    except Exception as exc:
        mp_ref = None
        out["cpu_overflow_mp_error"] = str(exc)
    out["cpu_overflow"] = {
        "cpu": cpu,
        "jax": jax_val,
        "mp": mp_ref,
        "claimed_cpu": 383.066358166,
        "claimed_ref": 322.317651492,
    }
    out["tiny_z_floor"] = {
        "log_kv_1_1e-50": float(log_kv(1.0, 1e-50)),
        "claimed": 69.07755,
        "true_leading": math.log(0.5) + math.lgamma(1.0) + 1.0 * math.log(2.0 / 1e-50),
    }
    g = GIG(-2.0, 1.0, 1e-22)
    out["gig_tiny_b"] = {
        "p": float(g.p),
        "a": float(g.a),
        "b": float(g.b),
        "jax_psi": float(g.log_partition()),
        "cpu_psi": float(type(g)._log_partition_cpu(np.asarray(g.natural_params()))),
        "claimed_jax": 69.077552790,
        "claimed_cpu": 690.775527898,
        "claimed_true": 102.700038453,
    }
    try:
        draws = np.asarray(GIG(-2.0, 1e-8, 1e-8).rvs(200, seed=42))
        out["gig_nan_samples"] = {
            "n": 200,
            "finite": int(np.isfinite(draws).sum()),
            "claimed_finite": 0,
        }
    except Exception as exc:
        out["gig_nan_samples"] = {"error": str(exc)}
    ig = InverseGamma(0.5, 1.0)
    out["invgamma_moments"] = {
        "mean": float(ig.mean()),
        "var": float(ig.var()),
        "claimed_mean": -2.0,
        "claimed_var": -8 / 3,
    }
    out["renyi"] = {
        "gamma": float(Gamma(0.2, 1.0).renyi(2.0)),
        "invgamma": float(ig.renyi(0.5)),
        "claimed_gamma": 1.324736,
        "claimed_invgamma": 2.260212,
    }
    return out


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def eval_methods(v: float, z: float) -> dict:
    vv = jnp.asarray(v)
    zz = jnp.asarray(z)
    cur_val = float(log_kv(vv, zz))
    cur_d1 = float(_cur_d1(vv, zz))
    cur_d2 = float(_cur_d2(vv, zz))
    k_d1 = float(_k_d1(vv, zz))
    k_d2 = float(_k_d2(vv, zz))
    mom = moment_bundle(v, z, n=128)
    mom256 = moment_bundle(v, z, n=256)
    gh = gauss_hermite_bundle(v, z, n=64)
    half = half_line_moments(v, z)
    sz = smallz_analytic(v, z) if abs(v) > 0.5 else None
    row = {
        "v": v,
        "z": z,
        "regime": fd_stencil_regimes(v, z),
        "current": {"log_k": cur_val, "d_order": cur_d1, "d2_order": cur_d2},
        "kernel_ad": {
            "log_k": float(_log_kv_scalar(vv, zz)),
            "d_order": k_d1,
            "d2_order": k_d2,
        },
        "moments_n128": {k: mom[k] for k in ("log_k", "d_order", "d2_order", "d_z")},
        "moments_n256": {k: mom256[k] for k in ("log_k", "d_order", "d2_order")},
        "gauss_hermite_n64": gh,
        "moments_interval_y": mom["interval_y"],
        "half_line_gl64": half,
        "hankel_ad": _safe_kernel(_hankel_log_kv, _h_d1, _h_d2, vv, zz),
        "olver_ad": (
            None if abs(v) < 1e-8
            else _safe_kernel(_olver_log_kv, _o_d1, _o_d2, jnp.asarray(abs(v)), zz, sign=1.0 if v >= 0 else -1.0)
        ),
        "quad_ad": _safe_kernel(_quadrature_log_kv, _q_d1, _q_d2, vv, zz),
        "smallz_analytic": sz,
        "smallz_ad": _safe_kernel(_smallz_log_kv, _s_d1, _s_d2, vv, zz),
    }
    return row


def _safe_kernel(val_fn, d1_fn, d2_fn, v, z, sign: float = 1.0):
    try:
        return {
            "log_k": float(val_fn(v, z)),
            "d_order": sign * float(d1_fn(v, z)),
            "d2_order": float(d2_fn(v, z)),
        }
    except Exception as exc:
        return {"error": str(exc)}


def errors_vs_ref(row: dict, ref: dict) -> dict:
    out = {}
    for name in (
        "current",
        "kernel_ad",
        "moments_n128",
        "moments_n256",
        "gauss_hermite_n64",
        "half_line_gl64",
        "hankel_ad",
        "olver_ad",
        "quad_ad",
        "smallz_analytic",
        "smallz_ad",
    ):
        block = row.get(name)
        if not block:
            continue
        out[name] = {
            q: scaled_err(block[q], ref[q])
            for q in ("log_k", "d_order", "d2_order")
            if q in block
        }
    return out


def gig_hessian_probe() -> list[dict]:
    from normix import GIG

    cases = [
        (0.5, 1.0, 1.0),
        (25.0, 1.0, 1.0),
        (25.0, 0.1, 0.1),
        (1.0, 1e4, 1e4),
        (0.0, 2.0, 2.0),
    ]
    rows = []
    for p, a, b in cases:
        g = GIG(p, a, b)
        theta = g.natural_params()
        H = np.asarray(type(g)._hessian_log_partition(theta))
        z = math.sqrt(a * b)
        mom = moment_bundle(p, z, n=256)
        rows.append({
            "p": p, "a": a, "b": b, "z": z,
            "H11_gig": float(H[0, 0]),
            "Lvv_moments": mom["d2_order"],
            "Lvv_kernel_ad": float(_k_d2(jnp.asarray(p), jnp.asarray(z))),
            "Lvv_current_nested": float(_cur_d2(jnp.asarray(p), jnp.asarray(z))),
            "eigmin": float(np.min(np.linalg.eigvalsh(H))),
        })
    return rows


def seam_slice(z: float = 1.0) -> dict:
    vs = np.concatenate([
        np.linspace(24.0, 24.999, 8),
        np.array([24.99999, 25.0, 25.00001]),
        np.linspace(25.001, 26.0, 8),
    ])
    rows = []
    for v in vs:
        vv = float(v)
        rows.append({
            "v": vv,
            "regime": regime(vv, z),
            "current_d1": float(_cur_d1(jnp.asarray(vv), jnp.asarray(z))),
            "current_d2": float(_cur_d2(jnp.asarray(vv), jnp.asarray(z))),
            "kernel_ad_d1": float(_k_d1(jnp.asarray(vv), jnp.asarray(z))),
            "kernel_ad_d2": float(_k_d2(jnp.asarray(vv), jnp.asarray(z))),
            "moments_d1": moment_bundle(vv, z, n=128)["d_order"],
            "moments_d2": moment_bundle(vv, z, n=128)["d2_order"],
        })
    anchors = [24.0, 24.9, 25.0, 25.1, 26.0]
    mp = []
    for v in anchors:
        try:
            ref = mp_spotcheck(v, z, dps=30)
            mp.append({"v": v, "d_order": ref[1], "d2_order": ref[2], "log_k": ref[0]})
        except Exception as exc:
            mp.append({"v": v, "error": str(exc)})
    return {"z": z, "rows": rows, "mp_anchors": mp}


def make_figures(audit_rows: list[dict], seam: dict) -> list[str]:
    try:
        from normix.utils.plotting import COLORS, FIG_H, FIG_W, set_theme
        import matplotlib.pyplot as plt
        set_theme()
    except Exception:
        import matplotlib.pyplot as plt
        COLORS = {
            "accent": "#1B365D", "green": "#28724F", "brick": "#9B3A34",
            "gold": "#A98324", "muted": "#6B6A64", "paper": "#F5F4ED",
            "ink": "#141413", "umber": "#8F5A2A", "teal": "#2F6F73",
        }
        FIG_W, FIG_H = 12.0, 12 / 1.618
    FIGDIR.mkdir(exist_ok=True)
    paths = []

    labels = [f"({r['v']:g},{r['z']:g})" for r in audit_rows]
    methods = [
        ("current", COLORS["brick"]),
        ("kernel_ad", COLORS["gold"]),
        ("moments_n128", COLORS["green"]),
        ("gauss_hermite_n64", COLORS["teal"] if "teal" in COLORS else COLORS.get("accent", "#2F6F73")),
        ("olver_ad", COLORS.get("violet", "#6D597A")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.85))
    x = np.arange(len(labels))
    width = 0.15
    for ax, qty, title in zip(axes, ("d_order", "d2_order"),
                             (r"$\partial_\nu \log K$", r"$\partial_{\nu\nu} \log K$")):
        for i, (name, color) in enumerate(methods):
            vals = []
            for r in audit_rows:
                e = r["errors"].get(name, {}).get(qty, np.nan)
                vals.append(max(e, 1e-18) if np.isfinite(e) else np.nan)
            ax.bar(x + (i - 2.0) * width, vals, width, label=name, color=color)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("scaled abs error vs mpmath")
        ax.set_title(title)
        ax.set_ylim(1e-16, 1e10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p1 = FIGDIR / "method_errors_audit_points.png"
    fig.savefig(p1, dpi=110, facecolor=COLORS.get("paper", "white"))
    plt.close(fig)
    paths.append(str(p1))

    rows = seam["rows"]
    vs = [r["v"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.85))
    axes[0].plot(vs, [r["current_d1"] for r in rows], color=COLORS["brick"], label="current FD JVP")
    axes[0].plot(vs, [r["kernel_ad_d1"] for r in rows], color=COLORS["gold"], label="kernel AD")
    axes[0].plot(vs, [r["moments_d1"] for r in rows], color=COLORS["green"], label="moments n=128")
    for a in seam["mp_anchors"]:
        if "d_order" in a:
            axes[0].scatter([a["v"]], [a["d_order"]], color=COLORS["accent"], zorder=5)
    axes[0].axvline(25.0, color=COLORS["muted"], ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\nu$ at $z=1$")
    axes[0].set_ylabel(r"$L_\nu$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(vs, [r["current_d2"] for r in rows], color=COLORS["brick"], label="current nested AD")
    axes[1].plot(vs, [r["kernel_ad_d2"] for r in rows], color=COLORS["gold"], label="kernel AD")
    axes[1].plot(vs, [r["moments_d2"] for r in rows], color=COLORS["green"], label="moments n=128")
    for a in seam["mp_anchors"]:
        if "d2_order" in a:
            axes[1].scatter([a["v"]], [a["d2_order"]], color=COLORS["accent"], zorder=5)
    axes[1].axvline(25.0, color=COLORS["muted"], ls="--", lw=0.8)
    axes[1].set_xlabel(r"$\nu$ at $z=1$")
    axes[1].set_ylabel(r"$L_{\nu\nu}$")
    axes[1].set_ylim(-5, 2)
    fig.tight_layout()
    p2 = FIGDIR / "seam_v25_z1.png"
    fig.savefig(p2, dpi=110, facecolor=COLORS.get("paper", "white"))
    plt.close(fig)
    paths.append(str(p2))
    return paths


def main():
    t0 = time.time()
    audit_payload, refs = load_audit()

    # Warm JIT
    _ = float(_cur_d2(jnp.asarray(1.0), jnp.asarray(1.0)))
    _ = float(_k_d2(jnp.asarray(1.0), jnp.asarray(1.0)))

    jax_vs_json = []
    for (v, z), rec in refs.items():
        got = {
            "log_k": float(log_kv(jnp.asarray(v), jnp.asarray(z))),
            "d_order": float(_cur_d1(jnp.asarray(v), jnp.asarray(z))),
            "d2_order": float(_cur_d2(jnp.asarray(v), jnp.asarray(z))),
        }
        jax_vs_json.append({
            "v": v, "z": z,
            "delta_vs_json": {
                "log_k": got["log_k"] - rec["normix"]["log_k"],
                "d_order": got["d_order"] - rec["normix"]["d_order"],
                "d2_order": got["d2_order"] - rec["normix"]["d2_order_nested_ad"],
            },
            "got": got,
        })

    spot = {}
    for key in ((25.0, 1.0), (25.0, 1e-6), (0.5, 1.0)):
        spot[str(key)] = {
            "mp": mp_spotcheck(*key, dps=40),
            "json": [refs[key][q] for q in ("log_k", "d_order", "d2_order")],
        }

    half_id = {
        "formula": float(half_integer_l_v(1.0)),
        "json_ref": refs[(0.5, 1.0)]["d_order"],
        "normix": float(_cur_d1(jnp.asarray(0.5), jnp.asarray(1.0))),
    }

    audit_rows = []
    for v, z in AUDIT_POINTS:
        row = eval_methods(v, z)
        ref = refs[(v, z)]
        row["errors"] = errors_vs_ref(row, ref)
        row["ref"] = {k: ref[k] for k in ("log_k", "d_order", "d2_order")}
        audit_rows.append(row)
        print(f"({v},{z}) regime={row['regime']['center']} seam={row['regime']['crosses_seam']}")
        for name in ("current", "kernel_ad", "moments_n128", "gauss_hermite_n64", "olver_ad"):
            e = row["errors"].get(name)
            if not e:
                continue
            print(f"  {name:18s}  d1={e['d_order']:.3e}  d2={e['d2_order']:.3e}")

    extra = []
    for v, z in [(24.9, 1.0), (25.1, 1.0), (30.0, 1.0), (1.0, 1.0),
                 (10.0, 0.01), (0.0, 1e-4), (2.0, 50.0)]:
        row = eval_methods(v, z)
        ref_vals = mp_spotcheck(v, z, dps=30)
        ref = {"log_k": ref_vals[0], "d_order": ref_vals[1], "d2_order": ref_vals[2]}
        extra.append({"point": [v, z], "regime": row["regime"], "errors": errors_vs_ref(row, ref),
                      "current": row["current"], "kernel_ad": row["kernel_ad"],
                      "moments_n128": row["moments_n128"], "ref": ref})
        print(f"extra ({v},{z}) kernel_ad d2 err={extra[-1]['errors']['kernel_ad']['d2_order']:.3e} "
              f"moments d2 err={extra[-1]['errors']['moments_n128']['d2_order']:.3e}")

    seam = seam_slice(1.0)
    gig = gig_hessian_probe()
    other = verify_other_claims()
    figs = make_figures(audit_rows, seam)

    payload = {
        "elapsed_s": time.time() - t0,
        "h": H,
        "jax_vs_audit_json": jax_vs_json,
        "mp_spotcheck": spot,
        "half_integer_identity": half_id,
        "audit_points": audit_rows,
        "extra_points": extra,
        "seam_z1": seam,
        "gig_hessian": gig,
        "other_claims": other,
        "figures": figs,
        "note": "Prototypes only; package code not changed.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print("saved", OUT_JSON)
    print("elapsed", payload["elapsed_s"])
    print("other", json.dumps(other, indent=2, default=str))


if __name__ == "__main__":
    main()
