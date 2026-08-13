# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Phase 1 — static full-history subordinator tracking on the S&P 500
#
# Theory: `dev-notes/research/subordinator_tracking_portfolio.md`.
# Plan: `dev-notes/research/subordinator_tracking_sp500_plan.md` §6.
# Phase 0: `00_synthetic_validation.py`.
#
# One NIG (and GH / VG) fit per universe on all 2552 days. Package code
# is not modified.

# %%
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

if "get_ipython" not in globals():
    matplotlib.use("Agg")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (HERE / "lib.py").exists():
    HERE = Path.cwd() / "notebooks" / "subordinator_tracking"
sys.path.insert(0, str(HERE))

from lib import (  # noqa: E402
    PRIMARY_D,
    PRIMARY_SEED,
    SIZES,
    acf,
    block_bootstrap_nig,
    cache_dir,
    cosine,
    data_hygiene,
    empirical_left_tail,
    ewma_smooth,
    figure_dir,
    fit_nig,
    gaussian_thin_bound,
    gamma_market_split,
    load_or_fit_generator,
    load_or_fit_gh_from_nig,
    load_or_fit_nig_named,
    load_or_fit_vg_named,
    load_sp500,
    maximize_sample_skew,
    min_var_weights,
    model_skew_at_t,
    nested_universe,
    nig_fast_stats,
    pc1_weights,
    pearson,
    rolling_sumsq,
    sample_central_moments,
    sample_skew,
    sign_flip_null_nig,
    signflip_pvalue,
    snr_row,
    spearman,
    t_of_weights,
    tracker_only,
    tracker_stats,
    unit_gross,
    weight_anatomy,
    y_estimators,
)
from normix.finance.risk import CVaR
from normix.utils.plotting import COLORS, FIG_H, FIG_W, set_theme

set_theme()
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
FIG = figure_dir()
TABLES = cache_dir() / "tables"
TABLES.mkdir(exist_ok=True)

# %%
D_NIG = list(SIZES)
D_GH = [5, 10, 25, 50]
D_VG = [5, 10]
SIGNFLIP_B = {5: 50, 10: 50, 25: 50, 50: 50, 100: 20, 200: 20, 468: 20}
BLOCK_B = 30
BLOCK_LEN = 21
SKEW_D = (10, 50)
UNIVERSE_SEED = PRIMARY_SEED

print("cache", cache_dir())

# %% [markdown]
# ## 1. Data, hygiene, nested universe (seed 0)

# %%
panel = load_sp500()
hygiene = data_hygiene(panel)
print(
    f"{hygiene['n_obs']} days × {hygiene['n_names']} names; "
    f"|r|>0.5: {hygiene['n_abs_gt_0.5']}; "
    f"zero-var: {hygiene['zero_var']}; nan: {hygiene['n_nan']}"
)
if hygiene["abs_gt_0.5"]:
    print("flagged |r|>0.5 (kept):")
    for dt, tic, val in hygiene["abs_gt_0.5"]:
        print(f"  {dt}  {tic:6s}  {val:+.3f}")

universe = nested_universe(list(panel.columns), SIZES, UNIVERSE_SEED)
dates = panel.index
tickers50 = universe[PRIMARY_D]
X50 = panel[tickers50].to_numpy(dtype=np.float64)
m50 = X50.mean(axis=1)
m_full = panel.to_numpy(dtype=np.float64).mean(axis=1)
print(f"d={PRIMARY_D} seed={UNIVERSE_SEED}: {tickers50[:8]} ...")

# %% [markdown]
# ## 2. Fits: NIG (all $d$), GH ($d\le 50$), VG ($d\le 10$), market univariate
#
# NIG $d=50$ reuses the Phase 0 generator cache. GH is a nested
# continuation from the NIG embedding — a family-sensitivity test, not a
# second cold start.

# %%
nig_models: dict[int, object] = {}
nig_meta: dict[int, dict] = {}
t0 = time.perf_counter()
for d in D_NIG:
    tickers = universe[d]
    if d == PRIMARY_D:
        model, meta = load_or_fit_generator(panel, tickers)
    else:
        model, meta = load_or_fit_nig_named(panel, tickers, f"nig_d{d}_seed{UNIVERSE_SEED}.npz")
    nig_models[d] = model
    nig_meta[d] = meta
    st = nig_fast_stats(model)
    print(
        f"  NIG d={d:3d}  q̃={st['q_tilde']:.4g}  κ={st['kappa']:.4g}  "
        f"iter={meta.get('n_iter')}  ({time.perf_counter()-t0:.1f}s)"
    )

gh_models: dict[int, object] = {}
gh_meta: dict[int, dict] = {}
for d in D_GH:
    tickers = universe[d]
    X = jnp.asarray(panel[tickers].to_numpy(), dtype=np.float64)
    model, meta = load_or_fit_gh_from_nig(
        nig_models[d], X, f"gh_d{d}_seed{UNIVERSE_SEED}.npz", tickers=tickers,
    )
    gh_models[d] = model
    gh_meta[d] = meta
    st = tracker_stats(model)
    print(
        f"  GH  d={d:3d}  q̃={st['q_tilde']:.4g}  κ={st['kappa']:.4g}  "
        f"p={float(model.p):.3g}  iter={meta.get('n_iter')}  "
        f"({time.perf_counter()-t0:.1f}s)"
    )

vg_models: dict[int, object] = {}
vg_meta: dict[int, dict] = {}
for d in D_VG:
    tickers = universe[d]
    model, meta = load_or_fit_vg_named(panel, tickers, f"vg_d{d}_seed{UNIVERSE_SEED}.npz")
    vg_models[d] = model
    vg_meta[d] = meta
    st = tracker_stats(model)
    print(
        f"  VG  d={d:3d}  q̃={st['q_tilde']:.4g}  κ={st['kappa']:.4g}  "
        f"α={float(model.alpha):.3g}  iter={meta.get('n_iter')}  "
        f"({time.perf_counter()-t0:.1f}s)"
    )

mkt_models = {}
for label, series in (("ew50", m50), ("ew468", m_full)):
    name = f"nig_{label}_univariate.npz"
    fake = pd.DataFrame({label: series}, index=dates)
    model, meta = load_or_fit_nig_named(fake, [label], name)
    mkt_models[label] = (model, meta)
    st = nig_fast_stats(model)
    print(
        f"  NIG {label}  q̃={st['q_tilde']:.4g}  κ={st['kappa']:.4g}  "
        f"iter={meta.get('n_iter')}"
    )

# %% [markdown]
# ## 3. Headline SNR table and sign-flip null (H1)
#
# Every $\hat{\tilde q}$ is tested against a day-wise sign-flip null that
# kills odd joint moments and preserves $\Sigma$. Phase 0's synthetic
# $c=0$ floor at $d=50$ was $0.0775$ (95% quantile); the sign-flip is the
# real-data analogue.

# %%
snr_rows = []
for d, model in nig_models.items():
    row = snr_row(model, family="nig", d=d, seed=UNIVERSE_SEED)
    row["n_iter"] = nig_meta[d].get("n_iter")
    row["converged"] = nig_meta[d].get("converged")
    snr_rows.append(row)
for d, model in gh_models.items():
    row = snr_row(model, family="gh", d=d, seed=UNIVERSE_SEED)
    row["n_iter"] = gh_meta[d].get("n_iter")
    row["converged"] = gh_meta[d].get("converged")
    snr_rows.append(row)
for d, model in vg_models.items():
    row = snr_row(model, family="vg", d=d, seed=UNIVERSE_SEED)
    row["n_iter"] = vg_meta[d].get("n_iter")
    row["converged"] = vg_meta[d].get("converged")
    snr_rows.append(row)
for label, (model, meta) in mkt_models.items():
    row = snr_row(model, family=f"nig_{label}", d=1, seed=UNIVERSE_SEED)
    row["n_iter"] = meta.get("n_iter")
    row["converged"] = meta.get("converged")
    snr_rows.append(row)

snr_tbl = pd.DataFrame(snr_rows)

nulls = {}
for d, B in SIGNFLIP_B.items():
    print(f"sign-flip NIG d={d} B={B}")
    X = panel[universe[d]].to_numpy(dtype=np.float64)
    nulls[d] = sign_flip_null_nig(
        X, B=B, seed=1000 + d, cache_stem=f"signflip/nig_d{d}_seed{UNIVERSE_SEED}",
    )

pvals = []
q95s = []
for _, row in snr_tbl.iterrows():
    d = int(row["d"])
    if row["family"] == "nig" and d in nulls:
        q95 = float(np.quantile(nulls[d]["q_tilde"], 0.95))
        p = signflip_pvalue(float(row["q_tilde"]), nulls[d]["q_tilde"])
    else:
        q95, p = np.nan, np.nan
    q95s.append(q95)
    pvals.append(p)
snr_tbl["q_tilde_null95"] = q95s
snr_tbl["p_signflip"] = pvals
snr_tbl.to_csv(TABLES / "phase1_snr.csv", index=False)
cols = [
    "family", "d", "q_tilde", "q_tilde_null95", "p_signflip",
    "e", "v", "cv2", "kappa_lev", "kappa", "corr_theory", "mse_rel",
    "t_dagger", "t_star_le_inv_q", "n_iter",
]
print(snr_tbl[cols].to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(FIG_W * 0.7, FIG_H * 0.55))
nig_s = snr_tbl[snr_tbl["family"] == "nig"].sort_values("d")
ax.plot(nig_s["d"], nig_s["q_tilde"], marker="o", color=COLORS["accent"], label=r"NIG $\hat{\tilde q}$")
ax.plot(
    nig_s["d"], nig_s["kappa"], marker="s", color=COLORS["green"],
    label=r"NIG $\hat\kappa$",
)
d_null = np.array(sorted(nulls))
ax.plot(
    d_null, [float(np.quantile(nulls[d]["q_tilde"], 0.95)) for d in d_null],
    color=COLORS["muted"], ls="--", marker="x",
    label=r"sign-flip 95% of $\hat{\tilde q}$",
)
gh_s = snr_tbl[snr_tbl["family"] == "gh"].sort_values("d")
if len(gh_s):
    ax.plot(gh_s["d"], gh_s["q_tilde"], marker="D", color=COLORS["umber"], label=r"GH $\hat{\tilde q}$")
ax.set_xlabel(r"$d$")
ax.set_ylabel(r"$\hat{\tilde q}$, $\hat\kappa$")
ax.set_title("static full-history SNR vs sign-flip floor (seed 0)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "01_snr_vs_d.png", dpi=110)
plt.show()

# %% [markdown]
# ## 4. Tracker time series at $d=50$ (H1, misspecification ACF)
#
# Volatility proxies: centered 21-day RV of the equal-weight mean,
# EWMA RV ($h=21$), and cross-sectional dispersion.

# %%
model50 = nig_models[PRIMARY_D]
st50 = tracker_stats(model50)
est = y_estimators(model50, jnp.asarray(X50))
Y_hat = np.asarray(est["Y_hat"])
Y_lin = np.asarray(est["Y_lin"])
Y_post = np.asarray(est["Y_post"])
q_perp = np.asarray(est["q_perp"])
disp = ((X50 - X50.mean(axis=1, keepdims=True)) / X50.std(axis=0, ddof=1)) ** 2
disp = disp.mean(axis=1)
rv21 = rolling_sumsq(m50, 21)
rv_ewma = ewma_smooth(m50 ** 2, 21)

mom_hat = sample_central_moments(Y_hat)
var_theory = st50["v"] + st50["e"] / st50["q_tilde"]
port = model50.project(jnp.asarray(st50["w_star"]))
skew_model = float(port.skewness())
acf_hat = acf(Y_hat, 21)
print(
    f"Var(Ŷ) sample={mom_hat['var']:.4g}  model v+e/q̃={var_theory:.4g}\n"
    f"skew(Ŷ) sample={mom_hat['skew']:.4g}  model={skew_model:.4g}\n"
    f"ACF1(Ŷ)={acf_hat[1]:.3f}  ACF5={acf_hat[5]:.3f}  ACF21={acf_hat[21]:.3f}  "
    f"(i.i.d. model: 0)"
)

proxy_tbl = pd.DataFrame([
    dict(
        a="Y_hat", b=name, pearson=pearson(Y_hat, series),
        spearman=spearman(Y_hat, series),
    )
    for name, series in (
        ("Y_lin", Y_lin), ("Y_post", Y_post), ("q_perp", q_perp),
        ("disp", disp), ("rv21", rv21), ("rv_ewma", rv_ewma),
    )
] + [
    dict(
        a="Y_post", b=name, pearson=pearson(Y_post, series),
        spearman=spearman(Y_post, series),
    )
    for name, series in (
        ("disp", disp), ("rv21", rv21), ("rv_ewma", rv_ewma), ("q_perp", q_perp),
    )
])
print(proxy_tbl.to_string(index=False))
proxy_tbl.to_csv(TABLES / "phase1_proxy_corr.csv", index=False)

# model-implied channel corrs (simulate from the fitted NIG)
X_sim, Y_sim = model50.joint.rvs(len(X50), seed=7)
est_sim = y_estimators(model50, X_sim)
print(
    "model-implied (sim): "
    f"corr(Ŷ, E[Y|X])={pearson(est_sim['Y_hat'], est_sim['Y_post']):.3f}  "
    f"corr(Ŷ, q⊥)={pearson(est_sim['Y_hat'], est_sim['q_perp']):.3f}  "
    f"corr(E[Y|X], q⊥)={pearson(est_sim['Y_post'], est_sim['q_perp']):.3f}"
)
print(
    "sample:              "
    f"corr(Ŷ, E[Y|X])={pearson(Y_hat, Y_post):.3f}  "
    f"corr(Ŷ, q⊥)={pearson(Y_hat, q_perp):.3f}  "
    f"corr(E[Y|X], q⊥)={pearson(Y_post, q_perp):.3f}"
)

# %%
fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H * 1.15), sharex=True)
t_idx = np.arange(len(dates))
axes[0].plot(dates, Y_hat, color=COLORS["accent"], lw=0.7, label=r"tracker $\hat Y$")
axes[0].plot(dates, Y_lin, color=COLORS["green"], lw=0.7, alpha=0.85, label="linear Bayes")
axes[0].plot(dates, Y_post, color=COLORS["umber"], lw=0.8, label=r"$E[Y\mid X]$")
axes[0].set_ylabel(r"$Y$ estimators")
axes[0].legend(frameon=False, fontsize=8, ncol=3)
axes[1].plot(dates, Y_post, color=COLORS["umber"], lw=0.8, label=r"$E[Y\mid X]$")
ax1b = axes[1].twinx()
ax1b.plot(dates, rv21, color=COLORS["muted"], lw=0.7, alpha=0.8, label="21d RV")
axes[1].set_ylabel(r"$E[Y\mid X]$")
ax1b.set_ylabel("21-day RV")
axes[2].plot(dates, Y_hat, color=COLORS["accent"], lw=0.6, alpha=0.5, label=r"$\hat Y$")
axes[2].plot(dates, disp, color=COLORS["violet"], lw=0.7, label="x-sec. dispersion")
axes[2].set_ylabel("tracker / dispersion")
axes[2].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.axvline(pd.Timestamp("2018-02-05"), color=COLORS["rule"], lw=0.8)
    ax.axvline(pd.Timestamp("2020-03-16"), color=COLORS["rule"], lw=0.8)
    ax.axvline(pd.Timestamp("2022-03-01"), color=COLORS["rule"], lw=0.8)
axes[0].set_title(rf"$d={PRIMARY_D}$ NIG tracker vs Bayes vs vol proxies")
fig.tight_layout()
fig.savefig(FIG / "01_tracker_series.png", dpi=110)
plt.show()

# COVID zoom
mask = (dates >= "2020-02-01") & (dates <= "2020-05-31")
fig, ax = plt.subplots(figsize=(FIG_W * 0.8, FIG_H * 0.5))
ax.plot(dates[mask], Y_hat[mask], color=COLORS["accent"], lw=1.0, label=r"$\hat Y$")
ax.plot(dates[mask], Y_post[mask], color=COLORS["umber"], lw=1.0, label=r"$E[Y\mid X]$")
ax2 = ax.twinx()
ax2.plot(dates[mask], rv21[mask], color=COLORS["muted"], lw=0.9, label="21d RV")
ax.set_ylabel("estimators")
ax2.set_ylabel("RV")
ax.set_title("COVID window")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "01_covid_zoom.png", dpi=110)
plt.show()

fig, ax = plt.subplots(figsize=(FIG_W * 0.5, FIG_H * 0.45))
ax.stem(np.arange(len(acf_hat)), acf_hat, linefmt="C0-", markerfmt="C0o", basefmt="k-")
ax.axhline(0, color=COLORS["rule"], lw=0.8)
band = 1.96 / np.sqrt(len(Y_hat))
ax.axhline(band, color=COLORS["muted"], ls="--", lw=0.8)
ax.axhline(-band, color=COLORS["muted"], ls="--", lw=0.8)
ax.set_xlabel("lag (days)")
ax.set_ylabel(r"ACF($\hat Y$)")
ax.set_title("tracker ACF (i.i.d. model: 0)")
fig.tight_layout()
fig.savefig(FIG / "01_tracker_acf.png", dpi=110)
plt.show()

# %% [markdown]
# ## 5. Max-skewness portfolio (H3)
#
# NIG identity $e\mu_3 = 3v^2$ forces $t^\dagger = -v/e < 0$, so the
# model's max-skewness portfolio is $w^\star$ for every fitted NIG. GH
# is the only place the GIG cumulant inequality is an empirical check.

# %%
print("t† vs 1/q̃:")
print(snr_tbl[snr_tbl["family"].isin(["nig", "gh", "vg"])][
    ["family", "d", "t_dagger", "inv_q", "t_star_le_inv_q", "q_tilde"]
].to_string(index=False))

Sigma50 = np.asarray(model50.sigma())
gamma50 = np.asarray(model50.gamma)
w_star = np.asarray(st50["w_star"])
w_eq = np.ones(PRIMARY_D) / PRIMARY_D
w_mv = min_var_weights(np.asarray(model50.L_Sigma))
w_pc = pc1_weights(Sigma50)
rng = np.random.default_rng(1)
random_ws = [rng.normal(size=PRIMARY_D) for _ in range(8)]
name_sk = [sample_skew(np.eye(PRIMARY_D)[i], X50) for i in range(PRIMARY_D)]
top_names = np.argsort(-np.abs(name_sk))[:3]

placements = [("w*", w_star), ("eq-wt", w_eq), ("min-var", w_mv), ("PC1", w_pc)]
for i, idx in enumerate(top_names):
    placements.append((tickers50[idx], np.eye(PRIMARY_D)[idx]))
for i, w in enumerate(random_ws):
    placements.append((f"rnd{i}", w))

place_rows = []
for name, w in placements:
    t = t_of_weights(w, Sigma50, gamma50)
    msk = model_skew_at_t(t, st50["e"], st50["v"], st50["mu3"]) if np.isfinite(t) else np.nan
    place_rows.append(dict(
        name=name, t=t, model_skew=msk, sample_skew=sample_skew(w, X50),
        m=float(np.asarray(w) @ gamma50),
    ))
place_tbl = pd.DataFrame(place_rows)
print(place_tbl.to_string(index=False))
place_tbl.to_csv(TABLES / "phase1_skew_placement.csv", index=False)

t_grid = np.linspace(1.0 / st50["q_tilde"], max(20.0, 1.0 / st50["q_tilde"] * 8), 200)
skew_curve = np.array([model_skew_at_t(t, st50["e"], st50["v"], st50["mu3"]) for t in t_grid])

fig, ax = plt.subplots(figsize=(FIG_W * 0.7, FIG_H * 0.55))
ax.plot(t_grid, skew_curve, color=COLORS["ink"], lw=1.2, label=r"model skew$(t)$")
marked = place_tbl[place_tbl["name"].isin(["w*", "eq-wt", "min-var", "PC1"])]
ax.scatter(marked["t"], marked["model_skew"], color=COLORS["accent"], zorder=3, label="model at portfolio")
ax.scatter(marked["t"], marked["sample_skew"], color=COLORS["umber"], marker="s", zorder=3, label="sample skew")
for _, r in marked.iterrows():
    ax.annotate(r["name"], (r["t"], r["sample_skew"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.axvline(1.0 / st50["q_tilde"], color=COLORS["rule"], ls="--", lw=0.8)
ax.set_xlabel(r"$t_w = w^\top\Sigma w / (w^\top\gamma)^2$")
ax.set_ylabel("skewness")
ax.set_title(rf"$d={PRIMARY_D}$ NIG: skewness vs $t$ (max at $t=1/\tilde q$)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "01_skew_placement.png", dpi=110)
plt.show()

# %%
skew_max_rows = []
for d in SKEW_D:
    Xd = panel[universe[d]].to_numpy(dtype=np.float64)
    md = nig_models[d]
    std = tracker_stats(md)
    w_tr = np.asarray(std["w_star"])
    res = maximize_sample_skew(Xd, n_starts=20, seed=3, w0_extra=[w_tr, np.ones(d)])
    # flip so wᵀγ ≥ 0 for comparison with the tracker
    if float(res["w"] @ np.asarray(md.gamma)) < 0:
        res["w"] = -res["w"]
        res["skew"] = -res["skew"]
    cos_tr = cosine(res["w"], w_tr)
    n = Xd.shape[0]
    Xa, Xb = Xd[: n // 2], Xd[n // 2:]
    res_a = maximize_sample_skew(Xa, n_starts=16, seed=4, w0_extra=[w_tr])
    if float(res_a["w"] @ np.asarray(md.gamma)) < 0:
        res_a["w"] = -res_a["w"]
    # tracker estimated on first half
    fit_a = fit_nig(jnp.asarray(Xa)).model
    w_tr_a = np.asarray(tracker_stats(fit_a)["w_star"])
    skew_max_rows.append(dict(
        d=d,
        sample_max_skew=res["skew"],
        tracker_sample_skew=sample_skew(w_tr, Xd),
        cosine_full=cos_tr,
        split_max_skew_oos=sample_skew(res_a["w"], Xb),
        split_tracker_skew_oos=sample_skew(w_tr_a, Xb),
        split_cosine=cosine(res_a["w"], w_tr_a),
    ))
    jax.clear_caches()
skew_max_tbl = pd.DataFrame(skew_max_rows)
print(skew_max_tbl.to_string(index=False))
skew_max_tbl.to_csv(TABLES / "phase1_skew_max.csv", index=False)

# %% [markdown]
# ## 6. Portfolio anatomy and Proposition 2 left tail
#
# Direction uncertainty: 21-day moving-block bootstrap, $B=30$, $d=50$.

# %%
anat = weight_anatomy(w_star, tickers50, n_top=12)
beta_i = np.array([
    np.cov(X50[:, i], m50, ddof=1)[0, 1] / np.var(m50, ddof=1) for i in range(PRIMARY_D)
])
ug = anat["unit_gross"]
P_star = X50 @ w_star
loc = float(w_star @ np.asarray(model50.mu))
print(
    f"gross={anat['gross']:.3g}  net={anat['net']:.3g}  "
    f"long_share={anat['long_share']:.3f}  n_long={anat['n_long']}  "
    f"n_short={anat['n_short']}  HHI={anat['herfindahl']:.3f}"
)
print("top |w| names:")
for tic, wi in anat["top"]:
    print(f"  {tic:6s}  {wi:+.4f}")
print(
    f"corr(Ŷ, m_t)={pearson(Y_hat, m50):.3f}  "
    f"corr(P*, m_t)={pearson(P_star, m50):.3f}  "
    f"corr(unit-gross w, beta)={pearson(ug, beta_i):.3f}"
)
print(
    f"E[P*] sample={P_star.mean():.5g}  location w*ᵀμ={loc:.5g}  "
    f"mean Ŷ={np.nanmean(Y_hat):.4g}  e={st50['e']:.4g}"
)
split = gamma_market_split(gamma50)
print(
    f"γ = g1+δ:  g={split['g']:.4g}  ||δ||²={split['delta_l2']:.4g}  "
    f"||γ||²={float(gamma50 @ gamma50):.4g}"
)

print(f"block bootstrap B={BLOCK_B} (21-day blocks)")
boot = block_bootstrap_nig(
    X50, B=BLOCK_B, block=BLOCK_LEN, seed=2026,
    cache_stem=f"blockboot/nig_d{PRIMARY_D}_seed{UNIVERSE_SEED}",
    w_star_ref=w_star,
)
cos_q = np.nanquantile(boot["cosine"], [0.05, 0.5, 0.95])
kap_q = np.nanquantile(boot["kappa"], [0.05, 0.5, 0.95])
q_q = np.nanquantile(boot["q_tilde"], [0.05, 0.5, 0.95])
print(
    f"cosine 5/50/95={cos_q}  κ 5/50/95={kap_q}  q̃ 5/50/95={q_q}"
)
pd.DataFrame(boot).to_csv(TABLES / "phase1_blockboot.csv", index=False)

c_grid = np.linspace(0.02, 2.0, 40)
emp_tail = empirical_left_tail(Y_hat, c_grid)
bnd = gaussian_thin_bound(c_grid, st50["q_tilde"])
fig, ax = plt.subplots(figsize=(FIG_W * 0.6, FIG_H * 0.5))
ax.plot(c_grid, emp_tail, color=COLORS["accent"], label="empirical")
ax.plot(c_grid, bnd, color=COLORS["muted"], ls="--", label=r"$2\Phi(-2\sqrt{c\tilde q})$")
ax.set_xlabel(r"$c$")
ax.set_ylabel(r"$P(\hat Y \leq -c)$")
ax.set_title("Prop. 2 left-tail bound vs tracker")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "01_left_tail.png", dpi=110)
plt.show()

# CVaR of ± tracker (descriptive; 5%)
Y_s = np.asarray(model50.subordinator().rvs(4000, seed=11))
cvar = CVaR(0.05)
port_plus = model50.project(jnp.asarray(w_star))
port_minus = model50.project(jnp.asarray(-w_star))
cvar_plus = float(cvar.value(port_plus, jnp.asarray(Y_s)))
cvar_minus = float(cvar.value(port_minus, jnp.asarray(Y_s)))
print(f"CVaR_5% long tracker={cvar_plus:.5g}  short tracker={cvar_minus:.5g}")

anatomy_meta = dict(
    gross=anat["gross"], net=anat["net"], long_share=anat["long_share"],
    n_long=anat["n_long"], n_short=anat["n_short"], herfindahl=anat["herfindahl"],
    corr_yhat_m=pearson(Y_hat, m50), corr_p_m=pearson(P_star, m50),
    corr_w_beta=pearson(ug, beta_i),
    mean_P=float(P_star.mean()), location=loc, mean_Yhat=float(np.nanmean(Y_hat)),
    e=st50["e"], g=split["g"], delta_l2=split["delta_l2"],
    cosine_boot05=float(cos_q[0]), kappa_boot=list(map(float, kap_q)),
    cvar_plus=cvar_plus, cvar_minus=cvar_minus,
    top=anat["top"],
)
(TABLES / "phase1_anatomy.json").write_text(json.dumps(anatomy_meta, indent=2, default=str))

# %% [markdown]
# ## 7. Phase 1 verdict
#
# Numbers: `_cache/tables/phase1_*.csv`. Figures: `_cache/figures/01_*.png`.

# %%
print("SNR (NIG):")
print(snr_tbl[snr_tbl["family"] == "nig"][cols].to_string(index=False))
print("\nmarket univariate:")
print(snr_tbl[snr_tbl["family"].str.startswith("nig_ew")][cols].to_string(index=False))
print("\nskew max:")
print(skew_max_tbl.to_string(index=False))
print("\nfigures:", sorted(p.name for p in FIG.glob("01_*.png")))
