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
# # Phase 3 — online EM dynamic tracker
#
# Warm start: batch NIG on days 1–504. Then Cappé–Moulines EWMA on the
# remainder. Phase 1: the static linear tracker is not a vol clock
# (corr with RV $=0.07$); $q_\perp$ is. This notebook scores both, plus
# parameter-path stability (H4).
#
# Plan §8, adjusted after Phase 1.

# %%
from __future__ import annotations

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
    cache_dir,
    cosine,
    dump_nig,
    ewma_neff,
    ewma_smooth,
    figure_dir,
    fit_nig,
    load_nig,
    load_or_compute,
    load_or_fit_generator,
    load_sp500,
    nested_universe,
    nig_fast_stats,
    online_em_path,
    pearson,
    rolling_sumsq,
    sample_central_moments,
    spearman,
)
from normix.utils.plotting import COLORS, FIG_H, FIG_W, set_theme

set_theme()
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
FIG = figure_dir()
TABLES = cache_dir() / "tables"
TABLES.mkdir(exist_ok=True)

WARM = 504
H_GRID = (21, 63, 126, 252, 504)
HS_STATE = (1, 5, 21)
ONLINE_TAU = tuple((h, 0.0) for h in H_GRID) + ((21, 0.1),)

print("cache", cache_dir())

# %%
panel = load_sp500()
universe = nested_universe(list(panel.columns), SIZES, PRIMARY_SEED)
tickers50 = universe[PRIMARY_D]
tickers20 = tickers50[:20]
X50 = panel[tickers50].to_numpy(dtype=np.float64)
X20 = panel[tickers20].to_numpy(dtype=np.float64)
dates = panel.index
m50 = X50.mean(axis=1)
X_run = X50[WARM:]
dates_run = dates[WARM:]
m_run = m50[WARM:]
n_run = len(X_run)
print(f"warm {WARM} days through {dates[WARM-1].date()}; online {n_run} days {dates_run[0].date()} → {dates_run[-1].date()}")

warm_path = cache_dir() / "fits" / "nig_d50_seed0_warm504.npz"
if warm_path.exists():
    model_warm = load_nig(warm_path)
else:
    result = fit_nig(jnp.asarray(X50[:WARM]), verbose=1)
    dump_nig(warm_path, result.model, n_iter=int(result.n_iter), converged=bool(result.converged))
    model_warm = result.model
st_warm = nig_fast_stats(model_warm)
print(f"warm q̃={st_warm['q_tilde']:.4g}  κ={st_warm['kappa']:.4g}")

static, _ = load_or_fit_generator(panel, tickers50)
st_static = nig_fast_stats(static)
print(f"static full-sample q̃={st_static['q_tilde']:.4g}  κ={st_static['kappa']:.4g}")

rv21 = rolling_sumsq(m_run, 21)
disp = ((X_run - X_run.mean(axis=1, keepdims=True)) / X50.std(axis=0, ddof=1)[:PRIMARY_D]) ** 2
disp = disp.mean(axis=1)

# %% [markdown]
# ## 1. EWMA grid at $d=50$, plus $1/t$ and a $d=20$ unshrunk check

# %%
runs = {}
t0 = time.perf_counter()
for h, tau in ONLINE_TAU:
    key = f"h{h}_tau{tau:g}"

    def _fn(h=h, tau=tau):
        return online_em_path(model_warm, jnp.asarray(X_run), half_life=h, tau=tau)

    runs[key] = load_or_compute(f"online_real/{key}.npz", _fn)
    print(f"  {key}  n_eff={ewma_neff(h):.0f}  ({time.perf_counter()-t0:.1f}s)")

def _sw():
    return online_em_path(
        model_warm, jnp.asarray(X_run), sample_weighted=True, n0=WARM,
    )
runs["sample_weighted"] = load_or_compute("online_real/sample_weighted.npz", _sw)
print(f"  sample_weighted  ({time.perf_counter()-t0:.1f}s)")

# d=20 unshrunk, short and long h
warm20_path = cache_dir() / "fits" / "nig_d20_seed0_warm504.npz"
if warm20_path.exists():
    model_warm20 = load_nig(warm20_path)
else:
    result = fit_nig(jnp.asarray(X20[:WARM]))
    dump_nig(warm20_path, result.model, n_iter=int(result.n_iter))
    model_warm20 = result.model
runs20 = {}
for h in (21, 252):
    key = f"d20_h{h}"

    def _fn(h=h):
        return online_em_path(model_warm20, jnp.asarray(X20[WARM:]), half_life=h, tau=0.0)

    runs20[key] = load_or_compute(f"online_real/{key}.npz", _fn)
    print(f"  {key}  ({time.perf_counter()-t0:.1f}s)")

# %%
frontier_rows = []
palette = [COLORS["accent"], COLORS["green"], COLORS["violet"], COLORS["umber"], COLORS["teal"], COLORS["brick"]]
fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H * 1.2), sharex=True)
for (h, tau), color in zip(ONLINE_TAU, palette):
    key = f"h{h}_tau{tau:g}"
    run = runs[key]
    label = rf"$h={h}$" + (rf", $\tau={tau}$" if tau else "")
    axes[0].plot(dates_run, run["kappa"], color=color, lw=0.8, label=label)
    cos21 = np.array([
        cosine(run["inv_sigma_gamma"][t], run["inv_sigma_gamma"][t - 21])
        if t >= 21 else np.nan
        for t in range(n_run)
    ])
    axes[1].plot(dates_run, cos21, color=color, lw=0.8, label=label)
    axes[2].plot(dates_run, run["q_perp_filt"], color=color, lw=0.7, alpha=0.85, label=label)

    row = dict(
        h=h, tau=tau, n_eff=ewma_neff(h),
        mean_kappa=float(np.nanmean(run["kappa"])),
        mean_q=float(np.nanmean(run["q_tilde"])),
        mean_e=float(np.nanmean(run["e"])),
        mean_turnover=float(np.mean(run["turnover"][1:])),
        mean_cos_lag1=float(np.nanmean(run["cos_lag1"][1:])),
        mean_cos_lag21=float(np.nanmean(cos21)),
        corr_yhat_rv=pearson(run["Y_hat_filt"], rv21),
        corr_qperp_rv=pearson(run["q_perp_filt"], rv21),
        corr_yhat_disp=pearson(run["Y_hat_filt"], disp),
        corr_qperp_disp=pearson(run["q_perp_filt"], disp),
        skew_P=sample_central_moments(run["P_filt"])["skew"],
        mean_P=float(np.nanmean(run["P_filt"])),
        std_P=float(np.nanstd(run["P_filt"])),
    )
    for hs in HS_STATE:
        row[f"corr_yhat_s{hs}"] = pearson(ewma_smooth(run["Y_hat_filt"], hs), rv21)
        row[f"corr_qperp_s{hs}"] = pearson(ewma_smooth(run["q_perp_filt"], hs), rv21)
    row["corr_rv_ewma"] = pearson(ewma_smooth(m_run ** 2, h), rv21)
    frontier_rows.append(row)

axes[2].plot(dates_run, rv21 / np.nanmean(rv21) * np.nanmean(runs["h252_tau0"]["q_perp_filt"]),
             color=COLORS["muted"], lw=0.6, alpha=0.7, label="21d RV (rescaled)")
for ax in axes:
    ax.axvline(pd.Timestamp("2018-02-05"), color=COLORS["rule"], lw=0.8)
    ax.axvline(pd.Timestamp("2020-03-16"), color=COLORS["rule"], lw=0.8)
    ax.axvline(pd.Timestamp("2022-03-01"), color=COLORS["rule"], lw=0.8)
axes[0].axhline(st_static["kappa"], color=COLORS["ink"], ls="--", lw=0.8, label="static κ")
axes[0].set_ylabel(r"$\kappa_t$")
axes[1].set_ylabel(r"$\cos\angle(w^\star_t, w^\star_{t-21})$")
axes[2].set_ylabel(r"$q_\perp^{\mathrm{filt}}$")
axes[0].legend(frameon=False, fontsize=7, ncol=3)
axes[0].set_title(r"online EM on S&P $d=50$: $\kappa_t$, direction, quadratic channel")
fig.tight_layout()
fig.savefig(FIG / "03_online_paths.png", dpi=110)
plt.show()

sw = runs["sample_weighted"]
frontier_rows.append(dict(
    h=np.inf, tau=0.0, n_eff=float(WARM + n_run),
    mean_kappa=float(np.nanmean(sw["kappa"])),
    mean_q=float(np.nanmean(sw["q_tilde"])),
    mean_e=float(np.nanmean(sw["e"])),
    mean_turnover=float(np.mean(sw["turnover"][1:])),
    mean_cos_lag1=float(np.nanmean(sw["cos_lag1"][1:])),
    mean_cos_lag21=float(np.nanmean([
        cosine(sw["inv_sigma_gamma"][t], sw["inv_sigma_gamma"][t - 21])
        if t >= 21 else np.nan for t in range(n_run)
    ])),
    corr_yhat_rv=pearson(sw["Y_hat_filt"], rv21),
    corr_qperp_rv=pearson(sw["q_perp_filt"], rv21),
    corr_yhat_disp=pearson(sw["Y_hat_filt"], disp),
    corr_qperp_disp=pearson(sw["q_perp_filt"], disp),
    skew_P=sample_central_moments(sw["P_filt"])["skew"],
    mean_P=float(np.nanmean(sw["P_filt"])),
    std_P=float(np.nanstd(sw["P_filt"])),
    corr_yhat_s1=pearson(sw["Y_hat_filt"], rv21),
    corr_qperp_s1=pearson(sw["q_perp_filt"], rv21),
    corr_yhat_s5=pearson(ewma_smooth(sw["Y_hat_filt"], 5), rv21),
    corr_qperp_s5=pearson(ewma_smooth(sw["q_perp_filt"], 5), rv21),
    corr_yhat_s21=pearson(ewma_smooth(sw["Y_hat_filt"], 21), rv21),
    corr_qperp_s21=pearson(ewma_smooth(sw["q_perp_filt"], 21), rv21),
    corr_rv_ewma=np.nan,
))

frontier = pd.DataFrame(frontier_rows)
print(frontier.to_string(index=False))
frontier.to_csv(TABLES / "phase3_frontier.csv", index=False)

print(
    "sample-weighted terminal vs static: "
    f"κ {sw['kappa'][-1]:.4g} vs {st_static['kappa']:.4g}  "
    f"q̃ {sw['q_tilde'][-1]:.4g} vs {st_static['q_tilde']:.4g}"
)

# %%
# d=20 comparison
print("d=20 unshrunk:")
for key, run in runs20.items():
    print(
        f"  {key}  mean κ={np.nanmean(run['kappa']):.3g}  "
        f"corr(Ŷ, RV)={pearson(run['Y_hat_filt'], rv21):.3f}  "
        f"corr(q⊥, RV)={pearson(run['q_perp_filt'], rv21):.3f}  "
        f"turnover={np.mean(run['turnover'][1:]):.3g}"
    )

# COVID zoom of q_perp vs RV
mask = (dates_run >= "2020-02-01") & (dates_run <= "2020-05-31")
fig, ax = plt.subplots(figsize=(FIG_W * 0.8, FIG_H * 0.5))
ax.plot(dates_run[mask], runs["h21_tau0"]["q_perp_filt"][mask], color=COLORS["accent"], lw=1.0, label=r"$h=21$ $q_\perp$")
ax.plot(dates_run[mask], runs["h252_tau0"]["q_perp_filt"][mask], color=COLORS["green"], lw=1.0, label=r"$h=252$ $q_\perp$")
ax2 = ax.twinx()
ax2.plot(dates_run[mask], rv21[mask], color=COLORS["muted"], lw=0.9, label="21d RV")
ax.set_title("COVID window: quadratic channel vs RV")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "03_covid_qperp.png", dpi=110)
plt.show()

# %% [markdown]
# ## 2. Phase 3 verdict
#
# Static $\kappa=0.048$. Short $h$ inflating $\kappa_t$ is H4's noise
# diagnostic. Extraction: $\mathrm{corr}(q_\perp, \mathrm{RV})$ vs
# $\mathrm{corr}(\hat Y, \mathrm{RV})$ vs EWMA RV at the same $h$.

# %%
print("\nfigures:", sorted(p.name for p in FIG.glob("03_*.png")))
