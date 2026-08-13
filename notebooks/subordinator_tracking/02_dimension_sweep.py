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
# # Phase 2 — does $\tilde q_d$ saturate? (H2)
#
# Nested universes, 5 seeds. Overlay the Phase 1 sign-flip 95% floor.
# Attribution: eigen-decomposition of $\Sigma$ and the split
# $\gamma = g\mathbf 1 + \delta$.
#
# Plan §7; Phase 1 already showed seed-0 $\hat{\tilde q}$ below the null
# at $d\ge 100$. This sweep asks whether that is seed-specific.

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if not (HERE / "lib.py").exists():
    HERE = Path.cwd() / "notebooks" / "subordinator_tracking"
sys.path.insert(0, str(HERE))

from lib import (  # noqa: E402
    SIZES,
    cache_dir,
    figure_dir,
    gamma_market_split,
    load_or_compute,
    load_or_fit_generator,
    load_or_fit_nig_named,
    load_sp500,
    nested_universe,
    nig_fast_stats,
    qtilde_eigen_attribution,
    tracker_stats,
)
from normix.utils.plotting import COLORS, FIG_H, FIG_W, set_theme

set_theme()
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
FIG = figure_dir()
TABLES = cache_dir() / "tables"
TABLES.mkdir(exist_ok=True)

SEEDS = (0, 1, 2, 3, 4)
D_GRID = list(SIZES)
PHASE0_FLOOR = {  # synthetic c=0, T=2552, R=20 (Phase 0)
    10: 0.0206,
    25: 0.0395,
    50: 0.0775,
}

print("cache", cache_dir())

# %%
panel = load_sp500()
tickers_all = list(panel.columns)
dates = panel.index

# Phase 1 sign-flip 95% (seed 0)
null95 = {}
for d in D_GRID:
    path = cache_dir() / f"signflip/nig_d{d}_seed0/summary.npz"
    if path.exists():
        with np.load(path) as z:
            null95[d] = float(np.quantile(z["q_tilde"], 0.95))
print("sign-flip 95% (seed 0):", null95)

# %% [markdown]
# ## 1. Nested NIG fits, 5 seeds

# %%
rows = []
t0 = time.perf_counter()
for seed in SEEDS:
    universe = nested_universe(tickers_all, D_GRID, seed)
    for d in D_GRID:
        tickers = universe[d]
        if d == 50 and seed == 0:
            model, meta = load_or_fit_generator(panel, tickers)
        else:
            model, meta = load_or_fit_nig_named(
                panel, tickers, f"nig_d{d}_seed{seed}.npz",
            )
        st = tracker_stats(model)
        split = gamma_market_split(np.asarray(model.gamma))
        attr = qtilde_eigen_attribution(model)
        n = attr["share"].size
        n_small = max(1, n // 10)
        # equal-weight univariate of this universe
        m = panel[tickers].to_numpy(dtype=np.float64).mean(axis=1)
        fake = pd.DataFrame({"ew": m}, index=dates)
        mkt, mkt_meta = load_or_fit_nig_named(
            fake, ["ew"], f"nig_ew_d{d}_seed{seed}.npz",
        )
        st_m = nig_fast_stats(mkt)
        Sigma = np.asarray(model.sigma())
        corr = Sigma / np.sqrt(np.outer(np.diag(Sigma), np.diag(Sigma)))
        rho = float((corr.sum() - n) / (n * (n - 1))) if n > 1 else 0.0
        sig2 = float(np.mean(np.diag(Sigma)))
        q_sat = (split["g"] ** 2) / (sig2 * rho) if rho > 1e-8 else np.nan
        rows.append(dict(
            seed=seed, d=d,
            q_tilde=st["q_tilde"], kappa=st["kappa"], kappa_lev=st["kappa_lev"],
            v=st["v"], e=st["e"],
            kappa_index=st_m["kappa"], q_tilde_index=st_m["q_tilde"],
            g=split["g"], delta_l2=split["delta_l2"],
            gamma_l2=float(np.asarray(model.gamma) @ np.asarray(model.gamma)),
            pc1_share=float(attr["share"][0]),
            top3_share=float(attr["share"][:3].sum()),
            small10_share=float(attr["share"][-n_small:].sum()),
            rho=rho, q_sat=q_sat,
            n_iter=meta.get("n_iter"),
            mkt_n_iter=mkt_meta.get("n_iter"),
        ))
        print(
            f"  seed={seed} d={d:3d}  q̃={st['q_tilde']:.4g}  "
            f"κ={st['kappa']:.4g}  κ_idx={st_m['kappa']:.4g}  "
            f"PC1={attr['share'][0]:.3f}  ({time.perf_counter()-t0:.1f}s)"
        )

sweep = pd.DataFrame(rows)
sweep.to_csv(TABLES / "phase2_sweep.csv", index=False)

# %%
agg = (
    sweep.groupby("d")
    .agg(
        q_mean=("q_tilde", "mean"), q_std=("q_tilde", "std"),
        k_mean=("kappa", "mean"), k_std=("kappa", "std"),
        kidx_mean=("kappa_index", "mean"), kidx_std=("kappa_index", "std"),
        pc1_mean=("pc1_share", "mean"),
        top3_mean=("top3_share", "mean"),
        small_mean=("small10_share", "mean"),
        g_mean=("g", "mean"),
        delta_mean=("delta_l2", "mean"),
        gamma_l2_mean=("gamma_l2", "mean"),
        rho_mean=("rho", "mean"),
        qsat_mean=("q_sat", "mean"),
    )
    .reset_index()
)
agg["q_null95"] = agg["d"].map(null95)
agg["q_phase0_floor"] = agg["d"].map(PHASE0_FLOOR)
print(agg.to_string(index=False))
agg.to_csv(TABLES / "phase2_sweep_agg.csv", index=False)

# %%
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.55))
dvals = agg["d"].to_numpy()
se = 1.96 * agg["q_std"].to_numpy() / np.sqrt(len(SEEDS))
axes[0].errorbar(
    dvals, agg["q_mean"], yerr=se, color=COLORS["accent"], marker="o",
    capsize=3, label=r"mean $\hat{\tilde q}$ (5 seeds)",
)
nd = np.array(sorted(null95))
axes[0].plot(
    nd, [null95[d] for d in nd], color=COLORS["muted"], ls="--", marker="x",
    label=r"sign-flip 95% (seed 0)",
)
p0d = np.array(sorted(PHASE0_FLOOR))
axes[0].plot(
    p0d, [PHASE0_FLOOR[d] for d in p0d], color=COLORS["umber"], ls=":",
    marker="s", label=r"Phase 0 $c=0$ 95%",
)
axes[0].set_xlabel(r"$d$")
axes[0].set_ylabel(r"$\hat{\tilde q}$")
axes[0].set_title(r"H2: $\hat{\tilde q}_d$ vs noise floor")
axes[0].legend(frameon=False, fontsize=8)

se_k = 1.96 * agg["k_std"].to_numpy() / np.sqrt(len(SEEDS))
axes[1].errorbar(
    dvals, agg["k_mean"], yerr=se_k, color=COLORS["green"], marker="o",
    capsize=3, label=r"panel $\hat\kappa$",
)
se_i = 1.96 * agg["kidx_std"].to_numpy() / np.sqrt(len(SEEDS))
axes[1].errorbar(
    dvals, agg["kidx_mean"], yerr=se_i, color=COLORS["violet"], marker="s",
    capsize=3, label=r"EW univariate $\hat\kappa_{\mathrm{index}}$",
)
axes[1].set_xlabel(r"$d$")
axes[1].set_ylabel(r"$\hat\kappa$")
axes[1].set_title("cross-section vs index ceiling")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "02_qtilde_sweep.png", dpi=110)
plt.show()

# %% [markdown]
# ## 2. Attribution: market PC vs small eigenvalues vs $\delta$

# %%
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.55))
axes[0].plot(dvals, agg["pc1_mean"], marker="o", color=COLORS["accent"], label="PC1")
axes[0].plot(dvals, agg["top3_mean"], marker="s", color=COLORS["green"], label="top 3 PCs")
axes[0].plot(
    dvals, agg["small_mean"], marker="D", color=COLORS["umber"],
    label=r"smallest 10% of $\lambda$",
)
axes[0].set_xlabel(r"$d$")
axes[0].set_ylabel(r"share of $\tilde q$")
axes[0].set_ylim(0, 1.05)
axes[0].set_title(r"eigen-attribution of $\tilde q$")
axes[0].legend(frameon=False, fontsize=8)

frac_delta = agg["delta_mean"] / (agg["delta_mean"] + (agg["d"] * agg["g_mean"] ** 2))
axes[1].plot(dvals, frac_delta, marker="o", color=COLORS["accent"], label=r"$\lVert\delta\rVert^2 / \lVert\gamma\rVert^2$")
axes[1].plot(dvals, agg["qsat_mean"], marker="s", color=COLORS["muted"], label=r"$g^2/(\bar\sigma^2\rho)$")
axes[1].set_xlabel(r"$d$")
axes[1].set_ylabel("share / bound")
axes[1].set_title(r"equicorrelation split $\gamma=g\mathbf{1}+\delta$")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "02_attribution.png", dpi=110)
plt.show()

print("mean ||δ||² / ||γ||² by d:")
print(pd.DataFrame(dict(d=dvals, frac_delta=frac_delta, q_sat=agg["qsat_mean"], rho=agg["rho_mean"])).to_string(index=False))

# %% [markdown]
# ## 3. Phase 2 verdict
#
# If mean $\hat{\tilde q}_d$ tracks the sign-flip floor, H2's
# "saturation of a real market-skewness signal" is the wrong picture:
# there is no signal to saturate. Attribution then says *where* the
# estimation noise lives (small $\lambda$ vs PC1).

# %%
print(agg[[
    "d", "q_mean", "q_std", "q_null95", "k_mean", "kidx_mean",
    "pc1_mean", "small_mean",
]].to_string(index=False))
print("\nfigures:", sorted(p.name for p in FIG.glob("02_*.png")))
