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
# # Phase 0 — synthetic validation of subordinator tracking
#
# Theory: `dev-notes/research/subordinator_tracking_portfolio.md`.
# Plan: `dev-notes/research/subordinator_tracking_sp500_plan.md` §5.
#
# The latent clock $Y$ is observable here. Generator: NIG fitted to a
# $d=50$ S&P 500 subset, then $\gamma \mapsto c\gamma$ for
# $c \in \{0,1,3,10\}$. Checks:
#
# 1. Tracker / linear-Bayes / posterior MSE vs $e/\tilde q$, $v/(1+\kappa)$,
#    and the bound $\le v/(1+\kappa)$.
# 2. $\operatorname{corr}(\hat Y, Y) = \sqrt{\kappa/(1+\kappa)}$.
# 3. Estimation noise of $\hat{\tilde q}$ across $(d,T)$, including the
#    $c=0$ floor.
# 4. Direction recovery $\cos\angle(\widehat{\Sigma^{-1}\gamma},\,\Sigma^{-1}\gamma)$.
# 5. Online-EM rehearsal on a path with a slow $\gamma$-rotation and a
#    subordinator-scale jump.
#
# Package code is not modified. Helpers live in `lib.py`.

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
    NIG_FIT_KW,
    cache_dir,
    cosine,
    data_hygiene,
    ewma_neff,
    ewma_smooth,
    figure_dir,
    load_or_compute,
    load_or_fit_generator,
    load_sp500,
    make_tv_path,
    mse_laws_trial,
    nested_universe,
    online_em_path,
    pearson,
    refit_trial,
    restrict_nig,
    scale_gamma,
    tracker_stats,
    y_estimators,
)
from normix.utils.plotting import COLORS, FIG_H, FIG_W, set_theme

set_theme()
np.set_printoptions(precision=4, suppress=True)
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
FIG = figure_dir()

# %%
# Study knobs. Refits are cached under _cache/refit/; re-running is cheap.
C_GRID = (0.0, 1.0, 3.0, 10.0)
D_GRID = (10, 25, 50)
T_GRID = (500, 2552)
N_REP_MSE = 20
N_REP_REFIT = 20
T_DATA = 2552
D_PRIMARY = 50
UNIVERSE_SEED = 0
ONLINE_TAU = ((21, 0.0), (21, 0.1), (63, 0.0), (252, 0.0))
THETA_MAX = np.pi / 4
JUMP_SCALE = 3.0

print("fit kwargs", NIG_FIT_KW)
print("cache", cache_dir())

# %% [markdown]
# ## 1. Generator: NIG on a nested $d=50$ S&P 500 subset
#
# Cold-start EM with `regularization='a_eq_b'` (gauge $E[Y]=1$). The package
# default `tol=1e-3` stops after one iteration on this panel — $\gamma$ is
# small enough that $\mathrm{rms}(\Delta\gamma)/(1+\mathrm{rms}(\gamma))$
# already meets that threshold — so the study uses `tol=1e-5`.

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

sizes = [5, 10, 25, 50, 100, 200, 468]
universe = nested_universe(list(panel.columns), sizes, UNIVERSE_SEED)
tickers50 = universe[D_PRIMARY]
print(f"d={D_PRIMARY} seed={UNIVERSE_SEED}: {tickers50[:8]} ...")

gen, gen_meta = load_or_fit_generator(panel, tickers50)
print("generator meta:", {k: gen_meta[k] for k in gen_meta if k != "tickers"})
st0 = tracker_stats(gen)
print(
    f"q̃={st0['q_tilde']:.4g}  e={st0['e']:.4g}  v={st0['v']:.4g}  "
    f"κ={st0['kappa']:.4g}  κ_lev={st0['kappa_lev']:.4g}  "
    f"t†={st0['t_dagger']:.4g}  1/q̃={1/st0['q_tilde']:.4g}"
)
print(
    f"IG closed form: e=μ={float(gen.mu_ig):.4g}, "
    f"v=μ³/λ={float(gen.mu_ig)**3/float(gen.lam):.4g}, "
    f"e μ₃ / v² = {st0['e']*st0['mu3']/st0['v']**2:.4g}  (NIG: 3)"
)
assert st0["t_dagger"] <= 1.0 / st0["q_tilde"]

# %% [markdown]
# ## 2. $\gamma$-scale sweep
#
# $\tilde q \mapsto c^2\tilde q$, $\kappa \mapsto c^2\kappa$. The fitted
# $c=1$ point is already $\kappa\sim 5\cdot 10^{-2}$; $c=10$ is the
# $\kappa\gg 1$ (near-dominance) regime.

# %%
scale_rows = []
models_c = {}
for c in C_GRID:
    m = scale_gamma(gen, c)
    models_c[c] = m
    st = tracker_stats(m)
    scale_rows.append(dict(
        c=c, q_tilde=st["q_tilde"], kappa=st["kappa"], kappa_lev=st["kappa_lev"],
        corr_theory=st["corr_theory"],
        mse_hat_over_v=(1.0 / st["kappa"] if st["kappa"] else np.inf),
        t_dagger=st["t_dagger"],
    ))
scale_tbl = pd.DataFrame(scale_rows)
print(scale_tbl.to_string(index=False))

# %% [markdown]
# ## 3. MSE and correlation laws (true parameters)
#
# $T=2552$, $d=50$, $R=20$ i.i.d. replications. Acceptance: empirical MSE
# within 5% of the closed form (tracker and linear Bayes). Posterior mean
# has only the bound $\mathrm{MSE}\le v/(1+\kappa)$.

# %%
def _mse_grid():
    rows = []
    t0 = time.perf_counter()
    for c in C_GRID:
        model = models_c[c]
        for r in range(N_REP_MSE):
            trial = mse_laws_trial(model, T_DATA, seed=10_000 + 100 * int(c) + r)
            trial["c"] = c
            trial["rep"] = r
            rows.append(trial)
        print(f"  mse c={c}  ({time.perf_counter()-t0:.1f}s)")
    keys = [k for k in rows[0] if k not in ("c", "rep")]
    out = {
        "c": np.array([row["c"] for row in rows]),
        "rep": np.array([row["rep"] for row in rows]),
    }
    for k in keys:
        out[k] = np.array([row[k] for row in rows])
    return out

mse_raw = load_or_compute("mse_laws_d50.npz", _mse_grid)
mse_df = pd.DataFrame(mse_raw)

def _agg_mse(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, sub in df.groupby("c"):
        st = tracker_stats(models_c[float(c)])
        row = dict(c=float(c), kappa=st["kappa"], q_tilde=st["q_tilde"], v=st["v"])
        for name, theory in (
            ("mse_hat", "mse_hat_theory"),
            ("mse_lin", "mse_lin_theory"),
            ("corr_hat", "corr_hat_theory"),
        ):
            emp = float(sub[name].mean())
            th = float(sub[theory].mean()) if theory in sub else np.nan
            row[name] = emp
            row[theory] = th
            row[f"{name}_relerr"] = (
                abs(emp - th) / abs(th) if np.isfinite(th) and abs(th) > 0 else np.nan
            )
        row["mse_post"] = float(sub["mse_post"].mean())
        row["mse_post_bound"] = float(sub["mse_post_bound"].mean())
        row["mse_post_over_bound"] = row["mse_post"] / row["mse_post_bound"]
        row["corr_post"] = float(sub["corr_post"].mean())
        row["corr_lin"] = float(sub["corr_lin"].mean())
        row["pass_5pct"] = (
            (not np.isfinite(row["mse_hat_relerr"]) or row["mse_hat_relerr"] < 0.05)
            and row["mse_lin_relerr"] < 0.05
            and (not np.isfinite(row["corr_hat_relerr"]) or row["corr_hat_relerr"] < 0.05)
        )
        rows.append(row)
    return pd.DataFrame(rows)

mse_sum = _agg_mse(mse_df)
print(mse_sum.to_string(index=False))
(cache_dir() / "tables").mkdir(exist_ok=True)
mse_sum.to_csv(cache_dir() / "tables" / "mse_laws.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.55))
k = mse_sum["kappa"].to_numpy()
vv = mse_sum["v"].to_numpy()
k_grid = np.logspace(-3, 1.5, 200)
axes[0].plot(k_grid, 1.0 / k_grid, color=COLORS["muted"], lw=1, label=r"tracker $1/\kappa$")
axes[0].plot(
    k_grid, 1.0 / (1.0 + k_grid), color=COLORS["muted"], lw=1, ls="--",
    label=r"Bayes $1/(1+\kappa)$",
)
axes[0].scatter(k, mse_sum["mse_hat"] / vv, color=COLORS["accent"], s=40, zorder=3, label="tracker")
axes[0].scatter(
    k, mse_sum["mse_lin"] / vv, color=COLORS["green"], s=40, zorder=3, marker="s",
    label="linear Bayes",
)
axes[0].scatter(
    k, mse_sum["mse_post"] / vv, color=COLORS["umber"], s=40, zorder=3, marker="D",
    label="posterior mean",
)
axes[0].set_xscale("symlog", linthresh=1e-3)
axes[0].set_yscale("log")
axes[0].set_xlabel(r"$\kappa$")
axes[0].set_ylabel(r"MSE / $\mathrm{Var}(Y)$")
axes[0].legend(frameon=False, fontsize=9)
axes[0].set_title("MSE laws")

axes[1].plot(
    k_grid, np.sqrt(k_grid / (1.0 + k_grid)), color=COLORS["muted"], lw=1,
    label=r"$\sqrt{\kappa/(1+\kappa)}$",
)
axes[1].scatter(k, mse_sum["corr_hat"], color=COLORS["accent"], s=40, zorder=3, label="tracker")
axes[1].scatter(
    k, mse_sum["corr_post"], color=COLORS["umber"], s=40, zorder=3, marker="D",
    label="posterior mean",
)
axes[1].set_xscale("symlog", linthresh=1e-3)
axes[1].set_xlabel(r"$\kappa$")
axes[1].set_ylabel(r"$\mathrm{corr}(\cdot, Y)$")
axes[1].legend(frameon=False, fontsize=9)
axes[1].set_title("correlation with true $Y$")
fig.tight_layout()
fig.savefig(FIG / "00_mse_corr_laws.png", dpi=110)
plt.show()

# %% [markdown]
# ## 4. Estimation noise of $\hat{\tilde q}$ and direction recovery
#
# Cold-start EM on samples from the scaled generator (and its leading-$d$
# submodels). The $c=0$ column is the null floor $\hat{\tilde q}_0(d,T)$
# used later as a Phase 1/2 yardstick.

# %%
def _refit_grid():
    rec = {k: [] for k in (
        "d", "T", "c", "rep", "q_tilde_true", "q_tilde_hat", "kappa_true",
        "kappa_hat", "cosine", "corr_hat_trueY", "corr_post_trueY",
        "n_iter", "converged", "elapsed",
    )}
    t0 = time.perf_counter()
    n_done = 0
    n_tot = len(D_GRID) * len(T_GRID) * len(C_GRID) * N_REP_REFIT
    for d in D_GRID:
        for c in C_GRID:
            true = restrict_nig(scale_gamma(gen, c), d)
            for T in T_GRID:
                for r in range(N_REP_REFIT):
                    name = f"refit/d{d}_T{T}_c{c:g}_r{r}.npz"

                    def _fn(true=true, T=T, r=r, d=d, c=c):
                        tr = refit_trial(true, T, seed=20_000 + 1000 * d + 100 * int(c) + r)
                        return {k: np.asarray(v) for k, v in tr.items()}

                    row = load_or_compute(name, _fn)
                    rec["d"].append(d)
                    rec["T"].append(T)
                    rec["c"].append(c)
                    rec["rep"].append(r)
                    for k, v in row.items():
                        rec[k].append(np.asarray(v).reshape(()))
                    n_done += 1
                    if n_done % 20 == 0:
                        jax.clear_caches()
                    if n_done % 20 == 0 or n_done == n_tot:
                        print(f"  refit {n_done}/{n_tot}  ({time.perf_counter()-t0:.1f}s)")
    return {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}

refit_raw = load_or_compute("refit_grid.npz", _refit_grid)
refit_df = pd.DataFrame(refit_raw)
print(
    refit_df.groupby(["d", "T", "c"])[["q_tilde_hat", "kappa_hat", "cosine", "n_iter"]]
    .agg(["mean", "std"])
    .round(4)
    .to_string()
)
refit_df.to_csv(cache_dir() / "tables" / "refit_grid.csv", index=False)

# %%
fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H * 0.55))
sub = refit_df[refit_df["T"] == T_DATA]
palette = [COLORS["muted"], COLORS["accent"], COLORS["green"], COLORS["umber"]]
for c, color in zip(C_GRID, palette):
    g = sub[sub["c"] == c].groupby("d")["q_tilde_hat"]
    dvals = np.array(list(g.mean().index))
    mean = g.mean().to_numpy()
    se = g.std().to_numpy() / np.sqrt(g.count().to_numpy())
    axes[0].errorbar(
        dvals, mean, yerr=1.96 * se, color=color, marker="o",
        label=rf"$c={c:g}$", capsize=3,
    )
    true_q = [
        tracker_stats(restrict_nig(scale_gamma(gen, c), int(dd)))["q_tilde"]
        for dd in dvals
    ]
    axes[0].plot(dvals, true_q, color=color, ls="--", lw=1)
axes[0].set_xlabel(r"$d$")
axes[0].set_ylabel(r"$\hat{\tilde q}$")
axes[0].set_title(rf"$T={T_DATA}$: $\hat{{\tilde q}}$ vs $d$ (dashed = truth)")
axes[0].legend(frameon=False, fontsize=9)

for T, marker in ((500, "s"), (2552, "o")):
    sT = refit_df[(refit_df["d"] == D_PRIMARY) & (refit_df["T"] == T) & (refit_df["c"] > 0)]
    g = sT.groupby("kappa_true")["cosine"]
    axes[1].errorbar(
        g.mean().index, g.mean(), yerr=g.std() / np.sqrt(g.count()),
        marker=marker, capsize=3, label=rf"$T={T}$",
    )
axes[1].set_xlabel(r"true $\kappa$")
axes[1].set_ylabel(r"$\cos\angle(\widehat{\Sigma^{-1}\gamma},\,\Sigma^{-1}\gamma)$")
axes[1].set_title(rf"$d={D_PRIMARY}$: direction recovery")
axes[1].legend(frameon=False, fontsize=9)
axes[1].set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(FIG / "00_qtilde_direction.png", dpi=110)
plt.show()

floor = (
    refit_df[(refit_df["c"] == 0) & (refit_df["T"] == T_DATA)]
    .groupby("d")["q_tilde_hat"]
    .agg(mean="mean", p95=lambda s: float(np.quantile(s, 0.95)), std="std")
)
print("c=0 null floor of q̃_hat at T=2552:")
print(floor.to_string())
floor.to_csv(cache_dir() / "tables" / "qtilde_null_floor.csv")

# Log scale: linear axis hides the c=0 floor against c=10.
fig, ax = plt.subplots(figsize=(FIG_W * 0.55, FIG_H * 0.55))
for c, color in zip(C_GRID, palette):
    g = sub[sub["c"] == c].groupby("d")["q_tilde_hat"]
    dvals = np.array(list(g.mean().index))
    mean = g.mean().to_numpy()
    se = g.std().to_numpy() / np.sqrt(g.count().to_numpy())
    ax.errorbar(
        dvals, mean, yerr=1.96 * se, color=color, marker="o",
        label=rf"$c={c:g}$", capsize=3,
    )
    true_q = [
        tracker_stats(restrict_nig(scale_gamma(gen, c), int(dd)))["q_tilde"]
        for dd in dvals
    ]
    ax.plot(dvals, true_q, color=color, ls="--", lw=1)
ax.set_yscale("log")
ax.set_xlabel(r"$d$")
ax.set_ylabel(r"$\hat{\tilde q}$")
ax.set_title(rf"$T={T_DATA}$: $\hat{{\tilde q}}$ vs $d$ (log; dashed = truth)")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "00_qtilde_floor_zoom.png", dpi=110)
plt.show()

# %% [markdown]
# ## 5. Online EM rehearsal
#
# Synthetic path: $\gamma_t$ rotates by $\pi/4$ in the $\Sigma$-metric
# (so $\tilde q$ is constant) and the IG scale jumps by $3\times$ at
# $t=T/2$. Online EM is initialised at the *true* $t=0$ model (oracle
# start — this isolates tracking from identification; Phase 3 on real
# data will warm-start from a batch window instead).
#
# Two knobs: half-life $h$ of the $\eta$-EWMA, and optional shrinkage
# $\tau$ toward the $t=0$ $\eta_0$ (needed when $n_{\mathrm{eff}}\approx 2.9h$
# is not $\gg d$).

# %%
def _tv_path():
    return make_tv_path(
        gen, T_DATA, jump_at=T_DATA // 2, jump_scale=JUMP_SCALE,
        theta_max=THETA_MAX, seed=42,
    )

tv = load_or_compute("tv_path_d50.npz", _tv_path)
X_tv, Y_tv = tv["X"], tv["Y"]
print(
    f"TV path: T={len(Y_tv)}, jump_at={int(tv['jump_at'])}, "
    f"q̃={float(tv['q_tilde'][0]):.4g}, "
    f"κ before/after={tv['kappa_t'][0]:.4g}/{tv['kappa_t'][-1]:.4g}"
)

online_runs = {}
t0 = time.perf_counter()
for h, tau in ONLINE_TAU:
    key = f"h{h}_tau{tau:g}"

    def _fn(h=h, tau=tau):
        return online_em_path(gen, jnp.asarray(X_tv), half_life=h, tau=tau)

    online_runs[key] = load_or_compute(f"online/{key}.npz", _fn)
    print(f"  online {key}  n_eff={ewma_neff(h):.1f}  ({time.perf_counter()-t0:.1f}s)")

# %%
frontier_rows = []
true_invg = tv["inv_sigma_gamma_t"]
true_kappa = tv["kappa_t"]
true_q = float(tv["q_tilde"][0])
hs_smooth = (1, 5, 21)

fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H * 1.15), sharex=True)
t = np.arange(T_DATA)
axes[0].plot(t, Y_tv, color=COLORS["muted"], lw=0.6, alpha=0.7, label=r"true $Y$")
palette = [COLORS["accent"], COLORS["violet"], COLORS["green"], COLORS["umber"]]
for (h, tau), color in zip(ONLINE_TAU, palette):
    key = f"h{h}_tau{tau:g}"
    run = online_runs[key]
    label = rf"$h={h}$" + (rf", $\tau={tau}$" if tau else "")
    if tau == 0.0:
        axes[0].plot(t, run["Y_hat_filt"], color=color, lw=0.8, alpha=0.85, label=label)
    axes[1].plot(t, run["kappa"], color=color, lw=0.9, label=label)
    cos_t = np.array([
        cosine(run["inv_sigma_gamma"][i], true_invg[i]) for i in range(T_DATA)
    ])
    axes[2].plot(t, cos_t, color=color, lw=0.9, label=label)

    row = dict(
        h=h, tau=tau, n_eff=ewma_neff(h),
        corr_filt=pearson(run["Y_hat_filt"], Y_tv),
        corr_in=pearson(run["Y_hat_in"], Y_tv),
        mean_turnover=float(np.mean(run["turnover"][1:])),
        mean_q_over_true=float(np.mean(run["q_tilde"])) / true_q,
        mean_cos=float(np.nanmean(cos_t)),
        mean_kappa=float(np.mean(run["kappa"])),
        true_kappa_mean=float(np.mean(true_kappa)),
    )
    for hs in hs_smooth:
        row[f"corr_smooth_{hs}"] = pearson(ewma_smooth(run["Y_hat_filt"], hs), Y_tv)
    mkt = X_tv.mean(axis=1)
    row["corr_rv_ewma"] = pearson(ewma_smooth(mkt ** 2, h), Y_tv)
    frontier_rows.append(row)

axes[0].axvline(T_DATA // 2, color=COLORS["rule"], lw=1)
axes[1].plot(t, true_kappa, color=COLORS["ink"], lw=1.0, ls="--", label=r"true $\kappa_t$")
axes[1].axvline(T_DATA // 2, color=COLORS["rule"], lw=1)
axes[2].axvline(T_DATA // 2, color=COLORS["rule"], lw=1)
axes[0].set_ylabel(r"$Y$, $\hat Y^{\mathrm{filt}}$")
axes[1].set_ylabel(r"$\kappa_t$")
axes[2].set_ylabel(r"$\cos\angle(\hat w^\star_t, w^\star_t)$")
axes[2].set_xlabel("day")
axes[0].legend(frameon=False, fontsize=8, ncol=3)
axes[1].legend(frameon=False, fontsize=8, ncol=3)
axes[0].set_title("online EM on a rotating / jumping synthetic path")
fig.tight_layout()
fig.savefig(FIG / "00_online_paths.png", dpi=110)
plt.show()

frontier = pd.DataFrame(frontier_rows)
print(frontier.to_string(index=False))
frontier.to_csv(cache_dir() / "tables" / "online_frontier.csv", index=False)

est0 = y_estimators(gen, jnp.asarray(X_tv))
print(
    f"static true-model tracker corr={pearson(np.asarray(est0['Y_hat']), Y_tv):.3f}  "
    f"posterior corr={pearson(np.asarray(est0['Y_post']), Y_tv):.3f}  "
    f"(TV path; γ_t and e_t are moving, so this is not the §3 law)"
)

# %% [markdown]
# ## 6. Phase 0 verdict
#
# Numbers are written to `_cache/tables/` and figures to `_cache/figures/`.
# The findings note `dev-notes/research/subordinator_tracking_sp500_results.md`
# copies the headline tables.

# %%
print("MSE 5% acceptance:")
print(mse_sum[[
    "c", "kappa", "mse_hat_relerr", "mse_lin_relerr", "corr_hat_relerr",
    "mse_post_over_bound", "pass_5pct",
]].to_string(index=False))
print("\nNull floor q̃_hat (c=0, T=2552):")
print(floor.to_string())
print("\nOnline frontier:")
print(frontier.to_string(index=False))
print("\nfigures:", sorted(p.name for p in FIG.glob("00_*.png")))
