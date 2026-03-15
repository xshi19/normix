"""
Compare pure-JAX log_kv against scipy reference across all regimes.

Generates three figures:
  1. bessel_accuracy_overview.png — relative error heatmap + 1D profiles
  2. bessel_regime_map.png       — which algorithm handles each (v, z) point
  3. bessel_value_profiles.png   — log K_v(z) curves for selected v values

Usage:
    uv run python scripts/compare_bessel_implementations.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import kve as scipy_kve

jax.config.update("jax_enable_x64", True)

from normix._bessel import log_kv


def scipy_log_kv(v, z):
    """Scipy reference: log K_v(z) = log(kve(|v|, z)) - z."""
    v, z = np.asarray(v, dtype=np.float64), np.asarray(z, dtype=np.float64)
    z_safe = np.maximum(z, np.finfo(np.float64).tiny)
    val = np.log(scipy_kve(np.abs(v), z_safe)) - z_safe
    return val


def _regime_label(v_abs, z):
    """Return integer code for which regime handles (v, z)."""
    if z > max(25.0, v_abs**2 / 4.0):
        return 0  # Hankel
    if v_abs > 25.0:
        return 1  # Olver
    if z < 1e-6 and v_abs > 0.5:
        return 2  # Small-z
    return 3      # Quadrature


def fig_accuracy_overview():
    """2D relative error heatmap + 1D profiles for selected v values."""
    vs = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    zs = np.logspace(-6, 3, 200)

    errors = np.zeros((len(vs), len(zs)))
    for i, v in enumerate(vs):
        jax_vals = np.asarray(log_kv(jnp.full(len(zs), v), jnp.array(zs)))
        ref_vals = scipy_log_kv(v, zs)
        denom = np.maximum(np.abs(ref_vals), 1e-15)
        err = np.abs(jax_vals - ref_vals) / denom
        err[~np.isfinite(err)] = 0.0
        errors[i] = err

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: heatmap
    ax = axes[0]
    im = ax.pcolormesh(zs, vs, np.maximum(errors, 1e-17),
                       norm=LogNorm(vmin=1e-16, vmax=1e-2),
                       cmap="RdYlGn_r", shading="nearest")
    ax.set_xscale("log")
    ax.set_xlabel("z")
    ax.set_ylabel("v")
    ax.set_title("Relative error: pure-JAX log_kv vs scipy")
    fig.colorbar(im, ax=ax, label="Relative error")

    # Right: 1D profiles for selected v
    ax = axes[1]
    for v in [0.5, 2.0, 10.0, 50.0, 100.0]:
        jax_vals = np.asarray(log_kv(jnp.full(len(zs), v), jnp.array(zs)))
        ref_vals = scipy_log_kv(v, zs)
        denom = np.maximum(np.abs(ref_vals), 1e-15)
        rel_err = np.abs(jax_vals - ref_vals) / denom
        rel_err[~np.isfinite(rel_err)] = 0.0
        ax.semilogy(zs, np.maximum(rel_err, 1e-17), label=f"v={v}")

    ax.axhline(1e-9, color="k", ls="--", lw=0.8, alpha=0.5, label="1e-9 target")
    ax.set_xscale("log")
    ax.set_xlabel("z")
    ax.set_ylabel("Relative error")
    ax.set_title("Accuracy profiles by v")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(1e-17, 1e-2)

    plt.tight_layout()
    plt.savefig("scripts/bessel_accuracy_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved scripts/bessel_accuracy_overview.png")

    max_err = errors.max()
    print(f"  Max relative error across all (v, z): {max_err:.2e}")
    for i, v in enumerate(vs):
        row_max = errors[i].max()
        if row_max > 1e-10:
            print(f"  v={v:6.1f}: max rel error = {row_max:.2e}")


def fig_regime_map():
    """Show which algorithm handles each (v, z) point."""
    vs = np.linspace(0, 250, 300)
    zs = np.logspace(-4, 4, 300)
    regime = np.zeros((len(vs), len(zs)), dtype=int)
    for i, v in enumerate(vs):
        for j, z in enumerate(zs):
            regime[i, j] = _regime_label(abs(v), z)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.colormaps.get_cmap("Set2").resampled(4)
    im = ax.pcolormesh(zs, vs, regime, cmap=cmap, vmin=-0.5, vmax=3.5, shading="nearest")
    ax.set_xscale("log")
    ax.set_xlabel("z")
    ax.set_ylabel("|v|")
    ax.set_title("Regime map: which algorithm handles each (v, z)")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["Hankel\n(large z)", "Olver\n(large v)",
                              "Small-z\nasymptotic", "Quadrature\n(moderate)"])

    plt.tight_layout()
    plt.savefig("scripts/bessel_regime_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved scripts/bessel_regime_map.png")


def fig_value_profiles():
    """Plot log K_v(z) for different v, comparing JAX vs scipy."""
    v_values = [0.5, 1.0, 5.0, 20.0, 50.0, 100.0]
    zs = np.logspace(-3, 3, 500)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for idx, v in enumerate(v_values):
        ax = axes[idx // 3, idx % 3]
        jax_vals = np.asarray(log_kv(jnp.full(len(zs), v), jnp.array(zs)))
        ref_vals = scipy_log_kv(v, zs)

        ax.plot(zs, ref_vals, "k-", lw=2, alpha=0.3, label="scipy (ref)")
        ax.plot(zs, jax_vals, "b-", lw=1.2, label="pure JAX")

        # Mark regime boundaries
        v_abs = abs(v)
        hankel_z = max(25.0, v_abs**2 / 4.0)
        if hankel_z < zs[-1]:
            ax.axvline(hankel_z, color="r", ls="--", lw=0.8, alpha=0.6, label=f"Hankel z>{hankel_z:.0f}")

        ax.set_xscale("log")
        ax.set_xlabel("z")
        ax.set_ylabel(r"$\log K_{v}(z)$")
        ax.set_title(f"v = {v}")
        ax.legend(fontsize=7)

    plt.suptitle(r"$\log K_v(z)$: Pure JAX vs scipy reference", fontsize=14)
    plt.tight_layout()
    plt.savefig("scripts/bessel_value_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved scripts/bessel_value_profiles.png")


if __name__ == "__main__":
    print("=" * 60)
    print("Pure-JAX log_kv: accuracy comparison vs scipy")
    print("=" * 60)
    print()

    print("--- Regime Map ---")
    fig_regime_map()
    print()

    print("--- Value Profiles ---")
    fig_value_profiles()
    print()

    print("--- Accuracy Overview ---")
    fig_accuracy_overview()
    print()
    print("Done.")
