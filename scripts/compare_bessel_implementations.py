"""
Compare pure-JAX log K_v(z) implementation against mpmath reference.

Generates figures showing accuracy across all regimes:
  1. Hankel asymptotic (large z)
  2. Numerical quadrature (moderate z, v)
  3. Olver uniform expansion (large v)

Similar to notebooks_numpy/bessel_function_comparison.ipynb but for the
new pure-JAX implementation.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import mpmath

jax.config.update("jax_enable_x64", True)
mpmath.mp.dps = 50

from normix._bessel import (
    log_kv, _hankel_large_z, _quadrature_log_kv, _olver_large_v,
    _hankel_threshold, _OLVER_V_THRESH,
)


def mpmath_log_kv(v, z):
    """High-precision reference."""
    return float(mpmath.log(mpmath.besselk(abs(v), z)))


# ──────────────────────────────────────────────────────────────────────
# Figure 1: log K_v(z) curves for various v
# ──────────────────────────────────────────────────────────────────────
def plot_log_kv_curves():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    z_vals = np.linspace(0.1, 50, 500)
    v_list = [0.5, 1.0, 2.0, 5.0, 10.0]

    ax = axes[0]
    ax.set_title("log K_v(z) — Pure JAX Implementation")
    for v in v_list:
        y = np.array([float(log_kv(jnp.array(v), jnp.array(z))) for z in z_vals])
        ax.plot(z_vals, y, label=f"v={v}")
    ax.set_xlabel("z")
    ax.set_ylabel("log K_v(z)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    z_vals2 = np.linspace(0.01, 10, 300)
    v_list2 = [0.5, 1.0, 2.0, 5.0, 10.0]

    ax = axes[1]
    ax.set_title("Relative Error vs mpmath (moderate z)")
    for v in v_list2:
        errs = []
        for z in z_vals2:
            ref = mpmath_log_kv(v, z)
            got = float(log_kv(jnp.array(v), jnp.array(z)))
            errs.append(abs(got - ref) / (abs(ref) + 1e-300))
        ax.semilogy(z_vals2, errs, label=f"v={v}")
    ax.set_xlabel("z")
    ax.set_ylabel("Relative error")
    ax.axhline(y=1e-9, color='r', linestyle='--', alpha=0.5, label="1e-9 target")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-18, 1e-6)

    v_large = [50, 100, 200, 500]
    z_vals3 = np.logspace(-1, 2.5, 200)

    ax = axes[2]
    ax.set_title("Relative Error vs mpmath (large v)")
    for v in v_large:
        errs = []
        for z in z_vals3:
            ref = mpmath_log_kv(v, z)
            got = float(log_kv(jnp.array(float(v)), jnp.array(z)))
            errs.append(abs(got - ref) / (abs(ref) + 1e-300))
        ax.semilogx(z_vals3, errs, label=f"v={v}")
    ax.set_xlabel("z")
    ax.set_ylabel("Relative error")
    ax.axhline(y=1e-9, color='r', linestyle='--', alpha=0.5, label="1e-9 target")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1e-16, 5e-15)

    plt.tight_layout()
    plt.savefig("scripts/bessel_accuracy_overview.png", dpi=150, bbox_inches='tight')
    print("Saved: scripts/bessel_accuracy_overview.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Figure 2: Regime map and per-regime accuracy
# ──────────────────────────────────────────────────────────────────────
def plot_regime_map():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    v_grid = np.linspace(0.1, 120, 80)
    z_grid = np.logspace(-2, 3, 80)
    V, Z = np.meshgrid(v_grid, z_grid)

    regime = np.zeros_like(V)
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v, z = V[i, j], Z[i, j]
            ht = max(20.0, v * v / 8.0)
            if z > ht:
                regime[i, j] = 1  # Hankel
            elif v > _OLVER_V_THRESH:
                regime[i, j] = 2  # Olver
            else:
                regime[i, j] = 0  # Quadrature

    ax = axes[0]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['#2196F3', '#4CAF50', '#FF9800'])
    im = ax.pcolormesh(v_grid, z_grid, regime, cmap=cmap, shading='auto')
    ax.set_yscale('log')
    ax.set_xlabel("v (order)")
    ax.set_ylabel("z (argument)")
    ax.set_title("Regime Map")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='Quadrature'),
        Patch(facecolor='#4CAF50', label='Hankel'),
        Patch(facecolor='#FF9800', label='Olver'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    v_test = np.array([0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500])
    z_test = np.array([1e-3, 0.01, 0.1, 1, 5, 10, 30, 100, 500, 1000])
    rel_errs = np.zeros((len(v_test), len(z_test)))

    for i, v in enumerate(v_test):
        for j, z in enumerate(z_test):
            ref = mpmath_log_kv(float(v), float(z))
            got = float(log_kv(jnp.array(float(v)), jnp.array(float(z))))
            rel_errs[i, j] = abs(got - ref) / (abs(ref) + 1e-300)

    ax = axes[1]
    im = ax.imshow(np.log10(rel_errs + 1e-18), aspect='auto',
                   extent=[0, len(z_test)-1, len(v_test)-1, 0],
                   cmap='RdYlGn_r', vmin=-17, vmax=-8)
    ax.set_xticks(range(len(z_test)))
    ax.set_xticklabels([f"{z:.0e}" if z < 1 else f"{z:.0f}" for z in z_test],
                       rotation=45, fontsize=8)
    ax.set_yticks(range(len(v_test)))
    ax.set_yticklabels([f"{v}" for v in v_test])
    ax.set_xlabel("z")
    ax.set_ylabel("v")
    ax.set_title("log10(relative error) vs mpmath")
    plt.colorbar(im, ax=ax, label="log10(rel error)")

    plt.tight_layout()
    plt.savefig("scripts/bessel_regime_map.png", dpi=150, bbox_inches='tight')
    print("Saved: scripts/bessel_regime_map.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Figure 3: Overflow stress test (like the notebook)
# ──────────────────────────────────────────────────────────────────────
def plot_overflow_stress():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    z_stress = np.logspace(-12, 3, 200)
    v_list = [1.0, 10.0, 100.0]

    for idx, v in enumerate(v_list):
        ax = axes[idx]
        refs = [mpmath_log_kv(v, z) for z in z_stress]
        gots = [float(log_kv(jnp.array(v), jnp.array(z))) for z in z_stress]

        ax.semilogx(z_stress, refs, 'b-', label='mpmath (reference)', linewidth=2)
        ax.semilogx(z_stress, gots, 'r--', label='pure JAX', linewidth=1.5)
        ax.set_xlabel("z")
        ax.set_ylabel("log K_v(z)")
        ax.set_title(f"v = {v}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Overflow Stress Test: log K_v(z) from z=1e-12 to z=1000", y=1.02)
    plt.tight_layout()
    plt.savefig("scripts/bessel_overflow_stress.png", dpi=150, bbox_inches='tight')
    print("Saved: scripts/bessel_overflow_stress.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────
def print_summary():
    print("\n" + "="*70)
    print("SUMMARY: Pure-JAX log K_v(z) accuracy vs mpmath reference")
    print("="*70)

    v_all = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    z_all = [1e-10, 1e-6, 1e-3, 0.01, 0.1, 1.0, 5.0, 10.0, 30.0, 100.0, 500.0, 1000.0]

    max_err = 0.0
    total = 0
    pass_count = 0

    for v in v_all:
        for z in z_all:
            ref = mpmath_log_kv(v, z)
            got = float(log_kv(jnp.array(v), jnp.array(z)))
            rel = abs(got - ref) / (abs(ref) + 1e-300)
            max_err = max(max_err, rel)
            total += 1
            if rel < 1e-9:
                pass_count += 1

    print(f"Test points: {total}")
    print(f"Passed (<1e-9): {pass_count}/{total}")
    print(f"Max relative error: {max_err:.2e}")
    print(f"Implementation: 100% pure JAX, zero scipy callbacks")
    print(f"Regimes: Hankel (large z) | Olver (large v) | Quadrature (moderate)")
    print("="*70)


if __name__ == "__main__":
    print("Generating comparison figures...")
    plot_log_kv_curves()
    plot_regime_map()
    plot_overflow_stress()
    print_summary()
    print("\nDone!")
