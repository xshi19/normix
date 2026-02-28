"""
Detailed iteration-by-iteration trace of VG EM with and without normalization.

For each of the first 3 EM iterations, prints:
  - GIG conditional parameters (p, a, b) and their summary stats
  - Mahalanobis distances  b = (x-μ)^T Σ^{-1} (x-μ)
  - Conditional expectations  E[Y|X], E[1/Y|X], E[log Y|X]
  - Gamma expectation parameters  η = [s3, s2]
  - Newton solver details: target, initial guess, each Newton step
  - Resulting shape (α) and rate (β)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import digamma, polygamma

from normix.distributions.mixtures.variance_gamma import VarianceGamma
from normix.distributions.univariate.gamma import Gamma
from normix.utils import log_kv, column_median_mad, robust_cholesky

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent.parent / "data" / "sp500_sample.csv"


def load_data(n_stocks):
    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    tickers = sorted(df.columns.tolist())[:n_stocks]
    return df[tickers].values


def newton_solve_traced(s3, s2):
    """
    Solve ψ(α) - log(α) = s3 - log(s2) with full trace of each step.
    Returns (alpha, beta, steps) where steps is a list of dicts.
    """
    target = s3 - np.log(s2)
    alpha = max(s2, 2.0)
    steps = []

    for i in range(100):
        psi_val = digamma(alpha)
        psi_prime = polygamma(1, alpha)
        f_val = psi_val - np.log(alpha) - target
        f_prime = psi_prime - 1.0 / alpha

        step_info = {
            "iter": i,
            "alpha": alpha,
            "f_val": f_val,
            "f_prime": f_prime,
            "newton_step": -f_val / f_prime,
        }
        steps.append(step_info)

        alpha_new = alpha - f_val / f_prime
        alpha_new = max(alpha_new, 0.5)

        if abs(alpha_new - alpha) / abs(alpha) < 1e-12:
            alpha = alpha_new
            break
        alpha = alpha_new

    beta = alpha / s2
    return alpha, beta, target, steps


def run_em_trace(X_data, label, n_iters=3):
    """Run VG EM on X_data for n_iters iterations with detailed diagnostics."""
    n, d = X_data.shape

    vg = VarianceGamma()
    vg._joint = vg._create_joint_distribution()
    vg._joint._d = d
    vg._initialize_params(X_data, random_state=42)

    p_init = vg.classical_params
    print(f"\n{'='*100}")
    print(f"  EM trace: {label}  (n={n}, d={d})")
    print(f"  Init: shape={p_init.shape:.6f}, rate={p_init.rate:.6f}")
    print(f"{'='*100}")

    for it in range(n_iters):
        print(f"\n  --- Iteration {it+1} ---")

        # Access current parameters
        mu = vg._joint._mu
        gamma_vec = vg._joint._gamma
        alpha = vg._joint._alpha
        beta = vg._joint._beta
        L_inv = vg._joint.L_Sigma_inv
        L = vg._joint._L_Sigma

        # GIG conditional parameters
        p_cond = alpha - d / 2.0
        gamma_z = L_inv @ gamma_vec
        gamma_quad = np.dot(gamma_z, gamma_z)
        a_cond = 2 * beta + gamma_quad

        diff = X_data - mu
        z = L_inv @ diff.T
        b_cond = np.sum(z**2, axis=0)
        b_cond_safe = np.maximum(b_cond, 1e-10)

        print(f"  Current params: α={alpha:.6f}, β={beta:.6f}")
        print(f"  GIG conditional: p={p_cond:.4f}, a={a_cond:.6f}")
        print(f"  γ^T Σ^{{-1}} γ = {gamma_quad:.6f}")
        print(f"  Mahalanobis b: mean={b_cond.mean():.4f}, std={b_cond.std():.4f}, "
              f"min={b_cond.min():.4f}, max={b_cond.max():.4f}")
        print(f"  √(ab): mean={np.sqrt(a_cond*b_cond_safe).mean():.4f}, "
              f"std={np.sqrt(a_cond*b_cond_safe).std():.4f}")

        # Compute conditional expectations (replicating the code)
        sqrt_ab = np.sqrt(a_cond * b_cond_safe)
        sqrt_b_over_a = np.sqrt(b_cond_safe / a_cond)

        log_kv_p = log_kv(p_cond, sqrt_ab)
        log_kv_pm1 = log_kv(p_cond - 1, sqrt_ab)
        log_kv_pp1 = log_kv(p_cond + 1, sqrt_ab)

        E_Y = sqrt_b_over_a * np.exp(log_kv_pp1 - log_kv_p)
        E_inv_Y = np.exp(log_kv_pm1 - log_kv_p) / sqrt_b_over_a

        eps = 1e-6
        log_kv_p_plus = log_kv(p_cond + eps, sqrt_ab)
        log_kv_p_minus = log_kv(p_cond - eps, sqrt_ab)
        d_log_kv_dp = (log_kv_p_plus - log_kv_p_minus) / (2 * eps)
        E_log_Y = d_log_kv_dp + 0.5 * np.log(b_cond_safe / a_cond)

        s1 = np.mean(E_inv_Y)
        s2 = np.mean(E_Y)
        s3 = np.mean(E_log_Y)

        print(f"  E[Y|X]:    mean={E_Y.mean():.6f}, std={E_Y.std():.6f}, "
              f"CV={E_Y.std()/E_Y.mean():.6f}")
        print(f"  E[1/Y|X]:  mean={E_inv_Y.mean():.6f}, std={E_inv_Y.std():.6f}")
        print(f"  E[logY|X]: mean={E_log_Y.mean():.6f}, std={E_log_Y.std():.6f}")
        print(f"  s1=mean(E[1/Y])={s1:.6f}, s2=mean(E[Y])={s2:.6f}, "
              f"s3=mean(E[logY])={s3:.6f}")

        # Gamma expectation params and Newton trace
        print(f"\n  Gamma Newton solver:")
        print(f"    η = [s3, s2] = [{s3:.6f}, {s2:.6f}]")
        alpha_solved, beta_solved, target, steps = newton_solve_traced(s3, s2)
        print(f"    target = s3 - log(s2) = {s3:.6f} - {np.log(s2):.6f} = {target:.8f}")
        print(f"    initial α guess = max(s2, 2.0) = {max(s2, 2.0):.6f}")
        for s in steps[:6]:
            print(f"    step {s['iter']}: α={s['alpha']:.6f}, "
                  f"f(α)={s['f_val']:.2e}, f'(α)={s['f_prime']:.2e}, "
                  f"Δα={s['newton_step']:.2e}")
        if len(steps) > 6:
            print(f"    ... ({len(steps)} total steps)")
        print(f"    solved: α={alpha_solved:.6f}, β={beta_solved:.6f}")

        # Now do the actual M-step
        cond_exp = {"E_Y": E_Y, "E_inv_Y": E_inv_Y, "E_log_Y": E_log_Y}
        vg._m_step(X_data, cond_exp)

        p_new = vg.classical_params
        print(f"\n  After M-step: shape={p_new.shape:.6f}, rate={p_new.rate:.6f}")


def main():
    d = 150

    X_raw = load_data(d)
    center, scale = column_median_mad(X_raw)
    X_norm = (X_raw - center) / scale

    print("PART A: Verify Mahalanobis invariance at initialization")
    print("=" * 80)

    # Diagnose: trace through _initialize_params to see where invariance breaks
    print("\n  Initialization formula: Σ_init = (X_cov - Var(Y)·γγ^T) / E[Y]")
    print("  with α_init=2, β_init=1, so E[Y]=2, Var(Y)=2")
    print("  If Var(Y)·γγ^T has entries > X_cov, Σ_init goes non-PD")
    print("  and robust_cholesky patches it in a SCALE-DEPENDENT way.\n")

    for label, X_data in [("normalized", X_norm), ("raw", X_raw)]:
        n_d = X_data.shape[1]
        X_mean = np.mean(X_data, axis=0)
        X_cov = np.cov(X_data, rowvar=False)
        X_centered = X_data - X_mean
        X_std = np.std(X_data, axis=0)
        X_std = np.maximum(X_std, 1e-10)
        skewness = np.mean((X_centered / X_std) ** 3, axis=0)
        gamma_init = skewness * X_std * 0.1
        Var_Y = 2.0
        E_Y = 2.0

        Sigma_formula = (X_cov - Var_Y * np.outer(gamma_init, gamma_init)) / E_Y
        eigvals = np.linalg.eigvalsh(Sigma_formula)
        n_neg = np.sum(eigvals < 0)

        L = robust_cholesky(Sigma_formula, eps=1e-6)
        Sigma_fixed = L @ L.T
        rel_change = np.linalg.norm(Sigma_fixed - Sigma_formula) / np.linalg.norm(Sigma_formula)

        L_inv = np.linalg.solve(L, np.eye(n_d))
        diff = X_data - (X_mean - gamma_init * E_Y)
        z = L_inv @ diff.T
        b_cond = np.sum(z**2, axis=0)
        gz = L_inv @ gamma_init
        gamma_quad = np.dot(gz, gz)

        print(f"  {label}:")
        print(f"    ||γ_init||                  = {np.linalg.norm(gamma_init):.6e}")
        print(f"    diag(Var_Y·γγ^T)  max       = {Var_Y * np.max(gamma_init**2):.6e}")
        print(f"    diag(X_cov)       max       = {np.max(np.diag(X_cov)):.6e}")
        print(f"    Σ_formula eigenvals: min={eigvals.min():.4e}, max={eigvals.max():.4e}")
        print(f"    Negative eigenvalues         = {n_neg}")
        print(f"    ||Σ_fixed - Σ_formula|| / ||Σ_formula|| = {rel_change:.4e}")
        print(f"    γ^T Σ_fixed^{{-1}} γ            = {gamma_quad:.4e}")
        print(f"    Mahalanobis b: mean          = {b_cond.mean():.4e}")
        print()

    print("  CONCLUSION: The initialization formula Σ = (Cov - 2γγ^T)/2 produces")
    print("  a non-PD matrix in BOTH spaces, but the robust_cholesky fix is")
    print("  scale-dependent. In normalized space ||γ||≈3.1 and Cov has entries")
    print("  up to ~6, so 2·γ_i^2 can exceed Cov_ii, creating larger negative")
    print("  eigenvalues. The fix inflates Σ differently, causing γ^T Σ^{-1} γ")
    print("  and the Mahalanobis distances to be 1000x larger than in raw space.")
    print("  This makes the GIG posterior Y|X extremely concentrated from")
    print("  iteration 1, driving the shape parameter to ~18 immediately.")

    print("\n\nPART B: Detailed EM trace for first 3 iterations")
    run_em_trace(X_norm, "NORMALIZED (d=150)", n_iters=3)
    run_em_trace(X_raw, "RAW (d=150)", n_iters=3)

    print("\n\nPART C: Same comparison at d=80 (where both agree)")
    X_raw_80 = load_data(80)
    center_80, scale_80 = column_median_mad(X_raw_80)
    X_norm_80 = (X_raw_80 - center_80) / scale_80
    run_em_trace(X_norm_80, "NORMALIZED (d=80)", n_iters=3)
    run_em_trace(X_raw_80, "RAW (d=80)", n_iters=3)


if __name__ == "__main__":
    main()
