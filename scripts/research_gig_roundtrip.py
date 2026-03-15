"""
Research script: Diagnose GIG expectation parameter roundtrip errors in GH EM.

Investigates why set_expectation_params for the GIG sub-problem sometimes
produces large roundtrip errors during the GH M-step.

Diagnostics performed at each EM iteration:
  1. Per-component eta breakdown: E[Y], E[1/Y], E[log Y]
  2. L-BFGS-B optimization convergence details (success, grad norm, nit)
  3. Objective function comparison: new theta vs current theta
  4. Single-start (matching notebook) vs multi-start comparison
"""

import numpy as np
import pandas as pd
import os
import sys
import warnings
from scipy.optimize import minimize, Bounds

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from normix.distributions.mixtures import GeneralizedHyperbolic
from normix.distributions.univariate import GeneralizedInverseGaussian
from normix.utils.bessel import log_kv


def gig_objective(theta, eta):
    """Objective for GIG expectation-to-natural: A(θ) - θ·η."""
    p = theta[0] + 1
    b = -2 * theta[1]
    a = -2 * theta[2]
    if a <= 0 or b <= 0:
        return -np.inf
    sqrt_ab = np.sqrt(a * b)
    psi = np.log(2) + log_kv(p, sqrt_ab) + (p / 2) * (np.log(b) - np.log(a))
    return psi - np.dot(theta, eta)


def run_single_optimization(gig, eta, x0):
    """
    Run a single L-BFGS-B optimization (matching the library behavior exactly).

    This replicates what _expectation_to_natural does for ONE starting point.
    """
    bounds_list = gig._get_natural_param_support()
    lb = np.array([b[0] if not np.isinf(b[0]) else -1e10 for b in bounds_list])
    ub = np.array([b[1] if not np.isinf(b[1]) else 1e10 for b in bounds_list])
    bounds = Bounds(lb=lb, ub=ub)

    def objective(theta_arr):
        theta_arr = gig._project_to_support(theta_arr)
        psi = gig._log_partition(theta_arr)
        return psi - np.dot(theta_arr, eta)

    def grad_func(theta_arr):
        theta_arr = gig._project_to_support(theta_arr)
        grad_psi = gig._natural_to_expectation(theta_arr)
        return grad_psi - eta

    x0 = gig._project_to_support(np.asarray(x0))
    result = minimize(
        objective, x0,
        method='L-BFGS-B',
        jac=grad_func,
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    grad_at_sol = grad_func(result.x)
    grad_norm = np.max(np.abs(grad_at_sol))

    return {
        'theta': result.x.copy(),
        'obj_val': result.fun,
        'success': result.success,
        'message': result.message,
        'nit': result.nit,
        'nfev': result.nfev,
        'grad_norm': grad_norm,
        'grad_components': grad_at_sol.copy(),
    }


def run_multistart_optimization(gig, eta, theta0):
    """
    Run multi-start optimization using theta0 + _get_initial_natural_params.
    """
    starting_points = [theta0.copy()]
    extra_starts = gig._get_initial_natural_params(eta)
    if isinstance(extra_starts, np.ndarray) and extra_starts.ndim == 1:
        extra_starts = [extra_starts]
    starting_points.extend(extra_starts)

    best = None
    for x0 in starting_points:
        res = run_single_optimization(gig, eta, x0)
        if best is None or res['obj_val'] < best['obj_val'] or \
           res['grad_norm'] < best['grad_norm'] * 0.1:
            best = res
    best['n_starts'] = len(starting_points)
    return best


def recover_eta(theta):
    """Set a fresh GIG from theta and compute its expectation params."""
    gig = GeneralizedInverseGaussian()
    safe = theta.copy()
    safe[1] = min(safe[1], -1e-15)
    safe[2] = min(safe[2], -1e-15)
    gig._set_from_natural(safe)
    return gig._compute_expectation_params(), gig


def run_diagnostic_em(X, max_iter=15, tol=1e-3, regularization='det_sigma_one'):
    """Run GH EM with detailed GIG diagnostics at each iteration."""

    gh = GeneralizedHyperbolic()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, d = X.shape

    from normix.distributions.mixtures.generalized_hyperbolic import (
        REGULARIZATION_METHODS, robust_cholesky,
    )

    gh._joint = gh._create_joint_distribution()
    gh._joint._d = d

    regularize_fn = REGULARIZATION_METHODS[regularization]

    gh._initialize_params(X, random_state=42)

    def _apply_regularization():
        jt = gh._joint
        mu = jt._mu
        gamma = jt._gamma
        sigma = jt._L_Sigma @ jt._L_Sigma.T
        p_val, a_val, b_val = jt._p, jt._a, jt._b
        log_det_sigma = jt.log_det_Sigma
        if regularization == 'det_sigma_one':
            regularized = regularize_fn(
                mu, gamma, sigma, p_val, a_val, b_val, d,
                log_det_sigma=log_det_sigma
            )
        else:
            regularized = regularize_fn(
                mu, gamma, sigma, p_val, a_val, b_val, d
            )
        a_reg = regularized['a']
        b_reg = regularized['b']
        if a_reg < 1e-6 or b_reg < 1e-6 or a_reg > 1e6 or b_reg > 1e6:
            regularized['a'] = np.clip(a_reg, 1e-6, 1e6)
            regularized['b'] = np.clip(b_reg, 1e-6, 1e6)
        gh._joint.set_classical_params(**regularized)

    _apply_regularization()

    init_ll = np.mean(gh.logpdf(X))
    print(f"Initial log-likelihood: {init_ll:.6f}")
    print(f"Initial GIG: p={gh._joint._p:.6f}, a={gh._joint._a:.6f}, b={gh._joint._b:.6f}")
    print("=" * 90)

    for iteration in range(max_iter):
        prev_p = gh._joint._p
        prev_a = gh._joint._a
        prev_b = gh._joint._b
        prev_mu = gh._joint._mu.copy()
        prev_gamma = gh._joint._gamma.copy()
        prev_L = gh._joint._L_Sigma.copy()

        # E-step
        cond_exp = gh._conditional_expectation_y_given_x(X)

        s1 = np.mean(cond_exp['E_inv_Y'])
        s2 = np.mean(cond_exp['E_Y'])
        s3 = np.mean(cond_exp['E_log_Y'])

        gig_eta = np.array([s3, s1, s2])  # [E[log Y], E[1/Y], E[Y]]

        current_gig_theta = np.array([
            prev_p - 1,
            -prev_b / 2,
            -prev_a / 2,
        ])

        obj_current = gig_objective(current_gig_theta, gig_eta)

        # === (A) Single-start: exactly what the notebook does ===
        gig_single = GeneralizedInverseGaussian()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            single_res = run_single_optimization(gig_single, gig_eta, current_gig_theta)
        single_eta, single_gig = recover_eta(single_res['theta'])
        single_diff = single_eta - gig_eta
        single_max_err = np.max(np.abs(single_diff))

        # === (B) Multi-start: what we could do with extra starts ===
        gig_multi = GeneralizedInverseGaussian()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multi_res = run_multistart_optimization(gig_multi, gig_eta, current_gig_theta)
        multi_eta, multi_gig = recover_eta(multi_res['theta'])
        multi_diff = multi_eta - gig_eta
        multi_max_err = np.max(np.abs(multi_diff))

        # Now do the actual M-step (which uses single-start internally)
        gh._m_step(X, cond_exp, fix_tail=False, verbose=0)
        _apply_regularization()
        ll = np.mean(gh.logpdf(X))

        # --- Print ---
        print(f"\n{'='*90}")
        print(f"  Iteration {iteration + 1}  |  Log-likelihood: {ll:.6f}")
        print(f"{'='*90}")
        print(f"  GIG target eta: E[log Y]={gig_eta[0]:.8f}  E[1/Y]={gig_eta[1]:.8f}  E[Y]={gig_eta[2]:.8f}")

        eta_labels = ["E[log Y]", "E[1/Y] ", "E[Y]   "]

        print(f"\n  (A) SINGLE-START (matches notebook): max|err| = {single_max_err:.2e}")
        print(f"      success={single_res['success']}, nit={single_res['nit']}, "
              f"grad_norm={single_res['grad_norm']:.2e}")
        print(f"      message: {single_res['message']}")
        print(f"      grad components: [{single_res['grad_components'][0]:+.2e}, "
              f"{single_res['grad_components'][1]:+.2e}, {single_res['grad_components'][2]:+.2e}]")
        print(f"      Per-component roundtrip error:")
        for j in range(3):
            marker = " <<<" if abs(single_diff[j]) > 1e-5 else ""
            print(f"        {eta_labels[j]}: target={gig_eta[j]:+.8e}  diff={single_diff[j]:+.2e}  "
                  f"rel={abs(single_diff[j]) / (abs(gig_eta[j]) + 1e-15):.2e}{marker}")
        obj_single = gig_objective(single_res['theta'], gig_eta)
        print(f"      obj(current θ)={obj_current:.8f}  obj(new θ)={obj_single:.8f}  "
              f"Δ={obj_current - obj_single:+.2e}")
        print(f"      θ_new = (p={single_gig._p:.6f}, a={single_gig._a:.6f}, b={single_gig._b:.6f})")

        print(f"\n  (B) MULTI-START ({multi_res['n_starts']} starts): max|err| = {multi_max_err:.2e}")
        print(f"      success={multi_res['success']}, nit={multi_res['nit']}, "
              f"grad_norm={multi_res['grad_norm']:.2e}")
        print(f"      Per-component roundtrip error:")
        for j in range(3):
            marker = " <<<" if abs(multi_diff[j]) > 1e-5 else ""
            print(f"        {eta_labels[j]}: target={gig_eta[j]:+.8e}  diff={multi_diff[j]:+.2e}  "
                  f"rel={abs(multi_diff[j]) / (abs(gig_eta[j]) + 1e-15):.2e}{marker}")
        obj_multi = gig_objective(multi_res['theta'], gig_eta)
        print(f"      obj(current θ)={obj_current:.8f}  obj(new θ)={obj_multi:.8f}  "
              f"Δ={obj_current - obj_multi:+.2e}")
        print(f"      θ_new = (p={multi_gig._p:.6f}, a={multi_gig._a:.6f}, b={multi_gig._b:.6f})")

        if single_max_err > 10 * multi_max_err:
            print(f"\n  ** Multi-start is {single_max_err/multi_max_err:.0f}x better — "
                  f"single-start is stuck in a poor basin **")

        print(f"\n  GIG params: prev (p={prev_p:.6f}, a={prev_a:.6f}, b={prev_b:.6f})")
        print(f"              post-regularization (p={gh._joint._p:.6f}, a={gh._joint._a:.6f}, b={gh._joint._b:.6f})")

        # Convergence check
        max_rel_change = max(
            np.max(np.abs(gh._joint._mu - prev_mu) / (np.abs(prev_mu) + 1e-8)),
            np.max(np.abs(gh._joint._gamma - prev_gamma) / (np.abs(prev_gamma) + 1e-8)),
            np.max(np.abs(gh._joint._L_Sigma - prev_L) / (np.abs(prev_L) + 1e-8)),
        )
        if max_rel_change < tol:
            print(f"\nConverged at iteration {iteration + 1} (max_rel_change={max_rel_change:.2e})")
            break

    print("\n" + "=" * 90)
    final_ll = np.mean(gh.logpdf(X))
    print(f"Final log-likelihood: {final_ll:.6f}")
    print(f"Final GIG: p={gh._joint._p:.6f}, a={gh._joint._a:.6f}, b={gh._joint._b:.6f}")

    return gh


def main():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sp500_returns.csv')
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Falling back to synthetic data for demonstration.")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 5)) * 0.02
        X[:, 0] += 0.001
        run_diagnostic_em(X)
        return

    log_returns = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    n_total = len(log_returns)
    n_train = n_total // 2
    returns_train = log_returns.iloc[:n_train].values

    print(f"Data: {returns_train.shape[0]} samples, {returns_train.shape[1]} dimensions")
    print()

    run_diagnostic_em(returns_train, max_iter=15, tol=1e-3)


if __name__ == '__main__':
    main()
