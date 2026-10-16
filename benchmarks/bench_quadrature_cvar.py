"""CPU comparison of quadrature and conditional Monte Carlo for portfolio CVaR.

Run ``uv run python benchmarks/bench_quadrature_cvar.py --save``. Timings are
medians after JIT warmup and synchronize every result. Accuracy uses independent
adaptive SciPy integrals; MC accuracy is RMSE over reproducible seeds. Generation
and evaluation are timed separately. No compilation time is included.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import numpy as np
from scipy.optimize import minimize_scalar

from benchmarks.quadrature_reference import CASES, Reference
from benchmarks.utils import save_result, system_info
from normix import (
    Gamma, GeneralizedInverseGaussian, InverseGamma, InverseGaussian,
    NormalInverseGaussian, UnivariateGeneralizedHyperbolic,
)
from normix.finance import CVaR, DiscreteMeasure, MeanRiskProblem
from normix.finance._conditional import quantile


def _seconds(fn, repeats: int) -> float:
    for _ in range(3):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def _subordinators() -> dict:
    return {"VG": Gamma(2.0, 2.0), "NIG": InverseGaussian(1.0, 1.0),
            "GH": GeneralizedInverseGaussian(1.0, 2.0, 1.0),
            "NInvG3": InverseGamma(3.0, 2.0), "NInvG2": InverseGamma(2.0, 1.0)}


def _evaluate(risk, model, Q):
    v = -quantile(model, risk.alpha, Q)
    value, gradient, hessian = risk._scalar_bundle(model, Q, v)
    return jnp.stack([value, v, Q.mean(), gradient[1], hessian[1, 1]])


def accuracy_and_runtime(seeds: int, repeats: int) -> list[dict]:
    """Sweep scenario counts; report relative errors and warm CPU costs."""
    rows = []
    risk = CVaR(0.05)
    evaluate = eqx.filter_jit(_evaluate)
    for name, sub in _subordinators().items():
        nu, A, B = CASES[name]
        model = UnivariateGeneralizedHyperbolic.from_classical(
            mu=0.0, gamma=-0.5, sigma=1.0, p=nu, a=2*A, b=2*B)
        ref = Reference(nu, A, B).evaluate(0.0, -0.5, 1.0, risk.alpha)
        truth = np.array([ref["cvar"], ref["var"], ref["mean_y"],
                          ref["gradient"][1], ref["hessian"][1][1]])
        for method, counts in [("quadrature", (64, 128, 256, 512, 1024)),
                               ("MC", (1000, 5000, 20000, 100000))]:
            for n in counts:
                if method == "quadrature":
                    build = eqx.filter_jit(lambda d, seed: d.quadrature(n))
                    n_seeds = 1
                else:
                    build = eqx.filter_jit(lambda d, seed: DiscreteMeasure.from_sample(d.rvs(n, seed=seed)))
                    n_seeds = seeds
                estimates = []
                for seed in range(n_seeds):
                    Q = build(sub, jnp.asarray(seed))
                    estimates.append(np.asarray(evaluate(risk, model, Q)))
                errors = np.sqrt(np.mean((np.asarray(estimates) - truth)**2, axis=0)) / np.abs(truth)
                row = {"family": name, "method": method, "count": n, "seeds": n_seeds,
                       "generation_ms": 1e3 * _seconds(lambda: build(sub, jnp.asarray(0)), repeats),
                       "evaluation_ms": 1e3 * _seconds(lambda: evaluate(risk, model, Q), repeats),
                       **{key: float(error) for key, error in zip(
                           ("cvar_rel_error", "var_rel_error", "mean_rel_error", "gradient_rel_error", "hessian_rel_error"), errors)},
                       "reference_cvar": float(truth[0]), "reference_var": float(truth[1])}
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def equal_accuracy(rows: list[dict]) -> list[dict]:
    """Smallest measured count meeting joint VaR/CVaR/mean error targets."""
    out = []
    for family in _subordinators():
        for tolerance in (1e-2, 1e-3, 1e-4):
            result = {"family": family, "relative_tolerance": tolerance}
            for method in ("quadrature", "MC"):
                passing = [r["count"] for r in rows if r["family"] == family and r["method"] == method
                           and max(r[k] for k in ("cvar_rel_error", "var_rel_error", "mean_rel_error")) <= tolerance]
                result[method] = min(passing) if passing else None
            out.append(result)
    return out


def portfolio_comparison(repeats: int) -> dict:
    """Compare optimized weights, true objective, frontier, and fused kernel."""
    model = NormalInverseGaussian.from_classical(
        mu=jnp.array([0.01, 0.03, -0.01]), gamma=jnp.array([-0.2, 0.1, 0.05]),
        sigma=jnp.array([[0.7, 0.1, 0.0], [0.1, 0.4, 0.05], [0.0, 0.05, 0.2]]),
        mu_ig=1.0, lam=1.0)
    risk, sub = CVaR(0.05), model.subordinator()
    problem = MeanRiskProblem(model, risk)
    ref = Reference(*CASES["NIG"])
    target = 0.03
    bounds = (-0.5, 0.5)
    def objective(gamma):
        mu = target - gamma
        sigma = float(jnp.sqrt(problem.dispersion(mu, gamma)))
        return ref.evaluate(mu, gamma, sigma, risk.alpha)["cvar"]
    opt = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-13})
    if not opt.success:
        raise RuntimeError(opt.message)
    ref_weights = np.asarray(problem.weights(target-opt.x, opt.x))
    rows = []
    targets = jnp.array([target])
    w = jnp.array([0.4, 0.3, 0.3])
    mu0, ga0 = problem.min_dispersion_point()
    mus, gas = jnp.linspace(mu0-.03, mu0+.03, 41), jnp.linspace(ga0-.05, ga0+.05, 41)
    search = eqx.filter_jit(lambda p, r, Q: p._search_return_line(r, Q, bounds, 48))
    measures = [("quadrature512", sub.quadrature()),
                ("MC20000", DiscreteMeasure.from_sample(sub.rvs(20000, seed=0)))]
    for method, Q in measures:
        frontier = lambda: problem.efficient_frontier(targets, Q, gamma_bounds=bounds)
        fr = frontier()
        actual = objective(float(fr.gamma_tilde[0]))
        weights = np.asarray(fr.weights[0])
        rows.append({"method": method, "weights": weights.tolist(),
                     "weight_max_abs_error": float(np.max(np.abs(weights-ref_weights))),
                     "reported_objective": float(fr.risk[0]), "reference_objective_at_weights": actual,
                     "true_objective_gap": actual - float(opt.fun),
                     "frontier_search_ms": 1e3 * _seconds(lambda: search(problem, targets, Q), repeats),
                     "surface_41x41_ms": 1e3 * _seconds(lambda: problem.efficient_surface(mus, gas, Q), repeats),
                     "fused_bundle_ms": 1e3 * _seconds(lambda: risk.value_grad_hess_w(model, w, Q), repeats)})
    return {"reference_weights": ref_weights.tolist(), "reference_objective": float(opt.fun), "rows": rows}


def step_sweep() -> list[dict]:
    """Measure the diagnostic and Hessian failure as normal smoothing vanishes."""
    rows = []
    ref, risk = Reference(*CASES["NIG"]), CVaR(0.05)
    for sigma in (1.0, 0.1, 0.03, 0.01, 0.003):
        model = UnivariateGeneralizedHyperbolic.from_classical(
            mu=0.0, gamma=-0.5, sigma=sigma**2, p=-0.5, a=1.0, b=1.0)
        truth = ref.evaluate(0.0, -0.5, sigma, risk.alpha)
        for n in (512, 2048):
            Q = model.subordinator().quadrature(n)
            got = np.asarray(eqx.filter_jit(_evaluate)(risk, model, Q))
            rows.append({"sigma": sigma, "count": n,
                         "resolution": float(risk.step_resolution(model, Q)),
                         "cvar_rel_error": abs(got[0]/truth["cvar"]-1),
                         "hessian_rel_error": abs(got[4]/truth["hessian"][1][1]-1)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError("run this comparison with JAX_PLATFORMS=cpu")
    rows = accuracy_and_runtime(args.seeds, args.repeats)
    data = {"accuracy_runtime": rows, "equal_accuracy": equal_accuracy(rows),
            "portfolio": portfolio_comparison(args.repeats), "near_step": step_sweep()}
    print(json.dumps(data, indent=2))
    if args.save:
        print(save_result("quadrature_cvar", data))
    if args.output:
        args.output.write_text(json.dumps({"system": system_info(), **data}, indent=2) + "\n")


if __name__ == "__main__":
    main()
