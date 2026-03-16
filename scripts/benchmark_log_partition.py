"""
Micro-benchmark: isolate JAX vs NumPy overhead at each layer.

Tests:
  1. Single log_kv call (the Bessel function)
  2. Gradient of GIG log-partition (∇ψ)
  3. Hessian of GIG log-partition (∇²ψ)
  4. Full eta→theta solve (warm-start)
  5. CPU-only JAX vs GPU JAX for tests 1-3

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/benchmark_log_partition.py
"""
import os, time, sys
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

W = 88
def header(s): print(f"\n{'='*W}\n{s}\n{'='*W}", flush=True)
def subheader(s): print(f"\n── {s} ──", flush=True)

def timeit(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times), np.array(times)

# Test point: GIG(p=0.5, a=1.0, b=1.0) — moderate parameters
theta_np = np.array([-0.5, -0.5, -0.5], dtype=np.float64)
eta_np   = np.array([0.3613, 1.0, 2.0], dtype=np.float64)
v_np, z_np = 0.5, 1.0

theta_jnp = jnp.array(theta_np)
eta_jnp   = jnp.array(eta_np)
v_jnp, z_jnp = jnp.array(v_np), jnp.array(z_np)

header("Micro-benchmark: JAX vs NumPy overhead per layer")
print(f"Device: {jax.devices()[0]}", flush=True)
print(f"Test point: theta={theta_np}, eta={eta_np.round(4)}", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Single log_kv(v, z) call
# ═══════════════════════════════════════════════════════════════════════
header("TEST 1: Single log_kv(v=0.5, z=1.0) call")

from normix_numpy.utils.bessel import log_kv as np_log_kv
from normix._bessel import log_kv as jax_log_kv

# NumPy
t, _ = timeit(lambda: np_log_kv(v_np, z_np), n=5000, warmup=100)
np_val = np_log_kv(v_np, z_np)
print(f"  NumPy scipy.kve:        {t*1e6:8.1f} μs   val={np_val:.10f}", flush=True)

# JAX eager (no JIT)
t, _ = timeit(lambda: float(jax_log_kv(v_jnp, z_jnp)), n=200, warmup=5)
jax_val = float(jax_log_kv(v_jnp, z_jnp))
print(f"  JAX eager (GPU):        {t*1e6:8.1f} μs   val={jax_val:.10f}", flush=True)

# JAX JIT GPU
jit_log_kv = jax.jit(jax_log_kv)
_ = jit_log_kv(v_jnp, z_jnp).block_until_ready()
t, _ = timeit(lambda: jit_log_kv(v_jnp, z_jnp).block_until_ready(), n=2000, warmup=50)
print(f"  JAX JIT (GPU):          {t*1e6:8.1f} μs", flush=True)

# JAX JIT CPU
v_cpu = jax.device_put(v_jnp, jax.devices("cpu")[0])
z_cpu = jax.device_put(z_jnp, jax.devices("cpu")[0])
jit_log_kv_cpu = jax.jit(jax_log_kv, device=jax.devices("cpu")[0])
_ = jit_log_kv_cpu(v_cpu, z_cpu).block_until_ready()
t, _ = timeit(lambda: jit_log_kv_cpu(v_cpu, z_cpu).block_until_ready(), n=2000, warmup=50)
print(f"  JAX JIT (CPU):          {t*1e6:8.1f} μs", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Gradient of log-partition  ∇ψ(θ) = η
# ═══════════════════════════════════════════════════════════════════════
header("TEST 2: Gradient ∇ψ(θ) — GIG log-partition")

from normix_numpy.distributions.univariate.generalized_inverse_gaussian import (
    GeneralizedInverseGaussian as LegacyGIG,
)

# NumPy analytical gradient (= expectation parameters)
legacy = LegacyGIG()
t, _ = timeit(lambda: legacy._natural_to_expectation(theta_np), n=5000, warmup=100)
np_grad = legacy._natural_to_expectation(theta_np)
print(f"  NumPy analytical:       {t*1e6:8.1f} μs   η={np_grad.round(6)}", flush=True)

# JAX JIT'd gradient (used by scipy-wrapped path)
from normix.distributions.gig import _grad_objective, _objective_jit, _objective
_ = _grad_objective(theta_jnp, eta_jnp)  # compile
t, _ = timeit(lambda: _grad_objective(theta_jnp, eta_jnp).block_until_ready(), n=1000, warmup=50)
jax_grad = np.array(_grad_objective(theta_jnp, eta_jnp))
print(f"  JAX JIT grad (GPU):     {t*1e6:8.1f} μs   g={jax_grad.round(6)}", flush=True)

# JAX JIT'd gradient on CPU
theta_cpu = jax.device_put(theta_jnp, jax.devices("cpu")[0])
eta_cpu = jax.device_put(eta_jnp, jax.devices("cpu")[0])
grad_obj_cpu = jax.jit(jax.grad(_objective), device=jax.devices("cpu")[0])
_ = grad_obj_cpu(theta_cpu, eta_cpu)
t, _ = timeit(lambda: grad_obj_cpu(theta_cpu, eta_cpu).block_until_ready(), n=1000, warmup=50)
print(f"  JAX JIT grad (CPU):     {t*1e6:8.1f} μs", flush=True)

# JAX eager gradient (no JIT — what the Newton solver does inside scan)
eager_grad = jax.grad(_objective)
_ = eager_grad(theta_jnp, eta_jnp)  # trace once
t, _ = timeit(lambda: eager_grad(theta_jnp, eta_jnp).block_until_ready(), n=100, warmup=5)
print(f"  JAX eager grad (GPU):   {t*1e6:8.1f} μs", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Hessian of log-partition  ∇²ψ(θ)
# ═══════════════════════════════════════════════════════════════════════
header("TEST 3: Hessian ∇²ψ(θ) — GIG log-partition")

# NumPy FD hessian (what legacy uses for fisher_information)
def numpy_fd_hessian(theta, eps=1e-4):
    n = len(theta)
    H = np.zeros((n, n))
    f0 = legacy._log_partition(theta)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                tp = theta.copy(); tp[i] += eps
                tm = theta.copy(); tm[i] -= eps
                H[i,i] = (legacy._log_partition(tp) - 2*f0 + legacy._log_partition(tm)) / eps**2
            else:
                tpp = theta.copy(); tpp[i]+=eps; tpp[j]+=eps
                tpm = theta.copy(); tpm[i]+=eps; tpm[j]-=eps
                tmp = theta.copy(); tmp[i]-=eps; tmp[j]+=eps
                tmm = theta.copy(); tmm[i]-=eps; tmm[j]-=eps
                H[i,j] = H[j,i] = (legacy._log_partition(tpp) - legacy._log_partition(tpm) - legacy._log_partition(tmp) + legacy._log_partition(tmm)) / (4*eps**2)
    return H

t, _ = timeit(lambda: numpy_fd_hessian(theta_np), n=1000, warmup=50)
np_hess = numpy_fd_hessian(theta_np)
print(f"  NumPy FD hessian:       {t*1e6:8.1f} μs   diag={np.diag(np_hess).round(4)}", flush=True)

# JAX autodiff hessian (JIT)
hess_fn_jit = jax.jit(jax.hessian(lambda t: _objective(t, eta_jnp)))
_ = hess_fn_jit(theta_jnp).block_until_ready()
t, _ = timeit(lambda: hess_fn_jit(theta_jnp).block_until_ready(), n=200, warmup=20)
jax_hess = np.array(hess_fn_jit(theta_jnp))
print(f"  JAX autodiff hess JIT (GPU): {t*1e6:8.1f} μs   diag={np.diag(jax_hess).round(4)}", flush=True)

# JAX autodiff hessian CPU
hess_fn_cpu = jax.jit(jax.hessian(lambda t: _objective(t, eta_cpu)), device=jax.devices("cpu")[0])
_ = hess_fn_cpu(theta_cpu).block_until_ready()
t, _ = timeit(lambda: hess_fn_cpu(theta_cpu).block_until_ready(), n=200, warmup=20)
print(f"  JAX autodiff hess JIT (CPU): {t*1e6:8.1f} μs", flush=True)

# JAX analytical hessian (JIT)
from normix.distributions.gig import _analytical_grad_hess_phi
phi_jnp = jnp.array([theta_np[0],
                      np.log(max(-theta_np[1], 1e-30)),
                      np.log(max(-theta_np[2], 1e-30))])
analytical_jit = jax.jit(lambda phi: _analytical_grad_hess_phi(phi, eta_jnp))
_ = analytical_jit(phi_jnp)
t, _ = timeit(lambda: analytical_jit(phi_jnp)[1].block_until_ready(), n=200, warmup=20)
ag, aH = analytical_jit(phi_jnp)
print(f"  JAX analytical hess JIT (GPU): {t*1e6:8.1f} μs   diag={np.diag(np.array(aH)).round(4)}", flush=True)

# JAX analytical hessian CPU
phi_cpu = jax.device_put(phi_jnp, jax.devices("cpu")[0])
analytical_cpu = jax.jit(lambda phi: _analytical_grad_hess_phi(phi, eta_cpu), device=jax.devices("cpu")[0])
_ = analytical_cpu(phi_cpu)
t, _ = timeit(lambda: analytical_cpu(phi_cpu)[1].block_until_ready(), n=200, warmup=20)
print(f"  JAX analytical hess JIT (CPU): {t*1e6:8.1f} μs", flush=True)

# JAX eager hessian (no JIT — what the Newton solver actually does inside scan)
eager_hess = jax.hessian(lambda t: _objective(t, eta_jnp))
_ = eager_hess(theta_jnp)  # trace once
t, _ = timeit(lambda: eager_hess(theta_jnp).block_until_ready(), n=20, warmup=2)
print(f"  JAX eager hess (GPU):   {t*1e6:8.1f} μs  <-- used inside scan loop", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Full eta→theta solve (warm-start)
# ═══════════════════════════════════════════════════════════════════════
header("TEST 4: Full η→θ solve (warm-start from true θ)")

from normix.distributions.gig import (
    _solve_jaxopt, _solve_newton_analytical, _solve_scipy, _initial_guesses,
)
from scipy.optimize import minimize

# Legacy NumPy (pure numpy+scipy, no JAX)
t, _ = timeit(lambda: legacy._expectation_to_natural(eta_np, theta0=theta_np), n=500, warmup=20)
print(f"  Legacy NumPy/SciPy:     {t*1e3:8.2f} ms", flush=True)

# scipy-wrapped JAX (current best JAX path)
def scipy_wrap():
    theta0 = np.array(theta_np); theta0[1]=min(theta0[1],-1e-8); theta0[2]=min(theta0[2],-1e-8)
    bounds = [(-np.inf,np.inf),(-np.inf,0),(-np.inf,0)]
    return minimize(
        fun=lambda t: float(_objective_jit(jnp.asarray(t, dtype=jnp.float64), eta_jnp)),
        x0=theta0, jac=lambda t: np.array(_grad_objective(jnp.asarray(t, dtype=jnp.float64), eta_jnp)),
        method='L-BFGS-B', bounds=bounds, options={'maxiter':500,'gtol':1e-10})
t, _ = timeit(scipy_wrap, n=100, warmup=10)
print(f"  scipy-wrapped JAX:      {t*1e3:8.2f} ms", flush=True)

# JAX Newton analytical scan=5
_ = _solve_newton_analytical(eta_jnp, theta_jnp, 500, 1e-10, scan_length=5)
t, _ = timeit(lambda: _solve_newton_analytical(eta_jnp, theta_jnp, 500, 1e-10, scan_length=5).block_until_ready(), n=5, warmup=1)
print(f"  JAX Newton analytical (scan=5, GPU): {t*1e3:8.1f} ms", flush=True)

# JAX Newton autodiff scan=5
_ = _solve_jaxopt(eta_jnp, theta_jnp, 500, 1e-10, scan_length=5)
t, _ = timeit(lambda: _solve_jaxopt(eta_jnp, theta_jnp, 500, 1e-10, scan_length=5).block_until_ready(), n=3, warmup=1)
print(f"  JAX Newton autodiff (scan=5, GPU):   {t*1e3:8.1f} ms", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
header("SUMMARY")
print("""
Each row shows median execution time after warmup. "Eager" means no jax.jit
wrapper — the function is traced/dispatched fresh each call (as happens inside
the Newton lax.scan body).

Key insight: the GIG problem is 3-dimensional. A 3x3 Hessian, 3-vector gradient.
These are too small for GPU parallelism to help — the per-operation dispatch
overhead dominates.

Known JAX issues:
  - #5986: lax.cond inside lax.scan is significantly slower on GPU (open P2, 2021)
  - #24411: JAX extremely slow on GPUs for small problems
  - #20326: JAX slower than NumPy for small-scale linear algebra in loops
  - JAX benchmarking docs: 10x10 inputs → JAX/GPU 10x slower than NumPy/CPU

References:
  https://github.com/google/jax/issues/5986
  https://github.com/jax-ml/jax/issues/24411
  https://github.com/google/jax/issues/20326
  https://docs.jax.dev/en/latest/benchmarking.html
""", flush=True)
