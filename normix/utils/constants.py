"""Shared numerical constants for normix."""

# ── Tiny floors (prevent log(0) and exact-zero divisions) ──────────────

# Floor for log-space clamping in JAX (jnp.maximum(x, LOG_EPS))
LOG_EPS: float = 1e-30

# Floor for numpy-side values (avoids exact zero in log)
TINY: float = 1e-300

# ── GIG-specific constants ─────────────────────────────────────────────

# Finite-difference step for Bessel order derivative ∂log K_v/∂v
BESSEL_EPS_V: float = 1e-5

# √(ab) threshold below which GIG delegates to Gamma/InverseGamma limits
GIG_DEGEN_THRESHOLD: float = 1e-10

# ── Optimisation constants ─────────────────────────────────────────────

# Tikhonov damping added to Newton Hessian for positive-definiteness
HESSIAN_DAMPING: float = 1e-6

# Floor for GIG θ₂, θ₃ during warm-start initialisation
THETA_FLOOR: float = -1e-8

# ── M-step constants ──────────────────────────────────────────────────

# Regularisation added to Σ in the M-step Cholesky factorisation
SIGMA_REG: float = 1e-8

# Floor for the denominator D = 1 − E[1/Y]·E[Y] in the M-step
SAFE_DENOMINATOR: float = 1e-10

# ── Finite-difference steps ───────────────────────────────────────────

# Central FD step for Fisher information (second-order)
FD_EPS_FISHER: float = 1e-4

# ── Backward compatibility aliases ────────────────────────────────────

GIG_EPS_V_HESS = BESSEL_EPS_V
GIG_EPS_NP = TINY
