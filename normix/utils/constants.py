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

# ── GIG parameter clamps ──────────────────────────────────────────────

# Near-zero perturbation for θ₂, θ₃ in GIG multi-start initialisation
GIG_THETA_PERTURB: float = 1e-4

# Clamp bounds for GIG parameters a, b in the GH M-step
GIG_CLAMP_LO: float = 1e-6
GIG_CLAMP_HI: float = 1e6

# Maximum allowed |p| from GIG solver before falling back
GIG_P_MAX: float = 50.0

# ── M-step constants ──────────────────────────────────────────────────

# Regularisation added to Σ in the M-step Cholesky factorisation
SIGMA_REG: float = 1e-8

# Floor for the denominator D = 1 − E[1/Y]·E[Y] in the M-step
SAFE_DENOMINATOR: float = 1e-10

# ── Initialisation constants ──────────────────────────────────────────

# Regularisation added to empirical Σ during moment-based initialisation
SIGMA_INIT_REG: float = 1e-4

# ── Finite-difference steps ───────────────────────────────────────────

# Central FD step for Fisher information (second-order)
FD_EPS_FISHER: float = 1e-4

# ── Bessel regime thresholds ─────────────────────────────────────────

# z threshold below which small-z asymptotic is used in log_kv
BESSEL_SMALLZ_THRESHOLD: float = 1e-6
