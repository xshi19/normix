"""Shared numerical constants for normix."""

# Floor for log-space clamping (jnp.maximum(x, LOG_EPS))
LOG_EPS: float = 1e-30

# Finite-difference step for GIG Hessian approximation
GIG_EPS_V_HESS: float = 1e-5

# Floor for numpy-side probability values (avoids exact zero in log)
GIG_EPS_NP: float = 1e-300
