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

# ── E-step constants ──────────────────────────────────────────────────

# Floor for the posterior GIG scale b_post = b + (x-μ)ᵀΣ⁻¹(x-μ) in the EM
# E-step. Bounds the conditional inverse moment E[1/Y|x] for observations near
# the mode (where (x-μ)ᵀΣ⁻¹(x-μ) → 0). Only binds for VG, whose prior b = 0;
# for GH / NIG / NInvG the prior b > 0 already keeps b_post above this floor.
# See dev-notes/tech_notes/vg_em_inverse_moment_singularity.md.
B_POST_FLOOR: float = 1e-6

# Lower floor on the (α−1) denominator of the VG/NInvG prior reconstruction
# moment β/(α−1) in `compute_eta_from_model`. The Gamma inverse moment
# E[1/Y] (VG) and the InverseGamma forward moment E[Y] (NInvG) both equal
# β/(α−1) and diverge to +∞ as α↓1 (and the bare closed form turns *negative*
# for α<1). Flooring the denominator keeps the reconstructed moment finite,
# positive, and continuous, while leaving the well-conditioned α>1+margin
# regime exact. Only this single moment is regularized — E[log Y] and the
# other finite moment keep their exact closed forms.
ALPHA_MOMENT_MARGIN: float = 0.1

# ── M-step constants ──────────────────────────────────────────────────

# Regularisation added to Σ in the M-step Cholesky factorisation
SIGMA_REG: float = 1e-8

# Magnitude floor for D = 1 − E[1/Y]·E[Y] in the M-step (applied as −max(|D|, floor))
SAFE_DENOMINATOR: float = 1e-10

# Positivity floor for the diagonal D in the factor-analysis M-step
# (Σ = F Fᵀ + diag(D)). See `docs/theory/factor_analysis.rst`.
D_FLOOR: float = 1e-8

# ── Initialisation constants ──────────────────────────────────────────

# Regularisation added to empirical Σ during moment-based initialisation
SIGMA_INIT_REG: float = 1e-4

# ── Finite-difference steps ───────────────────────────────────────────

# Central FD step for Fisher information (second-order)
FD_EPS_FISHER: float = 1e-4

# ── Bessel regime thresholds ─────────────────────────────────────────

# z threshold below which small-z asymptotic is used in log_kv
BESSEL_SMALLZ_THRESHOLD: float = 1e-6
