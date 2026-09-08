"""Shared numerical constants for normix."""

# ── Tiny floors (prevent log(0) and exact-zero divisions) ──────────────

# Floor for log-space clamping in JAX (jnp.maximum(x, LOG_EPS))
LOG_EPS: float = 1e-30

# Floor for numpy-side values (avoids exact zero in log)
TINY: float = 1e-300

# ── GIG-specific constants ─────────────────────────────────────────────

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

# Margin ε used by the opt-in VG α-lower-bound sentinels in `fit(alpha_min=...)`.
# The VG marginal density is unbounded at x=μ for α ≤ d/2, and E[1/Y|x] is
# unbounded for α ≤ d/2 + 1; the floored b_post keeps EM finite but cannot stop
# it parking near that spike. When `alpha_min='density'` the M-step clamps the
# Gamma shape at d/2 + ε (density bounded); `alpha_min='inverse_moment'` clamps
# at d/2 + 1 + ε (E[1/Y|x] also bounded). Opt-in only — default is no clamp.
ALPHA_MIN_MARGIN: float = 0.1

# ── M-step constants ──────────────────────────────────────────────────

# Regularisation added to Σ in the M-step Cholesky factorisation
SIGMA_REG: float = 1e-8

# Magnitude floor for D = 1 − E[1/Y]·E[Y] in the M-step (applied as −max(|D|, floor))
SAFE_DENOMINATOR: float = 1e-10

# Positivity floor for the diagonal D in the factor-analysis M-step
# (Σ = F Fᵀ + diag(D)). See `docs/theory/factor_analysis.md`.
D_FLOOR: float = 1e-8

# ── Initialisation constants ──────────────────────────────────────────

# Regularisation added to empirical Σ during moment-based initialisation
SIGMA_INIT_REG: float = 1e-4

# ── Finite-difference steps ───────────────────────────────────────────

# Half-width of the Taylor window for Rényi entropy about α = 1
# (H_α = H − ½ V_H (α−1) + O((α−1)²)); keeps jax.grad(renyi) defined at α = 1
RENYI_TAYLOR_EPS: float = 1e-6

# ── Bessel moment-quadrature kernel (S10) ─────────────────────────────

# Gauss–Legendre nodes per whole-line integral (two equal panels at the mode).
# Must be even. Sweep vs mpmath (n ∈ {64,96,128,192,256}): 192/40/2 is the
# unique setting that holds the 1e-12 scaled-error contract on the frozen
# table. 128/40/2 misses ∂_z at (0, 10^{-10}) (3.8e-11).
BESSEL_QUAD_NODES: int = 192

# Target log-density drop at the window edge for every tilt |k| ≤ T.
BESSEL_QUAD_LOG_DROP: float = 40.0

# Largest |k| for which the window is required to capture the tilted tail.
BESSEL_WINDOW_TILT: int = 2

# Fixed bisection steps used to solve each window edge (jit/vmap-safe).
BESSEL_WINDOW_ITERS: int = 40

# Absolute cap on |x| during window search (covers z down to ~1e-300).
BESSEL_WINDOW_HI_MAX: float = 800.0

# Degeneracy floor on each GL panel half-width (mode-centered).
BESSEL_PANEL_FLOOR: float = 1e-12

# ── Diversification / torsion constants ───────────────────────────────

# Relative spectral floor for Meucci torsion diagonalizations. Before
# correlation scaling, diag(H) is floored at TORSION_SPECTRAL_FLOOR·max(diag H);
# eigenvalues more negative than −TORSION_SPECTRAL_FLOOR·‖S‖_∞ mark the
# decomposition invalid (ENB → NaN), while roundoff-scale negatives are
# projected to zero. Relative so daily vs annualized returns share one value.
TORSION_SPECTRAL_FLOOR: float = 1e-12
