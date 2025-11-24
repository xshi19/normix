"""
Generalized Inverse Gaussian (GIG) distribution.

The GIG distribution belongs to the exponential family with:

Classical parameters: (λ, χ, ψ)
    p(x|λ,χ,ψ) ∝ x^(λ-1) exp(-1/2 (χ/x + ψx))

Natural parameters: η = (η₁, η₂, η₃)
    η₁ = λ - 1
    η₂ = -χ/2
    η₃ = -ψ/2

Sufficient statistics: T(x) = (log x, 1/x, x)
"""

