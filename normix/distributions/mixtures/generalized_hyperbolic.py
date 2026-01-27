"""
Generalized Hyperbolic (GH) marginal distribution f(x).

This is the marginal distribution obtained by integrating out y:
    f(x) = ∫ f(x, y) dy = ∫ f(x|y) f(y) dy

where:
    X|Y ~ N(μ + ΓY, ΣY)
    Y ~ GIG(λ, χ, ψ)

The marginal distribution has a closed form involving Bessel K functions.
"""

