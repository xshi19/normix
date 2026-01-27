"""
Normal Inverse Gaussian (NIG) marginal distribution f(x).

This is the marginal distribution obtained by integrating out y:
    f(x) = ∫ f(x|y) f(y) dy

where:
    X|Y ~ N(μ + ΓY, ΣY)
    Y ~ IG(μ, λ)
"""

