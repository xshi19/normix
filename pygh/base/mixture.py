"""
Base class for mixture distributions and normal mixture distributions.

These distributions have the form:
    f(x) = ∫ f(x|y) f(y) dy

where:
    - x|y ~ N(μ + Γy, Σy) (conditional normal)
    - y ~ Mixing distribution (GIG, IG, Gamma, etc.)

This representation makes them belong to the exponential family,
with the joint distribution f(x, y) also in exponential family.
"""

