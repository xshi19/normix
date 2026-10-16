# Probability measures for portfolio scenarios

Finance calculations use one fixed measure of the mixing variable $Y$.
The conditional normal distribution is integrated analytically. This keeps
value, gradient, and Hessian consistent with one smooth convex CVaR objective
({ref}`RockafellarUryasev2000 <rockafellaruryasev2000>`).
The proofs are in {doc}`../theory/quadrature_scenarios`.

## Construction and reuse

{py:class}`~normix.utils.quadrature.DiscreteMeasure` is an immutable
[Equinox](https://docs.kidger.site/equinox/) pytree containing positive nodes,
normalized log probabilities, and the source law's positive-moment threshold.
Its constructors validate nodes and weights, including under
[JAX](https://docs.jax.dev/en/latest/) transformations. Negative weights are
rejected because a signed sum of normal CDFs need not be a CDF.

`subordinator.quadrature()` constructs 512 nodes on eight Gauss–Legendre
panels in a sinh coordinate around the log-density mode. The shared Bessel
edge search determines the window; no Bessel normalization is evaluated.
The window covers density and first-moment tilts with log drop 40. Nodes are
restricted to representable log coordinates in $[-700,700]$.
Gamma, inverse gamma, and inverse Gaussian use their exact GIG embeddings.

`scenarios=None` selects quadrature in all CVaR consumers. `MeanRiskProblem`
and `WeightFunctional` construct and retain a measure. Other calls build it
from their supplied model. An explicit measure stays fixed even if the model
changes: rebuild it after a refit. The scenario argument retains the former
sample argument's positional slot. `efficient_frontier` takes `gamma_bounds`
by keyword.

An array remains an equal-weight sample. `DiscreteMeasure.from_weights(Y, w)`
normalizes nonnegative historical weights without sorting or changing the nodes.
The empirical constructors carry an infinite moment threshold; only quadrature
carries the original law's moment restrictions. Portfolio expected returns
continue to use the model's mean, including when empirical scenarios are supplied.
`CVaR.var` retains its existing PINV quantile semantics.

## Quantiles and derivatives

The extreme component quantiles bracket every positive weighted mixture.
Sixty bisections operate in $x=c+h\sinh t$ to handle the wide brackets induced
by heavy-tail nodes. The center and scale use $E_Q[Y]$. When the source law
has no first moment, they instead use $\exp(E_Q[\log Y])$; a truncated arithmetic
mean can be enormous even when the lower-tail risk is finite.

The value uses the Rockafellar–Uryasev objective at a frozen quantile.
Quadrature nodes and geometric weights are frozen, while density weights
remain differentiable. This supports first derivatives in subordinator
parameters. Portfolio Hessians use the analytic conditional formulas.
Second autodifferentiation of the frozen-root value is not a portfolio Hessian.

The source moment threshold prevents a finite truncation from hiding an
infinite risk or derivative. At zero skew the skew Hessian needs
$E[Y^{3/2}]$. A divergent skew derivative has sign $-\infty$; a divergent
CVaR or skew curvature is $+\infty$. Zero coefficients in portfolio chain
rules do not multiply these infinities.

## Resolution and scope

`CVaR.step_resolution` reports the conditional-CDF transition width divided
by local spacing in $\log Y$. Values below roughly two call for more nodes
and a convergence study. This diagnostic is a heuristic, not a guarantee.
There is no automatic refinement or Monte Carlo fallback.

A fixed rule can miss the narrow Hessian peak when conditional normal
variance is small, or when a small tail probability moves the transition
far into a heavy tail. The default first-moment window also does not promise
accurate zero-skew Hessians near the inverse-gamma $3/2$ moment boundary.
Finite node-count agreement alone cannot detect an omitted tail moment.

Adaptive panels following the quantile are deferred. They change the measure
across an optimization surface and require separate accuracy, convexity, and
runtime checks. Negative control-variate weights and a subordinator base
class are unnecessary for the fixed-measure interface.

For a reproducible comparison, run
`uv run python benchmarks/bench_quadrature_cvar.py --save` from a checkout.
It measures independent-reference errors, warm CPU time, generation costs,
scenario counts at matched accuracy, portfolio changes, and step resolution.
See {doc}`../tutorials/finance/08_quadrature_scenarios` for a smaller example.
