# Autodiff surface

Log-densities, moments, and the JAX log-partition triad are differentiable.
`log_prob` takes one observation.
`vmap` applies that function to a batch.
The table is what Gamma, GIG, and variance-gamma accept.
A dash means that cell was not checked.
`fit` and reverse-mode `from_expectation` are outside this surface.

| Operation | Eager | JIT | vmap | JVP | VJP | Second derivative |
|---|---|---|---|---|---|---|
| `log_prob`, `pdf` | yes | yes | yes | yes | yes | yes |
| `mean` | yes | yes | yes | yes | yes | yes |
| `expectation_params` | yes | yes | yes | yes | yes | yes |
| `from_expectation` (Gamma) | yes | yes | yes | yes | no | — |
| `from_expectation` (GIG) | yes | with `theta0` | with `theta0` | no | no | — |
| `cdf`, `ppf` | yes | yes | yes | yes | yes | yes in $x$ |
| `renyi(alpha)` | yes | yes | yes | in $\alpha$ | in $\alpha$ | no in $\theta$ |
| `entropy`, `varentropy` | yes | yes | yes | — | yes | — |
| `fit` | yes | no | no | no | no | no |

`jax.jit` takes a function of arrays.
`equinox.filter_jit` traces a method on the module.

Gamma `from_expectation` has a forward Jacobian.
Variance-gamma uses that Gamma solve.
The GIG JVP returns zeros.
`jit` or `vmap` of GIG `from_expectation` without `theta0` raises `ConcretizationTypeError`.

Reverse mode of `from_expectation` stops in `lax.fori_loop` or `lax.while_loop`.
The digamma Newton loop and the GIG line search have no reverse rule.

`renyi` differentiates in the order $\alpha$ through a custom JVP.
A derivative of `renyi` in a distribution parameter raises `UnexpectedTracerError`.
`entropy` and `varentropy` do differentiate in the parameters.

`cdf` and `ppf` accuracy, including the derivative of the approximation, is in {doc}`cdf_ppf`.

`fit` checks the sample on the host.
It is not a JAX program.
