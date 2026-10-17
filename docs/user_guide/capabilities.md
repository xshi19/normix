# Autodiff surface

Log-densities, moments, and the JAX log-partition triad are differentiable.
`vmap` batches one observation.
The table is what Gamma, GIG, and variance-gamma accept.
A dash means that cell was not checked.
`fit` and reverse-mode `from_expectation` are outside this surface.

| Operation | Eager | JIT | vmap | JVP | VJP | Second derivative |
|---|---|---|---|---|---|---|
| `log_prob`, `pdf` | yes | yes | yes | yes | yes | yes |
| `mean` | yes | — | — | yes | yes | yes |
| `expectation_params` | yes | — | — | yes | yes | yes |
| `from_expectation` | yes | yes | yes | yes | no | — |
| `cdf`, `ppf` | yes | yes | yes | yes | yes | yes in $x$ |
| `renyi(alpha)` | yes | — | — | in $\alpha$ | in $\alpha$ | no in $\theta$ |
| `entropy`, `varentropy` | yes | — | — | — | yes | — |
| `fit` | yes | no | no | no | no | no |

`jax.jit` takes a function of arrays.
`equinox.filter_jit` traces a method on the module.

`from_expectation` has a forward Jacobian.
Reverse mode stops in `lax.fori_loop` or `lax.while_loop`.
The digamma Newton loop and the GIG line search have no reverse rule.

`renyi` differentiates in the order $\alpha$ through a custom JVP.
A derivative of `renyi` in a distribution parameter raises `UnexpectedTracerError`.
`entropy` and `varentropy` do differentiate in the parameters.

`cdf` and `ppf` accuracy, including the derivative of the approximation, is in {doc}`cdf_ppf`.

`fit` checks the sample on the host.
It is not a JAX program.
