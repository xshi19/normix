# Autodiff surface

Log-densities, moments, and the JAX log-partition triad are differentiable.
`log_prob` takes one observation.
`vmap` applies that function to a batch.
`jax.jit` takes a function of arrays.
`equinox.filter_jit` traces a method on the module.
A bound method is not a valid `jax.jit` target: the module arrays are not hashable.

The cells are a probe of Gamma, GIG, and variance-gamma.
A dash means that cell was not run.

| Operation | Eager | JIT | vmap | JVP | VJP |
|---|---|---|---|---|---|
| `log_prob` | yes | yes | yes | yes | yes |
| `mean` | yes | yes | — | yes | — |
| `from_expectation` (Gamma, JAX) | yes | yes | yes | yes | yes |
| `from_expectation` (GIG warm start, JAX) | yes | yes | yes | yes | yes |
| `from_expectation` (GIG cold start, JAX) | yes | yes | yes | yes | yes |
| `from_expectation` (variance-gamma pytree, JAX) | yes | yes | yes | yes | yes |
| `from_expectation` (CPU) | yes | no | — | — | — |
| `cdf` | yes | yes | — | yes | yes |
| `renyi`, order | yes | yes | yes | yes | yes |
| `renyi`, distribution parameter | yes | — | — | yes | yes |
| `entropy`, `varentropy` | yes | yes | — | — | yes |
| `fit` (Gamma) | yes | yes | yes | yes | yes |
| `fit` (GIG) | yes | yes | — | — | — |
| `fit` (variance-gamma) | yes | no | — | — | — |

On the JAX backend, `from_expectation` has a forward derivative and a reverse derivative.
The map is the inverse Fisher matrix at the solved point.
A Hessian of Gamma `from_expectation` is finite.
The CPU backend is eager-only.

A GIG cold start accepts `jit`, `vmap`, JVP, and VJP.
A warm start does the same.
Variance-gamma uses the pytree of subordinator moments.
That path differentiates through the Gamma solve.

`renyi` differentiates in the order and in a distribution parameter.
Variance-gamma uses the joint Rényi entropy.
The marginal class has no `renyi` method.
`entropy` and `varentropy` also differentiate in the parameters.

Gamma `fit` and GIG `fit` are the moment map plus `from_expectation`.
Variance-gamma `fit` runs EM and checks the sample on the host.

`cdf` and `ppf` accuracy, including the derivative of the approximation, is in {doc}`cdf_ppf`.
