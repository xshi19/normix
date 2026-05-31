# Exponential-family structure

Every distribution in normix is an exponential family. This is not an
implementation detail — it is the organizing principle that gives the package
its three parametrizations, its uniform fitting interface, and its closed-form
divergences.

## The canonical form

A density in the family is

$$
p(x \mid \theta) = h(x)\, \exp\!\big(\langle \theta,\, t(x)\rangle - \psi(\theta)\big),
$$

with three ingredients each distribution must define:

- the **log base measure** $\log h(x)$ — `log_base_measure(x)`,
- the **sufficient statistics** $t(x)$ — `sufficient_statistics(x)`,
- the **log-partition** $\psi(\theta)$ — `_log_partition_from_theta(theta)`.

Everything else — moments, MLE, the EM M-step, divergences — is derived from
$\psi$.

## Three parametrizations

The same distribution can be described three equivalent ways, and normix
converts between them losslessly:

| Parametrization | Symbol | API |
|---|---|---|
| Classical | $(\alpha, \beta), \dots$ | constructor / `from_classical` |
| Natural | $\theta$ | `natural_params()` / `from_natural(theta)` |
| Expectation | $\eta = \nabla\psi(\theta) = \mathbb{E}[t(X)]$ | `expectation_params()` / `from_expectation(eta)` |

```python
theta = dist.natural_params()        # natural θ
eta = dist.expectation_params()      # expectation η = E[t(X)]
dist2 = type(dist).from_natural(theta)
dist3 = type(dist).from_expectation(eta)
```

`from_expectation` is the workhorse of the EM M-step: given any valid moment
vector $\eta$, it solves the strictly convex problem $\eta \mapsto \theta$ to
produce a distribution. The walkthrough in
{doc}`../tutorials/core/01_exponential_family` makes each of these concrete.

## The log-partition triad

Moments come from derivatives of $\psi$:

$$
\nabla\psi(\theta) = \mathbb{E}[t(X)] = \eta, \qquad
\nabla^2\psi(\theta) = \operatorname{Cov}[t(X)] = I(\theta),
$$

the second being the Fisher information. Each distribution therefore provides a
**triad** — log-partition, gradient, Hessian — in two backends:

- a **JAX** backend (`expectation_params()`, `fisher_information()`,
  default `backend="jax"`) that is JIT-able, differentiable, and `vmap`-friendly;
- a **CPU** backend (`backend="cpu"`) using numpy/scipy, used inside the EM hot
  loop where scipy's Bessel routines are fastest.

Defaults use `jax.grad` / `jax.hessian`; distributions with closed forms (e.g.
`Gamma` via `digamma`/`trigamma`) override them. The two backends are
numerically interchangeable — the choice is about performance and execution
context.

## Why it matters

- **One fitting interface.** Maximum likelihood is moment matching:
  $\hat\eta = \frac1n\sum_i t(x_i)$, then `from_expectation`. `fit_mle` is a
  one-liner that works for every family.
- **EM falls out naturally.** The E-step computes conditional moments
  $\mathbb{E}[t(Y)\mid X]$; the M-step is again `from_expectation`. See
  {doc}`em_fitting`.
- **Divergences are closed-form.** Hellinger and KL between two members reduce to
  evaluations of $\psi$ — no Monte Carlo. See {doc}`divergences`.

## Immutability

Distributions are `equinox.Module` pytrees: immutable, hashable, and traceable
by JAX. Parameter updates return a *new* instance rather than mutating in place,
which is what lets the EM loop, `jax.vmap`, and `jax.jit` treat models as plain
data.

For the full mathematical development, see the {doc}`design rationale
<../design/exponential_family>` and the {doc}`theory notes <../theory/index>`.
