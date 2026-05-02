# normix

JAX package for Generalized Hyperbolic distributions as exponential families.

Built on [Equinox](https://docs.kidger.site/equinox/) with Float64 precision throughout.

## Installation

```bash
pip install normix
```

Install optional plotting helpers with:

```bash
pip install "normix[plotting]"
```

For local development:

```bash
uv sync
pip install -e .
```

## Quick Start

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from normix import GeneralizedHyperbolic
from normix.fitting.em import BatchEMFitter

# Fit GH distribution to data via EM
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (1000, 3))

model = GeneralizedHyperbolic.from_classical(
    mu=jnp.zeros(3), gamma=jnp.zeros(3),
    sigma=jnp.eye(3), p=-0.5, a=2.0, b=1.0,
)
result = BatchEMFitter(max_iter=100).fit(model, X)

# Evaluate log-density (batched via vmap)
log_p = jax.vmap(result.model.log_prob)(X)   # shape (1000,)
```

## Distributions

### Univariate (exponential family)

| Class | Parameters | Description |
|---|---|---|
| `Gamma` | `alpha`, `beta` | Shape α > 0, rate β > 0 |
| `InverseGamma` | `alpha`, `beta` | Shape α > 0, rate β > 0 |
| `InverseGaussian` | `mu`, `lam` | Mean μ > 0, shape λ > 0 |
| `GIG` / `GeneralizedInverseGaussian` | `p`, `a`, `b` | Generalized Inverse Gaussian |

### Multivariate

| Class | Parameters | Description |
|---|---|---|
| `MultivariateNormal` | `mu`, `L_Sigma` | Mean μ, Cholesky L_Sigma of Σ |

### Normal Variance-Mean Mixtures (marginal)

| Class | Subordinator | Parameters |
|---|---|---|
| `VarianceGamma` | Gamma | `mu`, `gamma`, `L_Sigma`, `alpha`, `beta` |
| `NormalInverseGamma` | InverseGamma | `mu`, `gamma`, `L_Sigma`, `alpha`, `beta` |
| `NormalInverseGaussian` | InverseGaussian | `mu`, `gamma`, `L_Sigma`, `mu_ig`, `lam` |
| `GeneralizedHyperbolic` | GIG | `mu`, `gamma`, `L_Sigma`, `p`, `a`, `b` |

### Joint distributions

The `Joint*` classes (e.g. `JointGeneralizedHyperbolic`) model the full joint $f(x,y)$ where Y is the mixing variable. They are exponential families and are used internally for the EM E-step.

## Exponential Family API

All univariate and joint distributions subclass `ExponentialFamily(eqx.Module)`:

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from normix import Gamma

X = jnp.array([1.0, 1.5, 2.0, 2.5])
dist = Gamma(alpha=jnp.array(2.0), beta=jnp.array(1.0))

# Log-density (single observation)
dist.log_prob(jnp.array(1.5))

# Three parametrizations
theta = dist.natural_params()       # natural parameters θ
eta   = dist.expectation_params()   # expectation parameters η = E[t(X)]
I     = dist.fisher_information()   # Fisher information I(θ) = ∇²ψ(θ)

# Constructors
dist2 = Gamma.from_natural(theta)
dist3 = Gamma.from_expectation(eta)
dist4 = Gamma.fit_mle(X)           # η̂ = mean t(xᵢ)
```

## EM Algorithm

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from normix import GeneralizedHyperbolic
from normix.fitting.em import BatchEMFitter

d = 3
X = ...  # (n, d) data array

# Initialise from classical parameters
model = GeneralizedHyperbolic.from_classical(
    mu=jnp.zeros(d), gamma=jnp.zeros(d), sigma=jnp.eye(d),
    p=-0.5, a=2.0, b=1.0,
)

# Fit with hybrid CPU/JAX backend for maximum speed
fitter = BatchEMFitter(max_iter=200, tol=1e-6,
                       e_step_backend='cpu', m_step_backend='cpu')
result = fitter.fit(model, X)
fitted = result.model
```

## Bessel Functions

```python
import jax

jax.config.update("jax_enable_x64", True)

from normix import log_kv        # or: from normix.utils.bessel import log_kv

# JIT-able, differentiable (backend='jax', default)
log_kv(v=0.5, z=2.0)

# Fast CPU path for EM hot path (not JIT-able)
log_kv(v=0.5, z=2.0, backend='cpu')
```

## Package Layout

```
normix/
├── exponential_family.py         # ExponentialFamily base class
├── distributions/                # All distribution implementations
│   ├── gamma.py
│   ├── inverse_gamma.py
│   ├── inverse_gaussian.py
│   ├── generalized_inverse_gaussian.py
│   ├── normal.py
│   ├── variance_gamma.py
│   ├── normal_inverse_gamma.py
│   ├── normal_inverse_gaussian.py
│   └── generalized_hyperbolic.py
├── mixtures/                     # Joint and marginal base classes
├── fitting/em.py                 # BatchEMFitter, EMResult
└── utils/
    ├── bessel.py                 # log_kv with custom JVP
    ├── constants.py              # Shared numerical constants
    ├── plotting.py               # Notebook helpers
    └── validation.py             # EM validation helpers
```

## Development

```bash
uv run pytest tests/              # fast default suite
uv run jupyter lab                # notebooks
make -C docs html                 # build docs
```

### Test Suites

The default pytest configuration excludes tests marked `slow`, `stress`,
`integration`, or `gpu`:

```bash
uv run pytest tests/
```

Use marker expressions to run targeted suites:

```bash
uv run pytest tests/ -m "smoke or contract"
uv run pytest tests/ -m "slow or stress or integration"
uv run pytest tests/ -m gpu
uv run pytest tests/ -m "not gpu"
```

Useful profiling recipes:

```bash
uv run pytest tests/ --durations=50
JAX_LOG_COMPILES=1 uv run pytest tests/ --durations=50
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run pytest tests/ --durations=50
```

Distribution `log_prob` and `pdf` methods are single-observation APIs. Batch
them explicitly with `jax.vmap`, for example `jax.vmap(dist.pdf)(xs)`. This
keeps vector-valued observations, such as multivariate normal-mixture samples,
unambiguous while still giving JAX a stable batched computation.

## References

- Barndorff-Nielsen, O. E. (1977). Exponentially decreasing distributions for the logarithm of particle size.
- Eberlein, E., & Keller, U. (1995). Hyperbolic distributions in finance.

## License

MIT — see [LICENSE](LICENSE).
