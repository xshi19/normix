# normix

Python package for Generalized Hyperbolic distributions and related distributions.

## Overview

`normix` provides a comprehensive, production-ready implementation of the Generalized Hyperbolic (GH) distribution family, including:

- **Univariate distributions**: Exponential, Gamma, Inverse Gamma, Generalized Inverse Gaussian (GIG), Inverse Gaussian
- **Multivariate distributions**: Multivariate Normal
- **Mixture distributions**: Generalized Hyperbolic (GH), Normal Inverse Gaussian (NIG), Variance Gamma (VG), Normal Inverse Gamma (NInvG)

All distributions are implemented as **exponential families** with support for:
- Three parametrizations: **classical**, **natural**, and **expectation** parameters
- **sklearn-style API**: `fit()` returns self, method chaining supported
- Efficient **EM algorithms** for parameter estimation
- Joint distributions $f(x,y)$ and marginal distributions $f(x)$

## Documentation

- Build locally: `make -C docs html`
- Published docs (GitHub Pages): https://xshi19.github.io/normix/

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Architecture

See [Architecture Overview](#architecture-overview) below for the complete package structure.

## Legacy Code

The original implementation has been moved to `normix/legacy/` for reference:
- `normix/legacy/gig.py` - Original GIG implementation
- `normix/legacy/gh.py` - Original GH implementation
- `normix/legacy/func.py` - Original utility functions

**Note:** The legacy code is kept for reference only during the refactoring process. Do not import from `normix.legacy` in new code. The new implementation provides better API design, numerical stability, and testing.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the detailed implementation plan.

**Current status:** Base classes and simple distributions implemented.

**Completed:**
1. ✓ Base exponential family class with three parametrizations
2. ✓ Exponential and Gamma distributions
3. ✓ Generic testing framework for scipy comparison
4. ✓ Jupyter notebooks with visualizations

**In Progress:**
- GIG and Inverse Gaussian distributions
- Mixture distributions (VG, NInvG, NIG, GH)

## Architecture Overview

```
normix/
├── normix/
│   ├── __init__.py
│   │
│   ├── base/
│   │   ├── __init__.py
│   │   ├── exponential_family.py    # Base exponential family class
│   │   └── mixture.py               # Base mixture/joint distribution
│   │
│   ├── distributions/
│   │   ├── __init__.py
│   │   ├── univariate/
│   │   │   ├── __init__.py
│   │   │   ├── exponential.py                   # Exponential (✓ implemented)
│   │   │   ├── gamma.py                         # Gamma (✓ implemented)
│   │   │   ├── inverse_gamma.py                 # InvGamma
│   │   │   ├── inverse_gaussian.py              # IG
│   │   │   └── generalized_inverse_gaussian.py  # GIG
│   │   │
│   │   ├── multivariate/
│   │   │   ├── __init__.py
│   │   │   └── normal.py            # Multivariate normal (exponential family)
│   │   │
│   │   └── mixtures/
│   │       ├── __init__.py
│   │       ├── joint_generalized_hyperbolic.py       # Joint f(x,y): normal + GIG
│   │       ├── joint_normal_inverse_gaussian.py      # Joint f(x,y): normal + IG
│   │       ├── joint_normal_inverse_gamma.py         # Joint f(x,y): normal + InvGamma
│   │       ├── joint_variance_gamma.py               # Joint f(x,y): normal + Gamma
│   │       ├── generalized_hyperbolic.py             # Marginal f(x)
│   │       ├── normal_inverse_gaussian.py            # Marginal f(x)
│   │       ├── normal_inverse_gamma.py               # Marginal f(x)
│   │       └── variance_gamma.py                     # Marginal f(x)
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── validation.py            # Input validation
│   │   ├── bessel.py                # Bessel function utilities
│   │   └── numerical.py             # Numerical stability helpers
│   │
│   └── legacy/                      # Original implementation (reference only)
│       ├── __init__.py
│       ├── gig.py
│       ├── gh.py
│       └── func.py
│
├── tests/                           # Comprehensive test suite
│   ├── __init__.py
│   ├── test_exponential_family.py       # Base class tests
│   ├── test_distributions_vs_scipy.py   # Generic scipy comparison framework
│   ├── test_gig.py
│   ├── test_gh.py
│   └── ...
│
├── docs/                            # Documentation
│   ├── conf.py
│   ├── index.rst
│   ├── api/
│   ├── tutorials/
│   └── pdfs/                        # Papers and references
│
├── examples/                        # Example scripts
│   └── (to be added)
│
├── notebooks/                       # Jupyter notebooks with visualizations
│   ├── exponential_distribution.ipynb   # Exponential demo
│   ├── gamma_distribution.ipynb         # Gamma demo
│   └── ...
│
├── setup.py
├── README.md
└── ROADMAP.md
```

## Mathematical Background

### Exponential Families

Distributions in exponential family form have the probability density:

$$p(x|\theta) = h(x) \exp(\theta^T t(x) - \psi(\theta))$$

where:
- $\theta$: natural parameters (vector)
- $t(x)$: sufficient statistics (vector)
- $\psi(\theta)$: log partition function (cumulant generating function)
- $h(x)$: base measure

**Key properties:**
- Expectation parameters: $\eta = \nabla\psi(\theta) = E[t(X)]$
- Fisher information: $I(\theta) = \nabla^2\psi(\theta) = \text{Cov}[t(X)]$
- MLE in closed form: $\hat{\eta} = \frac{1}{n}\sum_{i=1}^n t(x_i)$

### Generalized Hyperbolic as Normal Mixture

The GH distribution can be represented as:

$$X|Y \sim N(\mu + \Gamma Y, \Sigma Y)$$
$$Y \sim \text{GIG}(\lambda, \chi, \psi)$$

The marginal distribution $f(x)$ has a closed form involving modified Bessel functions of the second kind $K_\lambda(z)$.

## Contributing

Contributions are welcome! Please see the [ROADMAP.md](ROADMAP.md) for current implementation priorities.

## Testing

Run tests with:
```bash
pytest tests/
```

With coverage:
```bash
pytest tests/ --cov=normix --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## References

- Barndorff-Nielsen, O. E. (1977). Exponentially decreasing distributions for the logarithm of particle size.
- Barndorff-Nielsen, O. E., & Halgreen, C. (1977). Infinite divisibility of the hyperbolic and generalized inverse Gaussian distributions.
- Eberlein, E., & Keller, U. (1995). Hyperbolic distributions in finance.

## Citation

If you use this package in academic work, please cite:

```bibtex
@software{normix,
  title = {normix: Generalized Hyperbolic Distributions for Python},
  author = {normix developers},
  year = {2024},
  url = {https://github.com/xshi19/normix}
}
```
