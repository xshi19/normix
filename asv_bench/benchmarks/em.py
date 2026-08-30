"""E-step and fixed-iteration EM timing on synthetic 2-d mixtures.

Source: ``benchmarks/bench_em_mixture.py``, shrunk off SP500. ``EStep``
times ``e_step`` (the batched conditional-expectation hot path).
``EMIteration`` runs five EM steps from the parent moment init — not
``GH.default_init``, which itself runs three nested 5-iter fits.
``tol=0`` so the five steps always run; family ``_fit_defaults`` still
apply (CPU E-step on VG/GH, ``|Σ|=1`` on GH).
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from benchmarks import block_pytree, require_requested_device
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from normix.distributions.generalized_hyperbolic import GeneralizedHyperbolic
from normix.distributions.normal_inverse_gaussian import NormalInverseGaussian
from normix.distributions.variance_gamma import VarianceGamma
from normix.utils.constants import SIGMA_INIT_REG

_DTYPE = jnp.float64
_MU = jnp.zeros(2, dtype=_DTYPE)
_GAMMA = jnp.array([0.3, -0.2], dtype=_DTYPE)
_SIGMA = jnp.eye(2, dtype=_DTYPE)

_DISTS = {
    "VG": lambda: VarianceGamma.from_classical(
        mu=_MU, gamma=_GAMMA, sigma=_SIGMA, alpha=2.5, beta=1.5
    ),
    "NIG": lambda: NormalInverseGaussian.from_classical(
        mu=_MU, gamma=_GAMMA, sigma=_SIGMA, mu_ig=1.0, lam=1.5
    ),
    "GH": lambda: GeneralizedHyperbolic.from_classical(
        mu=_MU, gamma=_GAMMA, sigma=_SIGMA, p=-0.5, a=1.5, b=1.0
    ),
}

_ITERS = 5
_EM_N = 1000
_EM_SEED = 0

# ASV's default repeat fills ~20 s per param even for µs kernels. Cap so
# the new classes stay inside the ~10 min/commit budget (cpu+cuda).
_REPEAT = (1, 3, 3.0)


def _moment_init(cls, X: jax.Array):
    """Parent moment init — skips GH's nested special-case EM fits."""
    n, d = X.shape
    mu = jnp.mean(X, axis=0)
    xc = X - mu
    sigma = (xc.T @ xc) / n + SIGMA_INIT_REG * jnp.eye(d, dtype=_DTYPE)
    return cls._from_init_params(mu, jnp.zeros(d, dtype=_DTYPE), sigma)


class EStep:
    """Batched ``e_step``: VG/NIG/GH × jax/cpu × N ∈ {1000, 10000}."""

    params = [list(_DISTS), ["jax", "cpu"], [1000, 10000]]
    param_names = ["dist", "backend", "n"]
    timeout = 180.0
    warmup_time = 0.0
    rounds = 1
    repeat = _REPEAT

    def setup(self, dist: str, backend: str, n: int) -> None:
        require_requested_device()
        model = _DISTS[dist]()
        X = model.rvs(int(n), seed=_EM_SEED)
        block_pytree(X)
        self.model = model
        self.X = X
        eta = model.e_step(X, backend=backend)
        block_pytree(eta)

    def time_conditional_expectations(
        self, dist: str, backend: str, n: int
    ) -> None:
        eta = self.model.e_step(self.X, backend=backend)
        block_pytree(eta)


class EMIteration:
    """Five EM iterations from moment init; family ``_fit_defaults`` apply."""

    params = [list(_DISTS)]
    param_names = ["dist"]
    timeout = 180.0
    warmup_time = 0.0
    rounds = 1
    repeat = _REPEAT

    def setup(self, dist: str) -> None:
        require_requested_device()
        truth = _DISTS[dist]()
        X = truth.rvs(_EM_N, seed=_EM_SEED)
        block_pytree(X)
        init = _moment_init(type(truth), X)
        self.init = init
        self.X = X
        result = init.fit(X, max_iter=_ITERS, tol=0.0)
        block_pytree(result.model)

    def time_fit(self, dist: str) -> None:
        result = self.init.fit(self.X, max_iter=_ITERS, tol=0.0)
        block_pytree(result.model)
