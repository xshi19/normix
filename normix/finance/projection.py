r"""
Portfolio projection: bridge from multivariate normal mixtures to univariate.

For a normal mixture :math:`X \stackrel{d}{=} \mu + \gamma Y + \sqrt{Y} Z` with
:math:`Z \sim \mathcal{N}(0, \Sigma)` and weights :math:`w \in \mathbb{R}^d`,
the portfolio return is itself a univariate normal mixture:

.. math::

    w^\top X \stackrel{d}{=} w^\top \mu
        + w^\top \gamma \, Y + \sqrt{w^\top \Sigma w} \, \sqrt{Y} \, Z_1,

where :math:`Z_1 \sim \mathcal{N}(0, 1)`. :func:`project_portfolio` returns
the corresponding ``Univariate*`` distribution instance.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from normix.mixtures.marginal import NormalMixture, _UnivariateNormalMixtureMixin


def project_portfolio(model: NormalMixture, w: jax.Array) -> _UnivariateNormalMixtureMixin:
    r"""Project a multivariate normal mixture onto portfolio weights ``w``.

    Returns a ``Univariate*`` instance (e.g. :class:`~normix.UnivariateNormalInverseGaussian`)
    representing the distribution of :math:`w^\top X`.
    """
    return model.project(w)
