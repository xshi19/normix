"""Linear algebra utilities for normix.

Provides numerically robust wrappers around common linear algebra operations,
particularly Cholesky decomposition with automatic regularization.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky, LinAlgError


def robust_cholesky(A: NDArray, *, eps: float = 1e-8) -> NDArray:
    r"""
    Compute lower Cholesky factor with eigenvalue-based regularization.

    Attempts :math:`L L^T = A`. If :math:`A` is not positive definite,
    computes the minimum eigenvalue and adds
    :math:`(|\lambda_{\min}| + \varepsilon) I` to guarantee positive
    definiteness.

    Since LAPACK's ``dpotrf`` only reads one triangle of the input matrix,
    explicit symmetrization is unnecessary before calling this function.

    Parameters
    ----------
    A : ndarray, shape (d, d)
        Matrix to decompose. Should be approximately symmetric positive
        definite (e.g., a covariance or precision matrix from an M-step).
    eps : float, optional
        Small positive constant added beyond :math:`|\lambda_{\min}|`
        when regularizing. Default is ``1e-8``.

    Returns
    -------
    L : ndarray, shape (d, d)
        Lower Cholesky factor satisfying :math:`L L^T \approx A`.
        If regularization was applied,
        :math:`L L^T = A + (|\lambda_{\min}| + \varepsilon) I`.

    Raises
    ------
    LinAlgError
        If decomposition fails even after eigenvalue-based regularization.

    Notes
    -----
    This function is intended for internal M-step and initialization
    computations where slight regularization is acceptable. For
    user-facing validation (e.g., checking that a user-provided
    covariance matrix is PD), use ``scipy.linalg.cholesky`` directly.

    Examples
    --------
    >>> import numpy as np
    >>> from normix.utils import robust_cholesky
    >>> A = np.array([[1.0, 0.5], [0.5, 1.0]])
    >>> L = robust_cholesky(A)
    >>> np.allclose(L @ L.T, A)
    True
    """
    try:
        return cholesky(A, lower=True)
    except LinAlgError:
        d = A.shape[0]
        min_eig = np.linalg.eigvalsh(A)[0]
        jitter = max(eps, abs(min_eig) + eps)
        return cholesky(A + jitter * np.eye(d), lower=True)
