import numpy as np
from scipy.special import kv, gammaln, factorial

def logkv(v, z):
    """ Log modified Bessel function of the second kind
    """
    y = np.log(kv(v, z))

    # y -> -infinity when z -> infinity
    # approximation formula: https://dlmf.nist.gov/10.40
    if np.any(y==-np.inf):
        k = np.arange(1, 10)
        akv = np.cumprod(4*v**2 - (2*k-1)**2) / factorial(k) / 8**k
        y[y==-np.inf] = -z[y==-np.inf] - 0.5*np.log(z[y==-np.inf]) + 0.5*np.log(np.pi/2) \
                        + np.log(1 + np.sum(akv/np.power.outer(z[y==-np.inf], k), axis=1))

    # y -> infinity when z -> 0
    # approximation formula: https://en.wikipedia.org/wiki/Bessel_function#Asymptotic_forms
    if np.any(y==np.inf):
        if np.abs(v) > 1e-10:
            y[y==np.inf] = gammaln(np.abs(v)) - np.log(2.) \
                            + np.abs(v) * (np.log(2.) - np.log(z[y==np.inf]))
        else:
            y[y==np.inf] = np.log(-np.log(z[y==np.inf]/2) - np.euler_gamma)

    return y

def logkvp(v, z):
    return -0.5*(np.exp(logkv(v+1, z)-logkv(v, z))+np.exp(logkv(v-1, z)-logkv(v, z)))

def kvratio(v1, v2, z):
    return np.exp(logkv(v1, z) - logkv(v2, z))

def mahalanobis(x, mu, sigma):
    """ (squared) Mahalanobis distance $(x-\mu)^T\Sigma^{-1}(x-\mu)$
        `scipy.spatial.distance.mahalanobis` cannot handle when x is a 2d matrix.
    """
    l = np.linalg.cholesky(sigma)
    z = solve_triangular(l, (x-mu).T, lower=True)
    return np.sum(z**2, axis=0)
