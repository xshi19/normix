"""
Base class for probability distributions with scipy-like API.

This module provides an abstract base class that defines the standard interface
for probability distributions, similar to ``scipy.stats``.

The API includes:

- **Density functions**: :meth:`pdf`, :meth:`logpdf`
- **Cumulative distribution**: :meth:`cdf`, :meth:`sf` (survival function)
- **Quantile functions**: :meth:`ppf`, :meth:`isf` (inverse survival)
- **Random sampling**: :meth:`rvs`
- **Fitting**: :meth:`fit` (returns self for method chaining)
- **Moments**: :meth:`mean`, :meth:`var`, :meth:`std`, :meth:`stats`
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import numpy as np
from numpy.typing import ArrayLike, NDArray


class Distribution(ABC):
    """
    Abstract base class for probability distributions.
    
    This class defines the standard API for probability distributions,
    similar to scipy.stats distributions. All concrete distributions
    should inherit from this class.
    
    The API includes:
    - pdf/pmf: Probability density/mass function
    - logpdf/logpmf: Log of the probability density/mass function
    - cdf: Cumulative distribution function
    - logcdf: Log of the cumulative distribution function
    - sf: Survival function (1 - CDF)
    - logsf: Log of the survival function
    - ppf: Percent point function (inverse of CDF)
    - isf: Inverse survival function
    - rvs: Random variate sampling
    - fit: Fit distribution parameters to data
    - stats: Return moments (mean, variance, skewness, kurtosis)
    - entropy: Differential entropy of the distribution
    - moment: Non-central moment of order n
    - median: Median of the distribution
    - mean: Mean of the distribution
    - var: Variance of the distribution
    - std: Standard deviation of the distribution
    - interval: Confidence interval
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the distribution.
        
        Parameters should be validated using Pydantic in subclasses.
        """
        pass
    
    @abstractmethod
    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """
        Probability density function.
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF.
            
        Returns
        -------
        pdf : ndarray
            Probability density at each point.
        """
        pass
    
    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """
        Log of the probability density function.
        
        Default implementation: log(pdf(x))
        Subclasses should override for numerical stability.
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log PDF.
            
        Returns
        -------
        logpdf : ndarray
            Log probability density at each point.
        """
        return np.log(self.pdf(x))
    
    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """
        Cumulative distribution function.
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the CDF.
            
        Returns
        -------
        cdf : ndarray
            Cumulative probability at each point.
        """
        raise NotImplementedError("CDF not implemented for this distribution")
    
    def logcdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """
        Log of the cumulative distribution function.
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log CDF.
            
        Returns
        -------
        logcdf : ndarray
            Log cumulative probability at each point.
        """
        return np.log(self.cdf(x))
    
    def sf(self, x: ArrayLike) -> NDArray[np.floating]:
        """
        Survival function (1 - CDF).
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the survival function.
            
        Returns
        -------
        sf : ndarray
            Survival probability at each point.
        """
        return 1.0 - self.cdf(x)
    
    def logsf(self, x: ArrayLike) -> NDArray[np.floating]:
        """
        Log of the survival function.
        
        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log survival function.
            
        Returns
        -------
        logsf : ndarray
            Log survival probability at each point.
        """
        return np.log(self.sf(x))
    
    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """
        Percent point function (inverse of CDF).
        
        Parameters
        ----------
        q : array_like
            Probabilities at which to evaluate the PPF.
            
        Returns
        -------
        ppf : ndarray
            Quantiles corresponding to the given probabilities.
        """
        raise NotImplementedError("PPF not implemented for this distribution")
    
    def isf(self, q: ArrayLike) -> NDArray[np.floating]:
        """
        Inverse survival function (inverse of SF).
        
        Parameters
        ----------
        q : array_like
            Probabilities at which to evaluate the ISF.
            
        Returns
        -------
        isf : ndarray
            Quantiles corresponding to the given survival probabilities.
        """
        return self.ppf(1.0 - q)
    
    @abstractmethod
    def rvs(self, size: Optional[Union[int, tuple]] = None, 
            random_state: Optional[Union[int, np.random.Generator]] = None) -> NDArray:
        """
        Random variate sampling.
        
        Parameters
        ----------
        size : int or tuple of ints, optional
            Shape of the output. If None, returns a scalar.
        random_state : int or numpy.random.Generator, optional
            Random state for reproducibility.
            
        Returns
        -------
        rvs : ndarray or scalar
            Random variates.
        """
        pass
    
    @abstractmethod
    def fit(self, data: ArrayLike, *args, **kwargs) -> 'Distribution':
        """
        Fit distribution parameters to data (sklearn-style).
        
        This method should fit the distribution parameters to the given data
        and return self for method chaining.
        
        Parameters
        ----------
        data : array_like
            Data to fit the distribution to.
        *args, **kwargs
            Additional parameters for fitting.
            
        Returns
        -------
        self : Distribution
            The fitted distribution instance (for method chaining).
        """
        pass
    
    def stats(self, moments: str = 'mv') -> Union[NDArray, tuple[NDArray, ...]]:
        """
        Return moments of the distribution.
        
        Parameters
        ----------
        moments : str, optional
            Composed of letters ['mvsk'] defining which moments to compute:
            'm' = mean, 'v' = variance, 's' = skewness, 'k' = kurtosis.
            Default is 'mv'.
            
        Returns
        -------
        stats : ndarray or tuple
            Requested moments.
        """
        results = []
        if 'm' in moments:
            results.append(self.mean())
        if 'v' in moments:
            results.append(self.var())
        if 's' in moments:
            results.append(self.skewness())
        if 'k' in moments:
            results.append(self.kurtosis())
        
        if len(results) == 1:
            return results[0]
        return tuple(results)
    
    def mean(self) -> NDArray[np.floating]:
        """
        Mean of the distribution.
        
        Returns
        -------
        mean : ndarray
            Mean value(s).
        """
        raise NotImplementedError("Mean not implemented for this distribution")
    
    def var(self) -> NDArray[np.floating]:
        """
        Variance of the distribution.
        
        Returns
        -------
        var : ndarray
            Variance value(s).
        """
        raise NotImplementedError("Variance not implemented for this distribution")
    
    def std(self) -> NDArray[np.floating]:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        std : ndarray
            Standard deviation value(s).
        """
        return np.sqrt(self.var())
    
    def skewness(self) -> NDArray[np.floating]:
        """
        Skewness of the distribution.
        
        Returns
        -------
        skewness : ndarray
            Skewness value(s).
        """
        raise NotImplementedError("Skewness not implemented for this distribution")
    
    def kurtosis(self) -> NDArray[np.floating]:
        """
        Kurtosis of the distribution.
        
        Returns
        -------
        kurtosis : ndarray
            Kurtosis value(s).
        """
        raise NotImplementedError("Kurtosis not implemented for this distribution")
    
    def median(self) -> NDArray[np.floating]:
        """
        Median of the distribution.
        
        Returns
        -------
        median : ndarray
            Median value(s).
        """
        return self.ppf(0.5)
    
    def entropy(self) -> float:
        """
        Differential entropy of the distribution.
        
        Returns
        -------
        entropy : float
            Entropy value.
        """
        raise NotImplementedError("Entropy not implemented for this distribution")
    
    def moment(self, n: int) -> NDArray[np.floating]:
        """
        Non-central moment of order n.
        
        Parameters
        ----------
        n : int
            Order of the moment.
            
        Returns
        -------
        moment : ndarray
            n-th moment.
        """
        raise NotImplementedError("Moment not implemented for this distribution")
    
    def interval(self, alpha: float) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Confidence interval with equal areas around the median.
        
        Parameters
        ----------
        alpha : float
            Confidence level (between 0 and 1).
            
        Returns
        -------
        a, b : tuple of ndarrays
            Lower and upper bounds of the confidence interval.
        """
        lower = self.ppf((1 - alpha) / 2)
        upper = self.ppf((1 + alpha) / 2)
        return lower, upper
    
    def score(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> float:
        """
        Compute the mean log-likelihood (sklearn-style scoring).
        
        This method follows the sklearn convention where higher scores are better.
        
        Parameters
        ----------
        X : array_like
            Data samples.
        y : array_like, optional
            Ignored. Present for sklearn API compatibility.
            
        Returns
        -------
        score : float
            Mean log-likelihood.
        """
        X = np.asarray(X)
        return np.mean(self.logpdf(X))
    
    def __repr__(self) -> str:
        """String representation of the distribution."""
        return f"{self.__class__.__name__}()"

