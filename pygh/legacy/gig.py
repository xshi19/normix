#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xiangshi
"""

import numpy as np
from math import sqrt
from scipy.special import kv
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from .func import *


def gig_hellinger(gig1, gig2):
    lam = (gig1.lam+gig2.lam)/2
    chi = (gig1.chi+gig2.chi)/2
    psi = (gig1.psi+gig2.psi)/2
    c1 = ((gig1.psi/gig1.chi1)**(gig1.lam/2))/kv(gig1.lam, np.sqrt(gig1.chi*gig1.psi))
    c2 = ((gig2.psi/gig2.chi1)**(gig2.lam/2))/kv(gig2.lam, np.sqrt(gig2.chi*gig2.psi))
    c = (psi/chi)^(lam/2)/kv(lam, np.sqrt(chi*psi));
    d = 1. - np.sqrt(c1*c2)/c;

class GIG(object):
    """
    Generalized Inverse Gaussian (GIG)
    """

    def __init__(self, lam, chi, psi):
        self.lam = lam
        self.chi = chi
        self.psi = psi

    def pdf(self, x):
        c = (self.psi/self.chi)**(self.lam/2.0) / 2.0 / \
            kv(self.lam, sqrt(self.chi*self.psi) + 0.0j).real
        p = c*x**(self.lam-1.0)*np.exp(-0.5*(self.chi*x**(-1.0)+self.psi*x))
        p[x <= 0] = 0
        return p

    def logpdf(self, x):
        logc = self.lam/2.0 * (np.log(self.psi) - np.log(self.chi)) \
                - np.log(2.0) - logkv(self.lam, sqrt(self.chi*self.psi))
        logp = logc + (self.lam-1.0)*np.log(x) - 0.5*(self.chi*x**(-1.0)+self.psi*x)
        return logp

    def cdf(self, x=None, n_grids=int(1e5)):
        mu = self.moment(1)
        z = np.r_[np.linspace(0., mu, n_grids+1)[1:],
                np.linspace(0., 1./mu, n_grids+1)[1:][::-1]**(-1)]
        p = self.pdf(z)
        p[0] = 0
        p[p<0] = 0
        y = np.cumsum(p)
        y = y/y[-1]
        y[y>1] = 1
        y[y<0] = 0
        y, idx = np.unique(y, return_index=True)
        z = z[idx]
        cdf = interp1d(z, y, kind='linear')
        if x is None:
            return cdf
        else:
            return cdf(x)

    def rvs(self, n, n_grids=int(1e5)):
        mu = self.moment(1)
        z = np.r_[np.linspace(0., mu, n_grids+1)[1:],
                np.linspace(0., 1./mu, n_grids+1)[1:][::-1]**(-1)]
        dz = np.diff(z)
        z = z[1:] - dz/2
        p = self.pdf(z)
        p[0] = 0
        p[p<0] = 0
        y = np.cumsum(p*dz)
        y = y/y[-1]
        y[y>1] = 1
        y[y<0] = 0
        y, idx = np.unique(y, return_index=True)
        z = z[idx]
        invcdf = interp1d(y, z, kind='linear')
        u = np.random.rand(n)
        x = invcdf(u)
        return x

    def moment(self, alpha):
        delta = np.sqrt(self.chi/self.psi)
        eta = np.sqrt(self.chi*self.psi)
        m = delta**alpha*np.exp(logkv(self.lam+alpha, eta)-logkv(self.lam, eta))
        return m

    def mean(self):
        return self.moment(1)

    def var(self):
        return self.moment(2)-self.moment(1)**2

    def suffstats2param(self, s1, s2, s3):
        def llh(param):
            lam = param[0]
            chi = param[1]
            psi = param[2]
            eta = np.array([np.sqrt(chi*psi)])
            delta = np.sqrt(chi/psi)
            l = 0.5*(chi*s1+psi*s2)-(lam-1)*s3+lam*np.log(delta)+logkv(lam, eta)
            return l[0]

        def grad(param):
            lam = param[0]
            chi = param[1]
            psi = param[2]
            eta = np.array([np.sqrt(chi*psi)])
            delta = np.sqrt(chi/psi)
            g = np.array([(-s3+np.log(delta)+(logkv(lam+1e-10, eta)[0]-logkv(lam-1e-10, eta)[0])/2e-10),
                          0.5*(s1+lam/chi+logkvp(lam, eta)[0]/delta),
                          0.5*(s2-lam/psi+logkvp(lam, eta)[0]*delta)])
            return g

        x0 = np.array([self.lam, self.chi, self.psi])
        bounds = [(-20., 20.), (1e-20, None), (1e-20, None)]
        res = minimize(fun=llh, x0=x0, jac=grad, bounds=bounds)

        if res.success:
            self.lam = res.x[0]
            self.chi = res.x[1]
            self.psi = res.x[2]
        return res

    def suffstats(self):
        s1 = self.moment(-1.)
        s2 = self.moment(1.)
        s3 = (self.moment(1e-10)-self.moment(-1e-10))/2e-10
        return s1, s2, s3

    def fit(self, x):
        s1_hat = np.mean(x**(-1))
        s2_hat = np.mean(x)
        s3_hat = np.mean(np.log(x))

        return self.suffstats2param(s1_hat, s2_hat, s3_hat)

    