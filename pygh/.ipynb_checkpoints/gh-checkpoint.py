#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xiangshi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.special import kv

from pygh.gig import GIG
from pygh.func import logkv, kvratio


class GH(object):
    """
    Generalized Hyperbolic (GH)
    """

    def __init__(self, mu, gamma, sigma, lam, chi, psi):
        
        self.mu = mu
        self.gamma = gamma
        self.sigma = sigma
        self.lam = lam
        self.chi = chi
        self.psi = psi
        self.dim = len(mu)
        
    def pdf(self, x):
        l = np.linalg.cholesky(self.sigma)
        z = solve_triangular(l, (x-self.mu).T, lower=True)
        gamma = solve_triangular(l, self.gamma)
        lam = self.lam-self.dim/2
        chi = self.chi+np.sum(z**2, axis=0)
        psi = self.psi+np.dot(gamma, gamma)

        c = (self.psi/self.chi)**(self.lam/2)/(2*np.pi)**(self.dim/2)/ \
            np.prod(np.diag(l))/kv(self.lam, np.sqrt(self.chi*self.psi) + 0.0j).real
        p = c*kv(lam, np.sqrt(chi*psi))*np.exp(np.dot(z.T, gamma))* \
            np.sqrt(psi/chi)**(self.dim/2-self.lam);
        return p
    
    def llh(self, x):
        l = np.linalg.cholesky(self.sigma)
        z = solve_triangular(l, (x-self.mu).T, lower=True)
        gamma = solve_triangular(l, self.gamma, lower=True)
        lam = self.lam-self.dim/2
        chi = self.chi+np.sum(z**2, axis=0)
        psi = self.psi+np.dot(gamma, gamma)
        
        delta = np.sqrt(chi/psi)
        eta = np.sqrt(chi*psi)
        
        llh = self.lam/2*np.log(self.psi/self.chi) \
            - logkv(self.lam, np.sqrt(self.chi*self.psi)) \
            + logkv(lam, eta) + np.dot(z.T, gamma) + lam*np.log(delta)
        return llh, lam, delta, eta
        
    
    def mean(self):
        return self.mu+self.gamma*GIG(self.lam, self.chi, self.psi).mean()
    
    def cov(self):
        gig = GIG(self.lam, self.chi, self.psi)
        return self.sigma*gig.mean() + np.outer(self.gamma, self.gamma)*gig.var()
    
    def fit_em(self, x, max_iter=100, fix_tail=False, eps=1e-4, reg='|sigma|=1', diff=1e-5, disp=True):
        if self.dim != x.shape[1]:
            raise ValueError('x dimension must be (, {})'.format(self.dim))
            
        suff_stats = [0]*6
        suff_stats[3] = np.mean(x, axis=0)
        
        llh_last = 0
        for i in range(max_iter):
            self.regulate(reg)
            llh, lam, delta, eta = self.llh(x)
            llh = np.mean(llh)
                
            if i>0:
                if disp:
                    print('iter=%s, llh=%.5f, change=%.5f'%(i, llh, llh-llh_last))
                
                if (np.abs(llh-llh_last)/np.abs(llh_last))<eps:
                    if disp:
                        print('success')
                    return True
            llh_last = llh
        
            # E-step
            a = kvratio(lam-1, lam, eta)/delta
            b = kvratio(lam+1, lam, eta)*delta
            c = (kvratio(lam+diff, lam, eta)-kvratio(lam-diff, lam, eta))/(2*diff)+np.log(delta)

            suff_stats[0] = np.mean(a)
            suff_stats[1] = np.mean(b)
            suff_stats[2] = np.mean(c)
            suff_stats[4] = np.mean(x.T*a, axis=1)
            suff_stats[5] = np.dot((x.T*a)/x.shape[0], x)
            
            if not fix_tail:
                gig = GIG(self.lam, self.chi, self.psi)
                res = gig.suffstats2param(suff_stats[0], suff_stats[1], suff_stats[2])
                if res.success:
                    self.lam = gig.lam
                    self.chi = gig.chi
                    self.psi = gig.psi
                
            self.mu = (suff_stats[3]-suff_stats[1]*suff_stats[4])/(1-suff_stats[0]*suff_stats[1])
            self.gamma = (suff_stats[4]-suff_stats[0]*suff_stats[3])/(1-suff_stats[0]*suff_stats[1])
            self.sigma = -np.outer(suff_stats[4], self.mu)
            self.sigma = self.sigma + self.sigma.T + suff_stats[5] \
                + suff_stats[0]*np.outer(self.mu, self.mu) \
                - suff_stats[1]*np.outer(self.gamma, self.gamma)
            # self.sigma = np.maximum(self.sigma, self.sigma.T)
            
        if disp:
            print('fail to converge')
        return False
    
    
    def rvs(self, n, n_grids=int(1e5)):
        t = GIG(self.lam, self.chi, self.psi).rvs(n, n_grids)
        z = np.random.multivariate_normal(np.zeros(self.dim), self.sigma, n)
        x = (np.outer(self.gamma, t) + z.T*np.sqrt(t)).T + self.mu
        return x
    
    
    def regulate(self, method='|sigma|=1'):
        """ GH parameterization (mu, gamma Sigma, lambda, chi, psi) is not unique, so regulation is needed.
        """
        if method=='chi=1':
            self.gamma = self.gamma*self.chi
            self.sigma = self.sigma*self.chi
            self.psi = self.psi*self.chi
            self.chi = 1.0
        elif method=='psi=1':
            self.gamma = self.gamma/self.psi
            self.sigma = self.sigma/self.psi
            self.chi = self.chi*self.psi
            self.psi = 1.0
        elif method=='chi=psi':
            self.gamma = self.gamma*np.sqrt(self.chi/self.psi)
            self.sigma = self.sigma*np.sqrt(self.chi/self.psi)
            self.chi = np.sqrt(self.chi*self.psi)
            self.psi = self.chi
        elif method=='|sigma|=1':
            _, logd = np.linalg.slogdet(self.sigma)
            const = np.exp(logd/self.dim)
            self.gamma = self.gamma/const
            self.sigma = self.sigma/const
            self.chi = self.chi*const
            self.psi = self.psi/const
