#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: xiangshi
"""

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import kv

from .gig import GIG
from .func import logkv


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
        gamma = solve_triangular(l, self.gamma);
        lam = self.lam-self.dim/2;
        chi = self.chi+np.sum(z**2, axis=0);
        psi = self.psi+np.dot(gamma, gamma);

        c = (self.psi/self.chi)**(self.lam/2)/(2*np.pi)**(self.dim/2)/ \
            np.prod(np.diag(l))/kv(self.lam, np.sqrt(self.chi*self.psi) + 0.0j).real
        p = c*kv(lam, np.sqrt(chi*psi))*np.exp(np.dot(z.T, gamma))* \
            np.sqrt(psi/chi)**(self.dim/2-self.lam);

        return p
