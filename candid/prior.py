import sys

import numpy as np
from statsmodels.stats.moment_helpers import corr2cov


class GMMPrior(object):
    def __init__(self, data, ncomp, dirchlet_weights=None, wishart_dof=None, wishart_matrix=None, mean_add=None, scale=None):
        """
        Holds the conjugate prior hyperparameters for a gaussian mixture model.
        An uniformative prior has:
            all dirchlet weights = 1
            wishart dof = (ndim + 1) / 2
            scale = 0
            wishart matrix = a non-singular matrix; choice of matrix will not affect the converged result for EM methods
                             but may affect speed of convergence. A good choice would be based on Σ0/n (see below).
                             Leave as None, to allow this to be set by data
            mean = a finite vector; the choice of mean will not affect the converged result for EM methods but may
                   affect the speed of convergence
        To set a a 0-effect prior (as if it were not applied), leave all values as None
        Parameters:
        :param n_components:
        :param ndim:
        :param ncomp:
        :param dirchlet_weights: Prior for component weights, gamma, given to each alpha: D(α|γ) = bΣα^(γ-1)
        :param wishart_dof: Degrees of freedom for the wishart matrix; the least informative, proper Wishart prior is
                            obtained by setting n = (d + 1) / 2
        :param wishart_matrix: The wishart scaling matrix, The prior mean of Wp(n, W) is nW, suggesting that a reasonable
                               choice for W would be Σ0/n, where Σ0 is some prior guess for the covariance matrix
        :param mean_add: The correction to the mean of the data to get the prior mean: mean(prior) = mean(data) + mean_add
                         An uniformative prior is a vector of 0s
        :param scale: The width of the prior normal distribution for the means is Vj x η^-1
        """
        self.ndim = data.shape[1]
        self.ncomp = ncomp
        self.dirchlet_weights = dirchlet_weights or np.ones(ncomp, dtype=float)
        self.wishart_dof = wishart_dof or (self.ndim + 1) / 2.
        self.wishart_matrix = wishart_matrix or np.zeros((self.ndim, self.ndim), dtype=float)
        self.mean_add = mean_add or np.zeros((self.ndim,), dtype=float)
        self.scale = scale or 0.
        self.check_hyperparameters()


    def check_hyperparameters(self):
        assert self.dirchlet_weights.shape == (self.ncomp,)
        assert self.wishart_matrix.shape == (self.ndim, self.ndim)
        assert self.mean_add.shape == (self.ndim, )

        assert self.wishart_dof > (self.ndim - 1) / 2., "wishart degrees of freedom must be > (ndim - 1)/2"

        assert ((self.wishart_matrix == 0).all() and (self.scale == 0)) or \
               (np.linalg.cond(self.wishart_matrix) > 1 / sys.float_info.epsilon), \
            "wishart matrix must not be singular; but it can be zero if scale is also 0"
        assert np.isfinite(self.mean_add).all(), "mean must be finite"
        assert self.scale >= 0, "scale must be a number >= 0"


class UniformativeGMMPrior(GMMPrior):
    def __init__(self, data, ncomp):
        super().__init__(data, ncomp)

        assert (data.shape[1] == self.ndim) & (data.ndim == 2)
        cov = np.cov(data.T)
        stds = np.sqrt(np.diag(cov))
        corr = np.eye(self.ndim)
        guess = corr2cov(corr, stds)
        self.wishart_matrix = guess / self.wishart_dof
