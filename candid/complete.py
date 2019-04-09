import warnings

import logging
import numpy as np
import theano
from scipy.stats import multivariate_normal
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture.gaussian_mixture import GaussianMixture as GMM
from statsmodels.stats.moment_helpers import corr2cov
from astroML.density_estimation.xdeconv import XDGMM as astroml_XDGMM
import theano.tensor as tt
from pymc3.distributions.dist_math import Cholesky
from tqdm import tqdm, trange
import inspect

from .backend import Backend, BackendBase, MultiBackend
from .prior import UniformativeGMMPrior
from . import compat


cholesky = Cholesky(nofail=True, lower=True)


def norm_covariance(x):
    return tt.tril(x) + tt.tril(x, -1).T


class CompleteXDGMMBase(astroml_XDGMM):
    engine = np
    tracked_names = ['loglike', 'mu', 'V', 'alpha']
    copy_names = ['n_components', 'ndim', 'labels', 'verbose', 'prior']


    def __init__(self, n_components, ndim, labels=None, verbose=False, prior=None, backend=None, debug=True):
        warnings.warn("""This local version of candid will be superseded by the version hosted at philastrophist/candid. 
        As such, edits to this module are not advised""")
        super().__init__(n_components, None, None, verbose, None)
        self.mu = np.ones((n_components, ndim), dtype=float)
        self.V = np.ones((n_components, ndim, ndim), dtype=float)
        self.alpha = np.ones((n_components, ), dtype=float) / n_components
        self.loglike = -np.inf

        self.ndim = ndim
        self.labels = labels or list(range(ndim))
        assert len(self.labels) == self.ndim
        self._shape_err = "Data array must be of shape (npoints, ndim={})".format(self.ndim)
        self.logsumexp = compat.logsumexp[self.engine]
        self.log_multivariate_gaussian = compat.log_multivariate_gaussian[self.engine]
        self.batched_product = compat.batched_product_3d[self.engine]
        self.linalg = compat.nlinalg[self.engine]
        self.stacked_svd = compat.stacked_svd[self.engine]
        self.asarray = compat.asarray[self.engine]

        self.debug = debug

        self.prior = prior
        backend = backend or Backend('XD')
        self.backend = Backend('XD', backend) if isinstance(backend, str) else backend
        if not self.backend.is_setup:
            self.loglike = -np.inf
            self.iteration = -1
        else:
            assert n_components == self.backend.mu.shape[1]
            assert ndim == self.backend.mu.shape[2]
            logging.info("Loading previous chain from backend at iteration {}".format(self.backend.iteration))
            self.load(-1)


    def marginalise(self, *dimensions):
        """returns new XDGMM with the dimensions given as input dropped from the distribution"""
        numbered_dimensions = [self.labels.index(d) if isinstance(d, str) else int(d) for d in dimensions]
        keep = [i for i in range(len(self.labels)) if i not in numbered_dimensions]
        return self.__getitem__(keep)


    def __getitem__(self, dimensions):
        numbered_dimensions = [self.labels.index(d) if isinstance(d, str) else int(d) for d in dimensions]
        new = self.copy()
        new.ndim = len(dimensions)
        new.labels = [l for i, l in enumerate(self.labels) if i in numbered_dimensions]
        for k, v in new.get_values().items():
            if 'V' in k:
                new.V = self.V[(slice(None),) + np.ix_(numbered_dimensions, numbered_dimensions)]
            if 'mu' in k:
                new.mu = self.mu[:, numbered_dimensions]
        return new


    def condition(self, X=None, Xerr=None, **kwargs):
        """Condition the model based on known values for some
        features.

        Parameters
        ----------
        X_input : array_like (optional), shape = (n_features, )
            An array of input values. Inputs set to NaN are not set, and
            become features to the resulting distribution. Order is
            preserved. Either an X array or an X dictionary is required
            (default=None).
        Returns
        -------
        cond_xdgmm: XDGMM object
            n_features = self.n_features-(n_features_conditioned)
            n_components = self.n_components
        https://github.com/tholoien/XDGMM/blob/master/xdgmm/xdgmm.py
        """
        if X is not None:
            assert len(kwargs) == 0, "You can only specify X=[x0, x1, ...] and Xerr=[xerr0, ...] without keyword args (dim1=[0,1])"
        X = X or [np.nan] * self.ndim
        Xerr = Xerr or [0.] * self.ndim
        X = [np.nan if x is None else x for x in X]
        Xerr = [0. if xerr is None else xerr for xerr in Xerr]

        for k, v in kwargs.items():
            i = self.labels.index(k)
            try:
                if len(v) == 2:
                    X[i], Xerr[i] = v
                elif len(v) == 1:
                    X[i] = v[0]
                elif len(v) > 2:
                    raise ValueError("Unknown specification, specify only the value and optionally its error for each dimension:"
                                     " dim1=[0, 1] or dim1=0")
            except TypeError:
                X[i] = v

        X = np.asarray(X)
        assert len(X) == self.ndim
        if np.isnan(X).all():
            return self

        new_mu = []
        new_V = []
        pk = []

        not_set_idx = np.nonzero(np.isnan(X))[0]
        set_idx = np.nonzero(True ^ np.isnan(X))[0]
        x = X[set_idx]
        covars = np.copy(self.V)

        if Xerr is not None:
            for i in set_idx:
                covars[:, i, i] += Xerr[i]


        for i in range(self.n_components):
            a = []
            a_ind = []
            A = []
            b = []
            B = []
            C = []

            for j in range(len(self.mu[i])):
                if j in not_set_idx:
                    a.append(self.mu[i][j])
                else:
                    b.append(self.mu[i][j])

            for j in not_set_idx:
                tmp = []
                for k in not_set_idx:
                    tmp.append(covars[i][j, k])
                A.append(np.array(tmp))

                tmp = []
                for k in set_idx:
                    tmp.append(covars[i][j, k])
                C.append(np.array(tmp))

            for j in set_idx:
                tmp = []
                for k in set_idx:
                    tmp.append(covars[i][j, k])
                B.append(np.array(tmp))

            a = np.array(a)
            b = np.array(b)
            A = np.array(A)
            B = np.array(B)
            C = np.array(C)

            mu_cond = a + np.dot(C, np.dot(np.linalg.inv(B), (x - b)))
            V_cond = A - np.dot(C, np.dot(np.linalg.inv(B), C.T))

            new_mu.append(mu_cond)
            new_V.append(V_cond)

            pk.append(multivariate_normal.pdf(x, mean=b, cov=B,
                                              allow_singular=True))

        new_mu = np.array(new_mu)
        new_V = np.array(new_V)
        pk = np.array(pk).flatten()
        new_weights = self.alpha * pk
        new_weights = new_weights / np.sum(new_weights)

        new_xdgmm = self.copy()
        new_xdgmm.ndim = new_mu.shape[1]
        new_xdgmm.V = new_V
        new_xdgmm.mu = new_mu
        new_xdgmm.alpha = new_weights
        new_xdgmm.labels = [l for l, n in zip(self.labels, np.isnan(X)) if n]
        return new_xdgmm


    def dump_states(self, backend):
        values = self.get_values()
        backend.setup(3, **values)
        backend.save(**self.backend.current[-3])
        backend.save(**self.backend.current[-2])
        backend.save(**self.backend.current[-1])


    @classmethod
    def from_backend(cls, backend, **kwargs):
        """
        :param backend: fname or backend object
        :return: instance of this XDGMM object
        """
        backend = Backend('XD', backend) if isinstance(backend, str) else backend
        gmm = cls(backend.mu.shape[1], backend.mu.shape[2], backend=backend, **kwargs)
        gmm.revert()
        return gmm


    @classmethod
    def from_combination(cls, data, *others):
        ncomponents = sum(o.n_components for o in others)
        new = cls(n_components=ncomponents, **{name: getattr(others[0], name) for name in cls.copy_names if name != 'n_components'})

        weights = np.concatenate([np.exp(o.logprob_k(data)) for o in others])
        values = {}
        for k in cls.tracked_names:
            array = [getattr(o, k) for o in others]
            try:
                values[k] =  np.concatenate(array)
            except ValueError:  # scalar values such as loglike or total number, etc
                values[k] = np.exp(np.average(np.log(array), weights=[o.loglike for o in others]))

        new.set_values(**values)

        new.alpha *= weights
        new.alpha /= new.alpha.sum()
        new.save(1)
        return new


    def load(self, index, chain_name=None):
        backend = self.backend.current if chain_name is None else self.backend.store[chain_name]
        for k, v in backend[index].items():
            setattr(self, k, v)
        self.iteration = index


    def initialise_prior(self, X):
        if self.prior is None:
            self.prior = UniformativeGMMPrior(X, self.n_components)


    def check_data(self, X):
        X = np.asarray(X)
        assert X.ndim == 2, self._shape_err
        assert X.shape[-1] == self.ndim, self._shape_err
        return X


    def initialise(self, X, n_iter=1000):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gmm = GMM(self.n_components, max_iter=n_iter, covariance_type='full',
                      random_state=self.random_state).fit(self.check_data(X))
        self.set_values(-np.inf, gmm.means_, gmm.covariances_, gmm.weights_)


    def copy(self):
        xdgmm = self.__class__(**{k: getattr(self, k) for k in self.copy_names})
        xdgmm.set_values(**self.get_values())
        return xdgmm


    @property
    def components(self):
        cs = []
        for i in range(self.n_components):
            xdgmm = self.copy()
            xdgmm.n_components = 1
            xdgmm.set_values(-np.inf, self.mu[[i]], self.V[[i]], self.alpha[[i]])
            cs.append(xdgmm)
        return cs


    def get_values(self):
        return {k: getattr(self, k) for k in self.tracked_names}


    def set_values(self, *args, **kwargs):
        for name, arg in zip(self.tracked_names, args):
            kwargs[name] = arg
        for name in self.tracked_names:
            try:
                value = kwargs[name]
                if 'alpha' in name:
                    value /= value.sum()
                setattr(self, name, value)
            except KeyError:
                pass


    def save(self, extend=None):
        values = {name: getattr(self, name) for name in self.tracked_names}
        if not self.backend.current.is_setup:
            self.backend.current.setup(extend, **values)
        elif extend is not None:
            self.backend.current.grow(extend)
        self.backend.current.save(**values)


    def check_values(self):
        infs = {name: np.where(np.isfinite(v)) for name, v in self.get_values().items()}
        assert all(np.all(infs.values())), "non-finite values in {}".format(infs)
        for name in self.tracked_names:
            value = getattr(self, name)
            if 'mu' in name:
                assert value.shape == (self.n_components, self.ndim), "means have shape (K, D) in {}".format(name)
            if 'V' in name:
                assert value.shape == (self.n_components, self.ndim, self.ndim), "covariances must have shape (K, D, D) in {}".format(name)
                assert all(np.all(np.diag(v) > 0) for v in value), "Covariance matrix has gone negative in {}".format(name)
            if 'alpha' in name:
                assert value.shape == (self.n_components, ), "weights must have shape (K,) in {}".format(name)
                assert np.allclose(sum(value), 1), "alpha must sum to 1 in {}".format(name)


    def converged(self, tol, i):
        if i > 5 and tol is not None:
            previous, current = self.backend.loglike[-2:]
            return current <= previous + tol
        return False


    def revert(self):
        self.load(int(np.argmax(self.backend.loglike)))


    def step(self, X, max_iter, tol, fix=None, fix_mu=False, fix_V=False, fix_alpha=False, random_state=None):
        fix = fix or []
        fix = tuple(np.asarray(fix).tolist())
        self.random_state = random_state
        X = self.check_data(X)
        self.initialise_prior(X)
        self.save(max_iter+1)

        previous_loglike = self.backend['loglike', -1]
        bar = tqdm(range(max_iter), disable=(not self.verbose))
        for i in bar:
            try:
                values = self.EMstep(X, fix_mu, fix_V, fix_alpha)
                if len(fix):
                    for i, k in enumerate(self.tracked_names):
                        if ('V' in k) or ('mu' in k) or ('alpha' in k):
                            values[i][[fix]] = getattr(self, k)[[fix]]
                self.set_values(*values)
                self.save()
                self.check_values()
            except (ValueError, AssertionError, np.linalg.linalg.LinAlgError) as e:
                if self.debug:
                    if isinstance(self.debug, MultiBackend):
                        backend = self.debug
                    else:
                        backend = Backend('XD', 'debug.h5')
                    self.dump_states(backend)
                raise e
            except KeyboardInterrupt:
                break
            bar.desc = "dlog(L) = {:+.2e}".format(self.loglike - previous_loglike)
            yield self.loglike
            if self.converged(tol, i):
                break
            previous_loglike = self.loglike


    def fit_once(self, X, max_iter=1000, tol=1e-5, reinitialise=True, fix=None, fix_mu=False, fix_V=False,
                 fix_alpha=False, random_state=None, revert=True):
        fix = fix or []
        if reinitialise and (not len(self.backend.current)):
            self.initialise(X)
        for _ in self.step(X, max_iter=max_iter, tol=tol, fix=fix, fix_mu=fix_mu, fix_V=fix_V, fix_alpha=fix_alpha,
                           random_state=random_state):
            pass
        if revert:
            self.revert()


    def fit(self, X, max_iter=1000, tol=1e-5, reinitialise=True, fix=None, split_merge_levels=0, split_merge_attempts=1,
            fix_mu=False, fix_V=False, fix_alpha=False, random_state=None, revert=True):
        fix = fix or tuple()
        self.fit_once(X, max_iter=max_iter, tol=tol, reinitialise=reinitialise, fix=fix,
                 fix_mu=fix_mu, fix_V=fix_V, fix_alpha=fix_alpha, random_state=random_state, revert=revert)
        self.backend.crop()

        if (self.n_components - len(fix) >= 0) and (split_merge_levels > 0) and (split_merge_attempts > 0):
            for level in trange(split_merge_levels, desc='SM level', disable=(not self.verbose)):
                candidates = self.split_merge_candidates(X, fix)[:split_merge_attempts]
                previous = self.backend.current_name
                bar = tqdm(candidates, desc='SM attempt', disable=(not self.verbose)) if len(candidates) > 1 else candidates
                for i, (split, merge) in enumerate(bar):
                    new = self.split(X, split).merge(X, *merge)
                    self.set_values(**new.get_values())
                    self.loglike = self.logL(X)
                    self.backend.branch_chain(max_iter*2, 'SM-{}: ({}|{},{})'.format(level, split, *merge))
                    logging.info("Split&Merge level {}, attempt {}: split {}, merge {}&{}".format(level, i, split, *merge))
                    if self.n_components - len(fix) > 3:
                        logging.info("Partial fitting of unaffected components")
                        self.fit_once(X, max_iter=max_iter, tol=tol, reinitialise=False, fix=fix+(split,)+merge,
                                fix_mu=fix_mu, fix_V=fix_V, fix_alpha=fix_alpha, random_state=random_state, revert=False)
                    logging.info("Full fitting of all components")
                    self.fit_once(X, max_iter=max_iter, tol=tol, reinitialise=False, fix=fix,
                             fix_mu=fix_mu, fix_V=fix_V, fix_alpha=fix_alpha, random_state=random_state, revert=revert)
                    self.backend.crop()
                    if self.better_logL(self.backend.current_name, previous):
                        self.backend.merge_chain(previous)
                        break
                    else:
                        self.backend.switch_chain(previous)
                else:
                    msg = "LogL has not increased over {} attempts ({} < {}), terminating hierarchy traversal and reverting to {}"
                    logging.info(msg.format(split_merge_attempts, self.backend.current.loglike.max(),
                                            self.backend.store[previous].loglike.max(), previous))
                    self.backend.switch_chain(previous)
                    self.revert()


    def better_logL(self, x, y):
        """
        :param x: backend chain name
        :param y: backend chain name
        :return: is x a better result than y [bool]
        """
        lx, ly = self.backend.store[x].loglike.max(), self.backend.store[y].loglike.max()
        better = lx > ly
        if better:
            logging.info("LogL increased {} > {}, terminating Split&Merge horizontal traversal".format(lx, ly))
        else:
            logging.info("LogL did not increase {} <= {}, continuing...".format(lx, ly))
        return better


    def _merge_candidates(self, X):
        # compute log_q (posterior for k given i), but use normalized probabilities
        # to allow for merging of empty components
        # log[p(x|k)] - log[sum_k p(x|k)]
        lnrik_a = self._lnresponsibilities(X) - self.engine.log(self.alpha[None, :])
        rik_a = self.engine.exp(lnrik_a)
        JM = self.engine.tril(self.engine.dot(rik_a.T, rik_a), -1)
        indices = self.asarray(np.tril_indices(self.n_components, -1))
        jm = JM[indices[0], indices[1]].argsort()[::-1]
        return self.engine.stack([indices[0][jm], indices[1][jm]]).T


    def _split_candidates(self, X):
        EV = self.stacked_svd(self.V)
        JS = EV[:, 0] * self.alpha
        return self.engine.argsort(JS)[::-1]  # return all candidates


    def split_merge_candidates(self, X, fix=None):
        fix = fix or []
        merges = self.merge_candidates(X)
        merges = list(map(tuple, [merge if not any(f in merge for f in fix) else [] for merge in merges]))
        splits = self.split_candidates(X)
        splits = [split for split in splits if split not in fix]
        splits = [[i for i in splits if i not in merge] for merge in merges]
        candidates = [(split[0], merge) for split, merge in zip(splits, merges) if len(split) > 0 and len(merge) > 0]
        return candidates


    def split(self, X, i):
        """Split component i according to Melchior+18, ie following SVD method in Zhang 2003, with alpha=1/2, u = 1/4"""
        new_alpha = self.alpha[i] / 2
        _, radius2, rotation = np.linalg.svd(self.V[i])
        dl = np.sqrt(radius2[0]) * rotation[0] / 4
        new_mu1 = self.mu[i] - dl
        new_mu2 = self.mu[i] + dl
        new_V = np.linalg.det(self.V[i]) ** (1 / self.ndim) * np.eye(self.ndim)

        new = self.copy()
        new.n_components += 1
        alpha = np.empty((new.n_components,))
        mu = np.empty((new.n_components, new.ndim))
        V = np.empty((new.n_components, new.ndim, new.ndim))
        alpha[:-1] = self.alpha
        mu[:-1] = self.mu
        V[:-1] = self.V
        alpha[i], alpha[-1] = new_alpha, new_alpha
        mu[i], mu[-1] = new_mu1, new_mu2
        V[i], V[-1] = new_V, new_V
        new.set_values(-np.inf, mu, V, alpha)
        logging.info("Split component {} into {} and {}".format(i, i, self.n_components-1))
        return new


    def merge(self, X, a, b):
        a, b = sorted([a, b])

        A = self.alpha * len(X)
        new_alpha = self.alpha[a] + self.alpha[b]
        new_mu = np.sum(self.mu[[a,b]] * A[[a,b]][:, None], axis=0) / A[[a,b]].sum()
        new_V = np.sum(self.V[[a,b]] * A[[a,b]][:, None, None], axis=0) / A[[a,b]].sum()

        alpha = np.delete(self.alpha, b, axis=0)
        mu = np.delete(self.mu, b, axis=0)
        V = np.delete(self.V, b, axis=0)

        alpha[a] = new_alpha
        mu[a] = new_mu
        V[a] = new_V

        new = self.copy()
        new.n_components -= 1
        new.set_values(-np.inf, mu, V, alpha)
        logging.info("Merged components {} and {} into {}".format(a, b, a))
        return new


    def _logprob_a(self, X):
        """
        Returns unnormalised responsibilities P(x | z = k, theta) for each data point,
        where x is the observation, z is the intrinsic value, k is the component id, and theta is [mu, V, alpha]
        :param X: Data values of shape (npoints, ndim)
        :return: array : shape == (npoints, ncomponents)
        """
        return self.log_multivariate_gaussian(X, self.mu, self.V, self.ndim, self.n_components) + self.engine.log(self.alpha)


    def _logprob_x(self, X):
        """
        Returns unnormalised responsibilities P(x | z = k, theta) for each data point integrated over all components
        where x is the observation, z is the intrinsic value, k is the component id, and theta is [mu, V, alpha]
        :param X: Data values of shape (npoints, ndim)
        :return: array : shape == (npoints, )
        """
        return self.logsumexp(self._logprob_a(X), axis=1)


    def _logprob_k(self, X):
        """data likelihood for each component, shape == (ncomponents, )"""
        return self.logsumexp(self._logprob_a(X), axis=0)


    def _unnormalised_lnresponsibilities(self, X, weights=None):
        if weights is None:
            weighting = 0
        else:
            weighting = self.engine.log(weights)[:, None]  # -inf if weight = 0
        return self._logprob_a(X) + weighting


    def _lnresponsibilities(self, X, weights=None):
        """
        Returns the normalised responsibilities over all components for each data point
        :param X: shape == (npoints, ndim)
        :return: array : shape == (npoints, ncomponents)
        """
        unnorm_lnrik = self._unnormalised_lnresponsibilities(X, weights)
        return unnorm_lnrik - self.logsumexp(unnorm_lnrik, axis=1)[:, None]


    def _logL(self, X):
        """
        Total loglikelihood
        :param X: shape == (npoints, ndim)
        :return: float
        """
        return self.engine.sum(self.logsumexp(self._logprob_a(X), axis=-1))


    def _EMstep_mu_unnormalised(self, X, rik):
        return self.engine.sum(rik[..., None] * X[:, None, :], axis=0)


    def _EMstep_V_unnormalised(self, X, mu, rik):
        diff = mu - X[:, None, :]  # (np, nc, nd)
        return self.batched_product(diff * rik[..., None], diff)


    def _EMstep_no_prior(self, X, fix_mu, fix_V, fix_alpha):
        n_samples, n_features = X.shape
        rik = self.engine.exp(self._lnresponsibilities(X))

        total = rik.sum(axis=0)
        if not fix_mu:
            mu = self._EMstep_mu_unnormalised(X, rik) / total[:, None]
        else:
            mu = self.mu
        if not fix_V:
            V = self._EMstep_V_unnormalised(X, mu, rik) / total[:, None, None]
        else:
            V = self.V
        if not fix_alpha:
            alpha = total / n_samples
        else:
            alpha = self.alpha

        return self._logL(X), mu, V, alpha


    def _EMstep(self, X, fix_mu=False, fix_V=False, fix_alpha=False, no_prior=False):
        if no_prior:
            return self._EMstep_no_prior(X, fix_mu, fix_V, fix_alpha)
        n_samples, n_features = X.shape
        data_mean = self.engine.mean(X, axis=0)
        prior_mean = data_mean + self.prior.mean_add
        prior_diff = self.mu - prior_mean
        prior_covariance = self.batched_product(prior_diff[None, ...], prior_diff[None, ...]) * self.prior.scale

        lnrik = self._lnresponsibilities(X)
        rik = self.engine.exp(lnrik)
        total = rik.sum(axis=0)  #self.engine.exp(self.logsumexp(lnrik, axis=0))
        if not fix_mu:
            mu = (self._EMstep_mu_unnormalised(X, rik) + (self.prior.scale * prior_mean)) / (total[:, None] + self.prior.scale)
        else:
            mu = self.mu
        if not fix_V:
            denominator = total[:, None, None] + 1 + (2 * (self.prior.wishart_dof - ((self.ndim + 1) / 2)))
            uncorrected_covariance = self._EMstep_V_unnormalised(X, mu, rik)
            V = (uncorrected_covariance + prior_covariance + (2 * self.prior.wishart_matrix)) / denominator
        else:
            V = self.V
        if not fix_alpha:
            alpha = (total + self.prior.dirchlet_weights - 1) / (n_samples + self.prior.dirchlet_weights.sum() - self.n_components)
            # We can use logaddexp to get around reeeeaaaally small alphas, but I guess if we get to that point we want to fail anyway!
            # unnorm_alpha = logaddexp(self.engine.log(total), self.engine.log(self.prior.dirchlet_weights - 1))
            # alpha = self.engine.exp(unnorm_alpha) / (n_samples + self.prior.dirchlet_weights.sum() - self.n_components)
        else:
            alpha = self.alpha

        return self._logL(X), mu, V, alpha


    EMstep = _EMstep
    logL = _logL
    logprob_a = _logprob_a
    logprob_x = _logprob_x
    logprob_k = _logprob_k
    lnresponsibilities = _lnresponsibilities
    split_candidates = _split_candidates
    merge_candidates = _merge_candidates



class CompleteXDGMMCompiled(CompleteXDGMMBase):
    engine = tt
    inputs = ['mu', 'V', 'alpha']

    def __init__(self, n_components, ndim, labels=None, verbose=False, prior=None, backend=None, debug=True):
        super().__init__(n_components, ndim, labels, verbose, prior, backend, debug)
        self.compiled = {}


    def step(self, X, max_iter=1000, tol=1e-5, fix=None, fix_mu=False, fix_V=False, fix_alpha=False, random_state=None):
        self.initialise_prior(X)
        self.compile('logL', [X])
        self.compile('EMstep', [X])
        return super().step(X, max_iter=max_iter, tol=tol, fix=fix, fix_mu=fix_mu, fix_V=fix_V, fix_alpha=fix_alpha,
                            random_state=random_state)


    def load_tensors(self):
        for k, v in self.get_values().items():
            setattr(self, k, tt.as_tensor(v).type(k))

    def compile(self, name, args, *optargs):
        key = (name,)+optargs
        if key not in self.compiled:
            old = self.get_values()
            self.load_tensors()
            names = getattr(self, '_'+name).__code__.co_varnames
            inputs = [tt.as_tensor(v).type(k) for k,v in zip(names, args)] + [getattr(self, i) for i in self.inputs]
            outputs = getattr(self, '_'+name)(*args, *optargs)
            self.compiled[key] = theano.function(inputs, outputs, on_unused_input='ignore')
            self.set_values(**old)
        inputs = [getattr(self, i) for i in self.inputs]
        return self.compiled[key](*args+inputs)


    def EMstep(self, X, fix_mu=False, fix_V=False, fix_alpha=False, no_prior=False):
        return self.compile('EMstep', [X], fix_mu, fix_V, fix_alpha, no_prior)


    def logL(self, X):
        return self.compile('logL', [X])

    def logprob_a(self, X):
        return self.compile('logprob_a', [X])

    def logprob_k(self, X):
        return self.compile('logprob_k', [X])

    def logprob_x(self, X):
        return self.compile('logprob_x', [X])

    def lnresponsibilities(self, X):
        return self.compile('lnresponsibilities', [X])

    def split_candidates(self, X):
        return self.compile('split_candidates', [X])

    def merge_candidates(self, X):
        return self.compile('merge_candidates', [X])

    @property
    def chol(self):
        return tt.stacklists([cholesky(norm_covariance(self.V[i])) for i in range(self.n_components)])


if __name__ == '__main__':
    means = np.array([[1, 2],
                      [3, 4],
                      [2, 2]])
    stds = np.array([[1, 3],
                     [1, 1],
                     [0.5, 5]])
    rhos = np.array([0.6, 0, -0.3])
    alphas = np.array([0.3, 0.3, 0.4])

    corrs = np.asarray([np.eye(2)]*len(alphas))
    corrs[:, 0, 1] = corrs[:, 1, 0] = rhos
    covs = np.zeros_like(corrs)
    covs[0] = corr2cov(corrs[0], stds[0])
    covs[1] = corr2cov(corrs[1], stds[1])
    covs[2] = corr2cov(corrs[2], stds[2])
    truth = CompleteXDGMMBase(3, 2)
    truth.V = covs
    truth.mu = means
    truth.alpha = alphas
    X = truth.sample(9000)
    truth.labels = list('ab')
    truth.condition(a=0)

    # estimator = CompleteXDGMMBase(3, 2, verbose=True, debug=True)
    # estimator.fit(X)