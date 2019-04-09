import numpy as np
from pymc3 import floatX
from pymc3.distributions.dist_math import Cholesky, alltrue_elemwise
from scipy import linalg
import theano.tensor as tt
from theano import shared, scan
from theano.ifelse import ifelse

__all__ = ['logaddexp', 'logsumexp', 'batched_product_3d', 'stacked_svd', 'asarray', 'set_subtensor']


def logaddexp(a, b):
    assert a.shape == b.shape

    vmax = max([a.max(), b.max()])
    out = np.log(np.exp(a - vmax) + np.exp(b - vmax))
    return out + vmax


class TheanoNumpyExpression(object):
    """
    Object to contain expressions compatible with both numpy and theano
    """
    engines = {np: 'numpy', tt: 'theano'}


    def theano(self, *args, **kwargs):
        pass

    def numpy(self, *args, **kwargs):
        pass

    def __getitem__(self, engine):
        return getattr(self, self.engines[engine])


class BatchedProduct3D(TheanoNumpyExpression):
    """returns the outer product of two 3d tensors looping over the first two axes"""
    def theano(self, x, y):
        return tt.batched_tensordot(x.transpose(1, 2, 0), y.transpose(1, 0, 2), axes=1)

    def numpy(self, x, y):
        return np.einsum('pcj,pck->cjk', x, y)

batched_product_3d = BatchedProduct3D()


class LogMultivariateGaussian(TheanoNumpyExpression):
    def numpy(self, x, mu, V, ndim=None, ncomp=None):
        """Evaluate a multivariate gaussian N(x|mu, V)

        This allows for multiple evaluations at once, using array broadcasting

        Parameters
        ----------
        x: array_like
            points, shape[-1] = n_features

        mu: array_like
            centers, shape[-1] = n_features

        V: array_like
            covariances, shape[-2:] = (n_features, n_features)

        Returns
        -------
        values: ndarray
            shape = broadcast(x.shape[:-1], mu.shape[:-1], V.shape[:-2])

        Examples
        --------

        >>> x = [1, 2]
        >>> mu = [0, 0]
        >>> V = [[2, 1], [1, 2]]
        >>> log_multivariate_gaussian(x, mu, V)
        -3.3871832107434003
        """
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        V = np.asarray(V, dtype=float)
        x, mu, V = x[:, np.newaxis, :], mu, V[np.newaxis, ...]

        ndim = x.shape[-1]
        x_mu = x - mu

        if V.shape[-2:] != (ndim, ndim):
            raise ValueError("Shape of (x-mu) and V do not match")

        Vshape = V.shape
        V = V.reshape([-1, ndim, ndim])

        Vchol = np.array([linalg.cholesky(V[i], lower=True)
                          for i in range(V.shape[0])])

        # we may be more efficient by using scipy.linalg.solve_triangular
        # with each cholesky decomposition
        VcholI = np.array([linalg.inv(Vchol[i])
                           for i in range(V.shape[0])])
        logdet = np.array([2 * np.sum(np.log(np.diagonal(Vchol[i])))
                           for i in range(V.shape[0])])

        VcholI = VcholI.reshape(Vshape)
        logdet = logdet.reshape(Vshape[:-2])

        VcIx = np.sum(VcholI * x_mu.reshape(x_mu.shape[:-1]
                                            + (1,) + x_mu.shape[-1:]), -1)
        xVIx = np.sum(VcIx ** 2, -1)

        r = -0.5 * ndim * np.log(2 * np.pi) - 0.5 * (logdet + xVIx)
        return r

    def theano(self, x, mu, V, ndim, ncomp):
        cholesky = Cholesky(nofail=True, lower=True)
        solve_lower = tt.slinalg.Solve(A_structure="lower_triangular")
        if x.ndim == 1:
            onedim = True
            x = x[None, :]
        else:
            onedim = False

        delta = x[:, None, :] - mu[None, ...]

        logps = []
        for i in range(ncomp):
            _chol_cov = cholesky(V[i])
            k = floatX(ndim)
            diag = tt.nlinalg.diag(_chol_cov)
            # Check if the covariance matrix is positive definite.
            ok = tt.all(diag > 0)
            # If not, replace the diagonal. We return -inf later, but
            # need to prevent solve_lower from throwing an exception.
            chol_cov = tt.switch(ok, _chol_cov, 1)

            delta_trans = solve_lower(chol_cov, delta[:, i].T).T
            _quaddist = (delta_trans ** 2).sum(axis=-1)
            logdet = tt.sum(tt.log(diag))
            if onedim:
                quaddist = _quaddist[0]
            else:
                quaddist = _quaddist
            norm = - 0.5 * k * floatX(np.log(2 * np.pi))
            logp = norm - 0.5 * quaddist - logdet
            safe_logp = tt.switch(alltrue_elemwise([ok]), logp, -np.inf)  # safe logp (-inf for invalid)
            logps.append(safe_logp)
        return tt.stacklists(logps).T

log_multivariate_gaussian = LogMultivariateGaussian()


class LogSumExp(TheanoNumpyExpression):
    def numpy(self, arr, b=1, axis=None, keepdims=False):
        """Computes the sum of arr assuming arr is in the log domain.

        Returns log(sum(exp(arr))) while minimizing the possibility of
        over/underflow.

        Examples
        --------

        >>> import numpy as np
        >>> a = np.arange(10)
        >>> np.log(np.sum(np.exp(a)))
        9.4586297444267107
        >>> logsumexp(a)
        9.4586297444267107
        """
        # if axis is specified, roll axis to 0 so that broadcasting works below
        if axis is not None:
            arr = np.rollaxis(arr, axis)
            axis = 0

        # Use the max to normalize, as with the log this is what accumulates
        # the fewest errors
        vmax = arr.max(axis=axis)
        out = np.log(np.sum(np.exp(arr - vmax), axis=axis, keepdims=keepdims))
        out += vmax
        return out


    def theano(self, arr, b=1, axis=None, keepdims=False):
        r = tt.log(tt.sum(tt.exp(arr + tt.log(b)), axis=axis, keepdims=keepdims))
        return ifelse(arr.shape[axis] > 0, r, tt.log(tt.zeros_like(arr).sum(axis=axis, keepdims=keepdims)))  # log(0) to force inf and not nan

logsumexp = LogSumExp()


class StackedSVD(TheanoNumpyExpression):
    def numpy(self, v):
        return np.linalg.svd(v, compute_uv=False)

    def theano(self, v, compute_uv=True):
        EV, _ = scan(lambda x: tt.nlinalg.svd(x, compute_uv=False), sequences=[v], outputs_info=None)
        return EV

stacked_svd = StackedSVD()


class AsArray(TheanoNumpyExpression):
    def numpy(self, x):
        return np.asarray(x)

    def theano(self, x):
        return shared(np.asarray(x))

asarray = AsArray()


class SetSubTensor(TheanoNumpyExpression):
    def numpy(self, array, selection, replacement):
        array[selection] = replacement

    def theano(self, array, selection, replacement):
        tt.set_subtensor(array[selection], replacement, inplace=True)

set_subtensor = SetSubTensor()


class Nlinalg(TheanoNumpyExpression):
    def numpy(self):
        return np.linalg

    def theano(self):
        return tt.nlinalg

nlinalg = Nlinalg()


class ErfInv(TheanoNumpyExpression):
    def numpy(self, x):
        from scipy.special import erfinv
        return erfinv(x)

    def theano(self, x):
        return tt.erfinv(x)

erfinv = ErfInv()