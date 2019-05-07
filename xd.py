import numpy as np
from scipy import linalg
from tqdm import trange

from candid.complete import CompleteXDGMMCompiled, Backend
from matplotlib.patches import Ellipse


def errors2covariance(errors):
    covar = np.zeros((errors.shape[0], errors.shape[1], errors.shape[1]), dtype=float)
    for i, err in enumerate(errors.T):
        covar[:, i, i] = err ** 2
    return covar


def covariance2errors(covariances):
    corr, std = zip(*[cov2corr(c, True) for c in covariances])
    return np.asarray(corr), np.asarray(std)



def log_multivariate_gaussian(x, mu, V):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)

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


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def orientation_from_covariance(cov, sigma):
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * sigma * np.sqrt(vals)
    return w, h, theta

def plot_ellipse(ax, mu, covariance, color, linewidth=2, alpha=0.5):
    x, y, angle = orientation_from_covariance(covariance, 2)
    e = Ellipse(mu, x, y, angle=angle)
    e.set_alpha(alpha)
    e.set_linewidth(linewidth)
    e.set_edgecolor(color)
    e.set_facecolor(color)
    e.set_fill(False)
    ax.add_artist(e)
    return e


def logsumexp(arr, b=1, axis=None, keepdims=False):
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

def conditional_2d_line(model, xlabel, xs):
    models = [model.condition(**{xlabel: x}) for x in xs]
    _ys = [np.sum(m.mu[:, 0] * m.alpha) for m in models]
    return np.asarray(_ys)


def plot_model(model, ax, color='k'):
    for component in model.components:
        plot_ellipse(ax, component.mu[0], component.V[0], color, alpha=0.3)


def _component_lnprobability(model, X, Xerr=None, norm=True):
    if Xerr is None:
        x, mu, V = X[:, np.newaxis, :], model.mu, model.V[np.newaxis, ...]
        logprob = log_multivariate_gaussian(x, mu, V)
    else:
        Xcov = errors2covariance(Xerr)
        x, mu, V = X[:, np.newaxis, :], model.mu, model.V[np.newaxis, ...] + Xcov[:, np.newaxis]
        logprob = log_multivariate_gaussian(x, mu, V)

    if norm:
        component_logprob = logsumexp(logprob, axis=1)  # sum along components
        return logprob - component_logprob[:, np.newaxis]
    else:
        return logprob

def component_lnprobability(model, X, Xerr=None, norm=True, batch_size=5000):
    bits = trange(0, len(X), batch_size, desc='calc component probs')
    if Xerr is None:
        probs = [_component_lnprobability(model, X[i:i + batch_size], norm) for i in bits]
    else:
        probs = [_component_lnprobability(model, X[i:i + batch_size], Xerr[i:i + batch_size], norm) for i in bits]
    return np.concatenate(probs, axis=0)

def model_lnprobability(models, X, Xerr=None, batch_size=5000):
    model_probs = np.asarray([logsumexp(component_lnprobability(model, X, Xerr, norm=False, batch_size=batch_size),
                                        axis=1) for model in models])
    return model_probs - logsumexp(model_probs, axis=0)[np.newaxis, :]



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from statsmodels.stats.moment_helpers import corr2cov, cov2corr


    def fit(ncomponents, colours, mags, steps=1000):
        X = np.stack([colours, mags]).T
        XDGMM = CompleteXDGMMCompiled
        model = XDGMM(ncomponents, 2, labels=['colour', 'mag'], verbose=True)
        model.initialise(X, 1000)
        model.fit(X, max_iter=steps, tol=None, reinitialise=False)
        return model

    corra = np.ones((2, 2))
    corra[1, 0] = corra[1, 0] = 0.7
    a = np.random.multivariate_normal([0, 0], corr2cov(corra, 2), 3000)

    corrb = np.ones((2, 2))
    corrb[1, 0] = corrb[1, 0] = 0.9
    b = np.random.multivariate_normal([4, 9], corr2cov(corrb, [1, 3]), 3000)
    X = np.concatenate([a, b])

    model = fit(2, X[:, 0], X[:, 1])
    samples = model.sample(6000)

    plt.scatter(*X.T, s=5)
    plt.axis('equal')
    ax = plt.gca()
    _xs = np.linspace(*ax.get_xlim())
    _ys = conditional_2d_line(model, 'colour', _xs)
    ax.plot(_xs, _ys, color='k', rasterized=True)


