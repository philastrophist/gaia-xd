import numpy as np
from candid.complete import CompleteXDGMMCompiled
from candid.visual import plot_ellipse
from candid.backend import Backend
from candid.visual import eigsorted





def conditional_2d_line(model, xlabel, xs):
    models = [model.condition(**{xlabel: x}) for x in xs]
    _ys = [np.sum(m.mu[:, 0] * m.alpha) for m in models]
    return np.asarray(_ys)


def plot_model(model, ax, color='k'):
    for component in model.components:
        plot_ellipse(ax, component.mu[0], component.V[0], color, alpha=0.3)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from statsmodels.stats.moment_helpers import corr2cov


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
    ax.plot(_xs, _ys, color='k')


