import os
from tqdm import trange
import numpy as np
from astropy.table import Table
from matplotlib import colors
import matplotlib.pyplot as plt

from xd import conditional_2d_line, CompleteXDGMMCompiled, plot_model, Backend, eigsorted


def exclude_components(model, components):
    """return a model without white dwarfs"""
    accepted = [c for i, c in enumerate(model.components) if i not in components]
    new = CompleteXDGMMCompiled(len(accepted), model.ndim, model.labels, model.verbose, model.prior)
    for i, component in enumerate(accepted):
        new.mu[i, :] = component.mu[0]
        new.V[i, :] = component.V[0]
        new.alpha[i] = component.alpha[0]
    new.alpha /= new.alpha.sum()
    return new


def exclude_wds(model):
    """
    return a model without white dwarfs
    this essentially finds the bit in the corner
    """
    rejected = [i for i, mu in enumerate(model.mu) if ((mu[0] < 1) and (mu[1] > 7.5))]
    return exclude_components(model, rejected)


def exclude_long_branch(model):
    """Attempts to remove the horizontal branch by removing the component with the bit that sticks out the most"""
    eigensorted = [eigsorted(v) for v in model.V]  # get (eigenvalues, eigenvectors)
    furthest = np.asarray([mu - (vals[0] * vects[:, 0] * 2.) for mu, (vals, vects) in zip(model.mu, eigensorted)])  # get 2sigma point along the major axis
    reject = np.where(furthest[:, 1] < -1)[0]  # remove the component whose 2sigma ellipse extends beyond Mg = -1
    return exclude_components(model, reject)


def _distance_to_line(line_X, data_X):
    """
    Compute pythagorian distance to the non-parametric line by taking the minimum distance.
    Not for user use (see distance_to_line)
    """
    return np.sqrt(np.sum((data_X[None, ...] - line_X[:, None, :]) ** 2, axis=-1)).min(axis=0)


def distance_to_line(line_X, data_X, batch_size=10000):
    """
    Compute pythagorian distance to the non-parametric line by taking the minimum distance
    We do this using batches in order to stop using all of the memory
    :param line_X: line array (n, [bp_rp, mg])
    :param data_X: data array (N, [bp_rp, mg])
    :param batch_size: how many to do at once
    :return:
    """
    # nbatches = (len(data_X) // batch_size) +
    dists = [_distance_to_line(line_X, data_X[start:start + batch_size]) for start in trange(0, len(data_X), batch_size,
                                                                                             desc='calc distances')]
    return np.concatenate(dists, axis=0)


if __name__ == '__main__':
    data = Table.read('async_20190310174529.vot').to_pandas()

    X = np.stack([data['bp_rp'].values, data['mg'].values]).T
    model = CompleteXDGMMCompiled(n_components=13, ndim=2, labels=['colour', 'mag'], verbose=True)
    fname = 'async_20190310174529.vot-{}components.h5'.format(model.n_components)

    if not os.path.exists(fname):
        model.fit(X, max_iter=9000, tol=1e-1)  # may take 10 mins or so
        model.dump_states(Backend('XD', fname))  # save to disk
    else:
        model = model.from_backend(fname)  # read from disk
        model.labels = ['colour', 'mag']

    # Here I remove the largest component (by its determinant) only because I didn't let it run long enough and that component failed to fit
    # You can remove this line to stop that happening or you can exclude components yourself using `exclude_components`
    largest = np.argmax([np.linalg.det(v) for v in model.V])
    model = exclude_components(model, [largest])
    main_sequence = exclude_long_branch(exclude_wds(model))  # get the main sequence
    # if the main sequence looks bad its because the model needs longer to run


    # plotting
    fig, ax = plt.subplots()
    h = ax.hist2d(data['bp_rp'], data['mg'], bins=300, cmin=10, norm=colors.PowerNorm(0.5), zorder=0.5)
    plot_model(model, ax, 'r')  # plot the components of the model
    line_colours = np.linspace(0, 4, 1000)
    line_mags = conditional_2d_line(main_sequence, 'colour', line_colours)  # plot the main sequence
    ax.plot(line_colours, line_mags, 'k')

    fig, ax = plt.subplots()
    line_X = np.stack([line_colours, line_mags]).T
    distances = distance_to_line(line_X, X)
    ax.hist(distances, bins=500)
    ax.set_xlabel('Distance to main sequence')

    fig, ax = plt.subplots()
    plt.hexbin(data['bp_rp'], data['mg'], distances, reduce_C_function=np.mean, gridsize=200)
    plt.colorbar()
    plt.title('Average distance to main sequence')
    plt.show()