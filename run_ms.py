import os
from tqdm import trange
import numpy as np
from astropy.table import Table
from matplotlib import colors
import matplotlib.pyplot as plt

import xd
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

def extract_submodel(model, components):
    accepted = [c for i, c in enumerate(model.components) if i in components]
    new = CompleteXDGMMCompiled(len(accepted), model.ndim, model.labels, model.verbose, model.prior)
    for i, component in enumerate(accepted):
        new.mu[i, :] = component.mu[0]
        new.V[i, :] = component.V[0]
        new.alpha[i] = component.alpha[0]
    new.alpha /= new.alpha.sum()
    return new

def get_wd_components(model):
    """
    return a model without white dwarfs
    this essentially finds the bit in the corner
    """
    return [i for i, mu in enumerate(model.mu) if ((mu[0] < 1) and (mu[1] > 7.5))]


def get_long_branch_components(model):
    """Attempts to remove the horizontal branch by removing the component with the bit that sticks out the most"""
    eigensorted = [eigsorted(v) for v in model.V]  # get (eigenvalues, eigenvectors)
    furthest = np.asarray([mu - (vals[0] * vects[:, 0] * 2.) for mu, (vals, vects) in zip(model.mu, eigensorted)])  # get 2sigma point along the major axis
    return np.where(furthest[:, 1] < -1)[0].tolist()  # remove the component whose 2sigma ellipse extends beyond Mg = -1


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
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('ncomponents', type=int)
    parser.add_argument('tol', type=float)
    parser.add_argument('maxiter', type=int)
    parser.add_argument('outdir')
    parser.add_argument('--cutn', type=int, default=None)

    args = parser.parse_args()
    data = Table.read(file).to_pandas()

    X = np.stack([data['bp_rp'].values, data['mg'].values]).T

    fname = os.path.join(args.outdir, args.file + '-{}components-tol{}-iter{}-cut{}.h5'.format(args.ncomponents, args.tol, args.maxiter, args.cutn))

    model = CompleteXDGMMCompiled(n_components=args.ncomponents, ndim=2, labels=['colour', 'mag'], verbose=True)
    

    if not os.path.exists(fname):
        model.fit(X[:args.cutn], max_iter=args.maxiter, tol=args.tol)  # may take 10 mins or so
        model.dump_states(Backend('XD', fname))  # save to disk
    else:
        model = model.from_backend(fname)  # read from disk
        model.labels = ['colour', 'mag']

    # Here I remove the largest component (by its determinant) only because I didn't let it run long enough and that component failed to fit
    # You can remove this line to stop that happening or you can exclude components yourself using `exclude_components`
    # largest = np.argmax([np.linalg.det(v) for v in model.V])
    # model = exclude_components(model, [largest])

    wd_components = get_wd_components(model)
    long_branch_components = get_long_branch_components(model)

    # get each sub branch of the CMD
    wd_model = extract_submodel(model, wd_components)
    long_branch_model = extract_submodel(model, long_branch_components)
    main_sequence = exclude_components(model, wd_components+long_branch_components)
    # if the main sequence looks bad its because the model needs longer to run

    Xerr = np.ones_like(X) / 100.  # example fake error

    # calculate probability of each datapoint belonging to each model, shape = (npoints, nmodels)
    # This takes about 10 minutes to do all the data, if it crashes, reduce the batch_size to a more manageable size
    slc = slice(5000)
    probs = np.exp(xd.model_lnprobability([main_sequence, long_branch_model, wd_model], X[slc], Xerr=Xerr[slc],
                                          batch_size=5000)).T


    #plotting
    from matplotlib.backends.backend_pdf import PdfPages
    from data import plot_cmd
    with PdfPages(os.path.join(args.outdir, 'results.pdf')) as pdf:
        fig, ax = plt.subplots()
        plot_cmd(data['bp_rp'], data['mg'], ax=ax)
        plot_model(model, ax, 'r')  # plot the components of the model
        line_colours = np.linspace(0, 4, 1000)
        line_mags = conditional_2d_line(main_sequence, 'colour', line_colours)  # plot the main sequence
        ax.plot(line_colours, line_mags, 'k')
        pdf.savefig(fig)

        fig, ax = plt.subplots()
        line_X = np.stack([line_colours, line_mags]).T
        distances = distance_to_line(line_X, X)
        ax.hist(distances, bins=500)
        ax.set_xlabel('Distance to main sequence')
        pdf.savefig(fig)

        fig, ax = plt.subplots()
        plt.hexbin(X[slc, 0], X[slc, 1], probs[:, 0], reduce_C_function=np.mean, gridsize=200)
        plt.colorbar()
        plt.title('Average probability of belonging to main sequence')
        pdf.savefig(fig)

        fig, ax = plt.subplots()
        plt.hexbin(data['bp_rp'], data['mg'], distances, reduce_C_function=np.mean, gridsize=200)
        plt.colorbar()
        plt.title('Average distance to main sequence')
        pdf.savefig(fig)
        