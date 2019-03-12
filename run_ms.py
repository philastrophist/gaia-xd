from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from xd import fit, conditional_2d_line, CompleteXDGMMCompiled, plot_model
from astropy.table import Table


def exclude_wds(model):
    """return a model without white dwarfs"""
    new = model.copy()
    accepted = [c for c in model.components if not ((c.mu[0, 0] < 1) and (c.mu[0, 1] > 7.5))]
    for i, component in enumerate(accepted):
        new.mu[i, :] = component.mu[0]
        new.V[i, :] = component.V[0]
        new.alpha[i] = component.alpha[0]
    new.alpha /= new.alpha.sum()
    return new


if __name__ == '__main__':
    data = Table.read('async_20190310174529.vot')
    model = fit(13, data['bp_rp'].data, data['mg'].data)

    fig, ax = plt.subplots()
    h = ax.hist2d(data['bp_rp'], data['mg'], bins=300, cmin=10, norm=colors.PowerNorm(0.5), zorder=0.5)
    main_sequence = exclude_wds(model)
    plot_model(model, ax)
    colours = np.linspace(0, 4)
    line = conditional_2d_line(model, 'colour', colours)
    ax.plot(colours, line, 'k')