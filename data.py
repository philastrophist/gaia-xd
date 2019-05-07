from astroquery.gaia import Gaia
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def gaia_query(n, distance=200, **kwargs):
    """
    Sends an archive query for d < 200 pc, with additional filters taken from
    Gaia Data Release 2: Observational Hertzsprung-Russell diagrams (Sect. 2.1)
    Gaia Collaboration, Babusiaux et al. (2018)
    (https://doi.org/10.1051/0004-6361/201832843)

    NOTE: 10000000 is a maximum query size (~76 MB / column)

    Additional keyword arguments are passed to TapPlus.launch_job_async method.
    """
    return Gaia.launch_job_async("select top {}".format(n)+
                    #" lum_val, teff_val,"
                    #" ra, dec, parallax,"
                    " bp_rp, phot_g_mean_mag+5*log10(parallax)-10 as mg"
             " from gaiadr2.gaia_source"
             " where parallax_over_error > 10"
             " and visibility_periods_used > 8"
             " and phot_g_mean_flux_over_error > 50"
             " and phot_bp_mean_flux_over_error > 20"
             " and phot_rp_mean_flux_over_error > 20"
             " and phot_bp_rp_excess_factor <"
                " 1.3+0.06*power(phot_bp_mean_mag-phot_rp_mean_mag,2)"
             " and phot_bp_rp_excess_factor >"
                " 1.0+0.015*power(phot_bp_mean_mag-phot_rp_mean_mag,2)"
             " and astrometric_chi2_al/(astrometric_n_good_obs_al-5)<"
                "1.44*greatest(1,exp(-0.4*(phot_g_mean_mag-19.5)))"
             +" and 1000/parallax <= {}".format(distance), **kwargs)

def plot_cmd(colours, mags, ax=None):
    plt.rc('text', usetex=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    # only show 2D-histogram for bins with more than 10 stars in them
    h = ax.hist2d(colours, mags, bins=300, cmin=10, norm=colors.PowerNorm(0.5), zorder=0.5)
    # fill the rest with scatter (set rasterized=True if saving as vector graphics)
    ax.scatter(colours, mags, alpha=0.05, s=1, color='k', zorder=0)
    ax.invert_yaxis()
    cb = fig.colorbar(h[3], ax=ax, pad=0.02)
    ax.set_xlabel(r'$G_{BP} - G_{RP}$')
    ax.set_ylabel(r'$M_G$')
    cb.set_label(r"$\mathrm{Stellar~density}$")
    return ax


if __name__ == '__main__':

    # job = gaia_query(10000000, dump_to_file=True)
    # data = job.get_data().to_pandas()

    from astropy.table import Table
    data = Table.read('async_20190310174529.vot')


    plot_cmd(data['bp_rp'], data['mg'])