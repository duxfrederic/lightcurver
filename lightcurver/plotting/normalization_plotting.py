import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pandas as pd

from ..structure.database import execute_sqlite_query
from ..utilities.chi2_selector import get_chi2_bounds


def plot_normalized_star_curves(combined_footprint_hash, save_path=None):
    """
    Given our config params (in particular chi2 rejection criterion), plot the fluxes of the stars that were
    used to build the normalization coefficient of the frames, normalized by this very coefficient.
    Args:
        combined_footprint_hash: footprint hash that we are working with.
        save_path: path or str, where we save the plot.

    Returns:
        None
    """
    # doing this because we set the stylesheet to dark for other plots, and we don't want it here (hard to read)
    mpl.rcParams.update(mpl.rcParamsDefault)

    # query the coefficients and fluxes
    fluxes_fit_chi2_min, fluxes_fit_chi2_max = get_chi2_bounds(psf_or_fluxes='fluxes')

    query_norm = """
    SELECT 
       nc.*, f.mjd
    FROM 
       normalization_coefficients nc
    JOIN
       frames f ON nc.frame_id = f.id
    WHERE   
       coefficient > coefficient_uncertainty
    AND
       combined_footprint_hash = ?
    """
    df_norm = execute_sqlite_query(query=query_norm,
                                   params=(combined_footprint_hash,),
                                   use_pandas=True, is_select=True)
    query_fluxes = """
    SELECT 
       sff.*, s.name AS name, s.gmag as gmag
    FROM 
       star_flux_in_frame sff
    JOIN 
       stars s ON sff.star_gaia_id = s.gaia_id
    WHERE 
       sff.chi2 > ? 
    AND
       sff.chi2 < ?
    AND
       sff.combined_footprint_hash = ?
    """
    df_fluxes = execute_sqlite_query(query=query_fluxes,
                                     params=(fluxes_fit_chi2_min, fluxes_fit_chi2_max, combined_footprint_hash,),
                                     use_pandas=True, is_select=True)
    df = pd.merge(df_norm, df_fluxes, on=['frame_id'])

    # normalize the fluxes
    df['normalized_flux'] = df['flux'] / df['coefficient']
    df['normalized_uncertainty'] = df['flux_uncertainty'] / df['coefficient']

    # ok, prepare plotting
    star_names = df['name'].unique()
    plot_size = 3.5
    n_rows = len(star_names) + 1

    fig, axs = plt.subplots(n_rows, 1, figsize=(2 * plot_size, n_rows * plot_size), sharex=True)
    axs = axs.flatten()

    # norm color scale across all stars based on chi2 values
    norm = colors.Normalize(vmin=fluxes_fit_chi2_min, vmax=fluxes_fit_chi2_max)
    cmap = plt.get_cmap('coolwarm')

    # norm coefficient plot
    axs[0].errorbar(df_norm['mjd'], df_norm['coefficient'],
                    yerr=df_norm['coefficient_uncertainty'],
                    fmt='.', ms=0,
                    ecolor='gray', alpha=0.8, zorder=-1000, elinewidth=0.8)
    sc = axs[0].scatter(df_norm['mjd'], df_norm['coefficient'], s=5, edgecolor='none', color='red')
    axs[0].set_ylabel('Normalization Coefficient')
    axs[0].grid(True)
    cl = fig.colorbar(sc, ax=axs[0], label='chi2')

    # norm star fluxes
    for i, name in enumerate(sorted(star_names)):
        ax = axs[i + 1]
        star_data = df[df['name'] == name]
        medflux = star_data['normalized_flux'].median()
        err = ((star_data['normalized_uncertainty'] / medflux)**2 + (star_data['coefficient_uncertainty'])**2)**0.5
        ax.errorbar(star_data['mjd'], star_data['normalized_flux'] / medflux,
                    yerr=err, fmt='.', ms=0, ecolor='gray', alpha=0.5, zorder=-1000, elinewidth=0.5)
        sc = ax.scatter(star_data['mjd'], star_data['normalized_flux'] / medflux,
                        c=star_data['chi2'], cmap=cmap, norm=norm, s=10,
                        edgecolor='none', label=f"Star {name} (gmag: {star_data['gmag'].unique()[0]:.1f})")
        ax.set_ylabel('Normalized flux / global median')
        ax.set_ylim((0.9, 1.1))
        ax.grid(True)
        ax.legend()

        fig.colorbar(sc, ax=ax, label='chi2')

    # time label
    axs[-1].set_xlabel('MJD')

    plt.tight_layout()
    cl.set_label('')
    cl.set_ticks([])

    if save_path is not None:
        plt.savefig(save_path, pad_inches=0, bbox_inches='tight', dpi=150)
    else:
        plt.show()
