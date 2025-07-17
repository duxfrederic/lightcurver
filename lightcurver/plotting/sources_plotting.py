import matplotlib.pyplot as plt

from .image_plotting import plot_image
from .footprint_plotting import plot_footprints


def plot_sources(sources, image, wcs=None, save_path=None, sources_label=None,
                 kwargs_imshow=None, **kwargs_plot):
    """
    Plot the image with detected sources marked (for debugging).

    Parameters:
    sources (astropy.table.Table): Table of detected sources.
    image (numpy.ndarray): Image data, 2D array.
    wcs (astropy.wcs.WCS object): the WCS corresponding to the data. default None.
    save_path (pathlib.Path or str): path to potential save location for image.
    """
    kwargs_imshow = {} if kwargs_imshow is None else kwargs_imshow
    fig, ax = plot_image(image=image,
                         wcs=wcs,
                         save_path=None,
                         **kwargs_imshow)

    base_plot_options = {'marker': 'o',
                         'ls': 'None',
                         'mfc': 'None',
                         'color': 'red',
                         'ms': 10,
                         'alpha': 0.7}
    base_plot_options.update(kwargs_plot)

    if wcs is not None:
        ra, dec = wcs.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)
        ax.plot(ra, dec, color='red', label=sources_label,
                transform=ax.get_transform('world'),
                **base_plot_options)
    else:
        ax.plot(sources['xcentroid'], sources['ycentroid'],
                label=sources_label,
                **base_plot_options)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)
        plt.close()
    else:
        return fig, ax


def plot_coordinates_and_sources_on_image(data, sources, gaia_coords, wcs, save_path, **kwargs_imshow):
    """
    This is similar to the above, but we want to focus on the quality of the WCS.
    Args:
        data: image
        sources: astropy Table
        gaia_coords: Skycoord, usually from gaia
        wcs: wcs object astropy
        save_path: where to save
        **kwargs_imshow:

    Returns:

    """

    kwargs_imshow = {} if kwargs_imshow is None else kwargs_imshow
    fig, ax = plot_image(image=data,
                         wcs=wcs,
                         save_path=None,
                         **kwargs_imshow)

    if gaia_coords is not None:
        ax.scatter(gaia_coords.ra, gaia_coords.dec, transform=ax.get_transform('world'), s=10, edgecolor='r',
                   facecolor='none', label='Gaia Stars')
    if sources is not None:
        ax.scatter(sources['x'], sources['y'], s=10, color='blue', label='Detections', alpha=0.7)

    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig, ax


def plot_footprints_with_stars(footprint_arrays, stars, save_path=None):
    """

    Args:
        footprint_arrays:  list of arrays where each array represents a footprint's corners.
        stars: pandas dataframe of stars, with columns 'name', 'ra', 'dec'
        save_path: str or path

    Returns:

    """
    fig, ax = plot_footprints(footprint_arrays, common_footprint=None, largest_footprint=None, save_path=None)
    for _, star in stars.iterrows():
        if star['name'] == 'roi':
            ax.plot(star['ra'], star['dec'], 'o', color='red', markersize=10, mfc='None')
        ax.plot(star['ra'], star['dec'], 'o', color='red', markersize=5, mfc='None')
        ax.text(star['ra'], star['dec'], star['name'], fontsize=8, ha='right')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
    else:
        return fig, ax
