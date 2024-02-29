import matplotlib.pyplot as plt

from .image_plotting import plot_image


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
                         save_path=save_path,
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

    return fig, ax
