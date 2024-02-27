import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.visualization.stretch import AsinhStretch


def plot_image(image, wcs=None, save_path=None, colorbar=False, **imshow_kwargs):
    """
    Plot the image with detected sources marked (for debugging).

    Parameters:
    image (numpy.ndarray): Image data, 2D array.
    wcs (astropy.wcs.WCS object): the WCS corresponding to the data. default None.
    save_path: do we save the plot somewhere? default None.
    imshow_kwargs: additional keyword arguments to be passed to matplotlib imshow.
    """
    fig = plt.figure(figsize=(11, 11))
    if wcs is not None:
        ax = plt.subplot(projection=wcs)
    else:
        ax = plt.subplot()
    norm = ImageNormalize(image,
                          interval=ZScaleInterval(contrast=0.1),
                          stretch=AsinhStretch()
                          )
    ax.imshow(image, cmap='gray', origin='lower', norm=norm, **imshow_kwargs)
    if colorbar:
        plt.colorbar()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.)

    return fig, ax
