import numpy as np
from scipy.ndimage import median_filter
from astropy.table import Table
import sep

from ..plotting.sources_plotting import plot_sources


def extract_stars(image_background_subtracted, background_rms, detection_threshold=3, min_area=10,
                  debug_plot_path=None):
    """
    Extract star positions from an image using SEP (Source Extractor as a Python library).

    Parameters:
    image_background_subtracted: image contained in numpy 2d array, with background subtracted!
    background_rms: float, rms of signal in the background -- noise estimate for significance calculation.
    detection_threshold: float, in units of sigma, default 4
    min_area: int, min number of pixels in detection for it to be considered, default 10

    Returns:
    astropy.table.Table: Table of detected sources.
    """
    image_filtered = median_filter(image_background_subtracted, size=2)

    objects = sep.extract(data=image_filtered,
                          thresh=detection_threshold,
                          err=background_rms,
                          minarea=min_area)

    sources = Table()
    for col in objects.dtype.names:
        sources[col] = objects[col]

    # just to stick to the daostarfinder way
    sources['xcentroid'] = sources['x']
    sources['ycentroid'] = sources['y']

    # remove flagged
    sources = sources[sources['flag'] == 0]

    # remove the weirdly elongated ones
    elongation = sources['a'] / sources['b']
    med_el = np.median(elongation)
    std_el = np.std(elongation)
    sources['elongation'] = elongation
    sources = sources[sources['elongation'] < med_el + std_el]

    # define some FWHM quantity
    sources['FWHM'] = 2 * (np.log(2) * (sources['a']**2 + sources['b']**2))**0.5

    # remove those occupying a weirdly small amount of space (likely hot pixels or cosmics)
    med_pix = np.median(sources['npix'])
    sources = sources[(sources['npix'] > 0.5 * med_pix)]

    # brightest first
    sources.sort('flux', reverse=True)

    if debug_plot_path is not None:
        debug_plot_path.parent.mkdir(exist_ok=True)
        plot_sources(sources=sources, image=image_background_subtracted, save_path=debug_plot_path)

    return sources
