import numpy as np
from astropy.table import Table
import sep
from astropy.io import fits
from ..plotting.sources_plotting import plot_sources


def extract_stars(image_background_subtracted, variance_map, detection_threshold=3, min_area=10,
                  debug_plot_path=None):
    """
    Extract star positions from an image using SEP (Source Extractor as a Python library).

    Parameters:
    image_background_subtracted: image contained in numpy 2d array, with background subtracted!
    variance_map: image, map of variance.
    detection_threshold: float, in units of sigma, default 3
    min_area: int, min number of pixels in detection for it to be considered, default 10

    Returns:
    astropy.table.Table: Table of detected sources.
    """

    objects = sep.extract(data=image_background_subtracted,
                          thresh=detection_threshold,
                          var=variance_map,
                          minarea=min_area)

    sources = Table()
    for col in objects.dtype.names:
        sources[col] = objects[col]

    # just to stick to the daostarfinder way
    sources['xcentroid'] = sources['x']
    sources['ycentroid'] = sources['y']

    # remove the weirdly elongated ones -- probably most stars, and we care about stars the most
    elongation = sources['a'] / sources['b']
    med_el = np.median(elongation)
    std_el = np.std(elongation)
    sources['elongation'] = elongation
    sources = sources[sources['elongation'] < med_el + 3*std_el]

    # define some FWHM quantity
    sources['FWHM'] = 2 * (np.log(2) * (sources['a']**2 + sources['b']**2))**0.5
    # and ellipticity
    sources['ellipticity'] = 1 - sources['b'] / sources['a']

    # brightest first
    sources.sort('flux', reverse=True)

    if debug_plot_path is not None:
        debug_plot_path.parent.mkdir(exist_ok=True)
        plot_sources(sources=sources, image=image_background_subtracted, save_path=debug_plot_path)

    return sources


def extract_sources_from_sky_sub_image(image_path, sources_path, detection_threshold, min_area,
                                       exptime, background_rms_electron_per_second, debug_plot_path):
    """
      Not used in the main pipeline but can be useful. Given an image, extracts
      sources, and save them in sources_path
    Args:
        image_path:  path, str: fits file containing an image.
        sources_path:  path, str: where the sources are saved
        detection_threshold: float, significance for acceptance
        min_area: int, minimum number of pixels above threshold
        exptime: float, exposure time of the frame (used to calculate a variance map,
                 because the pipeline stores the frames in e- / second)
        background_rms_electron_per_second: float, scatter in the background in e- / second.
                                            Used to calculate the variance map as well.
        debug_plot_path: where we (potentially) save a plot of our sources.
    Returns:
        Nothing
    """
    image_electrons = exptime * fits.getdata(image_path).astype(float)
    background_rms_electrons = exptime * background_rms_electron_per_second

    variance_map = background_rms_electrons**2 + np.abs(image_electrons)

    sources = extract_stars(image_background_subtracted=image_electrons,
                            detection_threshold=detection_threshold,
                            variance_map=variance_map,
                            min_area=min_area,
                            debug_plot_path=debug_plot_path)

    sources.write(sources_path, overwrite=True)

