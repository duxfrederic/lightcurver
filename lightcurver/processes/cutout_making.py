import h5py
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ..structure.user_config import get_user_config


def extract_stamp(data, header, exptime, sky_coord, cutout_size):
    """
    :param data: 2d numpy array containing the full image
    :param header: fits header for WCS info
    :param exptime: float, exposure time to convert from e-/s to e- and back.
    :param sky_coord: astropy SkyCoord: center of cutout
    :param cutout_size: int, pixels
    :return: 2d cutout array, 2d cutout noisemap array, wcs string of cutout
    """

    wcs = WCS(header)
    data_cutout = Cutout2D(data, sky_coord, cutout_size, wcs=wcs, mode='partial')
    # let's also carry the WCS of the cutouts
    wcs_header = data_cutout.wcs.to_header()
    wcs_header_string = wcs_header.tostring()

    # now just take the numpy array
    data_cutout_electrons = exptime * data_cutout.data

    # noise map given that the data is now in electrons ...
    stddev = 0.25 * (
                       np.nanstd(data_cutout_electrons[:, 0])
                       + np.nanstd(data_cutout_electrons[0, :])
                       + np.nanstd(data_cutout_electrons[:, -1])
                       + np.nanstd(data_cutout_electrons[-1, :])
                     )
    nmap = stddev + np.sqrt(np.abs(data_cutout_electrons))
    # remove zeros if there are any ...
    nmap[nmap < 1e-7] = 1e-7

    return data_cutout_electrons / exptime, nmap / exptime, wcs_header_string


def extract_all_stamps():
    user_config = get_user_config()

    # where we'll save our stamps
    regions_file = user_config['regions_path']

    if user_config['redo_stamp_extraction']:
        # start from scratch ...
        if regions_file.exists():
            # delete if exists.
            regions_file.unlink()

