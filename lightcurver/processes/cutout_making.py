import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroscrappy import detect_cosmics
from astropy.time import Time
# to suppress the warnings telling us we don't have the parallax:
# we do not care about the radial component of proper motion, we just want
# roughly centered stars.
import warnings
import erfa

from ..structure.user_config import get_user_config
from ..structure.database import get_pandas, query_stars_for_frame_and_footprint
from ..utilities.footprint import get_combined_footprint_hash


def extract_stamp(data, header, exptime, sky_coord, cutout_size, background_rms_electron_per_second):
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

    noisemap = exptime * background_rms_electron_per_second + np.sqrt(np.abs(data_cutout_electrons))
    # remove zeros if there are any ...
    noisemap[noisemap < 1e-7] = 1e-7

    return data_cutout_electrons / exptime, noisemap / exptime, wcs_header_string


def extract_all_stamps():
    """
    This is the routine that the workflow manager will call.
    It interfaces with the user config and the database to locate the
    files and objects to extract, then extracts to a hdf5 file.

    Returns:
        Nothing
    """
    user_config = get_user_config()

    # where we'll save our stamps
    regions_file = user_config['regions_path']

    # query frames
    frames_to_process = get_pandas(columns=['id', 'image_relpath', 'exptime', 'mjd',
                                            'background_rms_electron_per_second'],
                                   conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])

    # we'll need to know what combined_footprint we're working with.
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_to_process['id'].to_list())

    # suppress the no parallax warning in proper motion corrections:
    warnings.filterwarnings('ignore', category=erfa.ErfaWarning)

    with h5py.File(regions_file, 'a') as reg_f:
        for i, frame in frames_to_process.iterrows():
            # check what stars need be extracted
            stars = query_stars_for_frame_and_footprint(frame_id=frame['id'],
                                                        combined_footprint_hash=combined_footprint_hash)

            # chance to skip this frame if both 'not redo' and 'all was extracted'
            if (not user_config['redo_stamp_extraction']) and (frame['image_relpath'] in reg_f.keys()):
                if not len(reg_f[frame['image_relpath']]['data'].keys()) == len(stars) + 1:
                    # not the right number of cutouts ...have to extract.
                    pass
                else:
                    # right number of cutouts, but gotta make sure the keys match.
                    keys = reg_f[frame['image_relpath']]['data'].keys()
                    all_there = True
                    for j, star in stars.iterrows():
                        if star['name'] not in keys:
                            all_there = False
                            break
                    if 'ROI' not in keys:
                        all_there = False
                    if all_there:
                        # all good, skip to next frame.
                        continue
            image_file = user_config['workdir'] / frame['image_relpath']
            data, header = fits.getdata(image_file), fits.getheader(image_file)
            global_rms = frame['background_rms_electron_per_second']
            # organize hdf5 file
            if frame['image_relpath'] not in reg_f.keys():
                frame_set = reg_f.create_group(frame['image_relpath'])
            else:
                frame_set = reg_f[frame['image_relpath']]
            if 'data' not in frame_set.keys():
                data_set = frame_set.create_group('data')
            else:
                data_set = frame_set['data']
            if 'noisemap' not in frame_set.keys():
                noise_set = frame_set.create_group('noisemap')
            else:
                noise_set = frame_set['noisemap']
            if 'wcs' not in frame_set.keys():
                wcs_set = frame_set.create_group('wcs')
            else:
                wcs_set = frame_set['wcs']
            if 'cosmicsmask' not in frame_set.keys():
                cosmic_mask = frame_set.create_group('cosmicsmask')
            else:
                cosmic_mask = frame_set['cosmicsmask']

            if 'ROI' not in cosmic_mask.keys():
                # extract the ROI -- assuming no proper motion.
                cutout, noisemap, wcs_str = extract_stamp(data=data, header=header,
                                                          exptime=frame['exptime'],
                                                          sky_coord=user_config['ROI_SkyCoord'],
                                                          cutout_size=user_config['stamp_size_ROI'],
                                                          background_rms_electron_per_second=global_rms)
                # clean the cosmics
                mask, cleaned = detect_cosmics(cutout, invar=noisemap**2)
                data_set['ROI'] = cleaned
                noise_set['ROI'] = noisemap
                wcs_set['ROI'] = wcs_str
                cosmic_mask['ROI'] = mask

            # set proper motion to 0 when not available
            if len(stars) > 0:  # if 0 stars, then frame will not be queried downstream.
                stars['pmra'][np.isnan(stars['pmra'])] = 0.
                stars['pmdec'][np.isnan(stars['pmdec'])] = 0.
            # extract the stars
            for j, star in stars.iterrows():
                star_name = str(star['name'])
                if star_name not in cosmic_mask.keys():
                    # make a sky coord with our star and its proper motion
                    star_coord = SkyCoord(ra=star['ra'] * u.deg,
                                          dec=star['dec'] * u.deg,
                                          pm_ra_cosdec=star['pmra'] * u.mas / u.yr,
                                          pm_dec=star['pmdec'] * u.mas / u.yr,
                                          frame='icrs',
                                          obstime=Time(star['ref_epoch'], format='decimalyear'))
                    # then correct the proper motion
                    obs_epoch = Time(frame['mjd'], format='mjd')
                    corrected_coord = star_coord.apply_space_motion(new_obstime=obs_epoch)
                    cutout, noisemap, wcs_str = extract_stamp(data=data, header=header,
                                                             exptime=frame['exptime'],
                                                             sky_coord=corrected_coord,
                                                             cutout_size=user_config['stamp_size_stars'],
                                                             background_rms_electron_per_second=global_rms)

                    # again, clean the cosmics.
                    mask, cleaned = detect_cosmics(cutout, invar=noisemap**2)
                    data_set[star_name] = cleaned
                    noise_set[star_name] = noisemap
                    wcs_set[star_name] = wcs_str
                    cosmic_mask[star_name] = mask

