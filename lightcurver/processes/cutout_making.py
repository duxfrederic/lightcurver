import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroscrappy import detect_cosmics
from astropy.time import Time
from ccdproc import CCDData, ccdmask
import logging
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
    :param background_rms_electron_per_second: float, found during global background model with sep: used for noisemap.
    :return: 2d cutout array, 2d cutout noisemap array, wcs string of cutout, center of cutout in original image (x,y)
    """

    wcs = WCS(header)
    data_cutout = Cutout2D(data, sky_coord, cutout_size, wcs=wcs, mode='partial')
    # let's also carry the WCS of the cutouts
    wcs_header = data_cutout.wcs.to_header()
    wcs_header_string = wcs_header.tostring()
    # in case we need to refer back to the original position of the cutout in the fits file:
    original_center_position = data_cutout.center_original

    # now just take the numpy array
    data_cutout_electrons = exptime * data_cutout.data

    noisemap_electrons = ((exptime * background_rms_electron_per_second)**2 + np.abs(data_cutout_electrons))**0.5
    # remove zeros if there are any ...
    noisemap_electrons[noisemap_electrons < 1e-7] = 1e-7

    return data_cutout.data, noisemap_electrons / exptime, wcs_header_string, np.array(original_center_position)


def mask_cutout(cutout_data, noisemap, do_mask_bad_columns, do_mask_cosmics, cosmics_masking_params):
    """
    Masks bad columns and cosmics.

    :param cutout_data: numpy array, stamp
    :param noisemap: numpy array, associated noisemap
    :param do_mask_bad_columns: bool, use ccdproc to find and mask bad columns (before masking cosmics)
    :param do_mask_cosmics: bool, should we mask cosmics with astroscrappy?
    :param cosmics_masking_params: extra arguments to pass to astroscrappy for the masking of cosmics.

    Returns: numpy array: True for masked value, False for good pixels, same shape as data.
    """
    # 1. identifying bad columns and rows
    row_column_mask = np.zeros_like(cutout_data, dtype=bool)
    if do_mask_bad_columns:
        ccd = CCDData(cutout_data, unit='electron/second')
        mask = ccdmask(ccd, findbadcolumns=True)
        # Now, ccdmask will mask anything above the noise. We have to be conservative here, let us only keep lines that
        # extend to both ends of the cutout, "bad columns or rows".
        # Bad columns: check if both top and bottom ends are masked
        bad_columns = np.all([mask[0, :], mask[-1, :]], axis=0)
        # bad rows: check if both left and right ends are masked
        bad_rows = np.all([mask[:, 0], mask[:, -1]], axis=0)
        # new mask with only full bad rows or columns

        row_column_mask[:, bad_columns] = True
        row_column_mask[bad_rows, :] = True

    # 2. masking cosmics
    if do_mask_cosmics:
        # not keeping the "cleaned" cutout.
        cosmics_mask, _ = detect_cosmics(cutout_data, invar=noisemap**2, **cosmics_masking_params)
    else:
        cosmics_mask = np.zeros_like(cutout_data, dtype=bool)

    # 3. combine the two masks: True for masked value, False for good pixels.
    mask = cosmics_mask + row_column_mask
    return mask


def extract_all_stamps():
    """
    This is the routine that the workflow manager will call.
    It interfaces with the user config and the database to locate the
    files and objects to extract, then extracts to a hdf5 file.

    Returns:
        Nothing
    """
    logger = logging.getLogger('lightcurver.cutout_making')
    user_config = get_user_config()

    # where we'll save our stamps
    regions_file = user_config['regions_path']

    # for the cosmic masking
    cosmics_masking_params = user_config['cosmics_masking_params']

    # query frames
    frames_to_process = get_pandas(columns=['id', 'image_relpath', 'exptime', 'mjd',
                                            'background_rms_electron_per_second'],
                                   conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])

    # we'll need to know what combined_footprint we're working with.
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_to_process['id'].to_list())
    logger.info(
        f'Will extract cutouts from (potentially) {len(frames_to_process)} frames, in file {regions_file}. '
        f'The combined footprint hash is {combined_footprint_hash}.'
    )
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
                        if star['gaia_id'] not in keys:
                            all_there = False
                            break
                    if 'ROI' not in keys:
                        all_there = False
                    if all_there:
                        # all good, skip to next frame.
                        logger.info(f"Frame with id {frame['id']} likely already extracted, skipping.")
                        continue
            image_file = user_config['workdir'] / frame['image_relpath']
            data, header = fits.getdata(image_file), fits.getheader(image_file)
            global_rms = frame['background_rms_electron_per_second']
            # organize hdf5 file
            if frame['image_relpath'] not in reg_f.keys():
                frame_set = reg_f.create_group(frame['image_relpath'])
            else:
                frame_set = reg_f[frame['image_relpath']]
            # keep track of the total shape of the image, so we can rescale image coordinates later,
            # useful for distortion.
            if 'frame_shape' not in frame_set:
                frame_set['frame_shape'] = data.shape
            # now the "object dependent" parts: needs groups as there will be one entry per extracted object.
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
            if 'image_pixel_coordinates' not in frame_set.keys():
                pixel_coord_set = frame_set.create_group('image_pixel_coordinates')
            else:
                pixel_coord_set = frame_set['image_pixel_coordinates']
            if 'cosmicsmask' not in frame_set.keys():
                cosmic_mask = frame_set.create_group('cosmicsmask')
            else:
                cosmic_mask = frame_set['cosmicsmask']

            if user_config['redo_stamp_extraction'] or ('ROI' not in cosmic_mask.keys()):
                # extract the ROI -- assuming no proper motion.
                cutout, noisemap, wcs_str, cutout_center = extract_stamp(data=data, header=header,
                                                                         exptime=frame['exptime'],
                                                                         sky_coord=user_config['ROI_SkyCoord'],
                                                                         cutout_size=user_config['stamp_size_ROI'],
                                                                         background_rms_electron_per_second=global_rms)
                # masking
                mask = mask_cutout(cutout_data=cutout, noisemap=noisemap,
                                   do_mask_bad_columns=user_config['mask_bad_rows_and_columns'],
                                   do_mask_cosmics=user_config['clean_cosmics'],
                                   cosmics_masking_params=cosmics_masking_params)

                if 'ROI' in data_set:
                    del data_set['ROI']
                data_set['ROI'] = cutout
                if 'ROI' in noise_set:
                    del noise_set['ROI']
                noise_set['ROI'] = noisemap
                if 'ROI' in wcs_set:
                    del wcs_set['ROI']
                wcs_set['ROI'] = wcs_str
                if 'ROI' in cosmic_mask:
                    del cosmic_mask['ROI']
                cosmic_mask['ROI'] = mask
                if 'ROI' in pixel_coord_set:
                    del pixel_coord_set['ROI']
                pixel_coord_set['ROI'] = cutout_center

            # set proper motion to 0 when not available
            if len(stars) > 0:  # if 0 stars, then frame will not be queried downstream.
                stars.loc[np.isnan(stars['pmra']), 'pmra'] = 0.0
                stars.loc[np.isnan(stars['pmdec']), 'pmdec'] = 0.0
            elif len(stars) == 0:
                logger.warning(
                    f"Frame with id {frame['id']} has no star available for extraction. "
                    "This is fine, it will simply not be used in the downstream steps."
                )
            # extract the stars
            for j, star in stars.iterrows():
                star_name = str(star['gaia_id'])
                if user_config['redo_stamp_extraction'] or (star_name not in cosmic_mask.keys()):
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
                    cutout, noisemap, wcs_str, cutout_center = extract_stamp(
                        data=data, header=header,
                        exptime=frame['exptime'],
                        sky_coord=corrected_coord,
                        cutout_size=user_config['stamp_size_stars'],
                        background_rms_electron_per_second=global_rms
                    )

                    # again, masking
                    mask = mask_cutout(cutout_data=cutout, noisemap=noisemap,
                                       do_mask_bad_columns=user_config['mask_bad_rows_and_columns'],
                                       do_mask_cosmics=user_config['clean_cosmics'],
                                       cosmics_masking_params=cosmics_masking_params)

                    if star_name in data_set:
                        del data_set[star_name]
                    data_set[star_name] = cutout
                    if star_name in noise_set:
                        del noise_set[star_name]
                    noise_set[star_name] = noisemap
                    if star_name in wcs_set:
                        del wcs_set[star_name]
                    wcs_set[star_name] = wcs_str
                    if star_name in cosmic_mask:
                        del cosmic_mask[star_name]
                    cosmic_mask[star_name] = mask
                    if star_name in pixel_coord_set:
                        del pixel_coord_set[star_name]
                    pixel_coord_set[star_name] = cutout_center

            logger.info(f"Frame with id {frame['id']}: completed making of cutouts.")
