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
from ..utilities.footprint import get_frames_hash


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
    stddev = np.nanmean([
           np.nanstd(data_cutout_electrons[:, 0]),
           np.nanstd(data_cutout_electrons[0, :]),
           np.nanstd(data_cutout_electrons[:, -1]),
           np.nanstd(data_cutout_electrons[-1, :]),
    ])
    noisemap = stddev + np.sqrt(np.abs(data_cutout_electrons))
    # remove zeros if there are any ...
    noisemap[noisemap < 1e-7] = 1e-7

    return data_cutout_electrons / exptime, noisemap / exptime, wcs_header_string


def extract_all_stamps():
    user_config = get_user_config()

    # where we'll save our stamps
    regions_file = user_config['regions_path']

    if user_config['redo_stamp_extraction']:
        # start from scratch ...
        if regions_file.exists():
            # delete if exists.
            regions_file.unlink()

    # query frames
    frames_to_process = get_pandas(columns=['id', 'image_relpath', 'exptime', 'mjd'],
                                   conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])

    # we'll need to know what combined_footprint we're working with.
    # so, calculate the hash! TODO factor out some day? how?
    if user_config['star_selection_strategy'] != 'ROI_disk':
        # then it depends on the frames we're considering.
        frames_hash = get_frames_hash(frames_to_process['id'].to_list())
    else:
        # if ROI_disk, it does not depend on the frames: unique region defined by its radius.
        frames_hash = hash(user_config['ROI_disk_radius_arcseconds'])

    # suppress the no parallax warning in proper motion corrections:
    warnings.filterwarnings('ignore', category=erfa.ErfaWarning)

    with h5py.File(regions_file, 'a') as regf:
        for i, frame in frames_to_process.iterrows():
            if frame['image_relpath'] in regf.keys():
                continue
            image_file = user_config['workdir'] / frame['image_relpath']

            try:
                data, header = fits.getdata(image_file), fits.getheader(image_file)
            except Exception as E:
                print(f"Problem with {user_config['image_relpath']}: {E}")
                continue

            # organize hdf5 file
            frame_set = regf.create_group(frame['image_relpath'])
            data_set = frame_set.create_group('data')
            noise_set = frame_set.create_group('noisemap')
            wcs_set = frame_set.create_group('wcs')
            cosmic_mask = frame_set.create_group('cosmicsmask')

            # extract the ROI -- assuming no proper motion.
            cutout, noisemap, wcsstr = extract_stamp(data=data, header=header,
                                                     exptime=frame['exptime'],
                                                     sky_coord=user_config['ROI_SkyCoord'],
                                                     cutout_size=user_config['stamp_size_ROI'])
            # clean the cosmics
            mask, cleaned = detect_cosmics(cutout, invar=noisemap**2)
            data_set['ROI'] = cleaned
            noise_set['ROI'] = noisemap
            wcs_set['ROI'] = wcsstr
            cosmic_mask['ROI'] = mask

            stars = query_stars_for_frame_and_footprint(frame_id=frame['id'],
                                                        combined_footprint_hash=frames_hash)

            # extract the stars
            for j, star in stars.iterrows():
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
                cutout, noisemap, wcsstr = extract_stamp(data=data, header=header,
                                                         exptime=frame['exptime'],
                                                         sky_coord=corrected_coord,
                                                         cutout_size=user_config['stamp_size_stars'])

                source_id = str(star['name'])
                # again, clean the cosmics.
                mask, cleaned = detect_cosmics(cutout, invar=noisemap**2)
                data_set[source_id] = cleaned
                noise_set[source_id] = noisemap
                wcs_set[source_id] = wcsstr
                cosmic_mask[source_id] = mask

