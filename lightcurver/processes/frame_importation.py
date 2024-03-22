import numpy as np
import sqlite3
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from time import sleep
import logging

from .background_estimation import subtract_background
from .star_extraction import extract_stars
from .frame_characterization import ephemeris, estimate_seeing
from ..structure.user_header_parser import load_custom_header_parser


def process_new_frame(fits_file, user_config):
    """
        crops, transforms to e- and writes the result to our workdir.

        fits_file: path or str to a fits file.

        :param fits_file: path
        :param user_config: dictionary containing user config
        :param logger: a logger.
    """
    trim_vertical = user_config.get('trim_vertical', 0)
    trim_horizontal = user_config.get('trim_horizontal', 0)
    copied_image_relpath = Path('frames') / f"{fits_file.stem}.fits"
    logger = logging.getLogger("lightcurver.importation")

    try:
        with fits.open(str(fits_file), mode='readonly', ignore_missing_end=False, memmap=True) as hdu:
            logger.info(f'  Importing {fits_file}.')
            hdu_index = 1 if len(hdu) > 1 else 0
            data_shape = hdu[hdu_index].data.shape
            cutout_data = hdu[hdu_index].section[
                          trim_vertical:data_shape[0] - trim_vertical,
                          trim_horizontal:data_shape[1] - trim_horizontal
                          ]
            header = hdu[hdu_index].header
            logger.info(f"Read fits file at {fits_file}")
    except ValueError:
        # then we have some memmap problems, if  bzero, bscale, or blank are in header.
        with fits.open(str(fits_file), mode='readonly') as hdu:
            logger.info(f'  Importing {fits_file}.')
            hdu_index = 1 if len(hdu) > 1 else 0
            data_shape = hdu[hdu_index].data.shape
            cutout_data = hdu[hdu_index].data[
                          trim_vertical:data_shape[0] - trim_vertical,
                          trim_horizontal:data_shape[1] - trim_horizontal
                          ]
            header = hdu[hdu_index].header
            logger.warning(f"Read fits file at {fits_file}, could not use memmap!!")
    cutout_data = cutout_data.astype(float)
    wcs = WCS(header)
    # so we cropped our data, thus we need to change the CRPIX of our WCS
    if wcs.is_celestial:
        wcs.wcs.crpix[0] -= trim_horizontal
        wcs.wcs.crpix[1] -= trim_vertical

    header['BUNIT'] = "ELPERSEC"

    # the user defines their header parser, returning a dictionary with {'mjd':, 'gain':, 'filter':, 'exptime':}
    # it needs be located at $workdir/header_parser/parse_header.py
    # the function needs be called parse_header, and it has to accept a fits header as argument and return the
    # dictionary above.
    header_parser_function = load_custom_header_parser()
    mjd_gain_filter_exptime_dict = header_parser_function(header)
    logger.info(f'  file {fits_file}: parsed header.')
    cutout_data *= mjd_gain_filter_exptime_dict['gain'] / mjd_gain_filter_exptime_dict['exptime']
    # unit: electron per second

    # ok, now subtract sky!
    cutout_data_sub, bkg = subtract_background(cutout_data,
                                               mask_sources_first=user_config['mask_sources_before_background'],
                                               n_boxes=user_config['background_estimation_n_boxes'])
    sky_level_electron_per_second = float(bkg.globalback)
    background_rms_electron_per_second = float(bkg.globalrms)
    logger.info(f'  file {fits_file}: background estimated.')

    # before we write, let's keep as much as we can from our previous header
    new_header = wcs.to_header()
    # then copy non-WCS entries from the original header
    for key in header:
        if key not in new_header and not key.startswith('WCSAXES') and not key.startswith(
                'CRPIX') and not key.startswith('CRVAL') and not key.startswith('CDELT') and not key.startswith(
                'CTYPE') and not key.startswith('CUNIT') and not key.startswith('CD') and key not in ['COMMENT',
                                                                                                      'HISTORY']:
            if key.strip():  # some headers are weird
                new_header[key] = header[key]
    # now we can write the file
    write_file = user_config['workdir'] / copied_image_relpath
    logger.info(f'    Writing file: {write_file}')
    fits.writeto(write_file, cutout_data_sub.astype(np.float32),
                 header=new_header, overwrite=True)
    # and find sources
    # (do plots if toggle set)
    do_plot = user_config.get('source_extraction_do_plots', False)
    plot_path = user_config['plots_dir'] / 'source_extraction' / f'{fits_file.stem}.jpg' if do_plot else None
    # we need a proper noise map here to correctly detect our sources.
    exptime = mjd_gain_filter_exptime_dict['exptime']
    cutout_data_sub_el = cutout_data_sub * exptime
    background_rms = bkg.globalrms * exptime
    variance_map = background_rms**2 + np.abs(cutout_data_sub_el)
    sources_table = extract_stars(image_background_subtracted=cutout_data_sub,
                                  variance_map=variance_map / exptime**2,  # scale back to (e-/s)^2
                                  detection_threshold=user_config.get('source_extraction_threshold', 3),
                                  min_area=user_config.get('source_extraction_min_area', 10),
                                  debug_plot_path=plot_path)
    # saving the sources in the same dir as the frame itself
    sources_file_filename = f"{copied_image_relpath.stem}_sources{copied_image_relpath.suffix}"
    sources_file_relpath = copied_image_relpath.parent / sources_file_filename
    sources_table.write(user_config['workdir'] / sources_file_relpath, format='fits', overwrite=True)

    seeing_pixels = estimate_seeing(sources_table)
    ellipticity = np.nanmedian(sources_table['ellipticity'])
    logger.info(
        f"Extracted {len(sources_table)} sources from {fits_file}. "
        f"Seeing pixels: {seeing_pixels:.02f}. "
        f"Ellipticity: {ellipticity:.02f}."
    )

    if 'telescope' in user_config:
        eph_dict = ephemeris(mjd=mjd_gain_filter_exptime_dict['mjd'],
                             ra_object=user_config['ROI_ra_deg'],
                             dec_object=user_config['ROI_dec_deg'],
                             telescope_longitude=user_config['telescope']['longitude'],
                             telescope_latitude=user_config['telescope']['latitude'],
                             telescope_elevation=user_config['telescope']['elevation'])
        telescope_information = {k: v for k, v in user_config['telescope'].items()}
        if eph_dict['weird_astro_conditions']:
            logger.warning(
                f'Ephemeris: weird for this frame ({fits_file}), please inspect and make sure your '
                'config (such as telescope location and ROI coordinates) and fits MJD are all correct. '
                f'Ephemeris dictionary for reference: {eph_dict}'
            )
    else:
        logger.warning(
            'Telescope information not provided in user config. No Ephemeris calculated. '
            'Consider adding them as this can help catch mistakes.'
        )
        telescope_information = None
        eph_dict = None

    # now we're ready for registering our new frame!
    add_frame_to_database(original_image_path=fits_file,
                          copied_image_relpath=copied_image_relpath,
                          sources_relpath=sources_file_relpath,
                          mjd=mjd_gain_filter_exptime_dict['mjd'],
                          gain=mjd_gain_filter_exptime_dict['gain'],
                          sky_level_electron_per_second=sky_level_electron_per_second,
                          background_rms_electron_per_second=background_rms_electron_per_second,
                          filter=mjd_gain_filter_exptime_dict['filter'],
                          exptime=mjd_gain_filter_exptime_dict['exptime'],
                          seeing_pixels=seeing_pixels,
                          ellipticity=ellipticity,
                          ephemeris_dictionary=eph_dict,
                          telescope_information=telescope_information,
                          user_config=user_config)
    return header


def add_frame_to_database(original_image_path, copied_image_relpath, sources_relpath,
                          mjd, gain,
                          sky_level_electron_per_second, background_rms_electron_per_second,
                          filter, exptime,
                          seeing_pixels, ellipticity, user_config,
                          telescope_information=None,
                          ephemeris_dictionary=None):
    """
    Adding our new image frame to our sqlite3 database. We will use the table "frames".
    The columns to be populated are:
      - filter
      - mjd
      - exptime
      - gain
      - original_image_path
      - copied_image_path
    the translation between database columns and header keywords is done by the dictionary
    "dictionary_of_keywords"
    if a 'telescope_information' dictionary is provided, we will also fill in the following columns:
      - telescope_latitude
      - telescope_longitude
      - telescope_altitude
      - telescope_name
      - imager_name

    :param original_image_path: Path to the original image
    :param copied_image_relpath: Path where the image was copied, relative to workdir
    :param sources_relpath: filename of the file of sources (fits table) as extracted by sep, relative to workdir
    :param mjd: float, mjd of frame
    :param gain: float
    :param sky_level_electron_per_second: float
    :param background_rms_electron_per_second: float
    :param filter: string, filter of the observations
    :param exptime: float, exposure time
    :param seeing_pixels: Seeing value measured for this frame, float.
    :param ellipticity: ellipticity of the psf, calculated as 1 - b/a
    :param user_config: dictionary containing the user config
    :param telescope_information: Optional dictionary with telescope information
    :param ephemeris_dictionary: dictionary as returned by the ephemeris function of frame_characterization.
    :return: None
    """
    columns = ['original_image_path', 'image_relpath', 'sources_relpath',
               'seeing_pixels', 'mjd', 'gain', 'sky_level_electron_per_second', 'background_rms_electron_per_second',
               'filter', 'exptime', 'ellipticity']

    values = [str(original_image_path), str(copied_image_relpath), str(sources_relpath),
              seeing_pixels, mjd, gain,
              sky_level_electron_per_second, background_rms_electron_per_second,
              filter, exptime, ellipticity]

    # if telescope information, add it to the columns and values
    if telescope_information is not None:
        for key, value in telescope_information.items():
            columns.append(f"telescope_{key}")
            values.append(value)

    if ephemeris_dictionary is not None:
        columns.append('airmass')
        values.append(float(ephemeris_dictionary['target_info']['airmass']))
        columns.append('degrees_to_moon')
        values.append(ephemeris_dictionary['moon_info']['distance_deg'])
        columns.append('moon_phase')
        values.append(ephemeris_dictionary['moon_info']['illumination'])
        columns.append('sun_altitude')
        values.append(ephemeris_dictionary['sun_info']['altitude_deg'])
        columns.append('azimuth')
        values.append(ephemeris_dictionary['target_info']['azimuth_deg'])
        columns.append('altitude')
        values.append(ephemeris_dictionary['target_info']['altitude_deg'])

    # we'll return this whole thing at the end for potential further use
    frame_info = {key: value for key, value in zip(columns, values)}

    # construct the SQL query based on the columns
    column_names = ', '.join(columns)
    placeholders = ', '.join(['?'] * len(columns))
    query = f'INSERT INTO frames ({column_names}) VALUES ({placeholders})'

    # the query
    try:
        inserted = False
        while not inserted:
            # We are being very careful, sqlite3 databases are not made for parallel writing.
            # Works well so long as the storage (on which the database is) is fairly responsive.
            # Local hard drives and SSDs all work, virtual storage volumes as well.
            # Only case of failure: an SSD I have that clearly has problems (response time 5-15 seconds for
            # touching a file)
            try:
                conn = sqlite3.connect(user_config['database_path'], timeout=2.5)
                cursor = conn.cursor()
                # cursor.execute("PRAGMA journal_mode=WAL")
                # cursor.execute("PRAGMA busy_timeout=2500")
                cursor.execute(query, values)
                conn.commit()
                inserted = True
                conn.close()
            except sqlite3.OperationalError:
                print('database locked, waiting')
                sleep(np.random.uniform(0, 0.5))  # retry after random wait
            finally:
                try:
                    conn.close()
                except:
                    pass
    except sqlite3.IntegrityError as IntE:
        logger = logging.getLogger("lightcurver.importation")
        logger.error(
            "Error: we might be inserting an already existing image again in the database. "
            "You most likely overwrote an existing file already, leaving us in an inconsistent state. "
            f"This should not have happened and will take manual fixing. Here was the original error: {IntE} "
            "We will raise the error again so you can check the traceback. "
            "If you see a UNIQUE violation on the image_relpath column, then the above is indeed what happened."
            "Force stopping pipeline."
        )
        raise IntE
    return frame_info

