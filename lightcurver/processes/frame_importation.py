import numpy as np
import sqlite3
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

from .background_estimation import subtract_background
from .star_extraction import extract_stars
from .frame_characterization import ephemeris, estimate_seeing
from ..structure.user_header_parser import load_custom_header_parser
# the user defines their header parser, returning a dictionary with {'mjd':, 'gain':, 'filter':, 'exptime':}
# it needs be located at $workdir/header_parser/parse_header.py
# the function needs be called parse_header, and it has to accept a fits header as argument and return the
# dictionary above.
header_parser_function = load_custom_header_parser()


def process_new_frame(fits_file, user_config, logger):
    """
        crops, transforms to e- and writes the result to our workdir.

        fits_file: path or str to a fits file.

        :param fits_file: path
        :param user_config: dictionary containing user config
        :param logger: a logger.
    """
    trim_vertical = user_config.get('trim_vertical', 0)
    trim_horizontal = user_config.get('trim_horizontal', 0)
    copied_image_relpath = Path('frames') / fits_file.name
    logger.info(f'  Importing {fits_file}.')
    with fits.open(str(fits_file), mode='readonly', ignore_missing_end=True, memmap=True) as hdu:
        hdu_index = 1 if len(hdu) > 1 else 0
        data_shape = hdu[hdu_index].data.shape
        cutout_data = hdu[hdu_index].section[
                      trim_vertical:data_shape[0] - trim_vertical,
                      trim_horizontal:data_shape[1] - trim_horizontal
                      ]
        cutout_data = cutout_data.astype(float)
        header = hdu[hdu_index].header
        wcs = WCS(header)
        # so we cropped our data, thus we need to change the CRPIX of our WCS
        if wcs.is_celestial:
            wcs.wcs.crpix[0] -= trim_horizontal
            wcs.wcs.crpix[1] -= trim_vertical

        header['BUNIT'] = "e-/s"
        mjd_gain_filter_exptime_dict = header_parser_function(header)
        cutout_data *= mjd_gain_filter_exptime_dict['gain'] / mjd_gain_filter_exptime_dict['exptime']
        # unit: electron per second

        # ok, now subtract sky!
        cutout_data_sub, bkg = subtract_background(cutout_data)
        sky_level_electron_per_second = float(bkg.globalback)
        background_rms_electron_per_second = float(bkg.globalrms)

        # before we write, let's keep as much as we can from our previous header
        new_header = wcs.to_header()
        # then copy non-WCS entries from the original header
        for key in header:
            if key not in new_header and not key.startswith('WCSAXES') and not key.startswith(
                    'CRPIX') and not key.startswith('CRVAL') and not key.startswith('CDELT') and not key.startswith(
                    'CTYPE') and not key.startswith('CUNIT') and not key.startswith('CD') and key not in ['COMMENT',
                                                                                                          'HISTORY']:
                new_header[key] = header[key]
        # now we can write the file
        fits.writeto(user_config['workdir'] / copied_image_relpath, cutout_data_sub.astype(np.float32),
                     header=new_header, overwrite=True)
        # and find sources
        # (do plots if toggle set)
        do_plot = user_config.get('source_extraction_do_plots', False)
        plot_path = user_config['plots_dir'] / 'source_extraction' / f'{fits_file.stem}.jpg' if do_plot else None
        sources_table = extract_stars(image_background_subtracted=cutout_data_sub,
                                      background_rms=bkg.globalrms,
                                      detection_threshold=user_config.get('source_extraction_threshold', 3),
                                      min_area=user_config.get('source_extraction_min_area', 10),
                                      debug_plot_path=plot_path)

        # saving the sources in the same dir as the frame itself
        sources_file_filename = f"{copied_image_relpath.stem}_sources{copied_image_relpath.suffix}"
        sources_file_relpath = copied_image_relpath.parent / sources_file_filename
        sources_table.write(user_config['workdir'] / sources_file_relpath, format='fits', overwrite=True)

        seeing_pixels = estimate_seeing(sources_table)
        ellipticity = np.nanmedian(sources_table['ellipticity'])
        if 'telescope' in user_config:
            eph_dict = ephemeris(mjd=mjd_gain_filter_exptime_dict['mjd'],
                                 ra_object=user_config['ROI_ra_deg'],
                                 dec_object=user_config['ROI_dec_deg'],
                                 telescope_longitude=user_config['telescope']['longitude'],
                                 telescope_latitude=user_config['telescope']['latitude'],
                                 telescope_elevation=user_config['telescope']['elevation'])
            telescope_information = {k: v for k, v in user_config['telescope'].items()}
        else:
            telescope_information = None
            eph_dict = None

    # now we're ready for registering our new frame!
    conn = sqlite3.connect(user_config['database_path'], timeout=15.0)
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
                          database_connexion=conn,
                          telescope_information=telescope_information)
    conn.commit()
    conn.close()
    return header


def add_frame_to_database(original_image_path, copied_image_relpath, sources_relpath,
                          mjd, gain,
                          sky_level_electron_per_second, background_rms_electron_per_second,
                          filter, exptime,
                          seeing_pixels, ellipticity,
                          database_connexion,
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
    :param database_connexion: SQLite3 connection object to the database
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

    # we'll return this whole thing at the end for further use
    frame_info = {key: value for key, value in zip(columns, values)}

    # construct the SQL query based on the columns
    column_names = ', '.join(columns)
    placeholders = ', '.join(['?'] * len(columns))
    query = f'INSERT INTO frames ({column_names}) VALUES ({placeholders})'

    # the query
    cursor = database_connexion.cursor()
    try:
        cursor.execute(query, values)
        database_connexion.commit()
    except sqlite3.IntegrityError as IntE:
        print("Error: most likely, we are inserting an already existing image again in the database.")
        print("You most likely overwrote an existing file already, leaving us in an inconsistent state.")
        print("This should not have happened and will take manual fixing. Here was the original error: {IntE}")
        print("We will raise it again so you can check the traceback.")
        print("If you see a UNIQUE violation on the image_relpath column, then the above is indeed what happened.")
        raise IntE
    return frame_info

