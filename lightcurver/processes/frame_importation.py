from astropy.io import fits
import sqlite3

from .background_estimation import subtract_background
from .star_extraction import extract_stars
from .frame_characterization import ephemeris, estimate_seeing


def process_new_frame(fits_file, user_config, dictionary_of_keywords):
    """
        crops, transforms to e- and writes the result to our workdir.

        fits_file: path or str to a fits file.

        :param fits_file: path
        :param user_config: dictionary containing user config
        :param dictionary_of_keywords: map between header keywords and database columns
    """
    trim_vertical = user_config.get('trim_vertical', 0)
    trim_horizontal = user_config.get('trim_horizontal', 0)
    out_file = user_config['workdir'] / fits_file.name
    with fits.open(str(fits_file), mode='readonly', ignore_missing_end=True, memmap=True) as hdu:
        hdu_index = 1 if len(hdu) > 1 else 0
        data_shape = hdu[hdu_index].data.shape
        cutout_data = hdu[hdu_index].section[
                      trim_vertical:data_shape[0] - trim_vertical,
                      trim_horizontal:data_shape[1] - trim_horizontal
                      ]
        header = hdu[hdu_index].header
        cutout_data *= header[dictionary_of_keywords['gain']]
        # while we're at it
        mjd = header[dictionary_of_keywords['mjd']]

        # ok, now subtract sky!
        cutout_data_sub, bkg = subtract_background(cutout_data)
        # we can write the file
        fits.writeto(out_file, cutout_data_sub, header=header, overwrite=True)
        # and find sources
        # (do plots if toggle set)
        do_plot = user_config.get('source_extraction_do_plots', False)
        plot_path = user_config['plots_dir'] / 'source_extraction' / f'{fits_file.stem}.jpg' if do_plot else None

        sources_table = extract_stars(image_background_subtracted=cutout_data_sub,
                                      background_rms=bkg.rms,
                                      detection_threshold=user_config.get('source_extraction_threshold', 3),
                                      min_area=user_config.get('source_extraction_min_area', 10),
                                      debug_plot_path=plot_path)

        seeing = estimate_seeing(sources_table)
        if 'telescope_longitude' in user_config:
            eph_dict = ephemeris(mjd=mjd,
                                 ra_object=user_config['ra_object'],
                                 dec_object=user_config['dec_object'],
                                 telescope_longitude=user_config['telescope_longitude'],
                                 telescope_latitude=user_config['telescope_latitude'],
                                 telescope_elevation=user_config['telescope_elevation'])
            telescope_information = {k: v for k, v in user_config.items() if 'telescope_' in k}
        else:
            telescope_information = None

    # now we're ready for registering our new frame!
    conn = sqlite3.connect(user_config['database_path'], timeout=15.0)
    add_frame_to_database(original_image_path=str(fits_file),
                          copied_image_path=str(out_file),
                          frame_fits_header=header,
                          dictionary_of_keywords=dictionary_of_keywords,
                          seeing=seeing,
                          ephemeris_dictionary=eph_dict,
                          database_connexion=conn,
                          telescope_information=telescope_information)
    conn.commit()
    conn.close()
    return header


def add_frame_to_database(original_image_path, copied_image_path,
                          frame_fits_header, dictionary_of_keywords,
                          seeing,
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
    :param copied_image_path: Path where the image was copied
    :param frame_fits_header: FITS header containing frame information
    :param dictionary_of_keywords: Dictionary translating FITS header keywords to database column names
    :param seeing: Seeing value measured for this frame, float.
    :param database_connexion: SQLite3 connection object to the database
    :param telescope_information: Optional dictionary with telescope information
    :param ephemeris_dictionary: dictionary as returned by the ephemerides function of frame_characterization.
    :return: None
    """
    columns = ['original_image_path', 'copied_image_path', 'seeing_pixels']
    values = [original_image_path, copied_image_path, seeing]

    for header_keyword, db_column in dictionary_of_keywords.items():
        columns.append(db_column)  # add the database column name
        values.append(frame_fits_header.get(header_keyword, None))  # corresponding value from the FITS header

    # if telescope information, add it to the columns and values
    if telescope_information is not None:
        for key, value in telescope_information.items():
            columns.append(key)
            values.append(value)

    if ephemeris_dictionary is not None:
        columns.append('airmass')
        values.append(ephemeris_dictionary['target_info']['airmass'])
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
    cursor.execute(query, values)
    database_connexion.commit()
    return frame_info

