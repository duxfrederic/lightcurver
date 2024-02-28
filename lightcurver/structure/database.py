import sqlite3


def initialize_database(db_path):
    """
    initializes the database we'll be working with to keep track of our images.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # table of frames
    column_definitions = [
        "id INTEGER PRIMARY KEY",
        "filter TEXT",
        "mjd REAL",
        "exptime REAL",
        "gain REAL",
        "original_image_path TEXT",
        "copied_image_path TEXT",
        "telescope_latitude REAL",
        "telescope_longitude REAL",
        "telescope_altitude REAL",
        "telescope_name TEXT",
        "imager_name TEXT",
        "eliminated INTEGER DEFAULT 0",
        "airmass REAL DEFAULT NULL",
        "degrees_to_moon REAL DEFAULT NULL",
        "moon_phase REAL DEFAULT NULL",
        "sun_altitude REAL DEFAULT NULL",
        "seeing_arcsecond REAL DEFAULT NULL"
        # potential new columns in future here
    ]

    # create the 'frames' table if it doesn't exist
    cursor.execute(f"CREATE TABLE IF NOT EXISTS frames ({', '.join(column_definitions)})")

    # add new columns to the 'frames' table if they're not already present
    for column_definition in column_definitions:
        column_name = column_definition.split()[0]
        try:
            cursor.execute(f"ALTER TABLE frames ADD COLUMN {column_definition}")
        except sqlite3.OperationalError:
            # operational error if column exists, ignore
            pass

    # table of stars
    # the stars will be filled in once by a python process.
    cursor.execute("""CREATE TABLE IF NOT EXISTS Stars (
                      id INTEGER PRIMARY KEY,
                      name TEXT, -- typically a letter
                      ra REAL, -- Right Ascension
                      dec REAL, -- Declination,
                      gmag REAL,
                      rmag REAL,
                      bmag REAL,                      
                      gaia_id TEXT -- Gaia ID);""")

    # linking stars and frame
    # again, a python process will check the footprint of each image
    # and fill in this table once for each image.
    cursor.execute("""CREATE TABLE IF NOT EXISTS FrameStars (
                      frame_id INTEGER,
                      star_id INTEGER,
                      FOREIGN KEY (frame_id) REFERENCES Frames(id),
                      FOREIGN KEY (star_id) REFERENCES Stars(id),
                      PRIMARY KEY (frame_id, star_id));""")

    # PSF is defined in yaml files (which stars compose it), here we keep track of which images have
    # a given PSF.
    # we trace back to the above link of stars and frame to know which stars entered
    # the PSF exactly.
    # the subsampling factor is also defined in yaml file.
    cursor.execute("""CREATE TABLE IF NOT EXISTS PSFs (
                      id INTEGER, 
                      frame_id INTEGER,
                      float chi2, -- chi2 of the fit of the PSF
                      psf_name TEXT, -- reference to the config file given in name
                      hdf5_route TEXT, -- where in the hdf5 file is the PSF,
                      FOREIGN KEY (frame_id) REFERENCES Frames(id),
                      PRIMARY KEY (id, frame_id);""")

    # very similarly, we keep track of normalization coefficients.
    # these are computed once per image in python.
    cursor.execute("""CREATE TABLE IF NOT EXISTS NormalizationCoefficients (
                      id INTEGER, 
                      frame_id INTEGER,
                      psf_id INTEGER,
                      coefficient_name TEXT, -- reference to the config file given in name
                      coefficient FLOAT,
                      coefficient_uncertainty FLOAT,
                      FOREIGN KEY (frame_id) REFERENCES Frames(id),
                      FOREIGN KEY (psf_id) REFERENCES PSFs(id),
                      PRIMARY KEY (id, psf_id, frame_id);""")

    # zero points are derived from normalization coefficients and are for APPROXIMATE magnitude calibration
    # as we are going to derive them from gaia colors and band conversions.
    cursor.execute("""CREATE TABLE IF NOT EXISTS ApproximateZeroPoints (
                      id INTEGER, 
                      norm_coefficient_id INTEGER,
                      zeropoint FLOAT,
                      zeropoint_uncertainty FLOAT,
                      FOREIGN KEY (norm_coefficient_id) REFERENCES NormalizationCoefficients(id),
                      PRIMARY KEY (id, norm_coefficient_id);""")

    conn.commit()
    conn.close()


def add_frame_to_database(original_image_path, copied_image_path,
                          frame_fits_header, dictionary_of_keywords,
                          database_connexion,
                          telescope_information=None):

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
    :param database_connexion: SQLite3 connection object to the database
    :param telescope_information: Optional dictionary with telescope information
    :return: None
    """
    columns = ['original_image_path', 'copied_image_path']
    values = [original_image_path, copied_image_path]

    for header_keyword, db_column in dictionary_of_keywords.items():
        columns.append(db_column)  # add the database column name
        values.append(frame_fits_header.get(header_keyword, None))  # corresponding value from the FITS header

    # if telescope information, add it to the columns and values
    if telescope_information is not None:
        for key, value in telescope_information.items():
            columns.append(key)
            values.append(value)

    # construct the SQL query based on the columns
    column_names = ', '.join(columns)
    placeholders = ', '.join(['?'] * len(columns))
    query = f'INSERT INTO frames ({column_names}) VALUES ({placeholders})'

    # the query
    cursor = database_connexion.cursor()
    cursor.execute(query, values)
    database_connexion.commit()

