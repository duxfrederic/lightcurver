import sqlite3
import pandas as pd

from ..structure.user_config import get_user_config


def get_pandas(conditions=None, columns=None, table='frames'):
    """
    Retrieve data from the database based on specified conditions and return as a Pandas DataFrame.

    Parameters:
        conditions (list of str): The SQL conditions to filter the data, default None.
        columns (str or list of str, optional): Columns to select from the database. Defaults to None,
            which selects all columns.
        table: string, which table to fetch data from.
    Returns:
        pandas.DataFrame: A DataFrame containing the results of the SQL query.
    """
    db_path = get_user_config()['database_path']
    conn = sqlite3.connect(db_path)
    if columns is None:
        columns = '*'
    request = f"SELECT {','.join(columns)} FROM {table}"
    if conditions is not None:
        conditions = ','.join(conditions)
        request += f" WHERE {conditions}"

    try:
        df = pd.read_sql_query(request, conn)
        return df
    finally:
        conn.close()


def execute_sqlite_query(query, params=(), is_select=True, timeout=15.0):
    db_path = get_user_config()['database_path']
    with sqlite3.connect(db_path, timeout=timeout) as conn:
        cursor = conn.cursor()
        if is_select:
            cursor.execute(query, params)
            return cursor.fetchall()
        else:
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount


def get_count_based_on_conditions(conditions, table='frames'):
    db_path = get_user_config()['database_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    request = f"select count(*) from {table} where {conditions}"

    return cursor.execute(request).fetchone()[0]


def initialize_database():
    """
    initializes the database we'll be working with to keep track of our images.
    """
    db_path = get_user_config()['database_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # table of frames
    column_definitions = [
        "id INTEGER PRIMARY KEY",
        "filter TEXT",
        "mjd REAL",
        "exptime REAL",
        "gain REAL",
        "original_image_path TEXT",  # e.g. /some/drive/2023-02-01T01:23:35.fits
        "image_relpath TEXT UNIQUE",  # e.g. images/2023-02-01T01:23:35.fits -- relative to $workdir
        "sources_relpath TEXT",  # convention: images/2023-02-01T01:23:35_sources.fits  -- same dir as above
        "telescope_latitude REAL",
        "telescope_longitude REAL",
        "telescope_elevation REAL",
        "telescope_name TEXT",
        "telescope_imager_name TEXT",
        "plate_solved INTEGER DEFAULT 0",
        "eliminated INTEGER DEFAULT 0",
        "airmass REAL DEFAULT NULL",
        "degrees_to_moon REAL DEFAULT NULL",
        "moon_phase REAL DEFAULT NULL",
        "sun_altitude REAL DEFAULT NULL",
        "seeing_pixels REAL DEFAULT NULL",
        "seeing_arcseconds REAL DEFAULT NULL",
        "azimuth REAL DEFAULT NULL",
        "altitude REAL DEFAULT NULL",
        "comment TEXT DEFAULT NULL",
        "roi_in_footprint INTEGER DEFAULT 1",
        # potential new stuff to keep track on in future here
    ]

    # create the 'frames' table if it doesn't exist
    cursor.execute(f"CREATE TABLE IF NOT EXISTS frames ({', '.join(column_definitions)})")

    # add new columns to the 'frames' table if they're not already present
    for column_definition in column_definitions:
        try:
            cursor.execute(f"ALTER TABLE frames ADD COLUMN {column_definition}")
        except sqlite3.OperationalError:
            # operational error if column exists, ignore
            pass

    # table of stars
    # the stars will be filled in once by a python process.
    cursor.execute("""CREATE TABLE IF NOT EXISTS stars (
                      id INTEGER PRIMARY KEY,
                      name TEXT, -- typically a letter
                      ra REAL, -- Right Ascension
                      dec REAL, -- Declination,
                      gmag REAL,
                      rmag REAL,
                      bmag REAL,                      
                      gaia_id TEXT -- Gaia ID
                      )""")

    # linking stars and frame
    # again, a python process will check the footprint of each image
    # and fill in this table once for each image.
    cursor.execute("""CREATE TABLE IF NOT EXISTS stars_in_frames (
                      frame_id INTEGER,
                      star_id INTEGER,
                      FOREIGN KEY (frame_id) REFERENCES frames(id),
                      FOREIGN KEY (star_id) REFERENCES stars(id),
                      PRIMARY KEY (frame_id, star_id)
                      )""")

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
                      FOREIGN KEY (frame_id) REFERENCES frames(id),
                      PRIMARY KEY (id, frame_id)
                      )""")

    # very similarly, we keep track of normalization coefficients.
    # these are computed once per image in python.
    cursor.execute("""CREATE TABLE IF NOT EXISTS normalization_coefficients (
                      id INTEGER, 
                      frame_id INTEGER,
                      psf_id INTEGER,
                      coefficient_name TEXT, -- reference to the config file given in name
                      coefficient FLOAT,
                      coefficient_uncertainty FLOAT,
                      FOREIGN KEY (frame_id) REFERENCES frames(id),
                      FOREIGN KEY (psf_id) REFERENCES PSFs(id),
                      PRIMARY KEY (id, psf_id, frame_id)
                      )""")

    # zero points are derived from normalization coefficients and are for APPROXIMATE magnitude calibration
    # as we are going to derive them from gaia colors and band conversions.
    cursor.execute("""CREATE TABLE IF NOT EXISTS approximate_zeropoints (
                      id INTEGER, 
                      norm_coefficient_id INTEGER,
                      zeropoint FLOAT,
                      zeropoint_uncertainty FLOAT,
                      FOREIGN KEY (norm_coefficient_id) REFERENCES normalization_coefficients(id),
                      PRIMARY KEY (id, norm_coefficient_id)
                      )""")

    # table of footprints
    cursor.execute("""CREATE TABLE IF NOT EXISTS footprints (
                      frame_id INTEGER PRIMARY KEY,
                      polygon TEXT NOT NULL,
                      FOREIGN KEY (frame_id) REFERENCES frames (id)
                      )""")

    # and also a smaller one: "combined" (intersection or union) of footprints given a hash
    # calculated from the sorted list of frames (image_relpath) composing the common footprint.
    cursor.execute("""CREATE TABLE IF NOT EXISTS combined_footprint ( 
                      id INTEGER PRIMARY KEY, 
                      type TEXT,  -- will be 'common' or 'largest' 
                      hash TEXT UNIQUE,  -- will be a hash of a concatenation of the image names used for this footprint
                      polygon TEXT
                      )""")

    conn.commit()
    conn.close()