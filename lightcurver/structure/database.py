import sqlite3
import pandas as pd

from ..structure.user_config import get_user_config


def get_pandas(conditions=None, columns=None):
    """
    Retrieve data from the database based on specified conditions and return as a Pandas DataFrame.

    Parameters:
        conditions (list of str): The SQL conditions to filter the data, default None.
        columns (str or list of str, optional): Columns to select from the database. Defaults to None,
            which selects all columns.
    Returns:
        pandas.DataFrame: A DataFrame containing the results of the SQL query.
    """
    db_path = get_user_config()['database_path']
    conn = sqlite3.connect(db_path)
    if columns is None:
        columns = '*'
    request = f"SELECT {','.join(columns)} FROM frames"
    if conditions is not None:
        conditions = ','.join(conditions)
        request += f" WHERE {conditions}"

    try:
        df = pd.read_sql_query(request, conn)
        return df
    finally:
        conn.close()


def execute_sqlite(command):
    db_path = get_user_config()['database_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return cursor.execute(command).fetchall()


def get_count_based_on_conditions(conditions):
    db_path = get_user_config()['database_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    request = f"select count(*) from frames where {conditions}"

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
        "seeing_pixels REAL DEFAULT NULL",
        "seeing_arcseconds REAL DEFAULT NULL",
        "azimuth REAL DEFAULT NULL",
        "altitude REAL DEFAULT NULL",
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
                      gaia_id TEXT -- Gaia ID
                      )""")

    # linking stars and frame
    # again, a python process will check the footprint of each image
    # and fill in this table once for each image.
    cursor.execute("""CREATE TABLE IF NOT EXISTS FrameStars (
                      frame_id INTEGER,
                      star_id INTEGER,
                      FOREIGN KEY (frame_id) REFERENCES Frames(id),
                      FOREIGN KEY (star_id) REFERENCES Stars(id),
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
                      FOREIGN KEY (frame_id) REFERENCES Frames(id),
                      PRIMARY KEY (id, frame_id)
                      )""")

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
                      PRIMARY KEY (id, psf_id, frame_id)
                      )""")

    # zero points are derived from normalization coefficients and are for APPROXIMATE magnitude calibration
    # as we are going to derive them from gaia colors and band conversions.
    cursor.execute("""CREATE TABLE IF NOT EXISTS ApproximateZeroPoints (
                      id INTEGER, 
                      norm_coefficient_id INTEGER,
                      zeropoint FLOAT,
                      zeropoint_uncertainty FLOAT,
                      FOREIGN KEY (norm_coefficient_id) REFERENCES NormalizationCoefficients(id),
                      PRIMARY KEY (id, norm_coefficient_id)
                      )""")

    conn.commit()
    conn.close()
