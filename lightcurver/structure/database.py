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
        conditions = ' AND '.join(conditions)
        request += f" WHERE {conditions}"

    try:
        df = pd.read_sql_query(request, conn)
        return df
    finally:
        conn.close()


def execute_sqlite_query(query, params=(), is_select=True, timeout=15.0, use_pandas=False):
    db_path = get_user_config()['database_path']
    with sqlite3.connect(db_path, timeout=timeout) as conn:
        cursor = conn.cursor()
        if is_select:
            if use_pandas:
                return pd.read_sql_query(sql=query, con=conn, params=params)
            cursor.execute(query, params)
            return cursor.fetchall()
        else:
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount


def get_count_based_on_conditions(conditions, table='frames'):
    """

    Args:
        conditions: string, e.g. 'plate_solved = 1 and eliminated = 0'
        table:

    Returns:

    """
    db_path = get_user_config()['database_path']
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    request = f"select count(*) from {table} where {conditions}"

    return cursor.execute(request).fetchone()[0]


def select_stars(combined_footprint_hash, stars_to_use=None):
    """
    Selects all the stars, either
     -- top 10 closest stars to ROI if stars_to_use is None
     -- top 'stars_to_use' closest stars to ROI if star s_to_use is int
     -- stars whose name is in 'stars_to_use' if stars_to_use is a list.

    Useful for selecting the stars we want to use when calculating a normalization coefficient.

    Args:
        combined_footprint_hash: since the stars were queried in a footprint, specify which. (or might get duplicates or stars
                        that are not in the current footprint)
        stars_to_use:   None or int or list, see docstring

    Returns:
        a pandas dataframe containing our stars (name, id coordinates ...)
    """
    base_query = "SELECT * FROM stars s WHERE combined_footprint_hash = ?"
    if stars_to_use is None:
        stars_to_use = 10  # make it an int for top 10 selection

    if type(stars_to_use) is int:
        # Query for top closest stars
        query = base_query + """
        ORDER BY s.distance_to_roi_arcsec ASC
        LIMIT ?
        """
        params = (combined_footprint_hash, stars_to_use)
    elif type(stars_to_use) is list:
        # Query for stars in the user-defined list
        placeholders = ','.join(['?'] * len(stars_to_use))
        query = base_query + f"""
        AND s.name IN ({placeholders})
        """
        params = (combined_footprint_hash, *stars_to_use)
    else:
        raise RuntimeError(f'stars_to_use argument: expected types None, int or list, got: {type(stars_to_use)}')

    return execute_sqlite_query(query, params, use_pandas=True)


def select_stars_for_a_frame(frame_id, combined_footprint_hash, stars_to_use=None):
    """
    Selects all the stars available in a given frame, either
     -- top 10 closest stars to ROI if stars_to_use is None
     -- top 'stars_to_use' closest stars to ROI if stars_to_use is int
     -- stars whose name is in 'stars_to_use' if stars_to_use is a list.

    Useful for selecting the stars we want to use when modelling the PSF
    or calculating a normalization coefficient.
    
    #TODO very similar to query_stars_for_frame_and_footprint,
    #TODO consider merging in the future.
    Args:
        frame_id:  database frame ID
        combined_footprint_hash: hash of the footprint in which the stars were originally queried.
        stars_to_use:   None or int or list, see docstring

    Returns:
        a pandas dataframe containing our stars (name, id coordinates ...)
    """
    base_query = """
        SELECT 
            sif.frame_id, 
            s.gaia_id, 
            s.name, 
            s.ra, 
            s.dec, 
            s.distance_to_roi_arcsec
        FROM stars_in_frames sif
        JOIN stars s ON sif.star_gaia_id = s.gaia_id AND sif.combined_footprint_hash = s.combined_footprint_hash
        WHERE sif.frame_id = ? AND s.combined_footprint_hash = ?"""
    if stars_to_use is None:
        stars_to_use = 10  # make it an int for top 10 selection

    if type(stars_to_use) is int:
        # Query for top closest stars
        query = base_query + """
        ORDER BY s.distance_to_roi_arcsec ASC
        LIMIT 10
        """
        params = (frame_id, combined_footprint_hash)
    elif type(stars_to_use) is list:
        # Query for stars in the user-defined list
        placeholders = ','.join(['?'] * len(stars_to_use))
        query = base_query + f"""
        AND s.name IN ({placeholders})
        """
        params = (frame_id, combined_footprint_hash, *stars_to_use)
    else:
        raise RuntimeError(f'stars_to_use argument: expected types None, int or list, got: {type(stars_to_use)}')

    return execute_sqlite_query(query, params, use_pandas=True)


def query_stars_for_frame_and_footprint(frame_id, combined_footprint_hash=None):
    """
    Queries and returns all stars linked to a specific frame, optionally filtered by a specific footprint hash.

    Parameters:
    - db_path: The path to the SQLite database file.
    - frame_id: The ID of the frame for which to find linked stars.
    - combined_footprint_hash: Optional. The combined footprint hash to filter the stars by.

    Returns:
    a pandas dataframe with the stars associated to the frame and footprint
    """

    sql_query = """
    SELECT stars.*
    FROM stars
    INNER JOIN stars_in_frames ON stars.gaia_id = stars_in_frames.star_gaia_id 
                                  AND stars.combined_footprint_hash = stars_in_frames.combined_footprint_hash
    WHERE stars_in_frames.frame_id = ?
    """

    # params for the query
    params = [frame_id]

    # if combined_footprint_hash, add it to the query and params
    if combined_footprint_hash is not None:
        sql_query += "AND stars.combined_footprint_hash = ?"
        params.append(combined_footprint_hash)

    stars = execute_sqlite_query(sql_query, params=params, is_select=True, use_pandas=True)

    return stars


def initialize_database(db_path=None):
    """
    initializes the database we'll be working with to keep track of our frames.

    Parameters:
        :param db_path: string or path, default None. Not used during execution of the pipeline,
                        but can be provided for tests.
    """
    if db_path is None:
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
        "image_relpath TEXT UNIQUE",  # e.g. frames/2023-02-01T01:23:35.fits -- relative to $workdir
        "sources_relpath TEXT",  # convention: frames/2023-02-01T01:23:35_sources.fits  -- same dir as above
        "telescope_latitude REAL",
        "telescope_longitude REAL",
        "telescope_elevation REAL",
        "telescope_name TEXT",
        "telescope_imager_name TEXT",
        "plate_solved INTEGER DEFAULT 0",
        "pixel_scale REAL DEFAULT NULL",
        "eliminated INTEGER DEFAULT 0",
        "airmass REAL DEFAULT NULL",
        "degrees_to_moon REAL DEFAULT NULL",
        "moon_phase REAL DEFAULT NULL",
        "sun_altitude REAL DEFAULT NULL",
        "seeing_pixels REAL DEFAULT NULL",
        "seeing_arcseconds REAL DEFAULT NULL",
        "sky_level_electron_per_second REAL DEFAULT NULL",
        "background_rms_electron_per_second REAL DEFAULT NULL",
        "ellipticity REAL DEFAULT NULL",
        "azimuth REAL DEFAULT NULL",
        "altitude REAL DEFAULT NULL",
        "comment TEXT DEFAULT NULL",
        "roi_in_footprint INTEGER DEFAULT 0",
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

    # now we'll need stars to calibrate the PSF and normalization of each frame.
    # these stars will be queried in a "footprint",  which can be changed to change the selection
    # of reference stars. So we'll refer to the footprint at hand in all steps.
    # a "footprint" is a polygon, composed of vertices (ra,dec).
    # we assume that the gnonomic projection is fine for checking what lands in a footprint
    # first, we define a footprint for each frame
    cursor.execute("""CREATE TABLE IF NOT EXISTS footprints (
                      frame_id INTEGER PRIMARY KEY,
                      polygon TEXT NOT NULL,
                      FOREIGN KEY (frame_id) REFERENCES frames (id)
                      )""")

    # then, a "combined" (intersection or union) footprint, to which we give a hash value that identifies it.
    # (this way, we don't recompute everything if we define a new footprint that actually is identical to one
    # we already processed).
    # this combined footprint is what we will refer to in every downstream step.
    cursor.execute("""CREATE TABLE IF NOT EXISTS combined_footprint ( 
                      id INTEGER PRIMARY KEY, 
                      hash INTEGER UNIQUE,  -- will be a hash of a concatenation of the used frames' ids
                      largest TEXT,
                      common TEXT
                      )""")

    # now the stars:
    # the stars will be filled in once by a python process.
    cursor.execute("""CREATE TABLE IF NOT EXISTS stars (
                      combined_footprint_hash INTEGER, -- which footprint was this star was queried from
                      name TEXT DEFAULT NULL, -- typically a letter
                      ra REAL,
                      dec REAL,
                      gmag REAL,
                      rmag REAL,
                      bmag REAL,               
                      pmra REAL,
                      pmdec REAL,    
                      ref_epoch REAL,   
                      gaia_id TEXT,
                      distance_to_roi_arcsec REAL,
                      FOREIGN KEY (combined_footprint_hash) REFERENCES combined_footprint(hash),
                      PRIMARY KEY (combined_footprint_hash, gaia_id)
                      )""")

    # linking stars and frame
    # again, a python process will check the footprint of each image
    # and fill in this table once for each image, the idea being able to query which stars are available in which image.
    cursor.execute("""CREATE TABLE IF NOT EXISTS stars_in_frames (
                      frame_id INTEGER,
                      star_gaia_id INTEGER,
                      combined_footprint_hash INTEGER,
                      FOREIGN KEY (frame_id) REFERENCES frames(id),
                      FOREIGN KEY (star_gaia_id) REFERENCES stars(gaia_id),
                      FOREIGN KEY (combined_footprint_hash) REFERENCES combined_footprint(hash),
                      PRIMARY KEY (combined_footprint_hash, frame_id, star_gaia_id)
                      )""")

    # here we keep track of which frames have a given PSF.
    # we trace back to which footprint the stars were queried in.
    # the subsampling factor is also defined in yaml file.
    cursor.execute("""CREATE TABLE IF NOT EXISTS PSFs (
                      combined_footprint_hash INTEGER,
                      frame_id INTEGER,
                      chi2 REAL, -- chi2 of the fit of the PSF
                      psf_ref TEXT, -- convention: sorted concatenation of all star names used in the model.
                      subsampling_factor INTEGER,  -- we do starred (pixelated) PSFs.
                      relative_loss_differential REAL, -- absolute change in last 10% of the iterations vs beginning
                      FOREIGN KEY (frame_id) REFERENCES frames(id),
                      FOREIGN KEY (combined_footprint_hash) REFERENCES combined_footprint(hash),
                      PRIMARY KEY (combined_footprint_hash, frame_id, psf_ref)
                      )""")

    # once we'll have PSF models, we'll do PSF photometry of the stars in the field.
    # the table below will keep track of the fitted fluxes.
    cursor.execute("""CREATE TABLE IF NOT EXISTS star_flux_in_frame (
                      frame_id INTEGER,
                      star_gaia_id INTEGER, 
                      combined_footprint_hash INTEGER,
                      flux REAL, -- in e- / second
                      flux_uncertainty REAL,
                      chi2 REAL, -- chi2 of fit in this specific frame.
                      relative_loss_differential REAL, -- absolute change in last 10% of the iterations vs beginning
                      FOREIGN KEY (frame_id) REFERENCES frames(id),
                      FOREIGN KEY (star_gaia_id) REFERENCES stars(gaia_id),
                      FOREIGN KEY (combined_footprint_hash) REFERENCES combined_footprint(hash),
                      PRIMARY KEY (combined_footprint_hash, frame_id, star_gaia_id)
                      )""")

    # very similarly to PSFs, we keep track of normalization coefficients.
    # these are computed once per image within python, from the star fluxes above.
    cursor.execute("""CREATE TABLE IF NOT EXISTS normalization_coefficients (
                      frame_id INTEGER,
                      combined_footprint_hash INTEGER,
                      coefficient REAL,
                      coefficient_uncertainty REAL,
                      FOREIGN KEY (frame_id) REFERENCES frames(id),
                      FOREIGN KEY (combined_footprint_hash) REFERENCES combined_footprint(hash),
                      PRIMARY KEY (combined_footprint_hash, frame_id)
                      )""")

    # zero points are derived from normalization coefficients and are for APPROXIMATE magnitude calibration
    # as we are going to derive them from gaia colors and band conversions.
    cursor.execute("""CREATE TABLE IF NOT EXISTS approximate_zeropoints (
                      frame_id INTEGER,
                      combined_footprint_hash INTEGER,
                      zeropoint REAL,
                      zeropoint_uncertainty REAL,
                      FOREIGN KEY (frame_id) REFERENCES frames(id),
                      FOREIGN KEY (combined_footprint_hash) REFERENCES combined_footprint(hash),
                      PRIMARY KEY (combined_footprint_hash, frame_id)
                      )""")

    conn.commit()
    conn.close()
