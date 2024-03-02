import pandas as pd
import sqlite3
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from shapely.geometry import Polygon, mapping
from functools import reduce
import hashlib
import json

from ..structure.database import execute_sqlite_query


def calc_footprints(list_of_fits_paths):
    """
    Calculate the common (intersection) and largest (union) footprints from a list of FITS files.

    This function loads World Coordinate System (WCS) objects from the headers of FITS files,
    calculates their footprints, and then determines both the intersection and union of these
    footprints using Shapely Polygons. The intersection represents the area common to all images,
    while the union covers the total area spanned by any of the images.

    Parameters:
    - list_of_fits_paths: A list of paths to FITS files.

    Returns:
    - wcs_objects: A list of WCS objects corresponding to the FITS files.
    - common_footprint: A Shapely Polygon representing the intersected footprint common to all WCS footprints.
    - largest_footprint: A Shapely Polygon representing the union of all WCS footprints.
    """
    wcs_objects = [WCS(fits.getheader(fits_path)) for fits_path in list_of_fits_paths]

    wcs_footprints = [wcs_obj.calc_footprint() for wcs_obj in wcs_objects]

    polygons = [Polygon(footprint) for footprint in wcs_footprints]

    common_footprint = reduce(lambda x, y: x.intersection(y), polygons)
    largest_footprint = reduce(lambda x, y: x.union(y), polygons)

    return wcs_objects, common_footprint, largest_footprint


def database_insert_single_footprint(frame_id, footprint_array):
    polygon_list = footprint_array.tolist()
    polygon_str = json.dumps(polygon_list)

    execute_sqlite_query(query="INSERT INTO footprints (frame_id, polygon) VALUES (?, ?)",
                         params=(frame_id, polygon_str),
                         is_select=False)


def database_get_footprint(frame_id):
    result = execute_sqlite_query(query="SELECT polygon FROM footprints WHERE frame_id = ?",
                                  params=(frame_id,),
                                  is_select=True)[0]

    polygon_list = json.loads(result)
    footprint_polygon = np.array(polygon_list)

    return footprint_polygon


def save_common_footprint_to_db(image_relpaths_df, footprint):
    # Generate a hash from the dataframe's image paths
    hash_str = hashlib.sha256(pd.util.hash_pandas_object(image_relpaths_df).values).hexdigest()
    polygon_str = json.dumps(mapping(footprint))

    # Connect to the database and create the table if it doesn't exist
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''INSERT INTO footprints (hash, polygon) VALUES (?, ?)''',
                   (hash_str, polygon_str))
    conn.commit()
    conn.close()


def load_common_footprint_from_db(db_path, hash_str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''SELECT polygon FROM footprints WHERE hash = ?''', (hash_str,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return Polygon(json.loads(result[0]))
    else:
        return None
