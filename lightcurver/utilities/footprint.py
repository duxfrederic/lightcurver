import pandas as pd
import sqlite3
import numpy as np
from shapely.geometry import Polygon, mapping
from functools import reduce
import hashlib
import json

from ..structure.database import execute_sqlite_query


def calc_common_and_total_footprint(list_of_footprints):
    """
    Calculate the common (intersection) and largest (union) footprints from a list of numpy arrays: products
    of astropy.wcs.WCS.calc_footprint.
    Then determines both the intersection and union of these
    footprints using Shapely Polygons. The intersection represents the area common to all images,
    while the union covers the total area spanned by any of the images.

    Parameters:
    - list_of_fits_paths: A list of paths to FITS files.

    Returns:
    - wcs_objects: A list of WCS objects corresponding to the FITS files.
    - common_footprint: A Shapely Polygon representing the intersected footprint common to all WCS footprints.
    - largest_footprint: A Shapely Polygon representing the union of all WCS footprints.
    """
    wcs_footprints = list_of_footprints

    polygons = [Polygon(footprint) for footprint in wcs_footprints]

    common_footprint = reduce(lambda x, y: x.intersection(y), polygons)
    largest_footprint = reduce(lambda x, y: x.union(y), polygons)

    return (common_footprint.simplify(tolerance=0.001, preserve_topology=True),
            largest_footprint.simplify(tolerance=0.001, preserve_topology=True))


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


def get_frames_hash(frames_ids):
    """
    when calculating footprints, we need a way to identify which footprint was calculated from which frames.
    I don't want to deal with the relational many-to-many situation that will arise in a relational database,
    so let's calculate a hash of the integers of the frames that were used to calculate a footprint.
    then to check for a footprint, which can just query the hash.
    Args:
        frames_ids: list of integers, frames.id in the database

    Returns:
        a text hash

    """
    assert len(set(frames_ids)) == len(frames_ids), "Non-unique frame ids passed to this function"
    sorted_frame_ids = sorted(frames_ids)
    frame_ids_tuple = tuple(sorted_frame_ids)
    return hash(frame_ids_tuple)


def save_combined_footprints_to_db(frames_hash, common_footprint, largest_footprint):

    common_str = json.dumps(mapping(common_footprint))
    largest_str = json.dumps(mapping(largest_footprint))
    save_query = "INSERT INTO combined_footprint (hash, largest, common) VALUES (?, ?, ?)"
    execute_sqlite_query(save_query,
                         params=(frames_hash, largest_str, common_str),
                         is_select=False)



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
