import numpy as np
from shapely.geometry import Polygon, mapping
from functools import reduce
import json
from astropy.io import fits
from astropy.wcs import WCS

from ..structure.database import execute_sqlite_query, get_pandas
from ..structure.user_config import get_user_config


def get_combined_footprint_hash(user_config, frames_id_list):
    """
    Calculates the hash of the combined footprint of the frames whose id is in frames_id_list.
    Args:
        user_config: dictionary obtained from structure.config.get_user_config
        frames_id_list: list of integers, which frames are we using.

    Returns:
        frames_hash, integer

    """
    if user_config['star_selection_strategy'] != 'ROI_disk':
        # then it depends on the frames we're considering.
        frames_hash = get_frames_hash(frames_id_list)
    else:
        # if ROI_disk, it does not depend on the frames: unique region defined by its radius.
        return hash(user_config['ROI_disk_radius_arcseconds'])


def calc_common_and_total_footprint(list_of_footprints):
    """
    Calculate the common (intersection) and largest (union) footprints from a list of numpy arrays: products
    of astropy.wcs.WCS.calc_footprint.
    Then determines both the intersection and union of these
    footprints using Shapely Polygons. The intersection represents the area common to all frames,
    while the union covers the total area spanned by any of the frames.

    Parameters:
    - list_of_fits_paths: A list of paths to FITS files.

    Returns:
    - wcs_objects: A list of WCS objects corresponding to the FITS files.
    - common_footprint: A Shapely Polygon representing the intersected footprint common to all WCS footprints.
    - largest_footprint: A Shapely Polygon representing the union of all WCS footprints.
    """
    wcs_footprints = list_of_footprints

    polygons = [Polygon(footprint) for footprint in wcs_footprints]
    try:
        common_footprint = reduce(lambda x, y: x.intersection(y), polygons)
        common_footprint = common_footprint.simplify(tolerance=0.001, preserve_topology=True)
    except TypeError:
        # we might have no common footprint?
        common_footprint = None

    largest_footprint = reduce(lambda x, y: x.union(y), polygons)
    largest_footprint = largest_footprint.simplify(tolerance=0.001, preserve_topology=True)

    return common_footprint, largest_footprint


def database_insert_single_footprint(frame_id, footprint_array):
    polygon_list = footprint_array.tolist()
    polygon_str = json.dumps(polygon_list)

    execute_sqlite_query(query="INSERT OR REPLACE INTO footprints (frame_id, polygon) VALUES (?, ?)",
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


def load_combined_footprint_from_db(frames_hash):

    query = "SELECT largest, common FROM combined_footprint WHERE hash = ?"
    result = execute_sqlite_query(query,
                                  params=(frames_hash,),
                                  is_select=True)[0]
    largest = json.loads(result[0])
    common = json.loads(result[1])
    if result:
        return largest, common
    else:
        return None


def check_in_footprint_for_all_images():
    """
    Just a wrapper for running the footprint check. Can be useful to manually execute in some cases.
    Will
     - load the footprints from db or load all the headers of the plate solved images depending on function argument
     - check if the ROI coord is in the footprint defined by the WCS
     - update the frames table.
    We skip the footprints table, as this is the ROI and not a simple star we want a better check
    than the simple gnonomic projection we are forced to rely on when using the footprint table.

    Returns:
        Nothing
    """
    frames_to_process = get_pandas(columns=['id', 'image_relpath'],
                                   conditions=['plate_solved = 1', 'eliminated = 0'])
    user_config = get_user_config()

    for i, frame in frames_to_process.iterrows():
        frame_id = frame['id']
        frame_path = user_config['workdir'] / frame['image_relpath']
        final_header = fits.getheader(frame_path)
        wcs = WCS(final_header)
        in_footprint = user_config['ROI_SkyCoord'].contained_by(wcs)
        execute_sqlite_query(query="UPDATE frames SET roi_in_footprint = ? WHERE id = ?",
                             params=(int(in_footprint), frame_id), is_select=False)


def identify_and_eliminate_bad_pointings():
    """
    Called after calculating the footprints. Will identify pointings that are ~really~ different, and
    flag them in the database ('eliminated = 1', 'comment = "bad_pointing"')
    Returns: nothing

    """

    select_query = """
    SELECT frames.id, footprints.polygon
    FROM footprints 
    JOIN frames ON footprints.frame_id = frames.id
    WHERE frames.eliminated != 1;
    """
    update_query = """
    UPDATE frames
    SET comment = 'bad_pointing', eliminated = 1
    WHERE id = ?;
    """

    results = execute_sqlite_query(select_query, is_select=True, use_pandas=True)
    mean_positions = []

    for i, row in results.iterrows():
        frame_id = row['id']
        polygon = row['polygon']
        polygon = np.array(json.loads(polygon))
        mean_position = np.mean(polygon, axis=0)
        mean_positions.append((frame_id, mean_position))

    all_means = np.array([pos for _, pos in mean_positions])
    overall_mean = np.mean(all_means, axis=0)

    # distance of each frame's mean position from the overall mean
    deviations = [(frame_id, np.linalg.norm(mean_pos - overall_mean)) for frame_id, mean_pos in mean_positions]

    # threshold
    deviation_values = [dev for _, dev in deviations]
    mean_deviation = np.mean(deviation_values)
    std_deviation = np.std(deviation_values)
    threshold = mean_deviation + 5 * std_deviation  # quite a generous threshold.

    # flag frames with significant deviation
    bad_frames = [frame_id for frame_id, dev in deviations if dev > threshold]

    for frame_id in bad_frames:
        execute_sqlite_query(update_query, params=(frame_id,))


def get_angle_wcs(wcs_object):
    """
     Takes a WCS object, and returns the angle in degrees to the North (so, angle relative to "North up, East left")
    Args:
        wcs_object: astropy WCS object

    Returns:
        angle: float, angle in degrees.
    """

    if hasattr(wcs_object.wcs, 'cd'):
        matrix = wcs_object.wcs.cd
    elif hasattr(wcs_object.wcs, 'pc'):
        matrix = wcs_object.wcs.pc
    else:
        raise ValueError("Neither CD nor PC matrix found in WCS.")

    cd1_1, cd1_2 = matrix[0, 0], matrix[0, 1]
    cd2_1, cd2_2 = matrix[1, 0], matrix[1, 1]

    angle = np.arctan2(-cd1_2, cd2_2) * 180.0 / np.pi

    return angle
