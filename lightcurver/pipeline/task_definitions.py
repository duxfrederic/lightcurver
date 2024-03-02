# this file wraps around the processes defined in the processes subpackage.
# the wrapper determine which images / regions / psfs (depending on task) need processing
# before proceeding, and adds structure around multiprocessing when needed.
from multiprocessing import Pool, Manager
import os
import numpy as np
import pandas as pd
import logging
import logging.handlers
from pathlib import Path
import functools
import json
from astropy.coordinates import SkyCoord
import astropy.units as u

from ..structure.user_config import get_user_config
from ..structure.database import get_pandas, execute_sqlite_query
from ..processes.frame_importation import process_new_frame
from ..processes.plate_solving import solve_one_image_and_update_database
from ..utilities.footprint import (calc_common_and_total_footprint, get_frames_hash,
                                   save_combined_footprints_to_db, load_combined_footprint_from_db)
from ..plotting.footprint_plotting import plot_footprints
from ..processes.find_gaia_stars import find_gaia_stars_in_polygon


def worker_init(log_queue):
    logger = logging.getLogger(f"worker-{os.getpid()}")
    logger.setLevel(logging.INFO)
    q_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(q_handler)


def log_process(func):
    @functools.wraps(func)
    def wrapper(args):
        frame_id_for_logger = args[-1]
        logger = logging.getLogger(f"Process-{os.getpid()}")
        logger.info(f"{func.__name__} .... Processing image with ID {frame_id_for_logger}")
        return func(*args[:-1])  # execute original function without the last arg (used for logging)
    return wrapper


@log_process
def process_new_frame_wrapper(*args):
    process_new_frame(*args)


def read_convert_skysub_character_catalog():
    log_queue = Manager().Queue()
    listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    listener.start()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("PlateSolveLogger")

    # find the new images, we compare on file name!
    user_config = get_user_config()
    available_frames = sum([list(raw_dir.glob('*.fits')) for raw_dir in user_config['raw_dirs']], start=[])
    df_available_frames = pd.DataFrame({'frame_name': [frame.name for frame in available_frames]})
    already_imported = get_pandas(columns=['original_image_path', 'id'])
    if not already_imported.empty:
        already_imported['name'] = already_imported.apply(lambda row: Path(row['original_image_path']).name, axis=1)
    else:
        # just a decoy
        already_imported['name'] = pd.Series(dtype='str')
    new_frames_df = df_available_frames[~df_available_frames['frame_name'].isin(already_imported['name'])]
    new_frames = [frame for frame in available_frames if frame.name in new_frames_df['frame_name'].tolist()]
    logger.info(f"Importing {len(new_frames)} new frames.")

    with Pool(processes=user_config['multiprocessing_cpu_count'],
              initializer=worker_init, initargs=(log_queue,)) as pool:
        pool.map(process_new_frame_wrapper, [
            (new_frame,
             user_config,
             logger,
             new_frame)  # duplicating so have an identifier for logger.
            for new_frame in new_frames
        ])

    listener.stop()


@log_process
def solve_one_image_and_update_database_wrapper(*args):
    solve_one_image_and_update_database(*args)


def plate_solve_all_images():
    # boilerplate logging queue and listener
    # TODO can we reduce the boiler plate?
    log_queue = Manager().Queue()
    listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    listener.start()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger("PlateSolveLogger")
    user_config = get_user_config()
    workdir = Path(user_config['workdir'])
    if user_config['already_plate_solved']:
        logger.info('The images are already plate solved according to user config. Stopping.')
        listener.stop()
        return

    frames_to_process = get_pandas(columns=['id', 'image_relpath', 'sources_relpath'],
                                   conditions=['plate_solved = 0', 'eliminated = 0'])
    logger.info(f"Ready to plate solve {len(frames_to_process)} images.")

    with Pool(processes=user_config['multiprocessing_cpu_count'],
              initializer=worker_init, initargs=(log_queue,)) as pool:
        pool.map(solve_one_image_and_update_database_wrapper, [
            (workdir / row['image_relpath'],
             workdir / row['sources_relpath'],
             user_config,
             row['id'],
             logger,
             row['id']) # duplicate row['id'] for logger identification.
            for index, row in frames_to_process.iterrows()
        ])

    listener.stop()


def calc_common_and_total_footprint_and_save():
    """
    verifies whether the footprint was already calculated for the set of frames at hand
    if no, calculates it and stores it.

    Returns: None

    """

    query = """
    SELECT frames.id, footprints.polygon
    FROM footprints 
    JOIN frames ON footprints.frame_id = frames.id 
    WHERE frames.eliminated != 1;
    """
    results = execute_sqlite_query(query)
    frames_ids = [result[0] for result in results]
    frames_hash = get_frames_hash(frames_ids)
    # check if already exists
    count = execute_sqlite_query("SELECT COUNT(*) FROM combined_footprint WHERE hash = ?",
                                 params=(frames_hash,))[0][0]
    if count > 0:
        return
    polygon_list = [np.array(json.loads(result[1])) for result in results]
    common_footprint, largest_footprint = calc_common_and_total_footprint(polygon_list)

    user_config = get_user_config()
    plots_dir = user_config['plots_dir']
    footprints_plot_path = plots_dir / 'footprints.jpg'
    plot_footprints(polygon_list, common_footprint, largest_footprint, save_path=footprints_plot_path)

    # ok, save it
    save_combined_footprints_to_db(frames_hash, common_footprint, largest_footprint)


def query_gaia_stars():
    user_config = get_user_config()
    frames_info = get_pandas(columns=['id', 'pixel_scale'], conditions=['frames.eliminated != 1'])
    frames_hash = get_frames_hash(frames_info['id'].to_list())
    # before doing anything, check whether we are already done

    count = execute_sqlite_query("SELECT COUNT(*) FROM stars WHERE combined_footprint_hash = ?",
                                 params=(frames_hash,), is_select=True)[0][0]
    if count > 0 and not user_config['gaia_query_redo']:
        # we're done
        return
    elif count > 0 and user_config['gaia_query_redo']:
        # then we need to purge the database from the stars queried with this footprint.
        # TODO I forgot we have two types of footprints for a given footprint hash dayum
        # TODO for now proceeding with the user having to set redo if changing footprint type
        execute_sqlite_query("DELETE FROM stars WHERE combined_footprint_hash = ?",
                             params=(frames_hash,), is_select=True)

    largest_footprint, common_footprint = load_combined_footprint_from_db(frames_hash)
    if user_config['star_selection_strategy'] == 'common_footprint_stars':
        query_footprint = common_footprint
        # then we want to make sure we use stars that are available in all frames.
        # this likely achieves the best precision, but is only possible typically in dedicated
        # monitoring programs with stable pointings.
    elif user_config['star_selection_strategy'] == 'stars_per_frame':
        query_footprint = largest_footprint
        # then, we must fall back to using stars selected in each individual frame.
        # here, we will query a larger footprint so that we have options in each
        # individual frame.

    stars_table = find_gaia_stars_in_polygon(
                        query_footprint['coordinates'][0],
                        release='dr3',
                        astrometric_excess_noise_max=user_config["star_max_astrometric_excess_noise"],
                        gmag_range=(user_config["star_min_gmag"], user_config["star_max_gmag"]),
                        max_phot_g_mean_flux_error=user_config["star_max_phot_g_mean_flux_error"]
    )

    message = "Too few stars compared to the config criterion! Only {len(stars_table)} stars available."
    assert len(stars_table) >= user_config['min_number_stars'], message

    columns = ['combined_footprint_hash', 'ra', 'dec', 'gmag', 'rmag', 'bmag', 'pmra', 'pmdec',
               'gaia_id', 'distance_to_roi_arcsec']
    insert_query = f"INSERT INTO stars ({', '.join(columns)}) VALUES ({', '.join(len(columns)*['?'])})"
    for star in stars_table:
        star_coord = SkyCoord(ra=star['ra'] * u.degree, dec=star['dec'] * u.degree)
        distance_to_roi = star_coord.separation(user_config['ROI_SkyCoord']).arcsecond

        star_data = (frames_hash, star['ra'], star['dec'], star['phot_g_mean_mag'],
                     star['phot_rp_mean_mag'],  star['phot_bp_mean_mag'],
                     star['pmra'], star['pmdec'], star['source_id'],
                     distance_to_roi)
        execute_sqlite_query(insert_query, params=star_data, is_select=False)
