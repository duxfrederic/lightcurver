# This file wraps around the processes defined in the processes subpackage.
# The wrappers typically determine which frames / regions / psfs (depending on task) need processing
# before proceeding, and adds structure around multiprocessing when needed.
# This is not needed for all processes.
from multiprocessing import Pool, Manager
import os
import numpy as np
import pandas as pd
import logging
import logging.handlers
from pathlib import Path
import functools
import json


from ..structure.user_config import get_user_config
from ..structure.database import get_pandas, execute_sqlite_query
from ..processes.frame_importation import process_new_frame
from ..processes.plate_solving import solve_one_image_and_update_database, select_frames_needing_plate_solving
from ..utilities.footprint import (calc_common_and_total_footprint, get_frames_hash,
                                   save_combined_footprints_to_db, identify_and_eliminate_bad_pointings)
from ..plotting.footprint_plotting import plot_footprints
from ..processes.star_extraction import extract_sources_from_sky_sub_image


def worker_init(log_queue):
    logger = logging.getLogger(f"worker-{os.getpid()}")
    logger.setLevel(logging.INFO)
    q_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(q_handler)
    logger.propagate = False


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
    # boiler plate logger setup
    log_queue = Manager().Queue()
    base_logger = logging.getLogger("lightcurver")
    listener = logging.handlers.QueueListener(log_queue, *base_logger.handlers)
    listener.start()
    logger = logging.getLogger("lightcurver.importation")

    # find the new frames, we compare on file name!
    user_config = get_user_config()
    available_frames = sum([list(raw_dir.glob('*')) for raw_dir in user_config['raw_dirs']], start=[])
    df_available_frames = pd.DataFrame({'frame_name': [frame.name for frame in available_frames]})
    already_imported = get_pandas(columns=['original_image_path', 'id'])
    if not already_imported.empty:
        already_imported['name'] = already_imported.apply(lambda row: Path(row['original_image_path']).name, axis=1)
    else:
        # just a decoy
        already_imported['name'] = pd.Series(dtype='str')
    new_frames_df = df_available_frames[~df_available_frames['frame_name'].isin(already_imported['name'])]
    new_frames = [frame for frame in available_frames if frame.name in new_frames_df['frame_name'].tolist()]
    logger.info(f"Importing {len(new_frames)} new frames from directories {user_config['raw_dirs']}.")
    logger.info(f"Will write them to {user_config['workdir'] / 'frames'}")
    logger.info(f"Database will be at {user_config['workdir'] / 'database.sqlite3'}")

    with Pool(processes=user_config['multiprocessing_cpu_count'],
              initializer=worker_init, initargs=(log_queue,)) as pool:
        pool.map(process_new_frame_wrapper, [
            (new_frame,
             user_config,
             new_frame)  # duplicating so to have an identifier for logger.
            for new_frame in new_frames
        ])

    listener.stop()


@log_process
def solve_one_image_and_update_database_wrapper(*args):
    solve_one_image_and_update_database(*args)


def plate_solve_all_frames():
    # boiler plate logger setup
    log_queue = Manager().Queue()
    base_logger = logging.getLogger("lightcurver")
    listener = logging.handlers.QueueListener(log_queue, *base_logger.handlers)
    listener.start()
    logger = logging.getLogger("lightcurver.plate_solving")

    user_config = get_user_config()
    workdir = Path(user_config['workdir'])
    frames_to_process = select_frames_needing_plate_solving(user_config=user_config, logger=logger)
    logger.info(f"Ready to plate solve {len(frames_to_process)} frames.")

    with Pool(processes=user_config['multiprocessing_cpu_count'],
              initializer=worker_init, initargs=(log_queue,)) as pool:
        pool.map(solve_one_image_and_update_database_wrapper, [
            (workdir / row['image_relpath'],
             workdir / row['sources_relpath'],
             user_config,
             row['id'],
             row['id'])  # duplicating row['id'] for logger naming
            for index, row in frames_to_process.iterrows()
        ])

    listener.stop()


def calc_common_and_total_footprint_and_save():
    """
    verifies whether the footprint was already calculated for the set of frames at hand
    if no, calculates it and stores it.

    Returns: None

    """
    logger = logging.getLogger("lightcurver.combined_footprint_calculation")
    # so, before we do anything, let us eliminate the really obvious bad pointings.
    identify_and_eliminate_bad_pointings()
    # ok, keep going.
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
        logger.info(f'This combined footprint (hash {frames_hash}) was already calculated.')
        return
    logger.info(f'Calculating combined footprint (hash {frames_hash}) (loaded polygons from database, now combining).')
    polygon_list = [np.array(json.loads(result[1])) for result in results]
    common_footprint, largest_footprint = calc_common_and_total_footprint(polygon_list)

    user_config = get_user_config()
    plots_dir = user_config['plots_dir']
    footprints_plot_path = plots_dir / 'footprints.jpg'
    plot_footprints(polygon_list, common_footprint, largest_footprint, save_path=footprints_plot_path)
    logger.info(f'Combined footprint plot (hash {frames_hash}) saved at {footprints_plot_path}.')

    # ok, save it
    save_combined_footprints_to_db(frames_hash, common_footprint, largest_footprint)
    logger.info(f'Combined footprint with (hash {frames_hash} saved to db')


@log_process
def extract_sources_from_sky_sub_image_wrapper(*args):
    extract_sources_from_sky_sub_image(*args)


def source_extract_all_images(conditions=None):
    """
    This is not called directly in the pipeline, but can be useful if you want to re-extract all the sources
    with different parameters.
    So, unlike a routine called by the pipeline, this one takes an argument.

    :param conditions: list of strings, e.g. ['eliminated = 0', 'plate_solved = 0']. Default: None.
                       To filter what frames will be source-extracted again.
    :returns: Nothing

    """
    log_queue = Manager().Queue()
    base_logger = logging.getLogger("lightcurver")
    listener = logging.handlers.QueueListener(log_queue, *base_logger.handlers)
    listener.start()
    logger = logging.getLogger("lightcurver.source_extraction")

    user_config = get_user_config()
    workdir = Path(user_config['workdir'])
    frames_to_process = get_pandas(columns=['id', 'image_relpath', 'sources_relpath',
                                            'exptime', 'background_rms_electron_per_second'],
                                   conditions=conditions)
    logger.info(f"Extracting sources from {len(frames_to_process)} frames.")

    with Pool(processes=user_config['multiprocessing_cpu_count'],
              initializer=worker_init, initargs=(log_queue,)) as pool:
        pool.map(extract_sources_from_sky_sub_image_wrapper, [
            (workdir / row['image_relpath'],
             workdir / row['sources_relpath'],
             user_config['source_extraction_threshold'],
             user_config['source_extraction_min_area'],
             row['exptime'],
             row['background_rms_electron_per_second'],
             user_config['plots_dir'] / 'source_extraction' / f"{Path(row['image_relpath']).stem}.jpg",
             row['id'])  # for logger naming
            for index, row in frames_to_process.iterrows()
        ])

    listener.stop()
