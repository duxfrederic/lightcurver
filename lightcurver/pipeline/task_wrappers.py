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
from ..processes.plate_solving import solve_one_image_and_update_database
from ..utilities.footprint import (calc_common_and_total_footprint, get_frames_hash,
                                   save_combined_footprints_to_db)
from ..plotting.footprint_plotting import plot_footprints


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

    # find the new frames, we compare on file name!
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
             new_frame)  # duplicating so to have an identifier for logger.
            for new_frame in new_frames
        ])

    listener.stop()


@log_process
def solve_one_image_and_update_database_wrapper(*args):
    solve_one_image_and_update_database(*args)


def plate_solve_all_frames():
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
        logger.info('The frames are already plate solved according to user config. Stopping.')
        listener.stop()
        return

    frames_to_process = get_pandas(columns=['id', 'image_relpath', 'sources_relpath'],
                                   conditions=['plate_solved = 0', 'eliminated = 0'])
    logger.info(f"Ready to plate solve {len(frames_to_process)} frames.")

    with Pool(processes=user_config['multiprocessing_cpu_count'],
              initializer=worker_init, initargs=(log_queue,)) as pool:
        pool.map(solve_one_image_and_update_database_wrapper, [
            (workdir / row['image_relpath'],
             workdir / row['sources_relpath'],
             user_config,
             row['id'],
             logger,
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
    logger = logging.getLogger("calc_footprints")
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
    polygon_list = [np.array(json.loads(result[1])) for result in results]
    common_footprint, largest_footprint = calc_common_and_total_footprint(polygon_list)

    user_config = get_user_config()
    plots_dir = user_config['plots_dir']
    footprints_plot_path = plots_dir / 'footprints.jpg'
    plot_footprints(polygon_list, common_footprint, largest_footprint, save_path=footprints_plot_path)

    # ok, save it
    save_combined_footprints_to_db(frames_hash, common_footprint, largest_footprint)
    logger.info(f'Footprint with (hash {frames_hash} saved to db')

