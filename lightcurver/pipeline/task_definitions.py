# this file wraps around the processes defined in the processes subpackage.
# the wrapper determine which images / regions / psfs (depending on task) need processing
# before proceeding.
from multiprocessing import Pool, Manager
import os
import logging
import logging.handlers
import pandas as pd
from pathlib import Path

from ..structure.user_config import get_user_config
from ..structure.database import get_pandas, execute_sqlite_query
from ..processes.frame_importation import process_new_frame
from ..processes.plate_solving import solve_one_image_and_update_database
# the user defines their header parser, returning a dictionary with {'mjd':, 'gain':, 'filter':, 'exptime':}
# it needs be located at $workdir/header_parser/parse_header.py
# the function needs be called parse_header, and it has to accept a fits header as argument and return the
# dictionary above.


def worker_init(log_queue):
    logger = logging.getLogger(f"worker-{os.getpid()}")
    logger.setLevel(logging.INFO)
    q_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(q_handler)


def process_new_frame_wrapper(args):
    frame_id_for_logger = args[-1]
    logger = logging.getLogger(f"Process-{os.getpid()}")
    logger.info(f"process_new_frame .... Processing image with ID {frame_id_for_logger}")
    process_new_frame(*args[:-1])


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
    already_imported['name'] = already_imported.apply(lambda row: Path(row['original_image_path']).name, axis=1)
    new_frames_df = df_available_frames[~df_available_frames['frame_name'].isin(already_imported['name'])]
    new_frames = [frame for frame in available_frames if frame.name in new_frames_df['frame_name'].tolist()]
    logger.info(f"Importing {len(new_frames)} new frames.")

    with Pool(initializer=worker_init, initargs=(log_queue,)) as pool:
        pool.map(process_new_frame_wrapper, [
            (new_frame,
             user_config,
             logger,
             new_frame)  # duplicating so have an identifier for logger.
            for new_frame in new_frames
        ])

    listener.stop()


def solve_one_image_and_update_database_wrapper(args):
    frame_id_for_logger = args[-1]
    logger = logging.getLogger(f"Process-{os.getpid()}")
    logger.info(f"solve_one_image .... Processing image with ID {frame_id_for_logger}")
    solve_one_image_and_update_database(*args[:-1])


def plate_solve_all_images():
    # Set up logging queue and listener
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
                                   conditions=['plate_solved = 0'])
    logger.info(f"Ready to plate solve {len(frames_to_process)} images.")

    with Pool(initializer=worker_init, initargs=(log_queue,)) as pool:
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

