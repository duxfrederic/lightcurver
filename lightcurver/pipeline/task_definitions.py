import numpy as np
from astropy.io import fits
from pathlib import Path

from ..structure.user_config import get_user_config
from ..structure.database import get_pandas, make_connection
from ..processes.frame_importation import process_new_frame


def read_convert_skysub_character_catalog():
    # find the new images, we compare on file name!
    user_config = get_user_config()
    available_frames = sum([raw_dir.glob('*.fits') for raw_dir in user_config['raw_dirs']], start=[])
    already_imported = get_pandas(columns=['original_image_path'])
    already_imported['name'] = already_imported.apply(lambda row: Path(row['original_image_path']).name)
    new_frames = [frame for frame in available_frames if frame.name not in already_imported]

    for new_frame in new_frames:
        header = fits.getheader(new_frame)


