import numpy as np
from astropy.io import fits
import pandas as pd
from pathlib import Path

from ..structure.user_config import get_user_config
from ..structure.database import get_pandas
from ..processes.frame_importation import process_new_frame
from ..structure.user_header_parser import load_custom_header_parser
# the user defines their header parser, returning a dictionary with {'mjd':, 'gain':, 'filter':, 'exptime':}
# it needs be located at $workdir/header_parser/parse_header.py
# the function needs be called parse_header, and it has to accept a fits header as argument and return the
# dictionary above.
custom_parser = load_custom_header_parser()


def read_convert_skysub_character_catalog():
    # find the new images, we compare on file name!
    user_config = get_user_config()
    available_frames = sum([list(raw_dir.glob('*.fits')) for raw_dir in user_config['raw_dirs']], start=[])
    df_available_frames = pd.DataFrame({'frame_name': [frame.name for frame in available_frames]})
    already_imported = get_pandas(columns=['original_image_path'])
    already_imported['name'] = already_imported.apply(lambda row: Path(row['original_image_path']).name, axis=1)
    new_frames_df = df_available_frames[~df_available_frames['frame_name'].isin(already_imported['name'])]
    new_frames = [frame for frame in available_frames if frame.name in new_frames_df['frame_name'].tolist()]

    for new_frame in new_frames:
        process_new_frame(new_frame, user_config, custom_parser)


