import numpy as np

from ..structure.database import get_count_based_on_conditions, get_pandas

def check_read_convert_skysub_character_catalog():
    already_imported = get_pandas(columns=['original_image_path'])