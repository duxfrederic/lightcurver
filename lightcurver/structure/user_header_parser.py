import importlib.util
import os

from .user_config import get_user_config


def load_custom_header_parser():
    """
    dynamically load the 'parse_header' function from the user-defined file
    located at '$workdir/header_parser/header_parser.py'.
    """
    workspace_dir = get_user_config()['workdir']

    file_path = os.path.join(workspace_dir, 'header_parser', 'parse_header.py')
    module_name = 'header_parser'

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        parse_header_function = getattr(module, 'parse_header')
        return parse_header_function
    except AttributeError:
        raise ImportError("The function 'parse_header' was not found in 'header_parser.py'.")
