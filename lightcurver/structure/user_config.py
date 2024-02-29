import yaml
import os
from pathlib import Path

from .exceptions import NoConfigFilePathInEnvironment


def get_user_config():
    if 'LIGHTCURVER_CONFIG' not in os.environ:
        raise NoConfigFilePathInEnvironment
    config_path = os.environ['LIGHTCURVER_CONFIG']
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    assert 'raw_dirs' in config
    raw_dirs = config['raw_dirs']
    if type(raw_dirs) is list:
        raw_dirs = [Path(pp) for pp in raw_dirs]
    elif type(raw_dirs) is str:
        raw_dirs = [Path(raw_dirs)]
    config['raw_dirs'] = raw_dirs

    assert 'workdir' in config
    config['workdir'] = Path(config['workdir'])
    config['database_path'] = config['workdir'] / 'database.sqlite3'
    config['plots_dir'] = config['workdir'] / 'plots'
    config['logs_dir'] = config['workdir'] / 'logs'
    config['images_dir'] = config['workdir'] / 'images'
    config['regions_path'] = config['workdir'] / 'regions.h5'
    config['psfs_path'] = config['workdir'] / 'psfs.h5'
    for directory in [config['plots_dir'], config['logs_dir'], config['images_dir']]:
        directory.mkdir(parents=True, exist_ok=True)
    assert 'redo' in config
    return config
