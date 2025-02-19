import yaml
import os
from pathlib import Path
from astropy.coordinates import SkyCoord
from astropy import units as u
from importlib import resources

from .exceptions import NoConfigFilePathInEnvironment


def get_user_config():
    """
    This reads the yaml file containing the user config.
    Then processes some of its parameters and builds others.

    Returns:
        dictionary of parameters.
    """
    if 'LIGHTCURVER_CONFIG' not in os.environ:
        raise NoConfigFilePathInEnvironment
    config_path = os.environ['LIGHTCURVER_CONFIG']
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    roi_keys = list(config['ROI'].keys())
    config['roi_name'] = roi_name = roi_keys[0]
    ra, dec = config['ROI'][config['roi_name']]['coordinates']
    config['ROI_ra_deg'] = ra
    config['ROI_dec_deg'] = dec
    config['ROI_SkyCoord'] = SkyCoord(ra*u.deg, dec*u.deg)

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
    config['frames_dir'] = config['workdir'] / 'frames'
    config['regions_path'] = config['workdir'] / 'regions.h5'
    config['psfs_path'] = config['workdir'] / 'psfs.h5'
    for directory in [config['plots_dir'], config['logs_dir'], config['frames_dir']]:
        directory.mkdir(parents=True, exist_ok=True)

    # star names: make it a list if user defined a string.
    # e.g. stars_to_use = 'abcd' --> ['a', 'b', 'c', 'd']
    if type(config['stars_to_use_psf']) is str:
        config['stars_to_use_psf'] = [c for c in config['stars_to_use_psf']]
    if type(config['stars_to_use_norm']) is str:
        config['stars_to_use_norm'] = [c for c in config['stars_to_use_norm']]
    if type(config['stars_to_exclude_psf']) is str:
        config['stars_to_exclude_psf'] = [c for c in config['stars_to_exclude_psf']]
    if type(config['stars_to_exclude_norm']) is str:
        config['stars_to_exclude_norm'] = [c for c in config['stars_to_exclude_norm']]

    # photometric bands check
    photom_band = config['photometric_band']
    if photom_band in ['r_sdss', 'i_sdss', 'g_sdss', 'V', 'R', 'Ic', 'B_T', 'V_T']:
        config['reference_absolute_photometric_survey'] = 'gaia'
    elif 'panstarrs' in photom_band:
        # check declination
        if dec < -30.5:
            raise RuntimeError('With this declination, '
                               'it is unlikely you will find pan-starrs magnitudes for absolute calibration.')
        config['reference_absolute_photometric_survey'] = 'panstarrs'
    else:
        raise RuntimeError(f'Config check: not a photometric band we implemented: {photom_band}')

    # constraints on ROI cutout prep:
    if 'constraints_on_frame_columns_for_roi' not in config:
        config['constraints_on_frame_columns_for_roi'] = {}
    if 'constraints_on_normalization_coeff' not in config:
        config['constraints_on_normalization_coeff'] = {}

    # fixing the astrometry: default false
    if 'fix_point_source_astrometry' not in config:
        config['fix_point_source_astrometry'] = False

    return config


def compare_config_with_pipeline_delivered_one():
    # rough loading of user config:
    if 'LIGHTCURVER_CONFIG' not in os.environ:
        raise NoConfigFilePathInEnvironment
    config_path = os.environ['LIGHTCURVER_CONFIG']
    with open(config_path, 'r') as file:
        user_config = yaml.safe_load(file)

    # rough loading of pipeline config:
    pipeline_config_path = resources.files('lightcurver.pipeline.example_config_file') / 'config.yaml'
    with open(pipeline_config_path, 'r') as file:
        pipeline_config = yaml.safe_load(file)

    user_config_keys = set(user_config.keys())
    pipeline_config_keys = set(pipeline_config.keys())

    user_extra_keys = user_config_keys.difference(pipeline_config_keys)
    pipeline_extra_keys = pipeline_config_keys.difference(user_config_keys)
    pipeline_extra_keys_values = {key: pipeline_config[key] for key in pipeline_extra_keys}

    return {
        'extra_keys_in_user_config': user_extra_keys,
        'extra_keys_in_pipeline_config': pipeline_extra_keys,
        'pipeline_extra_keys_values': pipeline_extra_keys_values
    }
