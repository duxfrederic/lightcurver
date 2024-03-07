import numpy as np
from astropy.stats import sigma_clipped_stats

from ..structure.user_config import get_user_config
from ..structure.database import execute_sqlite_query


def get_chi2_bounds(psf_or_fluxes):
    """
    This is a wrapper around the user config. We need chi2 bounds to only query the frames
    with a good enough PSF fit. There are several options to decide on the bounds:
    Params:
        psf_or_fluxes: string, either 'psf' or 'fluxes'.
    Returns:
        tuple (chi2_min, chi2_max)
    """
    user_config = get_user_config()
    assert psf_or_fluxes in ['psf', 'fluxes'], f'get_chi2_bounds: not something I know of: {psf_or_fluxes}'
    conf = user_config[f'{psf_or_fluxes}_fit_exclude_strategy']
    if conf is None:
        return -np.inf, np.inf
    elif type(conf) is dict:
        # then we're either dealing with bounds, or sigma clip
        assert len(conf.keys()) == 1
        key = list(conf.keys())[0]
        conf, val = key, conf[key]
        if conf == 'sigma_clip':
            # here we need to query the chi2. we'll just query them all to have
            # a feeling of what chi2 values we have, can be a systematic in the noisemaps.
            if psf_or_fluxes == 'psf':
                chi2val = execute_sqlite_query("select chi2 from PSFs", is_select=True, use_pandas=True)
            else:
                chi2val = execute_sqlite_query("select chi2 from star_flux_in_frame",
                                               is_select=True, use_pandas=True)
            mean, median, std = sigma_clipped_stats(chi2val['chi2'], sigma=val)
            chi2_min = median - val * std
            chi2_max = median + val * std
            return chi2_min, chi2_max
        elif conf == 'threshold':
            return val
        else:
            raise RuntimeError(f"Unexpected psf_fit_exclude_strategy: {conf}. valid: None, 'sigma_clip' or 'threshold'")
