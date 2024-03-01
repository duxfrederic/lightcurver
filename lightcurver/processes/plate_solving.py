import os
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from widefield_plate_solver import plate_solve

from ..structure.database import execute_sqlite_query


def solve_one_image(image_path, sources_path, user_config, logger):
    sources = Table(fits.getdata(sources_path))

    if user_config['astrometry_net_api_key'] is None:
        use_api = False
    else:
        use_api = True
        os.environ['astrometry_net_api_key'] = user_config['astrometry_net_api_key']

    roi_keys = list(user_config['ROI'].keys())
    ra, dec = user_config['ROI'][roi_keys[0]]['coordinates']
    plate_scale_min, plate_scale_max = user_config['plate_scale_interval']

    wcs = plate_solve(fits_file_path=image_path, sources=sources,
                      use_existing_wcs_as_guess=False,
                      use_api=use_api,
                      redo_if_done=True,  # we check for this upstream in this package
                      ra_approx=ra, dec_approx=dec,
                      scale_min=plate_scale_min, scale_max=plate_scale_max,
                      logger=logger,
                      do_debug_plot=False)

    return WCS(wcs).is_celestial


def solve_one_image_and_update_database(image_path, sources_path, user_config, frame_id, logger):
    success = solve_one_image(image_path, sources_path, user_config, logger)
    execute_sqlite_query(query="UPDATE frames SET plate_solved = ? WHERE id = ?",
                         params=(1 if success else 0, frame_id), is_select=False)

