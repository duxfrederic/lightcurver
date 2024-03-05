import os
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from widefield_plate_solver import plate_solve

from ..structure.database import execute_sqlite_query
from ..utilities.footprint import database_insert_single_footprint


def solve_one_image(image_path, sources_path, user_config, logger):
    sources = Table(fits.getdata(sources_path))

    if user_config['astrometry_net_api_key'] is None:
        use_api = False
    else:
        use_api = True
        os.environ['astrometry_net_api_key'] = user_config['astrometry_net_api_key']

    ra, dec = user_config['ROI_ra_deg'], user_config['ROI_dec_deg']
    plate_scale_min, plate_scale_max = user_config['plate_scale_interval']

    wcs = plate_solve(fits_file_path=image_path, sources=sources,
                      use_existing_wcs_as_guess=False,
                      use_api=use_api,
                      redo_if_done=True,  # we check for this upstream in this package
                      ra_approx=ra, dec_approx=dec,
                      scale_min=plate_scale_min, scale_max=plate_scale_max,
                      logger=logger,
                      do_debug_plot=False)

    return WCS(wcs)


def solve_one_image_and_update_database(image_path, sources_path, user_config, frame_id, logger):
    wcs = solve_one_image(image_path, sources_path, user_config, logger)
    success = wcs.is_celestial
    if success:
        # but our object might be out of the footprint of the image!
        final_header = fits.getheader(image_path)
        # replace the wcs above with the WCS we saved in the header of the image (contains naxis)
        wcs = WCS(final_header)
        in_footprint = user_config['ROI_SkyCoord'].contained_by(wcs)
        if not in_footprint:
            execute_sqlite_query(query="UPDATE frames SET roi_in_footprint = ? WHERE id = ?",
                                 params=(0, frame_id), is_select=False)
        # also, let us save the actual footprint
        footprint_array = wcs.calc_footprint()
        database_insert_single_footprint(frame_id, footprint_array)
        # and let us compute the pixel scale!
        psx, psy = proj_plane_pixel_scales(wcs)
        # these are most likely in deg/pixel. astropy says that wcs should carry a cunit attribute,
        # but it does not. Anyway, let us assume deg/pixel -- never seen anything else when working with wcs
        # of wide field images.
        anisotropy = abs(psx-psy)/(psx+psy)
        message = "Your pixels seem to be a bit rectangular! I did not implement support for this. "
        message += f"Anisotropy: {anisotropy:.01%}%"
        assert abs(psx-psy)/(psx+psy) < 1e-2, message
        pixel_scale = 0.5 * (psx + psy) * 3600  # to arcsecond / pixel
        execute_sqlite_query(query="UPDATE frames SET pixel_scale = ? WHERE id = ?",
                             params=(pixel_scale, frame_id), is_select=False)

    # at the end, set the image to plate solved in db
    execute_sqlite_query(query="UPDATE frames SET plate_solved = ? WHERE id = ?",
                         params=(1 if success else 0, frame_id), is_select=False)
