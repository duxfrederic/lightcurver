import os
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from widefield_plate_solver import plate_solve
from widefield_plate_solver.exceptions import CouldNotSolveError
import logging

from ..structure.database import execute_sqlite_query
from ..utilities.footprint import database_insert_single_footprint, get_angle_wcs


def solve_one_image(image_path, sources_path, user_config):

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
                      do_debug_plot=False,
                      odds_to_solve=1e8)

    return WCS(wcs)


def post_plate_solve_steps(frame_path, user_config, frame_id):
    """
    This is called after an astrometric solution has been found for an image (it is also called if the image
    is already plate solved, runs then on the existing solution)

    it
    - calculates a footprint for the image (represented by a polygon, inserted as json in the database)
    - checks if the ROI is contained by the image (if no, eliminates it)
    - has a bit of a check on the anisotropy of the pixels (bad solutions are not likely to have square pixels)
    - updates database with rotation of the field, pixel scale ...

    Args:
        frame_path: path to the fits file with WCS
        user_config: dictionary containing the user config
        frame_id: integer, database, frames column id.

    Returns:

    """
    logger = logging.getLogger("lightcurver.plate_solving")
    logger.info(f'Post plate solve steps for frame {frame_id} (path {frame_path})')
    # our object might be out of the footprint of the image!
    final_header = fits.getheader(frame_path)
    # replace the wcs above with the WCS we saved in the header of the image (contains naxis)
    wcs = WCS(final_header)
    in_footprint = user_config['ROI_SkyCoord'].contained_by(wcs)
    if in_footprint:
        execute_sqlite_query(query="UPDATE frames SET roi_in_footprint = ? WHERE id = ?",
                             params=(1, frame_id), is_select=False)
    # also, let us save the actual footprint
    footprint_array = wcs.calc_footprint()
    database_insert_single_footprint(frame_id, footprint_array)
    # and let us compute the pixel scale!
    psx, psy = proj_plane_pixel_scales(wcs)
    # these are most likely in deg/pixel. astropy says that wcs should carry a cunit attribute,
    # but it does not. Anyway, let us assume deg/pixel -- never seen anything else when working with wcs
    # of wide field images.
    anisotropy = float(abs(psx - psy) / (psx + psy))
    suspicious_astrometry = abs(psx - psy) / (psx + psy) > float(user_config['max_pixel_anisotropy'])
    if suspicious_astrometry:
        message = "Your pixels are more rectangular than your config tolerance! Flagging (eliminating) this frame."
        message += f"Anisotropy: {anisotropy:.01%}, path: {frame_path}, db id: {frame_id})."
        logger.info(message)
        execute_sqlite_query(query="""UPDATE 
                                          frames 
                                      SET 
                                          eliminated = 1,
                                          comment='suspicious_plate_solved'
                                      WHERE 
                                          id = ?""",
                             params=(frame_id,), is_select=False)
    angle_to_north = get_angle_wcs(wcs)
    pixel_scale = 0.5 * (psx + psy) * 3600  # to arcsecond / pixel
    # first, set the pixel scale
    execute_sqlite_query(query="UPDATE frames SET pixel_scale = ? WHERE id = ?",
                         params=(pixel_scale, frame_id), is_select=False)
    # then, use it to compute the seeing in arcseconds, and insert the angle at the same time to combine queries
    execute_sqlite_query(query="""UPDATE    
                                      frames 
                                  SET 
                                      seeing_arcseconds = pixel_scale * seeing_pixels,
                                      angle_to_north = ?
                                  WHERE 
                                      id = ?""",
                         params=(angle_to_north, frame_id), is_select=False)
    logger.info(f'Updated pixel scale: {pixel_scale:.03f}"/pixel for frame {frame_id} (path {frame_path}).')


def solve_one_image_and_update_database(image_path, sources_path, user_config, frame_id):
    """
    solves image using the sources in sources_path, then adds useful things to the database.
    If already solved according to user_config, just does the database part.
    Args:
        image_path: path to fits file containing the image
        sources_path: path to fits file containing the sources extracted from the image
        user_config: dictionary read by the pipeline
        frame_id: the database frame id
    Returns:
        nothing
    """
    logger = logging.getLogger("lightcurver.plate_solving")
    if not user_config['already_plate_solved']:
        logger.info(f'Attempting astrometric solution for frame {frame_id} (path: {image_path}).')
        try:
            wcs = solve_one_image(image_path, sources_path, user_config)
            success = wcs.is_celestial
        except CouldNotSolveError:
            success = False
    else:
        logger.info(f'Frame {frame_id} (path: {image_path}) is already solved according to user config.')
        success = True

    if success:
        post_plate_solve_steps(frame_id=frame_id, frame_path=image_path, user_config=user_config)
    # at the end, set the image to plate solved in db, and flag it as having had a plate solve attempt.
    execute_sqlite_query(query="UPDATE frames SET plate_solved = ?, attempted_plate_solve = 1 WHERE id = ?",
                         params=(1 if success else 0, frame_id), is_select=False)
