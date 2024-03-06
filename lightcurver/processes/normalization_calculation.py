import numpy as np

from ..structure.database import execute_sqlite_query, get_pandas
from ..structure.user_config import get_user_config
from ..utilities.footprint import get_combined_footprint_hash


def get_unique_stars_with_flux_in_footprint(combined_footprint_hash):
    """
    Given our freshly populated table of 'star_flux_in_frame', which keeps track of the
    initial footprint the stars were queried from, we can obtain all the unique stars that
    have a flux.

    Args:
        combined_footprint_hash: int, the hash of the footprint we are working with.

    Returns:
        pandas df of gaia_ids of the stars we can query.
    """
    query = "SELECT DISTINCT(star_gaia_id) FROM star_flux_in_frame WHERE combined_footprint_hash = ?"
    return execute_sqlite_query(query, params=(combined_footprint_hash,), is_select=True, use_pandas=True)


def get_fluxes(combined_footprint_hash, photometry_chi2_min, photometry_chi2_max):
    """
    Retrieves all the available star fluxes in all frames.
    If a given frame does not have a flux for this star, a NaN placeholder will be used.
    We filter by the chi2 of the fit: fluxes with an out of bounds chi2 will be replaced by a NaN
    value as well.

    :param combined_footprint_hash: int, the hash of the footprint we are working with.
    :param photometry_chi2_min: minimum acceptable chi2 value for the fit of the photometry of the star in this frame
    :param photometry_chi2_max: ditto but max
    :return: A list of frames that meet the criteria.
    """
    #            IFNULL(sff.flux, 'NaN') AS flux,
    #            IFNULL(sff.flux_uncertainty, 'NaN') AS d_flux
    query = """
    SELECT s.name,
           f.id AS frame_id, 
           sff.star_gaia_id, 
           sff.combined_footprint_hash,
           IFNULL(sff.flux, 'NaN') AS flux,
           IFNULL(sff.flux_uncertainty, 'NaN') AS d_flux
    FROM 
       frames f
    JOIN stars s ON sff.star_gaia_id = s.gaia_id
    LEFT JOIN star_flux_in_frame sff ON f.id = sff.frame_id 
    WHERE 
        sff.combined_footprint_hash = ?
    AND 
        sff.chi2 BETWEEN ? AND ?
    ORDER BY 
       s.name, f.id"""
    params = (combined_footprint_hash, photometry_chi2_min, photometry_chi2_max)

    return execute_sqlite_query(query, params, is_select=True, use_pandas=True)


def calculate_coefficient():
    user_config = get_user_config()

    # query initial frames, so we can calculate the footprint at hand
    frames_ini = get_pandas(columns=['id', 'image_relpath', 'exptime', 'mjd', 'seeing_pixels', 'pixel_scale'],
                            conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_ini['id'].to_list())

    # get the unique stars with flux available:
    # stars = get_unique_stars_with_flux_in_footprint(combined_footprint_hash=combined_footprint_hash)

    # TODO fill in user defined chi2 values
    df = get_fluxes(combined_footprint_hash=combined_footprint_hash,
                    photometry_chi2_min=0.1,
                    photometry_chi2_max=1.5)

    reference_flux = df.groupby('star_gaia_id')['flux'].mean().reset_index()
    reference_flux.rename(columns={'flux': 'reference_flux'}, inplace=True)

    breakpoint()
