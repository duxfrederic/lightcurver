import numpy as np

from ..structure.database import execute_sqlite_query


def get_fluxes_for_star(gaia_id, photometry_chi2_min, photometry_chi2_max):
    """
    Retrieves all the fluxes in all frames, ordered by frame_id, for a given star.
    If a given frame does not have a flux for this star, a NaN placeholder will be used.
    We filter by the chi2 of the fit: fluxes with an out of bounds chi2 will be replaced by a NaN
    value as well.

    :param gaia_id: The Gaia ID of the star.
    :param photometry_chi2_min: minimum acceptable chi2 value for the fit of the photometry of the star in this frame
    :param photometry_chi2_max: ditto but max
    :return: A list of frames that meet the criteria.
    """

    query = """
    SELECT f.id AS frame_id, 
           IFNULL(sff.flux, 'NaN') AS flux,
           IFNULL(sff.flux_uncertainty, 'NaN') AS dflux
    FROM frames f
    LEFT JOIN star_flux_in_frame sff ON f.id = sff.frame_id AND sff.star_gaia_id = ?
    AND sff.chi2 BETWEEN ? AND ?
    ORDER BY f.id"""
    params = (gaia_id, photometry_chi2_min, photometry_chi2_max)

    return execute_sqlite_query(query, params, is_select=True, use_pandas=True)


def calculate_coefficient():
    pass
