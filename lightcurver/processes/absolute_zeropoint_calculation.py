import numpy as np
import pandas as pd
import sqlite3

from ..utilities.footprint import get_combined_footprint_hash
from ..structure.user_config import get_user_config
from ..structure.database import execute_sqlite_query, get_pandas
from ..utilities.absolute_magnitudes_from_panstarrs import save_panstarrs_catalog_photometry_to_database
from ..utilities.absolute_magnitudes_from_gaia import save_gaia_catalog_photometry_to_database


magnitude_calculation_functions = {
    'gaia': save_gaia_catalog_photometry_to_database,
    'panstarrs': save_panstarrs_catalog_photometry_to_database
}


def get_gaia_ids_with_flux_in_frame(combined_footprint_hash):
    """
    Queries all gaia_id of stars for which there is at least one flux_in_frame
    entry in the given combined_footprint_hash.

    Args:
        combined_footprint_hash: The combined footprint hash to filter the stars.

    Returns:
        List of gaia_id of stars.
    """
    query = """
    SELECT DISTINCT star_gaia_id
    FROM star_flux_in_frame
    WHERE combined_footprint_hash = ?
    """
    params = (combined_footprint_hash,)
    result = execute_sqlite_query(query, params)
    gaia_ids = [row[0] for row in result]
    return gaia_ids


def calculate_zeropoints():
    """
    Calculates zeropoints for each frame based on provided magnitudes and updates the database.
    Args:
        -
    Returns:
        None
    """

    # boiler plate
    user_config = get_user_config()
    frames_ini = get_pandas(columns=['id'],
                            conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_ini['id'].to_list())

    # first, trigger the calculation of the magnitudes of the reference stars in the band of the config
    gaia_ids = get_gaia_ids_with_flux_in_frame(combined_footprint_hash)
    source_catalog = user_config['reference_absolute_photometric_survey']
    absolute_mag_func = magnitude_calculation_functions[source_catalog]
    for gaia_id in pd.unique(gaia_ids):
        absolute_mag_func(gaia_id)

    # now, query the star fluxes and their reference magnitudes from our database.
    # we also join on the table of calibrated magnitudes obtained from gaia or panstarrs, etc.
    flux_query = """
    SELECT 
         sff.frame_id, 
         sff.flux, 
         s.gaia_id,
         csp.mag as catalog_mag
    FROM 
         star_flux_in_frame sff
    JOIN 
         stars s ON sff.star_gaia_id = s.gaia_id
    AND
         s.combined_footprint_hash = sff.combined_footprint_hash
    JOIN 
         frames f ON f.id = sff.frame_id
    JOIN
         catalog_star_photometry csp ON csp.star_gaia_id = s.gaia_id
    WHERE 
         sff.combined_footprint_hash = ?
    AND 
         csp.catalog = ?
    """

    # get the fluxes measured on the frames
    flux_data = execute_sqlite_query(flux_query,
                                     params=(combined_footprint_hash,
                                             user_config['reference_absolute_photometric_survey']),
                                     is_select=True, use_pandas=True)
    if flux_data.empty:
        return

    # continue with zeropoint calculation now
    flux_data['instrumental_mag'] = -2.5 * np.log10(flux_data['flux'])
    flux_data['mag_difference'] = flux_data['catalog_mag'] - flux_data['instrumental_mag']

    # zeropoint and uncertainty (std) for each frame
    zeropoint_results = flux_data.groupby('frame_id')['mag_difference'].agg(['median', 'std']).reset_index()
    zeropoint_results.rename(columns={'median': 'zeropoint', 'std': 'zeropoint_uncertainty'}, inplace=True)

    # Update database
    insert_query = """
    INSERT INTO absolute_zeropoints (frame_id, combined_footprint_hash, zeropoint, 
                                     zeropoint_uncertainty, source_catalog)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(frame_id, combined_footprint_hash) DO UPDATE SET
    zeropoint = excluded.zeropoint,
    zeropoint_uncertainty = excluded.zeropoint_uncertainty;
    """
    data_to_insert = [(row['frame_id'],
                       combined_footprint_hash,
                       row['zeropoint'],
                       row['zeropoint_uncertainty'],
                       source_catalog) for _, row in zeropoint_results.iterrows()]

    db_path = get_user_config()['database_path']
    with sqlite3.connect(db_path, timeout=15.0) as conn:
        conn.executemany(insert_query, data_to_insert)
