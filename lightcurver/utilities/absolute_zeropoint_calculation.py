import numpy as np
import sqlite3

from .footprint import get_combined_footprint_hash
from ..structure.user_config import get_user_config
from ..structure.database import execute_sqlite_query, get_pandas


def calculate_zeropoints(magnitudes_df, source_catalog):
    """
    Calculates zeropoints for each frame based on provided magnitudes and updates the database.
    Args:
        magnitudes_df (pd.DataFrame): DataFrame containing star magnitudes with columns:
                                      'gaia_id', 'catalog_mag', 'catalog_mag_err'
                                      (mag_err not used at the moment but keep it for the future)
        source_catalog: str, for reference, either 'gaia', 'panstarrs', ...

    Returns:
        None
    """
    flux_query = """
    SELECT 
         sff.frame_id, 
         sff.flux, 
         s.gaia_id
    FROM 
         star_flux_in_frame sff
    JOIN 
         stars s ON sff.star_gaia_id = s.gaia_id
    AND
         s.combined_footprint_hash = sff.combined_footprint_hash
    JOIN 
         frames f ON f.id = sff.frame_id
    WHERE 
         sff.combined_footprint_hash = ?
    """
    # boiler plate
    user_config = get_user_config()
    frames_ini = get_pandas(columns=['id'],
                            conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_ini['id'].to_list())

    # get the fluxes measured on the frames
    flux_data = execute_sqlite_query(flux_query, (combined_footprint_hash,), is_select=True, use_pandas=True)
    if flux_data.empty:
        return

    # merge flux_data with magnitudes_df to get the magnitudes for each star
    flux_data = flux_data.merge(magnitudes_df, on='gaia_id')

    flux_data['instrumental_mag'] = -2.5 * np.log10(flux_data['flux'])
    flux_data['mag_difference'] = flux_data['catalog_mag'] - flux_data['instrumental_mag']

    # Zeropoint and its uncertainty (std) for each frame
    zeropoint_results = flux_data.groupby('frame_id')['mag_difference'].agg(['median', 'std']).reset_index()
    zeropoint_results.rename(columns={'median': 'zeropoint', 'std': 'zeropoint_uncertainty'}, inplace=True)

    # Update database
    insert_query = """
    INSERT INTO approximate_zeropoints (frame_id, combined_footprint_hash, zeropoint, 
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
