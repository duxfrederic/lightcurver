import sqlite3
import numpy as np

from ..structure.database import execute_sqlite_query, get_pandas
from ..structure.user_config import get_user_config
from ..utilities.footprint import get_combined_footprint_hash


def r_sdss_from_gaia(gaia_like_pandas_table):
    """
        Returns r-sdss magnitudes from gaia magnitudes, from table 5.6 of
        https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html
        (see row g - r, here we do r - g, and add g.)
    Args:
        gaia_like_pandas_table: a pandas data frame with columns phot_g_mean_mag, phot_rp_mean_mag, phot_bp_mean_mag

    Returns:
        a pandas series containing r-sdss magnitudes.
    """

    bp_rp = gaia_like_pandas_table['phot_bp_mean_mag'] - gaia_like_pandas_table['phot_rp_mean_mag']
    g = gaia_like_pandas_table['phot_g_mean_mag']

    coef0 = 0.09837
    coef1 = -0.08592
    coef2 = -0.1907
    coef3 = 0.1701
    coef4 = -0.02263

    r_sdss = g + coef0 + coef1 * bp_rp + coef2 * bp_rp ** 2 + coef3 * bp_rp ** 3 + coef4 * bp_rp ** 4
    return r_sdss


def calculate_zeropoints():
    """
    for a given the current combined_footprint_hash, queries all the available star fluxes.
    compares them with gaia magnitudes to calculate a magnitude in your band
    (ONLY IMPLEMENTED FOR R-BAND SO FAR)
    Then calculates a zeropoint for each frame, and updates the database.
    Args:

    Returns:

    """
    flux_query = """
    SELECT 
         sff.frame_id, 
         sff.flux, 
         s.gaia_id, 
         s.gmag as phot_g_mean_mag, 
         s.bmag as phot_bp_mean_mag,
         s.rmag as phot_rp_mean_mag
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
    user_config = get_user_config()

    frames_ini = get_pandas(columns=['id'],
                            conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_ini['id'].to_list())

    flux_data = execute_sqlite_query(flux_query, (combined_footprint_hash,), is_select=True, use_pandas=True)
    if flux_data.empty:
        return

    # so, Gaia magnitudes to r-sdss (TODO make more general, we do have band information in the database)
    flux_data['r_SDSS'] = r_sdss_from_gaia(flux_data)

    flux_data['instrumental_mag'] = -2.5 * np.log10(flux_data['flux'])
    flux_data['mag_difference'] = flux_data['r_SDSS'] - flux_data['instrumental_mag']

    # zeropoint and its uncertainty (std) for each frame
    zeropoint_results = flux_data.groupby('frame_id')['mag_difference'].agg(['median', 'std']).reset_index()
    zeropoint_results.rename(columns={'median': 'zeropoint', 'std': 'zeropoint_uncertainty'}, inplace=True)

    # and update database
    insert_query = """
    INSERT INTO approximate_zeropoints (frame_id, combined_footprint_hash, zeropoint, zeropoint_uncertainty)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(frame_id, combined_footprint_hash) DO UPDATE SET
    zeropoint = excluded.zeropoint,
    zeropoint_uncertainty = excluded.zeropoint_uncertainty;
    """
    data_to_insert = [(row['frame_id'], combined_footprint_hash, row['zeropoint'], row['zeropoint_uncertainty']) for
                      _, row in zeropoint_results.iterrows()]

    db_path = get_user_config()['database_path']
    with sqlite3.connect(db_path, timeout=15.0) as conn:
        conn.executemany(insert_query, data_to_insert)

