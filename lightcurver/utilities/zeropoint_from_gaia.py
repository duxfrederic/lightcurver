import sqlite3
import numpy as np

from ..structure.database import execute_sqlite_query, get_pandas
from ..structure.user_config import get_user_config
from ..utilities.footprint import get_combined_footprint_hash


def convert_gaia_to_photometric_band(gaia_like_pandas_table, band):
    """
    Converts Gaia magnitudes to other photometric magnitudes using polynomial coefficients.
    For example, the sdss coefficients are from table 5.6 of
    https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html
    (see, e.g., row g - r, here we do r - g, and add g.)
    Args:
        gaia_like_pandas_table (pd.DataFrame): frame with columns phot_g_mean_mag, phot_rp_mean_mag, phot_bp_mean_mag
        band (str): The desired band:
                'r_sdss', 'i_sdss', 'g_sdss',
                'V', 'R', or 'Ic' for Johnson-Cousins,
                'B_T' or 'V_T' for Tycho2
    Returns:
        pd.Series: A pandas Series containing the specified magnitudes.
    """
    # taken from tables 5.6 and 5.7, see link in docstring
    coefficients = {
        'r_sdss': [-0.09837, 0.08592, 0.1907, -0.1701, 0.02263],
        'i_sdss': [-0.293, 0.6404, -0.09609, -0.002104],
        'g_sdss': [0.2199, -0.6365, -0.1548, 0.0064],
        'V': [-0.02704, 0.01424, -0.2156, 0.01426],
        'R': [-0.02275, 0.3961, -0.1243, -0.01396, 0.003775],
        'Ic': [0.01753, 0.76, -0.0991],
        'V_T': [-0.01077, -0.0682, -0.2387, 0.02342],
        'B_T': [-0.004288, -0.8547, 0.1244, -0.9085, 0.4843, -0.06814]
    }

    if band not in coefficients:
        raise ValueError(f"Unsupported band. Choose among {list(coefficients.keys())}.")

    coef = coefficients[band]
    bp_rp = gaia_like_pandas_table['phot_bp_mean_mag'] - gaia_like_pandas_table['phot_rp_mean_mag']
    g = gaia_like_pandas_table['phot_g_mean_mag']

    mag = g - sum(coef[i] * bp_rp**i for i in range(len(coef)))

    return mag


def calculate_zeropoints():
    """
    for a given the current combined_footprint_hash, queries all the available star fluxes.
    compares them with gaia magnitudes to calculate a magnitude in your band
    (ONLY IMPLEMENTED FOR R-BAND SO FAR)
    TODO
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

    # so, Gaia magnitudes to other photometric band
    flux_data['converted_mag'] = convert_gaia_to_photometric_band(flux_data, user_config['photometric_band'])

    flux_data['instrumental_mag'] = -2.5 * np.log10(flux_data['flux'])
    flux_data['mag_difference'] = flux_data['converted_mag'] - flux_data['instrumental_mag']

    # zeropoint and its uncertainty (std) for each frame
    zeropoint_results = flux_data.groupby('frame_id')['mag_difference'].agg(['median', 'std']).reset_index()
    zeropoint_results.rename(columns={'median': 'zeropoint', 'std': 'zeropoint_uncertainty'}, inplace=True)

    # and update database
    insert_query = """
    INSERT INTO absolute_zeropoints (frame_id, combined_footprint_hash, zeropoint, zeropoint_uncertainty)
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

