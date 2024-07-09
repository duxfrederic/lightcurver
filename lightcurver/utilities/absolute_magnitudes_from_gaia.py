from ..structure.database import execute_sqlite_query
from ..structure.user_config import get_user_config


def save_gaia_catalog_photometry_to_database(gaia_id):
    """
    Converts Gaia magnitudes to other photometric magnitudes using polynomial coefficients.
    For example, the sdss coefficients are from table 5.6 of
    https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html
    (see, e.g., row g - r, here we do r - g, and add g.)
    Args:
        gaia_id (int): star at hand for which we would like to compute photometry in the band of the config.
    Returns:
        None
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
    user_config = get_user_config()
    band = user_config['photometric_band']
    if band not in coefficients:
        raise ValueError(f"Unsupported band. Choose among {list(coefficients.keys())}.")

    # get the photometry from our database
    flux_query = """
    SELECT 
         gaia_id, 
         gmag as phot_g_mean_mag, 
         bmag as phot_bp_mean_mag,
         rmag as phot_rp_mean_mag
    FROM 
         stars
    WHERE 
         gaia_id = ?
    LIMIT 
         1
    """

    gaia_mags = execute_sqlite_query(flux_query, (gaia_id,), is_select=True, use_pandas=True)
    coef = coefficients[band]
    bp_rp = gaia_mags['phot_bp_mean_mag'] - gaia_mags['phot_rp_mean_mag']
    g = gaia_mags['phot_g_mean_mag']
    band_mag = (g - sum(coef[i] * bp_rp**i for i in range(len(coef))))[0]  # pandas series, extract 0th element

    # now we insert in the database:
    query = """
        INSERT OR REPLACE INTO catalog_star_photometry (catalog, band, mag, mag_err, original_catalog_id, 
                                                        star_gaia_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    params = ('gaia',
              band,
              band_mag,
              0.03,  # nominal mag scatter for relations above
              gaia_id,  # not important here
              gaia_id)
    # insert and hopefully done.
    execute_sqlite_query(query, params, is_select=False)
