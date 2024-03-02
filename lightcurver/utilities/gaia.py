from astroquery.gaia import Gaia


def find_gaia_stars_in_polygon(vertices, release='dr3', astrometric_excess_noise_max=None, gmag_range=None,
                               max_phot_g_mean_flux_error=None):
    """
    Query Gaia stars within a polygon defined by a list of vertices, with additional filtering options.

    :param vertices: list of tuples, each tuple contains RA and Dec in degrees [(ra1, dec1), (ra2, dec2), ..., ]
    :param release: str, default 'dr3'. Either 'dr3' or 'dr2'
    :param astrometric_excess_noise_max: float, maximum allowed astrometric excess noise (None for no filter)
    :param gmag_range: tuple, magnitude range in g-band as (min_gmag, max_gmag) (None for no filter)
    :param max_phot_g_mean_flux_error: float, flux error correlates to variability. (None for no filter)

    Returns: astropy table of gaia sources
    """
    assert release.lower() in ['dr2', 'dr3'], "Release must be either 'dr2' or 'dr3'"
    query_table = f"gaia{release.lower()}.gaia_source"
    Gaia.MAIN_GAIA_TABLE = query_table
    Gaia.ROW_LIMIT = 3000

    polygon_string = ', '.join([f"{vertex[0]},{vertex[1]}" for vertex in vertices])
    where_conditions = [
        f"1=CONTAINS(POINT('ICRS', {query_table}.ra, {query_table}.dec), POLYGON('ICRS', {polygon_string}))"
    ]

    if astrometric_excess_noise_max is not None:
        where_conditions.append(f"{query_table}.astrometric_excess_noise < {astrometric_excess_noise_max}")

    if gmag_range is not None:
        where_conditions.append(f"{query_table}.phot_g_mean_mag BETWEEN {gmag_range[0]} AND {gmag_range[1]}")

    if max_phot_g_mean_flux_error is not None:
        where_conditions.append(f"{query_table}.phot_g_mean_flux_error < {max_phot_g_mean_flux_error}")

    where_clause = " AND ".join(where_conditions)

    adql_query = f"""
    SELECT * FROM {Gaia.MAIN_GAIA_TABLE}
    WHERE {where_clause}
    """
    job = Gaia.launch_job_async(adql_query)
    result = job.get_results()
    return result

