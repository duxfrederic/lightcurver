from astroquery.utils.tap.core import TapPlus
from astropy.table import Table, Column
import numpy as np

"""
This file implements queries to Gaia, the aim being finding appropriate calibration stars in the field of interest.
We save the astrometry and photometry of the queried stars, as the photometry can be used later for absolute
zeropoint calibration.
"""

# 2024-04-19: Gaia archive down.
# let us query our Gaia stars from Vizier instead.
vizier_to_gaia_column_mapping = {
    'RA_ICRS': 'ra',
    'DE_ICRS': 'dec',
    'Gmag': 'phot_g_mean_mag',
    'RPmag': 'phot_rp_mean_mag',
    'BPmag': 'phot_bp_mean_mag',
    'pmRA': 'pmra',
    'pmDE': 'pmdec',
    'Source': 'source_id',
    'sepsi': 'astrometric_excess_noise_sig',
    'RFG': 'phot_g_mean_flux_over_error'
}
gaia_to_vizier_column_mapping = {value: key for key, value in vizier_to_gaia_column_mapping.items()}


def construct_where_conditions(gaia_provider, astrometric_excess_noise_max=None, gmag_range=None,
                               min_phot_g_mean_flux_over_error=None):
    """
    utility function for the find functions below, just checks on the conditions and returns a list of potential
    statements to us after 'WHERE'. Also takes care of formatting the table from which we will query.
    Args:
        gaia_provider: string, 'gaia' or 'vizier'
        astrometric_excess_noise_max: float, default None
        gmag_range: tuple of floats (min, max), default None
        min_phot_g_mean_flux_over_error: float, default None

    Returns:
        list of strings containing the conditions, and string containing the name of the table to query.

    """
    gaia_provider = gaia_provider.lower()
    assert gaia_provider in ['gaia', 'vizier'], "gaia_provider must be either 'gaia' or 'vizier'"
    if gaia_provider == 'gaia':
        query_table = "gaiadr3.gaia_source as gdr3 "
    else:
        query_table = f'"I/355/gaiadr3" AS gdr3 '

    where_conditions = []

    if astrometric_excess_noise_max is not None:
        col_name = 'astrometric_excess_noise_sig'
        if gaia_provider == 'vizier':
            col_name = gaia_to_vizier_column_mapping[col_name]
        where_conditions.append(f"gdr3.{col_name} < {astrometric_excess_noise_max}")

    if gmag_range is not None:
        col_name = 'phot_g_mean_mag'
        if gaia_provider == 'vizier':
            col_name = gaia_to_vizier_column_mapping[col_name]
        where_conditions.append(f"gdr3.{col_name} BETWEEN {gmag_range[0]} AND {gmag_range[1]}")

    if min_phot_g_mean_flux_over_error is not None:
        col_name = 'phot_g_mean_flux_over_error'
        if gaia_provider == 'vizier':
            col_name = gaia_to_vizier_column_mapping[col_name]
        where_conditions.append(f"gdr3.{col_name} > {min_phot_g_mean_flux_over_error}")

    return where_conditions, query_table


def find_gaia_stars(region_type, *args, **kwargs):
    """
    Main function to query Gaia stars based on region type (Circle or Polygon).

    :param region_type: str, 'circle' or 'polygon' to define the type of region for the query.
    :param args: Arguments passed to the specific region query function.
    :param kwargs: Keyword arguments for filtering options, passed to the specific region query function.
    """
    if region_type.lower() == 'circle':
        stars_table = find_gaia_stars_in_circle(*args, **kwargs)
    elif region_type.lower() == 'polygon':
        stars_table = find_gaia_stars_in_polygon(*args, **kwargs)
    else:
        raise ValueError("region_type must be either 'Circle' or 'Polygon'")

    # it seems that there is some variation when querying gaia? sometimes columns are capitalized, others not.
    # so, force lower:
    for name in stars_table.colnames:
        new_name = name.lower()
        stars_table.rename_column(name, new_name)
    return stars_table


def run_query(gaia_provider, adql_query):
    """
    Utility function to run the adql query constructed by other functions, given a gaia provider (gaia or vizier)
    Args:
        gaia_provider: str, 'gaia' or 'vizier'
        adql_query: str, an adql query.

    Returns:
        astropy table of sources, with columns following the gaia archive labelling.
    """

    # import Gaia here: on servers without an internet connection, this line waits until timeout
    # before printing a warning. For daily runs without an internet connection, better not import
    # Gaia when not needed.
    from astroquery.gaia import Gaia

    if gaia_provider.lower() == 'gaia':
        Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
        Gaia.ROW_LIMIT = 2000
        job = Gaia.launch_job_async(adql_query)
        result = job.get_results()
    elif gaia_provider.lower() == 'vizier':
        tap = TapPlus(url="https://TAPVizieR.cds.unistra.fr/TAPVizieR/tap")
        job = tap.launch_job(adql_query)
        result_vizier = job.get_results()
        # change column names to what we expect from Gaia:
        result = Table()
        for vizier_col, gaia_col in vizier_to_gaia_column_mapping.items():
            if vizier_col in result_vizier.colnames:
                result[gaia_col] = result_vizier[vizier_col]
        # vizier does not provide the reference epoch.
        result['ref_epoch'] = Column(name='ref_epoch', data=2016.0 * np.ones(len(result), dtype=float))
        # check that we're using dr3, for which the ref epoch is indeed 2016:
        if 'gaiadr3' not in adql_query:
            raise FutureWarning("Using Vizier and 2016 as ref epoch, but not using Gaia DR3. Ref epoch changed? check.")
    else:
        raise RuntimeError("gaia_provider must be 'gaia' or 'vizier'")

    return result


def find_gaia_stars_in_circle(center_radius, gaia_provider='gaia', astrometric_excess_noise_max=None, gmag_range=None,
                              min_phot_g_mean_flux_over_error=None):
    """
    Query Gaia stars within a circle defined by a central point and radius, with additional filtering options.

    :param center_radius: dictionary {'center': tuple(ra,dec), 'radius': radius_degrees}
    :param gaia_provider: str, default 'gaia'. Either 'gaia' or 'vizier'
    :param astrometric_excess_noise_max: float, maximum allowed astrometric excess noise (None for no filter)
    :param gmag_range: tuple, magnitude range in g-band as (min_gmag, max_gmag) (None for no filter)
    :param min_phot_g_mean_flux_over_error: float, flux error correlates to variability. (None for no filter)

    Returns: astropy table of gaia sources
    """
    where_conditions, query_table = construct_where_conditions(gaia_provider, astrometric_excess_noise_max,
                                                               gmag_range, min_phot_g_mean_flux_over_error)

    # Constructing the circle condition
    c = center_radius['center']
    r = center_radius['radius']
    # while handling the vizier weird column naming:
    ra_col = 'ra'
    dec_col = 'dec'
    if gaia_provider == 'vizier':
        ra_col = gaia_to_vizier_column_mapping[ra_col]
        dec_col = gaia_to_vizier_column_mapping[dec_col]
    where_conditions.append(
        f"1=CONTAINS(POINT('ICRS', gdr3.{ra_col}, gdr3.{dec_col}), CIRCLE('ICRS', {c[0]}, {c[1]}, {r}))"
    )

    where_clause = " AND ".join(where_conditions)

    adql_query = f"""
    SELECT * FROM {query_table}
    """
    if where_clause:
        adql_query += f" WHERE {where_clause}"

    result = run_query(gaia_provider=gaia_provider, adql_query=adql_query)
    return result


def find_gaia_stars_in_polygon(vertices, gaia_provider='gaia', astrometric_excess_noise_max=None, gmag_range=None,
                               min_phot_g_mean_flux_over_error=None):
    """
    Query Gaia stars within a polygon defined by a list of vertices, with additional filtering options.

    :param vertices: list of tuples, each tuple contains RA and Dec in degrees [(ra1, dec1), (ra2, dec2), ..., ]
    :param gaia_provider: str, default 'gaia'. Either 'gaia' or 'vizier'
    :param astrometric_excess_noise_max: float, maximum allowed astrometric excess noise (None for no filter)
    :param gmag_range: tuple, magnitude range in g-band as (min_gmag, max_gmag) (None for no filter)
    :param min_phot_g_mean_flux_over_error: float, flux error correlates to variability. (None for no filter)

    Returns: astropy table of gaia sources
    """

    from astroquery.gaia import Gaia
    where_conditions, query_table = construct_where_conditions(gaia_provider, astrometric_excess_noise_max,
                                                               gmag_range, min_phot_g_mean_flux_over_error)

    polygon_string = ', '.join([f"{vertex[0]},{vertex[1]}" for vertex in vertices])

    # handle the vizier weird column naming:
    ra_col = 'ra'
    dec_col = 'dec'
    if gaia_provider == 'vizier':
        ra_col = gaia_to_vizier_column_mapping[ra_col]
        dec_col = gaia_to_vizier_column_mapping[dec_col]
    where_conditions.append(
        f"1=CONTAINS(POINT('ICRS', gdr3.{ra_col}, gdr3.{dec_col}), POLYGON('ICRS', {polygon_string}))"
    )

    where_clause = " AND ".join(where_conditions)

    adql_query = f"""
    SELECT * FROM {Gaia.MAIN_GAIA_TABLE}
    WHERE {where_clause}
    """

    result = run_query(gaia_provider=gaia_provider, adql_query=adql_query)
    return result

