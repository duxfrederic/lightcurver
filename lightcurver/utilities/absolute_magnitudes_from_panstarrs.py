from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import logging

from ..structure.database import execute_sqlite_query
from ..structure.user_config import get_user_config

"""
 This file is used for photometric calibration, optionally. We query pan-starrs and save magnitudes to our database.
"""


def save_panstarrs_catalog_photometry_to_database(gaia_id):
    """
    Filtering the mast results: we want stars with precise PSF photometry and astrometry.
    Warning: no correction implemented for proper motion. Gaia ref epoch and pan-starrs dr2 roughly match in time frame
    though.
       Args:
           gaia_id:  gaia_id of the star as saved in the database
       returns:
           Nothing
    """
    logger = logging.getLogger('lightcurver.save_panstarrs_catalog_photometry_to_database')
    # 0. check whether we already have data for this star. If yes, we will not redo this step (slow).
    check_query = """
    SELECT COUNT(*) FROM catalog_star_photometry
    WHERE star_gaia_id = ? AND catalog = 'panstarrs'
    """
    check_params = (gaia_id, )
    result = execute_sqlite_query(check_query, check_params)
    if result[0][0] > 0:
        # magnitude already exists for this star, no need to proceed further
        print(f"Data already exists for Gaia ID {gaia_id} and catalog panstarrs")
        return

    # 1. query pan-starrs.
    mast_results = search_panstarrs_around_coordinates(gaia_id)
    # 2. check that we have the relevant data
    mag_dict = photometric_selection_heuristic(mast_results)
    if mag_dict is None:
        # no relevant information ended up being available.
        logger.warning(f"No relevant Pan-STARRS photometry found for star {gaia_id}.")
        return
    # 3. if pan-starrs had the right information, we insert.
    query = """
        INSERT OR REPLACE INTO catalog_star_photometry (catalog, band, mag, mag_err, original_catalog_id, 
                                                        star_gaia_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    params = ('panstarrs',
              mag_dict['band'],
              mag_dict['mag'],
              mag_dict['mag_err'],
              mag_dict['catalog_ID'],
              gaia_id)
    # insert and done!
    execute_sqlite_query(query, params, is_select=False)


def search_panstarrs_around_coordinates(gaia_id):
    """
    Just using astorquery's MAST interface to find sources around the coordinates
    Args:
       gaia_id: int, gaia_id. ra and dec will be queried from our database.
    returns:
       MAST results in an astropy Table, with columns such as raMean, gMeanPSFMag, etc.
    """
    ra, dec = execute_sqlite_query('SELECT ra, dec FROM stars WHERE gaia_id = ?', (gaia_id, ))[0]

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    radius = 1.5 * u.arcsecond  # this is generous given the magnitude of the proper motion of most stars.
    result = Catalogs.query_region(coord, radius=radius, catalog="PanSTARRS", data_release="dr2")
    return result


def photometric_selection_heuristic(mast_results):
    """
    Just a helper function to eliminate bad photometry information: compares mast_result to actual band needed
    from user config
    Args:
        mast_results: astropy table output of search_panstarrs_around_coordinates above
    Return:
        dictionary with band, mag, mag_err, catalog ID, or None if mast_result does not contain the right information.
    """
    # ok, now we start cleaning up. Pan-Starrs has a tendency to not properly cluster identical detections.
    # Thus, we'll do some rough filtering on the nDetections column first.
    if len(mast_results) > 1:
        n_detections = mast_results['nDetections']
        max_index = np.argmax(n_detections)
        # in such cases, the second source will typically have been detecting one or two times max,
        # against many times for the main (correct) one.
        mast_results = mast_results[mast_results['nDetections'] > 0.2 * n_detections[max_index]]
        # so, eliminate such wrongly unmerged detections.

    # next, either we have nothing found in pan-starrs, or still multiple detections:
    if len(mast_results) != 1:
        # can't do anything with this danger of selecting the wrong source
        return None

    # now check that the detection has photometry in the right bands depending on what we need.
    result = mast_results[0]
    config = get_user_config()
    band = config['photometric_band']
    # just a sanity check:
    if 'panstarrs' not in band:
        raise RuntimeError('Running a Pan-STARRS related function when the config file does not mention Pan-STARRS?')
    # see what coverage we have for this star:
    available_bands = [b for b in 'grizy' if result[f'{b}MeanPSFMag']]
    # now the band we need:
    band = band.replace('_panstarrs', '')
    if (band in 'grizy') and (band not in available_bands):
        # then, no can do.
        return None
    elif band in 'grizy':
        # then, simple.
        mag = result[f'{band}MeanPSFMag']
        mag_err = result[f'{band}MeanPSFMagErr']
    elif band == 'c':
        # composite, we need both g and r
        if ('g' not in available_bands) or ('r' not in available_bands):
            return None
        # we combine according to https://iopscience.iop.org/article/10.1088/1538-3873/aabadf/pdf (Eq 2)
        mag = 0.49 * result['gMeanPSFMag'] + 0.51 * result['rMeanPSFMag']
        # approx, same for uncertainty
        mag_err = 0.49 * result['gMeanPSFMagErr'] + 0.51 * result['rMeanPSFMagErr']
    elif band == 'o':
        # another composite, we need r and i
        if ('r' not in available_bands) or ('i' not in available_bands):
            return None
        # same as above, Eq (2) of link
        mag = 0.55 * result['rMeanPSFMag'] + 0.45 * result['iMeanPSFMag']
        # approx, same for uncertainty
        mag_err = 0.55 * result['rMeanPSFMagErr'] + 0.45 * result['iMeanPSFMagErr']
    else:
        # how did we end up here? For sanity, raise.
        raise RuntimeError(f'User config provided a band related to pan-starrs that we do not know about: {band}')

    # if we made it here, then we have a relatively safe catalog magnitude for the star at hand
    return {'band': band, 'mag': mag, 'mag_err': mag_err, 'catalog_ID': result['objID']}
