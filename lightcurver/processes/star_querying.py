import logging
from astropy.coordinates import SkyCoord
import json
import numpy as np
import pandas as pd

from ..utilities.footprint import load_combined_footprint_from_db, get_frames_hash
from ..structure.user_config import get_user_config
from ..processes.frame_star_assignment import populate_stars_in_frames
from ..utilities.gaia import find_gaia_stars
from ..utilities.star_naming import generate_star_names
from ..structure.database import get_pandas, execute_sqlite_query
from ..plotting.sources_plotting import plot_footprints_with_stars


def query_gaia_stars():
    """
    this is called by the workflow manager.
    Queries the frames to be used, and checks the star selection strategy from the user config.
    Then queries the stars with the additional criteria from the config (mag range, astrometric noise, photometric
    noise ...) and inserts them in the database for easy access.
    Also, runs the function that assigns stars to each frame depending on the respective footprint of the frame.
    Returns:

    """
    logger = logging.getLogger("lightcurver.querying_ref_stars_from_gaia")
    user_config = get_user_config()
    frames_info = get_pandas(columns=['id', 'pixel_scale'], conditions=['frames.eliminated != 1'])
    if user_config['star_selection_strategy'] != 'ROI_disk':
        # then it depends on the frames we're considering.
        frames_hash = get_frames_hash(frames_info['id'].to_list())
    else:
        # if ROI_disk, it does not depend on the frames: unique region defined by its radius.
        frames_hash = hash(user_config['ROI_disk_radius_arcseconds'])

    # before doing anything, check whether we are already done
    count = execute_sqlite_query("SELECT COUNT(*) FROM stars WHERE combined_footprint_hash = ?",
                                 params=(frames_hash,), is_select=True)[0][0]
    if count > 0 and not user_config['gaia_query_redo']:
        logger.info(f'Gaia stars already fetched for this footprint: {frames_hash}')
        logger.info('Still re-calculating which star is in which frame again.')
        # we still need to populate the new frames though
        populate_stars_in_frames()
        return
    elif count > 0 and user_config['gaia_query_redo']:
        logger.info(f'Gaia stars already fetched for this footprint: {frames_hash} but redo is True.')
        # then we need to purge the database from the stars queried with this footprint.
        # TODO I forgot we have two types of footprints for a given footprint hash dayum
        # TODO for now proceeding with the user having to set redo if changing footprint type
        execute_sqlite_query("DELETE FROM stars WHERE combined_footprint_hash = ?",
                             params=(frames_hash,), is_select=True)
        logger.info(f'  deleted previously queried stars.')

    if user_config['star_selection_strategy'] == 'common_footprint_stars':
        logger.info(f'config star selection strategy: common footprint of the frames')
        _, common_footprint = load_combined_footprint_from_db(frames_hash)
        region_type = 'polygon'
        query_footprint = common_footprint['coordinates'][0]
        # then we want to make sure we use stars that are available in all frames.
        # this likely achieves the best precision, but is only possible typically in dedicated
        # monitoring programs with stable pointings.
    elif user_config['star_selection_strategy'] == 'stars_per_frame':
        logger.info(f'config star selection strategy: combined largest footprint of the frames')
        largest_footprint, _ = load_combined_footprint_from_db(frames_hash)
        region_type = 'polygon'
        query_footprint = largest_footprint['coordinates'][0]
        # then, we must fall back to using stars selected in each individual frame.
        # here, we will query a larger footprint so that we have options in each
        # individual frame.
    elif user_config['star_selection_strategy'] == 'ROI_disk':
        logger.info(f'config star selection strategy: in a disk around the ROI.')
        center = user_config['ROI_ra_deg'], user_config['ROI_dec_deg']
        radius = user_config['ROI_disk_radius_arcseconds'] / 3600.0
        region_type = 'circle'
        query_footprint = {'center': center, 'radius': radius}
    else:
        raise RuntimeError("Not an agreed upon strategy for star selection:", user_config['star_selection_strategy'])

    kwargs_query = {
        'astrometric_excess_noise_max': user_config['star_max_astrometric_excess_noise'],
        'gmag_range': (user_config['star_min_gmag'], user_config['star_max_gmag']),
        'min_phot_g_mean_flux_over_error': user_config['min_phot_g_mean_flux_over_error'],
        'gaia_provider': user_config['gaia_provider']
    }
    logging.info(f'Querying stars with the following parameters: {kwargs_query}')

    stars_table = find_gaia_stars(region_type, query_footprint, **kwargs_query)

    message = f"Too few stars compared to the config criterion! Only {len(stars_table)} stars available."
    enough_stars = len(stars_table) >= user_config['min_number_stars']
    if not enough_stars:
        logging.error(message + ' Force stopping.')
    assert enough_stars, message

    columns = ['combined_footprint_hash', 'name', 'ra', 'dec', 'gmag', 'rmag', 'bmag', 'pmra', 'pmdec', 'ref_epoch',
               'gaia_id', 'distance_to_roi_arcsec']
    insert_query = f"INSERT INTO stars ({', '.join(columns)}) VALUES ({', '.join(len(columns)*['?'])})"
    stars_coord = SkyCoord(ra=stars_table['ra'], dec=stars_table['dec'])
    stars_table['distance_to_roi'] = stars_coord.separation(user_config['ROI_SkyCoord']).arcsecond
    # we do not want the ROI itself as a reference:
    stars_table = stars_table[stars_table['distance_to_roi'] > user_config['ROI_size']]
    stars_table.sort('distance_to_roi')
    # add a friendly name to each star (a, b, c, ....)
    stars_table['name'] = generate_star_names(len(stars_table))
    for star in stars_table:
        # loads of floats because float32 does weird stuff to sqlite? not sure, but explicit casting fixes issues.
        star_data = (frames_hash, star['name'], float(star['ra']), float(star['dec']), float(star['phot_g_mean_mag']),
                     float(star['phot_rp_mean_mag']),  float(star['phot_bp_mean_mag']),
                     float(star['pmra']), float(star['pmdec']), float(star['ref_epoch']), int(star['source_id']),
                     star['distance_to_roi'])
        execute_sqlite_query(insert_query, params=star_data, is_select=False)
    logger.info('Calculating which star is in which frame.')
    populate_stars_in_frames()
    # let us also make a plot of how the gaia stars we queried are distributed within our footprint.
    query = """
    SELECT frames.id, footprints.polygon
    FROM footprints 
    JOIN frames ON footprints.frame_id = frames.id 
    WHERE frames.eliminated != 1;
    """
    results = execute_sqlite_query(query)
    polygon_list = [np.array(json.loads(result[1])) for result in results]
    save_path = user_config['plots_dir'] / 'footprints_with_gaia_stars.jpg'
    stars_table = stars_table.to_pandas()
    roi_coord = user_config['ROI_SkyCoord']
    roi_row = {'name': 'roi', 'ra': roi_coord.ra.value, 'dec': roi_coord.dec.value}
    stars_table = pd.concat([stars_table, pd.DataFrame([roi_row])], ignore_index=True)
    plot_footprints_with_stars(footprint_arrays=polygon_list, stars=stars_table, save_path=save_path)
    logger.info(f'Plot of the queried reference Gaia stars saved at {save_path}.')
