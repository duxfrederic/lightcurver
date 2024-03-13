from scipy.stats import median_abs_deviation
import sqlite3
import numpy as np
from scipy.optimize import minimize
import pandas as pd

from ..structure.database import execute_sqlite_query, get_pandas
from ..structure.user_config import get_user_config
from ..utilities.footprint import get_combined_footprint_hash
from ..utilities.chi2_selector import get_chi2_bounds
from ..plotting.normalization_plotting import plot_normalized_star_curves


def get_fluxes(combined_footprint_hash, photometry_chi2_min, photometry_chi2_max):
    """
    Retrieves all the available star fluxes in all frames.
    If a given frame does not have a flux for this star, a NaN placeholder will be used.
    We filter by the chi2 of the fit: fluxes with an out of bounds chi2 will be replaced by a NaN
    value as well.

    :param combined_footprint_hash: int, the hash of the footprint we are working with.
    :param photometry_chi2_min: minimum acceptable chi2 value for the fit of the photometry of the star in this frame
    :param photometry_chi2_max: ditto but max
    :return: A list of frames that meet the criteria.
    """
    query = """
    SELECT s.name,
           f.id AS frame_id, 
           f.mjd,
           sff.star_gaia_id, 
           sff.combined_footprint_hash,
           IFNULL(sff.flux, 'NaN') AS flux,
           IFNULL(sff.flux_uncertainty, 'NaN') AS d_flux
    FROM 
       frames f
    JOIN star_flux_in_frame sff ON f.id = sff.frame_id 
    JOIN stars s ON sff.star_gaia_id = s.gaia_id AND sff.combined_footprint_hash = s.combined_footprint_hash
    JOIN stars_in_frames sif ON sif.star_gaia_id = s.gaia_id
                                AND sif.frame_id = f.id 
                                AND sif.combined_footprint_hash = s.combined_footprint_hash 
    WHERE 
        sff.combined_footprint_hash = ?
    AND 
        sff.chi2 BETWEEN ? AND ?
    ORDER BY 
       s.name, f.id"""
    params = (combined_footprint_hash, photometry_chi2_min, photometry_chi2_max)

    return execute_sqlite_query(query, params, is_select=True, use_pandas=True)


def update_normalization_coefficients(norm_data):
    db_path = get_user_config()['database_path']
    with sqlite3.connect(db_path, timeout=15.0) as conn:
        cursor = conn.cursor()

        # insert query with ON CONFLICT clause for bulk update
        # this handles the case where we try to insert normalization coefficients for a given frame again
        # on conflict, we update the existing record with the new coefficient values
        insert_query = """
        INSERT INTO normalization_coefficients (frame_id, combined_footprint_hash, coefficient, coefficient_uncertainty)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(combined_footprint_hash, frame_id) DO UPDATE SET
        coefficient=excluded.coefficient, coefficient_uncertainty=excluded.coefficient_uncertainty
        """

        cursor.executemany(insert_query, norm_data)
        conn.commit()


def cost_function_scatter_in_frame(scaling_factors, normalized_flux_pivot, normalized_d_flux_pivot):
    """
    This is a utility for the calculation of the coef. We compare the fluxes of stars divided by their global median
    in each frame. We scale each star light-curve by a global coefficient.
    This function gives us the average scatter in each frame between the different star fluxes given a set of
    scaling factors.
    Args:
        scaling_factors: 1d array, one value per star
        normalized_flux_pivot: pivot table with frame_id as columns and star_gaia_id as rows, for fluxes
        normalized_d_flux_pivot: same but for uncertainties

    Returns: scalar, a variance representative of the scatter in each frame with the available stars in that frame.
    """
    scaled_fluxes = normalized_flux_pivot.mul(scaling_factors, axis=0)

    weights = 1 / normalized_d_flux_pivot

    weighted_means = (scaled_fluxes * weights).sum(axis=0) / weights.sum(axis=0)
    weighted_variance = (
        (weights.mul((scaled_fluxes.sub(weighted_means, axis='columns'))**2)).sum(axis=0) / weights.sum(axis=0)).sum()
    return weighted_variance


def filter_outliers(fluxes, uncertainties, threshold=3.0):
    """
    Another utility function, just filtering the very off fluxes in each frame.
    Args:
        fluxes: pandas series index by star_gaia_id
        uncertainties:  pandas series index by star_gaia_id
        threshold: float

    Returns:
        same as input types, but filtered.

    """
    median_flux = fluxes.median()
    mad = median_abs_deviation(fluxes, scale='normal')
    is_outlier = abs(fluxes - median_flux) > (threshold * mad)
    # Ensure both fluxes and uncertainties are filtered by the same outlier mask
    return fluxes[~is_outlier], uncertainties[fluxes.index[~is_outlier]]


def weighted_std(values, weights):
    """
    last utility function: weighted standard deviation for our estimation of the scatter in each coefficient.
    Args:
        values: array
        weights:  array

    Returns:
        float, some standard deviation.

    """
    isnan = np.isnan(values) + np.isnan(weights)
    values = values[~isnan]
    weights = weights[~isnan]
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)


def calculate_coefficient():
    """
    this is a routine called by the workflow manager. It interfaces with the user config and the database
    to calculate a representative norm of each frame, which will be used as a normalization coefficient later.

    Returns:
        nothing

    """
    user_config = get_user_config()

    # query initial frames, so we can calculate the footprint at hand
    # TODO for all these frames_ini requests, if by the end of building this pipeline I do not
    # TODO see a variation in the pattern, just put them in the get_combined_footprint_hash function
    # TODO so it is called only when necessary.
    frames_ini = get_pandas(columns=['id', 'image_relpath', 'exptime', 'mjd', 'seeing_pixels', 'pixel_scale'],
                            conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_ini['id'].to_list())

    fluxes_fit_chi2_min, fluxes_fit_chi2_max = get_chi2_bounds(psf_or_fluxes='fluxes')
    df = get_fluxes(combined_footprint_hash=combined_footprint_hash,
                    photometry_chi2_min=fluxes_fit_chi2_min,
                    photometry_chi2_max=fluxes_fit_chi2_max)
    # 1. normalize by median in each star -- get a 'norm' of each frame for each individual star.
    median_flux_per_star = df.groupby('star_gaia_id')['flux'].median().rename('median_flux')
    df2 = df.merge(median_flux_per_star, on='star_gaia_id')
    df2['normalized_flux'] = df2['flux'] / df2['median_flux']
    df2['normalized_d_flux'] = df2['d_flux'] / df2['median_flux']

    # we'll pivot the table to have frame_id as columns and star_gaia_id as rows
    # (useful to conduct operations within each frame)
    normalized_flux_pivot = df2.pivot(index='star_gaia_id', columns='frame_id', values='normalized_flux')
    normalized_d_flux_pivot = df2.pivot(index='star_gaia_id', columns='frame_id', values='normalized_d_flux')

    # now, these fluxes are just, for each star, 'star flux / median star flux'.
    # we could easily have scaling differences, so we'll align the curves by indivudally scaling them,
    # minimizing the dispersion in eah frame.
    # we will do this with the constraint that the mean of the coefs should be 1
    # (else, the optimizer will just set everything to 0 to minimize dispersion ...)
    constraint = ({'type': 'eq', 'fun': lambda coeffs: 1 - np.nanmean(coeffs)})
    initial_star_scaling_factors = np.ones(normalized_flux_pivot.shape[0])
    result = minimize(cost_function_scatter_in_frame, initial_star_scaling_factors,
                      args=(normalized_flux_pivot, normalized_d_flux_pivot), constraints=constraint, method='SLSQP')
    optimized_star_scaling_factors_with_reference = result.x

    # ok, scale each star light-curve by its optimized scaling
    adjusted_normalized_fluxes = normalized_flux_pivot.mul(optimized_star_scaling_factors_with_reference, axis=0)
    adjusted_normalized_d_fluxes = normalized_d_flux_pivot.mul(optimized_star_scaling_factors_with_reference, axis=0)

    # now, let's eliminate the obvious outliers
    filtered_fluxes = adjusted_normalized_fluxes.copy()
    filtered_uncertainties = adjusted_normalized_d_fluxes.copy()

    filtered_weights = 1. / filtered_uncertainties
    norm_err = filtered_fluxes.columns.map(
        lambda frame_id: weighted_std(filtered_fluxes[frame_id], filtered_weights[frame_id])
    )
    norm_coeff = (filtered_fluxes.multiply(filtered_weights)).sum(axis=0) / filtered_weights.sum(axis=0)
    # restore index:
    norm_err = pd.Series(norm_err, index=filtered_fluxes.columns)
    # case with only one star: norm_err is 0, just set it to something relatively big.
    norm_err.loc[norm_err == 0.] = 0.1 * norm_coeff.loc[norm_err == 0.]

    # ok, prepare the insert into the db
    norm_data = []
    for frame_id in norm_coeff.keys():
        norm = float(norm_coeff[frame_id])
        err = float(norm_err[frame_id])
        norm_data.append((frame_id, combined_footprint_hash, norm, err))

    update_normalization_coefficients(norm_data)

    # ok, query it again in the plot function.
    plot_norm_dir = user_config['plots_dir'] / 'normalization' / str(combined_footprint_hash)
    plot_norm_dir.mkdir(exist_ok=True, parents=True)

    plot_file = plot_norm_dir / "normalization_fluxes_plot.pdf"
    plot_normalized_star_curves(combined_footprint_hash=combined_footprint_hash, save_path=plot_file)

