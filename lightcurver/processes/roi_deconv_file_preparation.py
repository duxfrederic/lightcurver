import h5py
import numpy as np
import logging

from ..structure.database import get_pandas, execute_sqlite_query
from ..utilities.footprint import get_combined_footprint_hash
from ..utilities.chi2_selector import get_chi2_bounds
from ..structure.user_config import get_user_config


def get_frames_for_roi(combined_footprint_hash, psf_fit_chi2_min, psf_fit_chi2_max,
                       constraints_on_frame_columns_dict, constraints_on_normalization_coeff_dict):
    """
    Retrieves frames and associated PSFs (built with stars of the given footprint)
    provided that those frames have a PSF with a chi2 between psf_fit_chi2_min and psf_fit_chi2_max.
    Optionally, can filter to include only frames without a flux measurement.

    :param combined_footprint_hash: int, the hash of the combined footprint we are processing.
    :param psf_fit_chi2_min: The minimum acceptable chi2 value for the PSF fit.
    :param psf_fit_chi2_max: The maximum acceptable chi2 value for the PSF fit.
    :param constraints_on_frame_columns_dict: a dictionary, with keys identical to that of the frames table and values
                                              intervals of allowed values. E.g., {'seeing_arcseconds': (0, 1.1), ...}
    :param constraints_on_normalization_coeff_dict: a dictionary, with keys identical to the columns of the
                                                    normalization_coefficients table columns, and values same as
                                                    argument constraints_on_frame_columns_dict.
    :return: A pandas dataframe of frames and associated PSFs that meet the criteria.
    """
    # so, select all frames with a normalization coefficient and psf.
    # if multiple psfs, select the one with the smallest chi2.
    # (done with the subquery)
    query = """
    SELECT f.*, ps.*, nc.*
    FROM frames f
    JOIN (
        SELECT *,
        ROW_NUMBER() OVER (PARTITION BY frame_id ORDER BY chi2 ASC) as rn
        FROM PSFs
    ) ps ON f.id = ps.frame_id AND ps.rn = 1
    JOIN normalization_coefficients nc ON f.id = nc.frame_id AND nc.combined_footprint_hash = ps.combined_footprint_hash
    WHERE nc.combined_footprint_hash = ?
    AND ps.chi2 BETWEEN ? AND ?
    """

    # params of the query:
    params = [combined_footprint_hash, psf_fit_chi2_min, psf_fit_chi2_max]

    # append constraints based on the provided constraints
    for column, (min_val, max_val) in constraints_on_frame_columns_dict.items():
        query += f" AND f.{column} BETWEEN ? AND ?"
        params.extend([min_val, max_val])

    # also append constraints on the normalization coefficient
    for column, (min_val, max_val) in constraints_on_normalization_coeff_dict.items():
        query += f" AND nc.{column} BETWEEN ? AND ?"
        params.extend([min_val, max_val])

    # and order by mjd ...
    query += " ORDER BY f.mjd"

    return execute_sqlite_query(query, tuple(params), is_select=True, use_pandas=True)


def fetch_and_adjust_zeropoints(combined_footprint_hash):
    """
    Just a helper function.
    We query our normalization coefficients and zeropoints.
    We adjust the zeropoints by the normalization coefficient.
    We check that the scatter in zeropoints after normalizing is smaller than the scatter before.
    (just a sanity check really)

    we return the global zeropoint of normalized data, and a scatter we can use as STATISTICAL uncertainty.

    Params:
        combined_footprint_hash: as usual the hash of the footprint we're working with at the moment.

    """

    zeropoint_query = """
    SELECT
        az.frame_id,
        az.zeropoint,
        az.zeropoint_uncertainty,
        nc.coefficient
    FROM
        approximate_zeropoints az
    JOIN
        normalization_coefficients nc ON az.frame_id = nc.frame_id
    AND
        az.combined_footprint_hash = nc.combined_footprint_hash
    WHERE
        az.combined_footprint_hash = ?
    """
    zeropoints_data = execute_sqlite_query(zeropoint_query, (combined_footprint_hash,), is_select=True, use_pandas=True)

    if zeropoints_data.empty:
        return None, None

    # convert normalization coefficient to magnitude adjustment,
    # then adjust the zeropoint by subtracting the magnitude adjustment
    zeropoints_data['norm_adjustment'] = -2.5 * np.log10(zeropoints_data['coefficient'])
    zeropoints_data['adjusted_zeropoint'] = zeropoints_data['zeropoint'] + zeropoints_data['norm_adjustment']

    zp_scatter_not_normalized = zeropoints_data['zeropoint'].std()
    zp_scatter_normalized = zeropoints_data['adjusted_zeropoint'].std()
    global_zp = zeropoints_data['adjusted_zeropoint'].median()

    message = "The scatter in zeropoints before normalizing is lower than after normalizing? Not normal, investigate."
    assert zp_scatter_normalized < zp_scatter_not_normalized, message

    return global_zp, zp_scatter_normalized


def prepare_roi_deconv_file():
    """
    This is called by the workflow manager.
    Given the previous steps, selects the frames that have the necessary quantities calculated
    (PSF, normalization coefficient).
    Then makes a new hdf5 file with the cutouts of the ROI (both data and noisemaps,
    divided by normalization coefficients), and the PSFs.
    Returns:
        Nothing
    """
    logger = logging.getLogger('lightcurver.roi_deconv_file_preparation')
    user_config = get_user_config()

    frames_ini = get_pandas(columns=['id'],
                            conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_ini['id'].to_list())

    psf_fit_chi2_min, psf_fit_chi2_max = get_chi2_bounds(psf_or_fluxes='psf')

    roi_constraints = user_config['constraints_on_frame_columns_for_roi']
    norm_constraints = user_config['constraints_on_normalization_coeff']
    frames = get_frames_for_roi(combined_footprint_hash=combined_footprint_hash,
                                psf_fit_chi2_min=psf_fit_chi2_min,
                                psf_fit_chi2_max=psf_fit_chi2_max,
                                constraints_on_frame_columns_dict=roi_constraints,
                                constraints_on_normalization_coeff_dict=norm_constraints)
    # ok, this frames database has everything: the PSF to use, the norm coeff, etc.
    logger.info(f'Preparing calibrated cutouts of the ROI from {len(frames)} frames.')
    # so, just like when we did photometry of stars, build the data for deconvolution
    with h5py.File(user_config['regions_path'], 'r') as h5f:
        data = []
        noisemap = []
        mask = []
        psf = []
        frame_id = []
        subsampling_factors = []  # should be a unique value but this way we can check
        angles_to_north = []
        # we'll include some more stuff for reference
        seeing, pixel_scale, wcs, mjd, exptime, sky_level_electron_per_second = [], [], [], [], [], []
        normalization_uncertainty = []
        for j, frame in frames.iterrows():
            coefficient = frame['coefficient']
            data.append(h5f[f"{frame['image_relpath']}/data/ROI"][...] / coefficient)
            noisemap.append(h5f[f"{frame['image_relpath']}/noisemap/ROI"][...] / coefficient)
            mask.append(h5f[f"{frame['image_relpath']}/cosmicsmask/ROI"][...])
            psf_ref = frame['psf_ref']
            psf.append(h5f[f"{frame['image_relpath']}/{psf_ref}/narrow_psf"][...])
            subsampling_factors.append(h5f[f"{frame['image_relpath']}/{psf_ref}/subsampling_factor"][...])
            seeing.append(frame['seeing_arcseconds'])
            pixel_scale.append(frame['pixel_scale'])
            wcs.append(h5f[f"{frame['image_relpath']}/wcs/ROI"][()])
            exptime.append(frame['exptime'])
            sky_level_electron_per_second.append(frame['sky_level_electron_per_second'])
            mjd.append(frame['mjd'])
            frame_id.append(frame['id'])
            normalization_uncertainty.append(frame['coefficient_uncertainty'])
            angles_to_north.append(frame['angle_to_north'])
        data, noisemap, mask, psf = np.array(data), np.array(noisemap), np.array(mask), np.array(psf)
        # just like for the PSF, we need to remove the NaNs ...
        isnan = np.where(np.isnan(data) * np.isnan(noisemap))
        data[isnan] = 0.
        noisemap[isnan] = 1e7
        # cosmics: masks are 'true' where cosmic, and we typically want it to "true" for good pixels
        mask = ~(np.array(mask).astype(bool))  # so we invert it.
        # oh, we invert it again to boost the noisemap where mask is False,
        # but better have the masks loaded the right way for the future.
        noisemap[~mask] *= 1000.
        # ok now that everything is ready let's get out of the context manager, also to close the file,
        # and we can open the deconvolution ready file.

    # get the zeropoint:
    global_zp, global_zp_scatter = fetch_and_adjust_zeropoints(combined_footprint_hash=combined_footprint_hash)

    # where we save the ready to deconvolve cutouts:
    save_path = user_config['prepared_roi_cutouts_path']

    if save_path is None:
        roi = user_config['roi_name']
        save_path = user_config['workdir'] / 'prepared_roi_cutouts' / f"cutouts_{combined_footprint_hash}_{roi}.h5"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with h5py.File(save_path, 'w') as f:
        f['frame_id'] = np.array(frame_id)
        f['data'] = np.array(data)
        f['noisemap'] = np.array(noisemap)
        f['psf'] = np.array(psf)
        f['seeing'] = np.array(seeing)
        f['sky_level_electron_per_second'] = np.array(sky_level_electron_per_second)
        f['mjd'] = np.array(mjd)
        f['global_zeropoint'] = np.array(float(global_zp))
        f['global_zeropoint_scatter'] = np.array(float(global_zp_scatter))
        f['relative_normalization_error'] = np.array(normalization_uncertainty)
        f['wcs'] = np.array(wcs)
        f['pixel_scale'] = np.array(pixel_scale)
        f['subsampling_factor'] = np.array(subsampling_factors)
        f['angle_to_north'] = np.array(angles_to_north)

    logger.info(f'Wrote the h5 file containing the calibrated cutouts at {save_path}.')
