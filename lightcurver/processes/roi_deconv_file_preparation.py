import h5py
import numpy as np

from lightcurver.structure.database import get_pandas, execute_sqlite_query
from lightcurver.utilities.footprint import get_combined_footprint_hash
from lightcurver.utilities.chi2_selector import get_chi2_bounds
from lightcurver.structure.user_config import get_user_config


def get_frames_for_roi(combined_footprint_hash, psf_fit_chi2_min, psf_fit_chi2_max, **constraints_on_frame_columns_dict):
    """
    Retrieves frames and associated PSFs (built with stars of the given footprint)
    provided that those frames have a PSF with a chi2 between psf_fit_chi2_min and psf_fit_chi2_max.
    Optionally, can filter to include only frames without a flux measurement.

    :param combined_footprint_hash: int, the hash of the combined footprint we are processing.
    :param psf_fit_chi2_min: The minimum acceptable chi2 value for the PSF fit.
    :param psf_fit_chi2_max: The maximum acceptable chi2 value for the PSF fit.
    :param constraints_on_frame_columns_dict: a dictionary, with keys identical to that of the frames table and values
                                              intervals of allowed values. E.g., {'seeing_arcseconds': (0, 1.1), ...}
    :return: A pandas dataframe of frames and associated PSFs that meet the criteria.
    """
    query = """
    SELECT f.*, ps.*, nc.*
    FROM frames f
    JOIN PSFs ps ON f.id = ps.frame_id 
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

    return execute_sqlite_query(query, tuple(params), is_select=True, use_pandas=True)


def prepare_roi_deconv_file():
    user_config = get_user_config()

    frames_ini = get_pandas(columns=['id'],
                            conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_ini['id'].to_list())

    psf_fit_chi2_min, psf_fit_chi2_max = get_chi2_bounds(psf_or_fluxes='psf')

    roi_constraints = user_config['constraints_on_frame_columns_for_roi']
    frames = get_frames_for_roi(combined_footprint_hash=combined_footprint_hash,
                                psf_fit_chi2_min=psf_fit_chi2_min,
                                psf_fit_chi2_max=psf_fit_chi2_max,
                                **roi_constraints)
    # ok, this frames database has everything: the PSF to use, the norm coeff, etc.

    # so, just like when we did photometry of stars, build the data for deconvolution
    with h5py.File(user_config['regions_path'], 'r') as h5f:
        data = []
        noisemap = []
        mask = []
        psf = []
        frame_id = []
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
            seeing.append(frame['seeing_arcseconds'])
            pixel_scale.append(frame['pixel_scale'])
            wcs.append(h5f[f"{frame['image_relpath']}/wcs/ROI"][()])
            exptime.append(frame['exptime'])
            sky_level_electron_per_second.append(frame['sky_level_electron_per_second'])
            mjd.append(frame['mjd'])
            frame_id.append(frame['id'])
            normalization_uncertainty.append(frame['coefficient_uncertainty'])
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

    save_path = user_config['prepared_roi_cutouts_path']
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with h5py.File(user_config['prepared_roi_cutouts_path'], 'w') as f:
        f['frame_id'] = np.array(frame_id)
        f['data'] = np.array(data)
        f['noisemap'] = np.array(noisemap)
        f['psf'] = np.array(psf)
        f['seeing'] = np.array(seeing)
        f['sky_level_electron_per_second'] = np.array(sky_level_electron_per_second)
        f['mjd'] = np.array(mjd)
        #f['global_zeropoint'] = None  # TODO
        f['relative_normalization_error'] = np.array(normalization_uncertainty)
        f['wcs'] = np.array(wcs)
        f['pixel_scale'] = np.array(pixel_scale)

