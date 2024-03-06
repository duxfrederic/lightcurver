import h5py
import numpy as np
import sqlite3
from datetime import datetime
from starred.deconvolution.deconvolution import setup_model
from starred.deconvolution.loss import Loss
from starred.deconvolution.parameters import ParametersDeconv
from starred.optim.optimization import Optimizer, FisherCovariance

from ..structure.database import execute_sqlite_query, select_stars, select_stars_for_a_frame, get_pandas
from ..structure.user_config import get_user_config
from ..utilities.chi2_selector import get_psf_chi2_bounds
from ..utilities.footprint import get_combined_footprint_hash
from ..plotting.star_photometry_plotting import plot_joint_deconv_diagnostic


def do_one_deconvolution(data, noisemap, psf, subsampling_factor, n_iter=2000):
    """
    Joint 'deconvolution' of N stamps of a star (in data), with noisemap, and associated PSF at each slice.
    the subsampling factor is that used for building the psf model.
    Equivalent to PSF photometry of all slices.
    Args:
        data: numpy array (N, nx, ny) containing the epochs of the star
        noisemap:  numpy array (N, nx, ny) noisemap of the above
        psf: numpy array (N, nxx, nyy) PSF model, one per slice
        subsampling_factor: int, subsampling_factor of the psf model.
        n_iter: int, number of adabelief iterations to do. default 2000

    Returns: dictionary, containing the fluxes (ready to be used) as a 1D array, the final kwargs of the optimization,
             the chi2, and by how much we rescaled the data before optimizing. (the fluxes are already scaled back
             to the data's actual scale, but not the kwargs.)

    """
    # so, rescale
    sigma_2 = noisemap**2
    scale = np.nanmax(data)
    data /= scale
    sigma_2 /= scale**2
    # image positions: just a point source in the center.
    xs = np.array([0.])
    ys = np.array([0.])
    # initial guess for fluxes, some kind of rough "aperture"  photometry with background subtraction
    background_values = np.nanmean([
            np.nanmedian(data[:, :1, :], axis=(1, 2)),
            np.nanmedian(data[:, :, :1], axis=(1, 2)),
            np.nanmedian(data[:, -1:, :], axis=(1, 2)),
            np.nanmedian(data[:, :, -1:], axis=(1, 2))
    ])
    a_est = np.nansum(data, axis=(1, 2)) - data[0].size * background_values
    a_est = list(a_est)

    model, kwargs_init, kwargs_up, kwargs_down, kwargs_fixed = setup_model(data, sigma_2, psf,
                                                                           xs, ys,
                                                                           subsampling_factor,
                                                                           a_est)
    kwargs_init['kwargs_background']['mean'] = background_values

    # fix the background. (except the mean component)
    kwargs_fixed['kwargs_background']['h'] = kwargs_init['kwargs_background']['h']
    # rotation pointless for single source.
    kwargs_fixed['kwargs_analytic']['alpha'] = kwargs_init['kwargs_analytic']['alpha']

    parameters = ParametersDeconv(kwargs_init=kwargs_init,
                                  kwargs_fixed=kwargs_fixed,
                                  kwargs_up=kwargs_up,
                                  kwargs_down=kwargs_down)

    loss = Loss(data=data, deconv_class=model, param_class=parameters, sigma_2=sigma_2,
                regularization_terms='l1_starlet',
                regularization_strength_scales=1,  # not needed since no free background...
                regularization_strength_hf=1)      # but the starred interface wants us to provide them anyway

    optim = Optimizer(loss, parameters, method='adabelief')

    optimiser_optax_option = {
                'max_iterations': n_iter, 'min_iterations': None,
                'init_learning_rate': 2e-3, 'schedule_learning_rate': True,
                'restart_from_init': True, 'stop_at_loss_increase': False,
                'progress_bar': True, 'return_param_history': True
    }

    optim.minimize(**optimiser_optax_option)
    kwargs_final = parameters.best_fit_values(as_kwargs=True)
    modelled_pixels = model.model(kwargs_final)
    residuals = data - modelled_pixels
    # let's calculate a chi2 in each frame!
    chi2_per_frame = np.nansum((residuals**2 / sigma_2), axis=(1, 2)) / model.image_size**2
    chi2 = np.nanmean(chi2_per_frame)
    fluxes = scale * np.array(kwargs_final['kwargs_analytic']['a'])

    # let us calculate the uncertainties ~ equivalent to photon noise / read noise
    fish = FisherCovariance(parameters, optim, diagonal_only=True)
    fish.compute_fisher_information()
    k_errs = fish.get_kwargs_sigma()
    fluxes_uncertainties = scale * np.array(k_errs['kwargs_analytic']['a'])

    result = {
        'scale': scale,
        'kwargs_final': kwargs_final,
        'kwargs_uncertainties': k_errs,
        'fluxes': fluxes,
        'fluxes_uncertainties': fluxes_uncertainties,
        'chi2': chi2,
        'chi2_per_frame': np.array(chi2_per_frame),
        'loss_curve': optim.loss_history,
        'residuals': scale * residuals
    }
    return result


def get_frames_for_star(combined_footprint_hash, gaia_id,
                        psf_fit_chi2_min, psf_fit_chi2_max, only_fluxless_frames=False):
    """
    Retrieves frames that include the given star, provided that those frames have a PSF with a chi2 between
    psf_fit_chi2_min and psf_fit_chi2_max. Optionally, can filter to include only frames without a flux measurement.

    :param combined_footprint_hash: int, the hash of the combined footprint we are processing.
    :param gaia_id: The Gaia ID of the star.
    :param psf_fit_chi2_min: The minimum acceptable chi2 value for the PSF fit.
    :param psf_fit_chi2_max: The maximum acceptable chi2 value for the PSF fit.
    :param only_fluxless_frames: If True, only returns frames without a flux measurement for the star. Default is False.
    :return: A list of frames that meet the criteria.
    """
    # start building the base query
    query = """
    SELECT f.*, ps.chi2, ps.psf_ref
    FROM frames f
    JOIN stars_in_frames sif ON f.id = sif.frame_id AND sif.combined_footprint_hash = ?
    """
    # add the LEFT JOIN for star_flux_in_frame, if we only are selecting the frames without a flux measurement.
    if only_fluxless_frames:
        query += "LEFT JOIN star_flux_in_frame sff ON f.id = sff.frame_id AND sif.star_gaia_id = sff.star_gaia_id\n"

    # keep building the query
    query += """
    JOIN PSFs ps ON f.id = ps.frame_id
    WHERE sif.star_gaia_id = ? 
    """
    # the condition for frames without flux measurements
    if only_fluxless_frames:
        query += "AND sff.frame_id IS NULL\n"

    # finish the query with PSF chi2 conditions and existence check for appropriate PSF, selecting
    # only 1 to avoid selecting the same frame multiple times if multiple psf models are available for a given frame.
    query += """
    AND EXISTS (
        SELECT 1
        FROM PSFs ps
        WHERE f.id = ps.frame_id
        AND ps.chi2 BETWEEN ? AND ?
    )"""
    params = (combined_footprint_hash, gaia_id, psf_fit_chi2_min, psf_fit_chi2_max)

    return execute_sqlite_query(query, params, is_select=True, use_pandas=True)


def update_star_fluxes(flux_data):
    db_path = get_user_config()['database_path']
    with sqlite3.connect(db_path, timeout=15.0) as conn:
        cursor = conn.cursor()

        # insert query with ON CONFLICT clause for bulk update
        # upon ON CONFLICT (integrity error due to trying to insert a flux for a given star and frame again),
        # as we would if we "redo", then we do indeed erase the previous values and add the new ones.
        insert_query = """
        INSERT INTO star_flux_in_frame (combined_footprint_hash, frame_id, star_gaia_id, flux, flux_uncertainty, chi2)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(combined_footprint_hash, frame_id, star_gaia_id) DO UPDATE SET
        flux=excluded.flux, flux_uncertainty=excluded.flux_uncertainty
        """

        cursor.executemany(insert_query, flux_data)

        conn.commit()


def do_star_photometry():
    """
    Here we do a joint deconvolution of all the frames at once, for each star.
    This is equivalent to PSF photometry, but we do it in the same way
    as the final deconvolution to eliminate potential systematics.
    Returns:

    """
    user_config = get_user_config()
    # so, we need the footprint we are working with.
    # below we will query only the frames we need for each frame, but first we need to
    # query all the frames that we actually started with when defining the footprints:
    # so, let's query those frames, then calculate the footprint.
    frames_ini = get_pandas(columns=['id', 'image_relpath', 'exptime', 'mjd', 'seeing_pixels', 'pixel_scale'],
                            conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_ini['id'].to_list())
    # now we can select the stars we need to do photometry of, within this footprint.
    stars = select_stars(stars_to_use=user_config['stars_to_use_norm'], combined_footprint_hash=combined_footprint_hash)
    # if not re-do ...select only the new frames that do not have a flux measurement yet.
    only_fluxless_frames = not user_config['redo_star_photometry']

    time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")  # for plots
    for i, star in stars.iterrows():
        psf_fit_chi2_min, psf_fit_chi2_max = get_psf_chi2_bounds()
        frames = get_frames_for_star(gaia_id=star['gaia_id'],
                                     psf_fit_chi2_min=psf_fit_chi2_min,
                                     psf_fit_chi2_max=psf_fit_chi2_max,
                                     only_fluxless_frames=only_fluxless_frames,
                                     combined_footprint_hash=combined_footprint_hash)
        if len(frames) == 0:
            # we up to date, nothing to do
            continue
        # build the data for deconvolution
        with h5py.File(user_config['regions_path'], 'r') as h5f:
            data = []
            noisemap = []
            mask = []
            psf = []
            for j, frame in frames.iterrows():
                data.append(h5f[f"{frame['image_relpath']}/data/{star['name']}"][...])
                noisemap.append(h5f[f"{frame['image_relpath']}/noisemap/{star['name']}"][...])
                # more difficult for the psf.
                # we need to reconstruct which stars were used in the psf given our
                # accepted stars in 'stars_to_use_psf'.
                stars_psf = select_stars_for_a_frame(frame_id=frame['id'],
                                                     stars_to_use=user_config['stars_to_use_psf'],
                                                     combined_footprint_hash=combined_footprint_hash)
                psf_ref = 'psf_' + ''.join(sorted(stars_psf['name']))
                mask.append(h5f[f"{frame['image_relpath']}/cosmicsmask/{star['name']}"][...])
                psf.append(h5f[f"{frame['image_relpath']}/{psf_ref}/narrow_psf"][...])
            data, noisemap, mask, psf = np.array(data), np.array(noisemap), np.array(mask), np.array(psf)
            # just like for the PSF, we need to remove the NaNs ...
            isnan = np.where(np.isnan(data) * np.isnan(noisemap))
            data[isnan] = 0.
            noisemap[isnan] = 1e7
            # cosmics: masks are 'true' where cosmic, and we typically want it to "true" for good pixels
            mask = ~(np.array(mask).astype(bool))  # so we invert it.
            # oh, we invert it again to boost the noisemap where mask is False,
            # but better have the masks loaded the right way for the future.
            noisemap[np.where(~mask)[0]] *= 1000.
            # ok now that everything is ready let's get out of the context manager, also to close the file.

        # ready to "deconvolve" using starred!
        result = do_one_deconvolution(data=data, noisemap=noisemap, psf=psf,
                                      subsampling_factor=user_config['subsampling_factor'],
                                      n_iter=user_config['star_deconv_n_iter'])
        # ok, plot the diagnostic
        plot_deconv_dir = user_config['plots_dir'] / 'deconvolutions' / str(combined_footprint_hash)
        plot_deconv_dir.mkdir(exist_ok=True, parents=True)

        plot_file = plot_deconv_dir / f"{time_now}_joint_deconv_star_{star['name']}.jpg"
        plot_joint_deconv_diagnostic(datas=data, noisemaps=noisemap,
                                     residuals=result['residuals'],
                                     chi2_per_frame=result['chi2_per_frame'], loss_curve=result['loss_curve'],
                                     save_path=plot_file)

        # now we can insert our results in our dedicated database table.
        # flux_data should be a list of tuples, each containing (frame_id, star_gaia_id, flux, flux_uncertainty)
        flux_data = []
        # the order of the frames is the same as before, and the same as that of our arrays and fluxes.
        for j, frame in frames.iterrows():
            gaia_id = star['gaia_id']
            frame_id = frame['id']
            flux = float(result['fluxes'][j])
            flux_uncertainty = float(result['fluxes_uncertainties'][j])
            chi2 = float(result['chi2_per_frame'][j])
            flux_data.append((combined_footprint_hash, frame_id, gaia_id, flux, flux_uncertainty, chi2))

        # big insert while updating if value already in DB...
        update_star_fluxes(flux_data)
        # done!
