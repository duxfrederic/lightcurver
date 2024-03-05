import h5py
import numpy as np
import sqlite3
from starred.deconvolution.deconvolution import setup_model
from starred.deconvolution.loss import Loss
from starred.deconvolution.parameters import ParametersDeconv
from starred.optim.optimization import Optimizer, FisherCovariance

from ..structure.database import execute_sqlite_query, select_stars, select_stars_for_a_frame
from ..structure.user_config import get_user_config


def do_one_deconvolution(data, noisemap, psf, subsampling_factor):
    """
    Joint 'deconvolution' of N stamps of a star (in data), with noisemap, and associated PSF at each slice.
    the subsampling factor is that used for building the psf model.
    Equivalent to PSF photometry of all slices.
    Args:
        data: numpy array (N, nx, ny) containing the epochs of the star
        noisemap:  numpy array (N, nx, ny) noisemap of the above
        psf: numpy array (N, nxx, nyy) PSF model, one per slice
        subsampling_factor: int, subsampling_factor of the psf model.

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
    background_values = 0.25 * (
            np.nanmedian(data[:, :1, :], axis=(1, 2)) +
            np.nanmedian(data[:, :, :1], axis=(1, 2)) +
            np.nanmedian(data[:, -1:, :], axis=(1, 2)) +
            np.nanmedian(data[:, :, -1:], axis=(1, 2))
    )
    a_est = np.nansum(data, axis=(1, 2)) - data[0].size * background_values
    a_est = list(a_est)

    model, kwargs_init, kwargs_up, kwargs_down, kwargs_fixed = setup_model(data, sigma_2, psf,
                                                                           xs, ys,
                                                                           subsampling_factor,
                                                                           a_est)
    kwargs_init['kwargs_background']['mean'] = background_values

    # fix the background. (except the mean component)
    kwargs_fixed['kwargs_background']['h'] = kwargs_init['kwargs_background']['h']
    # single point source, c_x, c_y will be degenerate with dx, dy.
    # rotation pointless as well.
    kwargs_fixed['kwargs_analytic']['c_x'] = kwargs_init['kwargs_analytic']['c_x']
    kwargs_fixed['kwargs_analytic']['c_y'] = kwargs_init['kwargs_analytic']['c_y']
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
                'max_iterations': 500, 'min_iterations': None,
                'init_learning_rate': 2e-3, 'schedule_learning_rate': True,
                'restart_from_init': True, 'stop_at_loss_increase': False,
                'progress_bar': True, 'return_param_history': True
    }

    optim.minimize(**optimiser_optax_option)
    kwargs_final = parameters.best_fit_values(as_kwargs=True)
    fluxes = scale * np.array(kwargs_final['kwargs_analytic']['a'])
    chi2 = -2 * loss._log_likelihood_chi2(kwargs_final) / (model.image_size**2)

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
        'loss_curve': optim.loss_history
    }
    return result


def get_frames_for_star(gaia_id):
    """
    Retrieves frames that include the given star

    :param gaia_id: The Gaia ID of the star.
    :return: A list of frames that need flux measurements for the given star.
    """
    query = """
    SELECT f.*
    FROM frames f
    JOIN stars_in_frames sif ON f.id = sif.frame_id
    WHERE sif.gaia_id = ?
    """
    params = (gaia_id,)
    return execute_sqlite_query(query, params, is_select=True, use_pandas=True)


def get_frames_for_star_without_flux(gaia_id):
    """
    Retrieves frames that include the given star but do not yet have a flux measurement.

    :param gaia_id: The Gaia ID of the star.
    :return: A list of frames that need flux measurements for the given star.
    """
    query = """
    SELECT f.*
    FROM frames f
    JOIN stars_in_frames sif ON f.id = sif.frame_id
    LEFT JOIN star_flux_in_frame sff ON f.id = sff.frame_id AND sif.gaia_id = sff.star_gaia_id
    WHERE sif.gaia_id = ? AND sff.frame_id IS NULL"""
    params = (gaia_id,)
    return execute_sqlite_query(query, params, is_select=True, use_pandas=True)


def update_star_fluxes(flux_data):
    db_path = get_user_config()['database_path']
    with sqlite3.connect(db_path, timeout=15.0) as conn:
        cursor = conn.cursor()

        # insert query with ON CONFLICT clause for bulk update
        # upon ON CONFLICT (integrity error due to trying to insert a flux for a given star and frame again),
        # as we would if we "redo", then we do indeed erase the previous values and add the new ones.
        insert_query = """
        INSERT INTO star_flux_in_frame (frame_id, star_gaia_id, flux, flux_uncertainty)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(frame_id, star_gaia_id) DO UPDATE SET
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
    # start by finding the stars we need to do photometry of:
    user_config = get_user_config()
    stars = select_stars(user_config['stars_to_use_norm'])
    # if re-do, select all the frames
    # if not re-do ...select only the new frames.
    frame_selector = get_frames_for_star if user_config['redo_star_photometry'] else get_frames_for_star_without_flux
    for i, star in stars.iterrows():
        frames = frame_selector(star['gaia_id'])
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
                stars_psf = select_stars_for_a_frame(frame['id'], user_config['stars_to_use_psf'])
                psf_ref = 'psf_' + ''.join(sorted(stars_psf['name']))
                mask.append(h5f[f"{frame['image_relpath']}/cosmicsmask/{star['name']}"][...])
                psf.append(h5f[f"{frame['image_relpath']}/{psf_ref}/narrow_psf"][...])
            data, noisemap, mask, psf = np.array(data), np.array(noisemap), np.array(mask), np.array(psf)
            # cosmics: masks are 'true' where cosmic, and we typically want it to "true" for good pixels
            mask = ~(np.array(mask).astype(bool))  # so we invert it.
            # oh, we invert it again to boost the noisemap where mask is False,
            # but better have the masks loaded the right way for the future.
            noisemap[np.where(~mask)[0]] *= 1000.
            # ok now that everything is ready let's get out of the context manager, also to close the file.

        # ready to "deconvolve" using starred!
        result = do_one_deconvolution(data=data, noisemap=noisemap, psf=psf,
                                      subsampling_factor=user_config['subsampling_factor'])

        # now we can insert our results in our dedicated database table.
        # flux_data should be a list of tuples, each containing (frame_id, star_gaia_id, flux, flux_uncertainty)
        flux_data = []
        # the order of the frames is the same as before, and the same as that of our arrays and fluxes.
        for j, frame in frames.iterrows():
            gaia_id = star['gaia_id']
            frame_id = frame['id']
            flux = float(result['fluxes'][j])
            flux_uncertainty = float(result['fluxes_uncertainties'][j])
            flux_data.append((frame_id, gaia_id, flux, flux_uncertainty))

        # big insert while updating if value already in DB...
        update_star_fluxes(flux_data)
        # done!
