import numpy as np
from datetime import datetime
from pathlib import Path
from astropy.io import fits
import h5py
import json
from copy import deepcopy
import pandas as pd
from scipy.ndimage import shift, rotate
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from photutils.aperture import CircularAperture, aperture_photometry
import logging

from starred.deconvolution.deconvolution import setup_model
from starred.deconvolution.loss import Loss
from starred.optim.optimization import Optimizer
from starred.deconvolution.parameters import ParametersDeconv
from starred.utils.noise_utils import propagate_noise

from ..structure.user_config import get_user_config
from ..structure.database import get_pandas, execute_sqlite_query
from ..utilities.footprint import get_combined_footprint_hash
from ..utilities.starred_utilities import get_flux_uncertainties
from ..plotting.star_photometry_plotting import plot_joint_deconv_diagnostic


def align_data_interpolation(array, starred_kwargs):
    """
    This is a utility function that takes in the data we use for deconvolution, and
    - de-rotates it
    - 'de-translates' it -- both according to our fitted values of translation and rotation.
    It uses interoplation to do so, hence to be used as a diagnostic tool only.

    Parameters:
        array: array containing our data, typically shape (N_epochs, nx, ny)
        starred_kwargs: the STARRED optimized keywords, that contain the rotations and translations.


    Returns:
        array_shift: the same array, but derotated and de-translated
    """
    dx = starred_kwargs['kwargs_analytic']['dx']
    dy = starred_kwargs['kwargs_analytic']['dy']
    alpha = starred_kwargs['kwargs_analytic']['alpha']

    array_shift = np.array(
        [rotate(shift(a, (-ddy, -ddx)), alph, reshape=False) for a, ddx, ddy, alph in zip(array, dx, dy, alpha)]
    )

    return array_shift


def do_deconvolution_of_roi():
    """
    Optionally called by the workflow manager.
    This is probably a bit too rigid given how complicated the joint modelling of 1000+ epochs of a blended ROI
    can be. But it can be used as a template for your own.
    This simply jointly models your ROI cutouts with the point sources given in the config file.
    Should work for most cases still!
    Returns:
        Nothing
    """
    logger = logging.getLogger('lightcurver.roi_modelling')
    user_config = get_user_config()
    if not user_config['do_ROI_model']:
        # we do nothing, this is an optional step.
        return

    frames_ini = get_pandas(columns=['id'],
                            conditions=['plate_solved = 1', 'eliminated = 0', 'roi_in_footprint = 1'])
    combined_footprint_hash = get_combined_footprint_hash(user_config, frames_ini['id'].to_list())
    deconv_file = user_config['prepared_roi_cutouts_path']

    if deconv_file is None:
        roi = user_config['roi_name']
        deconv_file = user_config['workdir'] / 'prepared_roi_cutouts' / f"cutouts_{combined_footprint_hash}_{roi}.h5"

    # load the data
    with h5py.File(deconv_file, 'r') as f:
        data = np.array(f['data'])
        noisemap = np.array(f['noisemap'])
        s = np.array(f['psf'])
        scale = np.nanmax(data)
        data /= scale
        noisemap /= scale
        seeings = np.array(f['seeing'])
        mjds = np.array(f['mjd'])
        zeropoint = np.array(f['global_zeropoint'])
        norm_errs = np.array(f['relative_normalization_error'])
        frame_ids = np.array(f['frame_id'])
        subsampling_factor = np.array(f['subsampling_factor'])
        pixel_scales = np.array(f['pixel_scale'])
        angles_to_north = np.array(f['angle_to_north'])
        wcs = np.array(f['wcs'])

    message = "The PSF models seem to have different subsampling factors! Incompatible with a STARRED deconvolution."
    unique_subsampling = (np.unique(subsampling_factor).size == 1)
    if not unique_subsampling:
        logger.error(message + ' Stopping the pipeline.')
    assert np.unique(subsampling_factor).size == 1, message
    subsampling_factor = int(subsampling_factor[0])
    im_size = data.shape[1]
    im_size_up = subsampling_factor * im_size
    epochs = data.shape[0]

    ps_coords = user_config['point_sources']
    ordered_ps = sorted(ps_coords.keys())
    logger.info(f'Jointly deconvolving {epochs} cutouts from your ROI, including {len(ordered_ps)} point sources.')

    # we use the first WCS as reference.
    # so the starred rotation angles will be w.r.t. to this one as well
    angles_to_north -= angles_to_north[0]
    wcs_ref = WCS(wcs[0].decode('utf-8'))
    ps_pixels = {lab: np.array(wcs_ref.world_to_pixel(SkyCoord(*val, unit=u.degree))).flatten() for lab, val in
                 ps_coords.items()}

    # so, initial guess for the positions of the point sources in pixels:
    xs = np.array([ps_pixels[ps][0] for ps in ordered_ps])
    ys = np.array([ps_pixels[ps][1] for ps in ordered_ps])

    # now, initial guess for the fluxes
    pixel_scale = np.nanmedian(pixel_scales)
    stack = np.nanmedian(data, axis=0)
    radii = 0.66 * np.average(seeings) / pixel_scale  # a bit bigger than FWHM.
    positions = [(x, y) for x, y in zip(xs, ys)]
    apertures = CircularAperture(positions, r=radii)
    photometry = aperture_photometry(stack, apertures)
    aperture_fluxes = list(photometry['aperture_sum'])

    # ok, ready to define the STARRED model
    offset = (im_size - 1) / 2.  # removing half image size, as starred has (0,0) = center of image
    initial_c_x = xs - offset
    initial_c_y = ys - offset
    initial_a = list(aperture_fluxes)
    initial_a = len(data) * initial_a
    model, kwargs_init, kwargs_up, kwargs_down, kwargs_fixed = setup_model(data,
                                                                           noisemap ** 2,
                                                                           s,
                                                                           initial_c_x,
                                                                           initial_c_y,
                                                                           subsampling_factor,
                                                                           initial_a)
    # prepare for the random rotations of the pointing ...
    kwargs_init['kwargs_analytic']['alpha'] = angles_to_north
    kwargs_fixed['kwargs_analytic']['alpha'] = angles_to_north
    # if we provide a background:
    if user_config['starting_background'] is not None:
        bck_path = Path(user_config['starting_background'])
        if not bck_path.is_absolute():
            bck_path = user_config['work_dir'] / bck_path
        if bck_path.name.endswith('fits'):
            bck = fits.getdata(bck_path)
        else:
            bck = np.load(bck_path)
        h = bck.flatten() / scale
        kwargs_init['kwargs_background']['h'] = h
        kwargs_fixed['kwargs_background']['h'] = h

    # ok, first step: translations and fluxes
    kwargs_fixed = deepcopy(kwargs_init)
    del kwargs_fixed['kwargs_analytic']['dx']
    del kwargs_fixed['kwargs_analytic']['dy']
    del kwargs_fixed['kwargs_analytic']['a']
    parameters = ParametersDeconv(kwargs_init=kwargs_init,
                                  kwargs_fixed=kwargs_fixed,
                                  kwargs_up=kwargs_up,
                                  kwargs_down=kwargs_down)

    loss = Loss(data, model, parameters, noisemap**2)

    optim = Optimizer(loss, parameters, method='l-bfgs-b')

    best_fit, logL_best_fit, extra_fields, runtime = optim.minimize(maxiter=user_config['roi_deconv_translations_iters'])
    kwargs_partial1 = deepcopy(parameters.best_fit_values(as_kwargs=True))
    logger.info('Finished first optimization, only varying the fluxes of the point sources and the translations.')

    # next, include the background.
    kwargs_fixed = deepcopy(kwargs_partial1)
    if user_config['further_optimize_background']:
        del kwargs_fixed['kwargs_background']['h']
    del kwargs_fixed['kwargs_background']['mean']
    del kwargs_fixed['kwargs_analytic']['a']
    del kwargs_fixed['kwargs_analytic']['c_x']
    del kwargs_fixed['kwargs_analytic']['c_y']
    del kwargs_fixed['kwargs_analytic']['dx']
    del kwargs_fixed['kwargs_analytic']['dy']
    W = propagate_noise(model, noisemap, kwargs_init, wavelet_type_list=['starlet'],
                        method='SLIT', num_samples=500, seed=1, likelihood_type='chi2',
                        verbose=False, upsampling_factor=subsampling_factor)[0]

    parameters = ParametersDeconv(kwargs_init=kwargs_partial1,
                                  kwargs_fixed=kwargs_fixed,
                                  kwargs_up=kwargs_up,
                                  kwargs_down=kwargs_down)

    loss = Loss(data, model, parameters, noisemap**2,
                regularization_terms='l1_starlet',
                regularization_strength_scales=1,
                regularization_strength_hf=1.,
                regularization_strength_positivity=100.,
                W=W)

    optim = Optimizer(loss, parameters, method='adabelief')
    optimiser_optax_option = {
        'max_iterations': user_config['roi_deconv_all_iters'],
        'init_learning_rate': 1e-4, 'schedule_learning_rate': False,
        'restart_from_init': False, 'stop_at_loss_increase': False,
        'progress_bar': True, 'return_param_history': True
    }

    best_fit, logL_best_fit, extra_fields, runtime = optim.minimize(**optimiser_optax_option)
    kwargs_final = deepcopy(parameters.best_fit_values(as_kwargs=True))

    out_dir = deconv_file.parent
    # the easy stuff, let's output the astrometry first:
    x_pixels = np.array(kwargs_final['kwargs_analytic']['c_x'] + kwargs_final['kwargs_analytic']['dx'][0])
    y_pixels = np.array(kwargs_final['kwargs_analytic']['c_y'] + kwargs_final['kwargs_analytic']['dy'][0])
    ps_coords_post = wcs_ref.pixel_to_world_values(np.array((x_pixels, y_pixels)).T)
    ps_coords_post = {ps: list(coord) for ps, coord in zip(ordered_ps, ps_coords_post)}
    with open(out_dir / f'{combined_footprint_hash}_astrometry.json', 'w') as ff:
        json.dump(ps_coords_post, ff)

    # the fluxes ...
    fluxes = kwargs_final['kwargs_analytic']['a']
    # get the uncertainties on the fluxes
    flux_photon_uncertainties = get_flux_uncertainties(kwargs=kwargs_final, kwargs_down=kwargs_down,
                                                       kwargs_up=kwargs_up,
                                                       data=data, noisemap=noisemap, model=model)
    curves = {}
    d_curves = {}
    # let's separate the fluxes by point source
    for i, ps in enumerate(ordered_ps):
        curve = fluxes[i::len(ordered_ps)] * scale
        curve_photon_uncertainties = flux_photon_uncertainties[i::len(ordered_ps)] * scale
        # these need be compounded with the normalisation errors.
        curve_norm_err = norm_errs * curve
        curves[ps] = curve
        d_curves[ps] = (curve_photon_uncertainties**2 + curve_norm_err**2)**0.5

    # ok, onto the chi2
    modelled_pixels = model.model(kwargs_final)
    residuals = data - modelled_pixels
    chi2_per_frame = np.nansum((residuals**2 / noisemap**2), axis=(1, 2)) / model.image_size ** 2

    # let's fold in some info, and save!
    df = []
    num_epochs = len(frame_ids)
    for epoch in range(num_epochs):
        row = {
            'frame_id': frame_ids[epoch],
            'mjd': mjds[epoch],
            'zeropoint': zeropoint,
            'reduced_chi2': chi2_per_frame[epoch],
        }
        for ps in ordered_ps:
            row[f'{ps}_flux'] = curves[ps][epoch]
            row[f'{ps}_d_flux'] = d_curves[ps][epoch]
        df.append(row)
    df = pd.DataFrame(df).set_index('frame_id')
    df.to_csv(out_dir / f'{combined_footprint_hash}_fluxes.csv')
    # ok, now some diagnostics.
    # first, subtract the point sources from the data, see what the stack looks like.
    # (helps to spot PSF model problems, or fit really gone wrong)
    kwargs_only_ps = deepcopy(kwargs_final)
    kwargs_only_ps['kwargs_background']['h'] *= 0.0

    data_no_ps = data - model.model(kwargs_only_ps)
    data_no_ps = align_data_interpolation(data_no_ps, kwargs_only_ps)
    stack_no_ps = np.nanmean(data_no_ps, axis=0)
    fits.writeto(out_dir / f'{combined_footprint_hash}_stack_without_point_sources.fits', scale * stack_no_ps,
                 overwrite=True)

    # and of course, output the deconvolution
    deconv, h = model.getDeconvolved(kwargs_final, 0)
    fits.writeto(out_dir / f'{combined_footprint_hash}_deconvolution.fits', scale * np.array(deconv),
                 overwrite=True)
    fits.writeto(out_dir / f'{combined_footprint_hash}_background.fits', scale * np.array(h),
                 overwrite=True)

    # now a diagnostic plot
    plot_deconv_dir = user_config['plots_dir'] / 'deconvolutions' / str(combined_footprint_hash)
    plot_deconv_dir.mkdir(exist_ok=True, parents=True)
    loss_history = optim.loss_history
    time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    plot_file = plot_deconv_dir / f"{time_now}_joint_deconv_roi_{roi}.jpg"
    plot_joint_deconv_diagnostic(datas=data, noisemaps=noisemap,
                                 residuals=residuals,
                                 chi2_per_frame=chi2_per_frame, loss_curve=loss_history,
                                 save_path=plot_file)
    logger.info(f'Finished modelling the ROI. Diagnostic plot at {plot_file}. '
                f"The global reduced chi2 was {np.mean(chi2_per_frame):.02f}. ")

