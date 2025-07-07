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
from astropy.wcs.utils import pixel_to_skycoord
from astropy import units as u
from astropy.nddata import CCDData
from photutils.aperture import CircularAperture, aperture_photometry
from ccdproc import Combiner
import logging

from starred.deconvolution.deconvolution import setup_model
from starred.deconvolution.loss import Loss, Prior
from starred.optim.optimization import Optimizer
from starred.deconvolution.parameters import ParametersDeconv
from starred.utils.noise_utils import propagate_noise

from ..structure.user_config import get_user_config
from ..structure.database import get_pandas
from ..utilities.footprint import get_combined_footprint_hash
from ..utilities.starred_utilities import get_flux_uncertainties
from ..plotting.joint_modelling_plotting import plot_joint_modelling_diagnostic
from ..plotting.html_visualisation import generate_lightcurve_html
from ..utilities.lightcurves_postprocessing import convert_flux_to_magnitude, group_observations


def align_data_interpolation(array, starred_kwargs):
    """
    This is a utility function that takes in the data we use for modelling, and
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


def stack_data_ccdproc(data, noisemap, n_sigma=3):
    """
    Using ccdproc to do average stacking with rejection.

    Args:
        data: 3D array of shape (N_epoch, nx, ny)
        noisemap: 3D array of shape (N_epoch, nx, ny)
        n_sigma: float, how many sigmas away from the median we clip columns of clip pixels at? Default 3.

    Returns:
        2D array of shape (nx, ny)
    """

    # here the units are not ADUs, but we provide our own weights, so it should not matter.
    ccds = [CCDData(epoch, unit=u.adu) for epoch in data]
    combiner = Combiner(ccds)
    # set the weights -- conservative here, not using 1/noisemap**2 even though this would maximize S/N for Gaussians
    combiner.weights = 1.0 / noisemap

    # apply sigma clipping, we don't do it iteratively, one clip should be enough for most artifacts.
    combiner.sigma_clipping(low_thresh=n_sigma, high_thresh=n_sigma, func=np.ma.median)
    average = combiner.average_combine()

    return average.data


def stack_data_diagnostic(data, noisemap, starred_kwargs, starred_model):
    """
    Stacks each data frame after interpolating them
    (rotation, translation) so they all match according to the fitted rotation and translations in starred_kwargs.
    Does it once for the data entire dataset, once with only the point sources, and once without the point sources.
    Parameters:
        data: numpy array, shape (N, nx, ny)
        noisemap: numpy array, shape (N, nx, ny)
        starred_kwargs: dictionary of keywords parameters of the starred model.
        starred_model: the starred Deconv class used to model the pixels.
    Returns:
        dictionary of 3 numpy arrays, each shape (nx, ny), with keys 'stack', 'stack_no_ps' and 'stack_no_background'
    """
    kwargs_only_ps = deepcopy(starred_kwargs)
    kwargs_only_ps['kwargs_background']['h'] *= 0.0

    kwargs_no_ps = deepcopy(starred_kwargs)
    kwargs_no_ps['kwargs_analytic']['a'] *= 0.0

    # data without point sources
    data_no_ps = data - starred_model.model(kwargs_only_ps)
    data_no_ps = align_data_interpolation(data_no_ps, kwargs_only_ps)

    # data without extended component
    data_no_background = data - starred_model.model(kwargs_no_ps)
    data_no_background = align_data_interpolation(data_no_background, kwargs_no_ps)

    # just data
    data_original = align_data_interpolation(data, starred_kwargs)

    # stacking.
    stack_no_ps = stack_data_ccdproc(data_no_ps, noisemap)
    stack_no_background = stack_data_ccdproc(data_no_background, noisemap)
    stack_original = stack_data_ccdproc(data_original, noisemap)

    return {
        'stack': stack_original,
        'stack_no_ps': stack_no_ps,
        'stack_no_background': stack_no_background
    }


def do_modelling_of_roi():
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
    roi_cutouts_file = user_config['prepared_roi_cutouts_path']

    roi = user_config['roi_name']
    if roi_cutouts_file is None:
        roi_cutouts_file = user_config['workdir'] / 'prepared_roi_cutouts' / f"cutouts_{combined_footprint_hash}_{roi}.h5"

    # load the data
    with h5py.File(roi_cutouts_file, 'r') as f:
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
        sky_level_electron_per_second = np.array(f['sky_level_electron_per_second'])

    message = "The PSF models seem to have different subsampling factors! Incompatible with STARRED modelling."
    unique_subsampling = (np.unique(subsampling_factor).size == 1)
    if not unique_subsampling:
        logger.error(message + ' Stopping the pipeline.')
    assert np.unique(subsampling_factor).size == 1, message
    subsampling_factor = int(subsampling_factor[0])
    im_size_x, im_size_y = data.shape[1:]
    epochs = data.shape[0]

    ps_coords = user_config['point_sources']
    ordered_ps = sorted(ps_coords.keys())
    logger.info(f'Jointly modelling {epochs} cutouts from your ROI, including {len(ordered_ps)} point sources.')

    # we use the first WCS as reference.
    # so the starred rotation angles will be w.r.t. to this one as well
    ref_index = 0
    angles_to_north -= angles_to_north[ref_index]
    wcs_ref = WCS(wcs[ref_index].decode('utf-8'))
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
    offset_x = (im_size_x - 1) / 2.  # removing half image size, as starred has (0,0) = center of image
    offset_y = (im_size_y - 1) / 2.
    initial_c_x = xs - offset_x
    initial_c_y = ys - offset_y
    initial_a = list(aperture_fluxes)
    initial_a = len(data) * initial_a
    model, kwargs_init, kwargs_up, kwargs_down, kwargs_fixed = setup_model(data,
                                                                           noisemap ** 2,
                                                                           s,
                                                                           initial_c_x,
                                                                           initial_c_y,
                                                                           subsampling_factor,
                                                                           initial_a)
    # prepare for the random rotations of the pointings ...
    kwargs_init['kwargs_analytic']['alpha'] = angles_to_north
    kwargs_fixed['kwargs_analytic']['alpha'] = angles_to_north
    
    # astrometric bit! 
    fix_astrometry = user_config['fix_point_source_astrometry']
    astrometric_prior = None  # default
    if type(fix_astrometry) is bool:
        # then we either fully fix the astrometry, or not at all
        if fix_astrometry:
            logger.info("Fully fixing the astrometry to the config values.")
            kwargs_fixed['kwargs_analytic']['c_x'] = initial_c_x
            kwargs_fixed['kwargs_analytic']['c_y'] = initial_c_y
        else:
            logger.info("Astrometry will be fully free during optimization.")

    elif type(fix_astrometry) is float:
        # we make a prior!
        logger.info("Setting a Gaussian astrometric prior on the astrometry: "
                    f"sigma = {fix_astrometry:.02f} data pixels.")
        astrometric_prior = Prior(prior_analytic=[
            ['c_x', initial_c_x, np.array(len(initial_c_x) * [fix_astrometry])],
            ['c_y', initial_c_y, np.array(len(initial_c_y) * [fix_astrometry])]
        ]
        )

    # if we provide a background:
    if user_config['starting_background'] is not None:
        bck_path = Path(user_config['starting_background'])
        if not bck_path.is_absolute():
            bck_path = user_config['workdir'] / bck_path
        if bck_path.name.endswith('fits'):
            bck = fits.getdata(bck_path)
        else:
            bck = np.load(bck_path)
        high_res_model_background_only = bck.flatten() / scale
        kwargs_init['kwargs_background']['h'] = high_res_model_background_only
        kwargs_fixed['kwargs_background']['h'] = high_res_model_background_only

    # ok, first step: translations and fluxes
    kwargs_fixed = deepcopy(kwargs_init)
    del kwargs_fixed['kwargs_analytic']['dx']
    del kwargs_fixed['kwargs_analytic']['dy']
    del kwargs_fixed['kwargs_analytic']['a']
    parameters = ParametersDeconv(kwargs_init=kwargs_init,
                                  kwargs_fixed=kwargs_fixed,
                                  kwargs_up=kwargs_up,
                                  kwargs_down=kwargs_down)

    roi_modeling_params = user_config.get('roi_model_regularization', {})
    if not roi_modeling_params:
        logger.warning('No background regularization params in config: using defaults.')

    regularization_scatter_fluxes_pre_optim = roi_modeling_params.get('regularization_scatter_fluxes_pre_optim', 10.0)

    loss = Loss(data, model, parameters, noisemap**2, prior=astrometric_prior,
                regularization_strength_flux_uniformity=regularization_scatter_fluxes_pre_optim)

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

    # same as before for the astrometry, re-fix it if desired!
    if type(fix_astrometry) is bool and fix_astrometry:
        kwargs_fixed['kwargs_analytic']['c_x'] = initial_c_x
        kwargs_fixed['kwargs_analytic']['c_y'] = initial_c_y
    starlet_layer_propagated_weights = propagate_noise(model, noisemap, kwargs_init, wavelet_type_list=['starlet'],
                                                       method='SLIT', num_samples=500, seed=1, likelihood_type='chi2',
                                                       verbose=False, upsampling_factor=subsampling_factor)[0]

    parameters = ParametersDeconv(kwargs_init=kwargs_partial1,
                                  kwargs_fixed=kwargs_fixed,
                                  kwargs_up=kwargs_up,
                                  kwargs_down=kwargs_down)

    regularization_strength_scales = roi_modeling_params.get('regularization_strength_scales', 1.0)
    regularization_strength_hf = roi_modeling_params.get('regularization_strength_hf', 1.0)
    regularization_strength_positivity = roi_modeling_params.get('regularization_strength_positivity', 100.0)
    regularization_strength_pts_source = roi_modeling_params.get('regularization_strength_pts_source', 0.01)
    regularization_scatter_fluxes_main_optim = roi_modeling_params.get('regularization_scatter_fluxes_main_optim', 10.0)
    loss = Loss(data, model, parameters, noisemap**2,
                regularization_terms='l1_starlet',
                regularization_strength_scales=regularization_strength_scales,
                regularization_strength_hf=regularization_strength_hf,
                regularization_strength_positivity=regularization_strength_positivity,
                regularization_strength_pts_source=regularization_strength_pts_source,
                regularization_strength_flux_uniformity=regularization_scatter_fluxes_main_optim,
                W=starlet_layer_propagated_weights,
                prior=astrometric_prior)
    if regularization_scatter_fluxes_main_optim > 0.0:
        logger.warning("From config: regularisation on flux scatter in final optimisation -- "
                       f"regularization_scatter_fluxes_main_optim = {regularization_scatter_fluxes_main_optim:.01f}")

    optim = Optimizer(loss, parameters, method='adabelief')
    optimiser_optax_option = {
        'max_iterations': user_config['roi_deconv_all_iters'],
        'init_learning_rate': 1e-4, 'schedule_learning_rate': False,
        'restart_from_init': False, 'stop_at_loss_increase': False,
        'progress_bar': True, 'return_param_history': True
    }

    best_fit, logL_best_fit, extra_fields, runtime = optim.minimize(**optimiser_optax_option)
    kwargs_final = deepcopy(parameters.best_fit_values(as_kwargs=True))

    out_dir = roi_cutouts_file.parent
    # the easy stuff, let's output the astrometry first:
    x_pixels = np.array(kwargs_final['kwargs_analytic']['c_x'] + kwargs_final['kwargs_analytic']['dx'][0]) + offset_x
    y_pixels = np.array(kwargs_final['kwargs_analytic']['c_y'] + kwargs_final['kwargs_analytic']['dy'][0]) + offset_y
    ps_coords_post = pixel_to_skycoord(x_pixels, y_pixels, wcs_ref)
    ps_coords_post = {ps: [coord.ra.deg, coord.dec.deg] for ps, coord in zip(ordered_ps, ps_coords_post)}
    with open(out_dir / f"{combined_footprint_hash}_{roi}_astrometry.json", 'w') as ff:
        json.dump(ps_coords_post, ff)

    # ok, now we extract the light curves from the fitted starred kwargs.
    mags_etc_per_epoch, mags_etc_per_night, residuals = get_fluxes_dataframe_from_model(
        starred_model=model,
        starred_kwargs=kwargs_final,
        starred_kwargs_down=kwargs_down,
        starred_kwargs_up=kwargs_up,
        data=data,
        noisemap=noisemap,
        point_sources_names=ordered_ps,
        model_scale=scale,
        normalization_errors=norm_errs,
        frame_ids=frame_ids,
        mjds=mjds,
        seeings=seeings,
        zeropoint=zeropoint,
        sky_level_electron_per_second=sky_level_electron_per_second)

    mags_etc_per_epoch.to_csv(out_dir / f"{combined_footprint_hash}_{roi}_photometry_per_epoch.csv")
    mags_etc_per_night.to_csv(out_dir / f"{combined_footprint_hash}_{roi}_photometry_per_night.csv")
    # make an html plot as well
    generate_lightcurve_html(mags_etc_per_night, out_dir / f"{combined_footprint_hash}_{roi}_photometry_per_night.html")
    
    # ok, now some diagnostics.
    # first, subtract the point sources from the data, see what the stack looks like.
    # (helps to spot PSF model problems, or fit really gone wrong)
    stacks = stack_data_diagnostic(
        data=data,
        noisemap=noisemap,
        starred_kwargs=kwargs_final,
        starred_model=model
    )

    for stack_type, stack in stacks.items():
        fits.writeto(
            out_dir / f"{combined_footprint_hash}_{roi}_{stack_type}.fits",
            scale * stack,
            overwrite=True,
            header=wcs_ref.to_header()
        )

    # and of course, output the fitted high-res model
    high_res_model, high_res_model_background_only = model.getDeconvolved(kwargs_final, 0)
    # make a higher res wcs
    wcs_highres = deepcopy(wcs_ref)
    # dividing the pixel scale by the subsampling factor to match the higher resolution
    wcs_highres.wcs.cdelt /= subsampling_factor
    wcs_highres.wcs.crpix *= subsampling_factor

    header_highres = wcs_highres.to_header()
    header_highres['ZPT'] = float(zeropoint) if zeropoint.ndim == 0 else float(zeropoint[0])
    fits.writeto(out_dir / f"{combined_footprint_hash}_{roi}_high_res_model.fits",
                 scale * np.array(high_res_model),
                 overwrite=True, header=header_highres)
    fits.writeto(out_dir / f"{combined_footprint_hash}_{roi}_background.fits",
                 scale * np.array(high_res_model_background_only),
                 overwrite=True, header=header_highres)

    # now a diagnostic plot
    plot_modelling_dir = user_config['plots_dir'] / 'pixel_modelling' / str(combined_footprint_hash)
    plot_modelling_dir.mkdir(exist_ok=True, parents=True)
    loss_history = optim.loss_history
    time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    plot_file = plot_modelling_dir / f"{time_now}_joint_modelling_roi_{roi}.jpg"
    red_chi2_per_frame = np.array(mags_etc_per_epoch['reduced_chi2'])
    plot_joint_modelling_diagnostic(datas=data, noisemaps=noisemap,
                                    residuals=residuals,
                                    chi2_per_frame=red_chi2_per_frame,
                                    loss_curve=loss_history,
                                    save_path=plot_file,
                                    starlet_background=np.array(high_res_model_background_only))
    logger.info(f'Finished modelling the ROI. Diagnostic plot at {plot_file}. '
                f"The global reduced chi2 was {np.mean(red_chi2_per_frame):.02f}. ")


def get_fluxes_dataframe_from_model(starred_model, starred_kwargs, starred_kwargs_down, starred_kwargs_up,
                                    data, noisemap, point_sources_names, model_scale, normalization_errors,
                                    frame_ids, mjds, seeings, zeropoint, sky_level_electron_per_second):
    """
    This has a lot of inputs, so it is not meant to be used outside the do_modelling_of_roi file.
    Instead, it helps comparimentalise what we do in the main function.

    Args:
        starred_model: the "deconvolution" starred model
        starred_kwargs: the fitted arguments, as a dictionary of keywords.
        starred_kwargs_down: has to be passed to starred, same as starred_kwargs but with lower bound values.
        starred_kwargs_up: has to be passed to starred, same as starred_kwargs but with higher bound values.
        data: 3D array, (N_epoch, nx, ny): data pixels, in units of model_scale below.
        noisemap:  3D array, (N_epoch, nx, ny): noise map pixels, in units of model_scale below.
        point_sources_names: for exporting, an iterable yielding strings, one per point source in the model
        model_scale: float, scale of the data above:
                            data and noisemaps were earlier normalised by this scale, so we undo that here
        normalization_errors: array of shape (N_epoch,), one per frame.
        frame_ids: array of shape (N_epoch,), one per frame.
        mjds:  array of shape (N_epoch,), one per frame.
        seeings:  array of shape (N_epoch,), one per frame.
        zeropoint: float, zeropoint of the cutouts.
        sky_level_electron_per_second:  array of shape (N_epoch,), one per frame.

    Returns: tuple:
     - data frame containing fluxes and magnitudes per epoch,
     - data frame, fluxes and others grouped per night,
     - residuals: 3D array of shape (N_epoch, nx, ny) containing residuals in units of the noise.

    """
    # the fluxes ...
    fluxes = np.array(starred_kwargs['kwargs_analytic']['a'])
    # get the uncertainties on the fluxes
    flux_photon_uncertainties = get_flux_uncertainties(kwargs=starred_kwargs, kwargs_down=starred_kwargs_down,
                                                       kwargs_up=starred_kwargs_up,
                                                       data=data, noisemap=noisemap, model=starred_model)
    # convert to numpy to avoid problems
    flux_photon_uncertainties = np.array(flux_photon_uncertainties)
    curves = {}
    d_curves = {}
    # let's separate the fluxes by point source
    for i, ps in enumerate(point_sources_names):
        curve = fluxes[i::len(point_sources_names)] * model_scale
        curve_photon_uncertainties = flux_photon_uncertainties[i::len(point_sources_names)] * model_scale
        # these need be compounded with the normalisation errors.
        curve_norm_err = normalization_errors * curve
        curves[ps] = curve
        d_curves[ps] = (curve_photon_uncertainties ** 2 + curve_norm_err ** 2) ** 0.5

    # ok, onto the chi2
    modelled_pixels = starred_model.model(starred_kwargs)
    residuals = data - modelled_pixels
    chi2_per_frame = np.nansum((residuals ** 2 / noisemap ** 2), axis=(1, 2)) / starred_model.image_size ** 2
    # to avoid problems:
    chi2_per_frame = np.array(chi2_per_frame)

    # let's fold in some info, and save!
    df = []
    num_epochs = len(frame_ids)
    for epoch in range(num_epochs):
        row = {
            'frame_id': frame_ids[epoch],
            'mjd': mjds[epoch],
            'zeropoint': zeropoint,
            'reduced_chi2': chi2_per_frame[epoch],
            'seeing': seeings[epoch],
            'sky_level_electron_per_second': sky_level_electron_per_second[epoch]
        }
        for ps in point_sources_names:
            row[f'{ps}_flux'] = curves[ps][epoch]
            row[f'{ps}_d_flux'] = d_curves[ps][epoch]
        df.append(row)

    df_per_epoch = pd.DataFrame(df).set_index('frame_id')
    df_per_night = group_observations(df_per_epoch)
    mags_per_epoch = convert_flux_to_magnitude(df_per_epoch)
    mags_per_night = convert_flux_to_magnitude(df_per_night)
    return mags_per_epoch, mags_per_night, residuals

