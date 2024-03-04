import numpy as np
import h5py
import io

from starred.psf.psf import PSF
from starred.psf.loss import Loss
from starred.optim.optimization import Optimizer
from starred.psf.parameters import ParametersPSF
from starred.utils.noise_utils import propagate_noise

settings = {}
dopsfplots = settings['dopsfplots']
psfname = settings['psfname']
psfstars = [char for char in psfname if char.islower()]

if dopsfplots:
    import matplotlib.pyplot as plt

    plt.switch_backend('agg')

# if h5 file storing psf has never been created,
psfdir = psfsfile.parent
psfdir.mkdir(parents=1, exist_ok=1)
if not psfsfile.exists():
    # just create it.
    with h5py.File(psfsfile, 'w') as f:
        pass

# get all image names: (we refered to the data with them in the hdf5 file):
with h5py.File(regionsfile, 'r') as f:
    imgnames = list(f.keys())

# add the psf key flag in the database:
if psfkeyflag not in db.getFieldNames(imgdb):
    db.addFields(imgdb, [f'{psfkeyflag}:bool'])

# set the flag to 0 ...
db.update(imgdb, ['gogogo'], ['True'], {psfkeyflag: False})
# and set the flag of the images for which we potentially build the psf to True:
for imgname in imgnames:
    db.update(imgdb, ['imgname'], [imgname], {psfkeyflag: True})


# a utility to load the data of one image:
def getData(imgname):
    data, noise, cosmicsmasks = [], [], []
    with h5py.File(regionsfile, 'r') as f:
        container = f[imgname]
        for s in psfstars:
            data.append(container['stamps'][s][()])
            noise.append(container['noise'][s][()])
            cosmicsmasks.append(container['cosmicsmasks'][s][()])

    cosmicsmasks = np.array(cosmicsmasks)
    data = np.array(data)
    noise = np.array(noise)
    # mask cosmics here.
    noise[cosmicsmasks] = 1e8
    return data, noise


# get an example:
_image, _ = getData(imgnames[0])

# so we can define a model here (instead of in the loop, avoids recompiling
# every time.)
model = PSF(image_size=_image[0].shape[0], number_of_sources=len(_image),
            upsampling_factor=subsampling_factor,
            convolution_method=convolution_method,
            include_moffat=include_moffat,
            elliptical_moffat=True)

smartguess = lambda im: model.smart_guess(im, fixed_background=True)


# main routine:
def buildPSF(image, noisemap):
    """


    Parameters
    ----------
    image : array, shape (imageno, nx, ny)
        array containing the data
    noisemap : array, shape (imageno, nx, ny)
        array containing the noisemaps.
    loss : starred.psf.loss, optional
        This
    optim : TYPE, optional
        DESCRIPTION. The default is None.
    lossfull : TYPE, optional
        DESCRIPTION. The default is None.
    optimfull : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    kwargs_final : TYPE
        DESCRIPTION.
    narrowpsf : TYPE
        DESCRIPTION.
    numpsf : TYPE
        DESCRIPTION.
    moffat : TYPE
        DESCRIPTION.
    optimtools : TYPE
        DESCRIPTION.

    """

    # normalize by max of data(numerical precision best with scale ~ 1)
    norm = image.max()
    image /= norm
    noisemap /= norm

    # Parameter initialization.

    kwargs_init, kwargs_fixed, kwargs_up, kwargs_down = smartguess(image)

    # smartguess doesn't know about cosmics, other stars ...
    # so we'll be a bit careful.
    medx0 = np.median(kwargs_init['kwargs_gaussian']['x0'])
    medy0 = np.median(kwargs_init['kwargs_gaussian']['y0'])
    kwargs_init['kwargs_gaussian']['x0'] = medx0 * np.ones_like(kwargs_init['kwargs_gaussian']['x0'])
    kwargs_init['kwargs_gaussian']['y0'] = medy0 * np.ones_like(kwargs_init['kwargs_gaussian']['y0'])

    parameters = ParametersPSF(kwargs_init,
                               kwargs_fixed,
                               kwargs_up=kwargs_up,
                               kwargs_down=kwargs_down)

    loss = Loss(image, model, parameters, noisemap ** 2, len(image),
                regularization_terms='l1_starlet',
                regularization_strength_scales=0,
                regularization_strength_hf=0)

    optim = Optimizer(loss,
                      parameters,
                      method=method_analytical)

    # fit the moffat:
    best_fit, logL_best_fit, extra_fields, runtime = optim.minimize(maxiter=n_iter_initial,
                                                                    restart_from_init=True)

    kwargs_partial = parameters.args2kwargs(best_fit)

    # now moving on to the background.
    # Release backgound, fix the moffat
    kwargs_fixed = {
        'kwargs_moffat': {'fwhm_x': kwargs_partial['kwargs_moffat']['fwhm_x'],
                          'fwhm_y': kwargs_partial['kwargs_moffat']['fwhm_y'],
                          'phi': kwargs_partial['kwargs_moffat']['phi'],
                          'beta': kwargs_partial['kwargs_moffat']['beta'],
                          'C': kwargs_partial['kwargs_moffat']['C']},
        'kwargs_gaussian': {
            # 'a': kwargs_partial['kwargs_gaussian']['a'],
            # 'x0': kwargs_partial['kwargs_gaussian']['x0'],
            # 'y0': kwargs_partial['kwargs_gaussian']['y0'],
        },
        'kwargs_background': {},
    }

    parametersfull = ParametersPSF(kwargs_partial,
                                   kwargs_fixed,
                                   kwargs_up,
                                   kwargs_down)

    topass = np.nanmedian(noisemap, axis=0)
    topass = np.expand_dims(topass, (0,))
    W = propagate_noise(model, topass, kwargs_init,
                        wavelet_type_list=['starlet'],
                        method='MC', num_samples=100,
                        seed=1, likelihood_type='chi2',
                        verbose=False,
                        upsampling_factor=subsampling_factor)[0]

    lossfull = Loss(image, model, parametersfull,
                    noisemap ** 2, len(image),
                    regularization_terms='l1_starlet',
                    regularization_strength_scales=1.,
                    regularization_strength_hf=2.,
                    regularization_strength_positivity=lambda_positivity,
                    W=W,
                    regularize_full_psf=regularize_full_psf)

    optimfull = Optimizer(lossfull, parametersfull, method='adabelief')

    optimiser_optax_option = {
        'max_iterations': n_iter, 'min_iterations': None,
        'init_learning_rate': 1e-4, 'schedule_learning_rate': True,
        # important: restart_from_init True
        'restart_from_init': True, 'stop_at_loss_increase': False,
        'progress_bar': True, 'return_param_history': True
    }

    best_fit, logL_best_fit, extra_fields, runtime = optimfull.minimize(**optimiser_optax_option)

    kwargs_final = parametersfull.args2kwargs(best_fit)

    ###########################################################################
    # book keeping
    narrowpsf = model.get_narrow_psf(**kwargs_final, norm=True)
    numpsf = model.get_background(kwargs_final['kwargs_background'])
    moffat = model.get_moffat(kwargs_final['kwargs_moffat'], norm=True)
    fullmodel = np.array([model.model(i, **kwargs_final) for i in range(image.shape[0])])
    residuals = image - fullmodel

    ###########################################################################
    return kwargs_final, narrowpsf, numpsf, moffat, fullmodel, residuals, extra_fields


from time import time

times = []

for i, image in enumerate(images):

    imgname = image['imgname']
    t0 = time()

    # load stamps and noise maps for this image
    data, noisemap = getData(imgname)
    # open the file in which we'll store the result
    with h5py.File(psfsfile, 'r+') as f:
        # check if we need to build again
        if not redo and (imgname + '_residuals') in f.keys():
            print(imgname, 'already done and redo is False.')
            continue

        # call the routine defined above
        kwargs_final, narrowpsf, numpsf, moffat, fullmodel, residuals, extra_fields = buildPSF(data,
                                                                                               noisemap)
        # time for storage. If key already exists, gotta delete it since
        # h5py does not like overwriting
        for tostore, name in zip([data, noisemap, narrowpsf, numpsf, moffat, fullmodel, residuals],
                                 ['data', 'noisemap', 'narrow', 'num', 'moffat', 'model', 'residuals']):
            key = f"{imgname}_{name}"
            if key in f.keys():
                del f[key]
            f[key] = tostore

    # write plots
    if dopsfplots:

        try:
            # try because the analytical methods don't have a 'loss_history'
            # field.
            fig = plt.figure(figsize=(2.56, 2.56))
            plt.plot(extra_fields['loss_history'])
            plt.title('loss history')
            plt.tight_layout()
            with io.BytesIO() as buff:
                # write the plot to a buffer, read it with numpy
                fig.savefig(buff, format='raw')
                buff.seek(0)
                plotimg = np.frombuffer(buff.getvalue(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                # white <-> black:
                lossim = 255 - plotimg.reshape((int(h), int(w), -1))[:, :, 0].T[:, ::-1]
            plt.close()
        except:
            print('no loss history in extra_fields')
            lossim = np.zeros((256, 256))
        plot_psf(image, noisemap, lossim)
    # time of iteration
    times.append(time() - t0)
