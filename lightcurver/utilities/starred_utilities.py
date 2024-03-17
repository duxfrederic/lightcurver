import numpy as np
from copy import deepcopy

from starred.deconvolution.loss import Loss
from starred.optim.optimization import Optimizer
from starred.deconvolution.parameters import ParametersDeconv
from starred.optim.inference_base import FisherCovariance


def get_flux_uncertainties(kwargs, kwargs_up, kwargs_down, data, noisemap, model):
    """
    Assuming the other parameters well constrained, this estimates the uncertainty on the flux.
    Args:
        kwargs: optimized starred parameters
        kwargs_up: upper bounds
        kwargs_down: lower bounds
        data: 3d array, the data
        noisemap: 3d array, same shape as data
        model: the model instance
    Returns:
        an array, the same shape as kwargs['kwargs_analytic']['a']: one uncertainty per flux.

    """
    kwargs_fixed = deepcopy(kwargs)
    del kwargs_fixed['kwargs_analytic']['a']

    parameters = ParametersDeconv(kwargs_init=kwargs,
                                  kwargs_fixed=kwargs_fixed,
                                  kwargs_up=kwargs_up,
                                  kwargs_down=kwargs_down)
    loss = Loss(data, model, parameters, noisemap ** 2,
                regularization_terms='l1_starlet')
    optim = Optimizer(loss, parameters, method='l-bfgs-b')
    optim.minimize(maxiter=10)

    fish = FisherCovariance(parameters, optim, diagonal_only=True)
    fish.compute_fisher_information()
    k_errs = fish.get_kwargs_sigma()
    return np.array(k_errs['kwargs_analytic']['a'])
