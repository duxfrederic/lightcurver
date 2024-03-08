import unittest
import numpy as np
from starred.procedures.psf_routines import build_psf

from lightcurver.processes.star_photometry import do_one_deconvolution


class TestStarredCalls(unittest.TestCase):

    def setUp(self):
        x, y = np.meshgrid(np.arange(-8, 8), np.arange(-8, 8))
        gauss = np.exp(-0.1 * (x**2 + y**2))
        self.data = 0.1*np.random.rand(5, 16, 16) + np.repeat(gauss[None, :, :], repeats=5, axis=0)
        self.noisemap = 0.1*np.ones((5, 16, 16))
        self.psf = np.repeat(gauss[None, :, :], repeats=5, axis=0)

        self.subsampling_factor = 1
        self.n_iter = 50

    def test_do_one_deconvolution(self):
        # call
        result = do_one_deconvolution(self.data, self.noisemap, self.psf, self.subsampling_factor, self.n_iter)

        self.assertIsInstance(result, dict)

        # check if expected keys are present there
        self.assertIn('scale', result)
        self.assertIn('kwargs_final', result)
        self.assertIn('kwargs_uncertainties', result)
        self.assertIn('fluxes', result)
        self.assertIn('fluxes_uncertainties', result)
        self.assertIn('chi2', result)
        self.assertIn('chi2_per_frame', result)
        self.assertIn('loss_curve', result)
        self.assertIn('residuals', result)

        # check if 'scale' is a positive float
        self.assertIsInstance(result['scale'], float)
        self.assertGreater(result['scale'], 0)

        # check if 'kwargs_final' and 'kwargs_uncertainties' are dictionaries
        self.assertIsInstance(result['kwargs_final'], dict)
        self.assertIsInstance(result['kwargs_uncertainties'], dict)

        # check if 'fluxes' and 'fluxes_uncertainties' are 1D numpy arrays
        self.assertIsInstance(result['fluxes'], np.ndarray)
        self.assertIsInstance(result['fluxes_uncertainties'], np.ndarray)
        self.assertEqual(result['fluxes'].ndim, 1)
        self.assertEqual(result['fluxes_uncertainties'].ndim, 1)
        self.assertEqual(result['fluxes'].size, result['fluxes_uncertainties'].size)
        self.assertEqual(result['fluxes'].size, self.data.shape[0])

        self.assertIsInstance(result['chi2'], float)  # float and not jax array
        self.assertGreaterEqual(result['chi2'], 0)

        self.assertIsInstance(result['chi2_per_frame'], np.ndarray)
        self.assertEqual(result['chi2_per_frame'].ndim, 1)
        self.assertEqual(len(result['chi2_per_frame']), self.data.shape[0])

        self.assertEqual(len(result['loss_curve']), self.n_iter)
        # if not the starred api might have changed and implemented some stop of the optimization
        # before the completion of the iterations. we do not want that typically, so
        # go in the code and make it stick to the set number of iterations.

        # check that 'residuals' is a 3D numpy array with same shape as the input data
        self.assertEqual(result['residuals'].shape, self.data.shape)

    def test_build_psf(self):
        result = build_psf(self.data,
                           self.noisemap,
                           subsampling_factor=self.subsampling_factor,
                           n_iter_analytic=5,
                           n_iter_adabelief=10,
                           masks=np.ones_like(self.data),
                           guess_method_star_position='center')
        self.assertIsInstance(result, dict)
        self.assertIn('full_psf', result)
        self.assertIn('adabelief_extra_fields', result)
        self.assertIn('loss_history', result['adabelief_extra_fields'])
        self.assertIn('narrow_psf', result)
        self.assertIn('chi2', result)
        self.assertIn('residuals', result)


if __name__ == '__main__':
    unittest.main()
