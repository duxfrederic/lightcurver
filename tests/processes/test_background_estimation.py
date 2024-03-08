import unittest
import numpy as np
from lightcurver.processes.background_estimation import subtract_background


class TestSubtractBackground(unittest.TestCase):

    def test_subtract_background(self):
        mean = 100
        sigma = 5
        test_image = np.random.normal(mean, sigma, size=(100, 100))

        image_subtracted, background = subtract_background(test_image)

        self.assertEqual(image_subtracted.shape, test_image.shape)

        self.assertTrue(np.isclose(background.globalback, mean, rtol=1e-1))
        self.assertTrue(np.isclose(background.globalrms, sigma, rtol=1e-1))


if __name__ == '__main__':
    unittest.main()
