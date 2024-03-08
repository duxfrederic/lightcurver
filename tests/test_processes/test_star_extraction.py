import unittest
import numpy as np
from astropy.table import Table

from lightcurver.processes.star_extraction import extract_stars


class TestExtractStars(unittest.TestCase):
    """
    This is more of a test to make sure the API of sep hasn't changed.
    """

    def setUp(self):
        # Create a sample background subtracted image
        self.image_background_subtracted = np.random.rand(100, 100)
        self.background_rms = 1.0
        self.detection_threshold = 3
        self.min_area = 10

    def test_extract_stars(self):
        sources = extract_stars(self.image_background_subtracted, self.background_rms,
                                self.detection_threshold, self.min_area)

        # check if the output is an astropy Table
        self.assertIsInstance(sources, Table)

        # check if the required columns are present
        self.assertIn('xcentroid', sources.colnames)
        self.assertIn('ycentroid', sources.colnames)
        self.assertIn('flag', sources.colnames)
        self.assertIn('a', sources.colnames)
        self.assertIn('b', sources.colnames)
        self.assertIn('flux', sources.colnames)
        self.assertIn('npix', sources.colnames)

        # check if indeed no source detected
        self.assertEqual(len(sources), 0, "There should be zero sources detected here.")


if __name__ == '__main__':
    unittest.main()
