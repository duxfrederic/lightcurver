import unittest
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord

from lightcurver.processes.cutout_making import extract_stamp


class TestExtractStamp(unittest.TestCase):

    def setUp(self):
        self.data = np.random.rand(100, 100)
        self.header = fits.Header()
        self.header['CRVAL1'] = 10.0
        self.header['CRVAL2'] = 20.0
        self.header['CRPIX1'] = 50.0
        self.header['CRPIX2'] = 50.0
        self.header['CD1_1'] = 0.1
        self.header['CD1_2'] = 0.0
        self.header['CD2_1'] = 0.0
        self.header['CD2_2'] = 0.1
        self.header['CTYPE1'] = 'RA---TAN'
        self.header['CTYPE2'] = 'DEC--TAN'

        # Sample parameters
        self.exptime = 1.0
        self.sky_coord = SkyCoord(10.0, 20.0, unit='deg')
        self.cutout_size = 10

    def test_extract_stamp(self):
        # check the function can be called
        cutout, noisemap, wcs_header_string, position = extract_stamp(
            self.data, self.header, self.exptime, self.sky_coord, self.cutout_size,
            background_rms_electron_per_second=0.5)

        # check ... output shapes?
        self.assertEqual(cutout.shape, (self.cutout_size, self.cutout_size))
        self.assertEqual(noisemap.shape, (self.cutout_size, self.cutout_size))

        # is the wcs propagated?
        self.assertTrue(wcs_header_string)


if __name__ == '__main__':
    unittest.main()
