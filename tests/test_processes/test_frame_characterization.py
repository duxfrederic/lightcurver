# tests/test_processes/test_frame_characterization.py
import unittest
from lightcurver.processes.frame_characterization import ephemeris


class EphemerisTestCase(unittest.TestCase):
    def test_ephemeris(self):
        # somehow realistic data
        mjd = 60365.13
        ra, dec = 141.23246, 2.32358
        altitude = 2400.0
        latitude = -29.256
        longitude = -70.738

        results = ephemeris(mjd, ra, dec, longitude, latitude, altitude)

        # Verify the structure and some key aspects of the results
        self.assertIsInstance(results, dict)
        self.assertIn('weird_astro_conditions', results)
        self.assertIn('comments', results)
        self.assertIn('target_info', results)
        self.assertIn('moon_info', results)
        self.assertIn('sun_info', results)


if __name__ == '__main__':
    unittest.main()
