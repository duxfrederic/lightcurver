# tests/processes/test_character.py
import unittest
from lightcurver.processes.character import ephemerides


class EphemeridesTestCase(unittest.TestCase):
    def test_ephemerides(self):
        # Test data
        mjd = 60365.13
        ra, dec = 141.23246, 2.32358
        altitude = 2400.0
        latitude = -29.256
        longitude = -70.738

        # Call the function with the test data
        results = ephemerides(mjd, ra, dec, longitude, latitude, altitude)

        # Verify the structure and some key aspects of the results
        self.assertIsInstance(results, dict)
        self.assertIn('astro_conditions', results)
        self.assertIn('comments', results)
        self.assertIn('target_info', results)
        self.assertIn('moon_info', results)
        self.assertIn('sun_info', results)


if __name__ == '__main__':
    unittest.main()
