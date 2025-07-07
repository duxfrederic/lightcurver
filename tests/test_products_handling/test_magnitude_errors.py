import unittest
import pandas as pd
import numpy as np

from lightcurver.utilities.lightcurves_postprocessing import convert_flux_to_magnitude


class TestConvertFluxToMagnitude(unittest.TestCase):
    def test_convert_flux_to_magnitude(self):
        # sample data with some error bar going to negative flux
        data = {
            'A_flux': [100, 50, 10, 5],
            'A_d_flux': [10, 5, 2, 6],
            'A_scatter_flux': [8, 4, 1.5, 3],
            'zeropoint': [25, 25, 25, 25]
        }
        df = pd.DataFrame(data)

        # expected output:
        expected_data = {
            'A_flux': [100, 50, 10, 5],
            'A_d_flux': [10, 5, 2, 6],
            'A_scatter_flux': [8, 4, 1.5, 3],
            'zeropoint': [25, 25, 25, 25],
            'A_mag': [20.0, 20.7526, 22.5, 23.253],
            'A_d_mag_down': [0.1035, 0.1035, 0.1980, 0.856],
            'A_d_mag_up': [0.1144, 0.1142, 0.2423, np.nan],
            'A_scatter_mag_down': [0.0835, 0.0835, 0.152, 0.510],
            'A_scatter_mag_up': [0.090, 0.090, 0.176, 0.995]
        }
        expected_df = pd.DataFrame(expected_data)

        result_df = convert_flux_to_magnitude(df)

        tol = 1e-2

        # Test 'A_mag'
        for i in range(len(expected_df)):
            expected_mag = expected_df.at[i, 'A_mag']
            result_mag = result_df.at[i, 'A_mag']
            if np.isnan(expected_mag):
                self.assertTrue(np.isnan(result_mag), f"Row {i} A_mag should be NaN")
            else:
                self.assertAlmostEqual(result_mag, expected_mag, delta=tol, msg=f"Row {i} A_mag mismatch")

        # Test 'A_d_mag_down'
        for i in range(len(expected_df)):
            expected_d_down = expected_df.at[i, 'A_d_mag_down']
            result_d_down = result_df.at[i, 'A_d_mag_down']
            if np.isnan(expected_d_down):
                self.assertTrue(np.isnan(result_d_down), f"Row {i} A_d_mag_down should be NaN")
            else:
                self.assertAlmostEqual(result_d_down, expected_d_down, delta=tol,
                                       msg=f"Row {i} A_d_mag_down mismatch")

        # Test 'A_d_mag_up'
        for i in range(len(expected_df)):
            expected_d_up = expected_df.at[i, 'A_d_mag_up']
            result_d_up = result_df.at[i, 'A_d_mag_up']
            if np.isnan(expected_d_up):
                self.assertTrue(np.isnan(result_d_up), f"Row {i} A_d_mag_up should be NaN")
            else:
                self.assertAlmostEqual(result_d_up, expected_d_up, delta=tol,
                                       msg=f"Row {i} A_d_mag_up mismatch")

        for i in range(len(expected_df)):
            expected_scatter_down = expected_df.at[i, 'A_scatter_mag_down']
            result_scatter_down = result_df.at[i, 'A_scatter_mag_down']
            if np.isnan(expected_scatter_down):
                self.assertTrue(np.isnan(result_scatter_down), f"Row {i} A_scatter_mag_down should be NaN")
            else:
                self.assertAlmostEqual(result_scatter_down, expected_scatter_down, delta=tol,
                                       msg=f"Row {i} A_scatter_mag_down mismatch")

            expected_scatter_up = expected_df.at[i, 'A_scatter_mag_up']
            result_scatter_up = result_df.at[i, 'A_scatter_mag_up']
            if np.isnan(expected_scatter_up):
                self.assertTrue(np.isnan(result_scatter_up), f"Row {i} A_scatter_mag_up should be NaN")
            else:
                self.assertAlmostEqual(result_scatter_up, expected_scatter_up, delta=tol,
                                       msg=f"Row {i} A_scatter_mag_up mismatch")


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
