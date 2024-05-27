import unittest
from unittest.mock import patch
import sqlite3
import tempfile
import shutil
from lightcurver.structure.database import initialize_database
from lightcurver.processes.normalization_calculation import update_normalization_coefficients


class TestNormalizationCoefficients(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = f"{self.temp_dir}/test.db"

        # initialize the sqlite3 at the temporary path
        initialize_database(self.db_path)

        # patch get_user_config to return the temporary database path
        self.patcher = patch('lightcurver.processes.normalization_calculation.get_user_config')
        self.mock_get_user_config = self.patcher.start()
        self.mock_get_user_config.return_value = {'database_path': self.db_path}

    def tearDown(self):
        self.patcher.stop()
        shutil.rmtree(self.temp_dir)

    def test_insert_normalization_coefficients(self):
        norm_data = [(1, -1, 1.0, 0.05), (2, -2, 0.9, 0.07)]
        update_normalization_coefficients(norm_data)

        # check data inserted correctly
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM normalization_coefficients")
            results = cursor.fetchall()
            self.assertEqual(len(results), 2)
            self.assertIn((1, -1, 1.0, 0.05), results)
            self.assertIn((2, -2, 0.9, 0.07), results)

        # now we're modifying an entry
        norm_data = [(1, -1, 1.1, 0.15)]
        update_normalization_coefficients(norm_data)
        # check new values
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM normalization_coefficients")
            results = cursor.fetchall()
            self.assertEqual(len(results), 2)
            self.assertIn((1, -1, 1.1, 0.15), results)
            self.assertIn((2, -2, 0.9, 0.07), results)


if __name__ == '__main__':
    unittest.main()
