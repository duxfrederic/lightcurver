import unittest
from unittest.mock import patch
import sqlite3
from lightcurver.structure.database import initialize_database
from lightcurver.processes.normalization_calculation import update_normalization_coefficients


class TestNormalizationCoefficients(unittest.TestCase):
    def setUp(self):
        # Mock get_user_config to return an in-memory database path
        self.patcher = patch('lightcurver.structure.user_config.get_user_config')
        self.mock_get_user_config = self.patcher.start()
        self.mock_get_user_config.return_value = {'database_path': ':memory:'}

        initialize_database(':memory:')

    def tearDown(self):
        self.patcher.stop()

    def test_insert_normalization_coefficients(self):
        norm_data = [(1, 'hash1', 0.5, 0.05), (2, 'hash2', 0.7, 0.07)]
        update_normalization_coefficients(norm_data)

        db_path = ':memory:'
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM normalization_coefficients")
            results = cursor.fetchall()
            self.assertEqual(len(results), 2)
            self.assertIn((1, 'hash1', 0.5, 0.05), results)
            self.assertIn((2, 'hash2', 0.7, 0.07), results)


if __name__ == '__main__':
    unittest.main()
