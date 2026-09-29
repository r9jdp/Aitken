import unittest

import pandas as pd

from audit_dft_traffic import coverage_table


class TrafficAuditTests(unittest.TestCase):
    def test_counts_unique_dates_not_hourly_rows(self):
        frame = pd.DataFrame(
            {
                "year": [2021, 2021, 2021],
                "count_date": ["2021-04-01", "2021-04-01", "2021-04-02"],
                "count_point_id": [1, 2, 1],
            }
        )
        result = coverage_table(frame, 2021, 2021)[0]
        self.assertEqual(result["unique_survey_days"], 2)
        self.assertEqual(result["unique_count_points"], 2)
        self.assertAlmostEqual(result["calendar_coverage_fraction"], 2 / 365)


if __name__ == "__main__":
    unittest.main()
