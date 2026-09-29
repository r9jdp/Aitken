from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from aurn import build_daily_features, parse_hourly_csv  # noqa: E402


class AurnTests(unittest.TestCase):
    def test_hourly_parser_enforces_18_ratified_hours(self) -> None:
        lines = [
            "Data supplied by UK-AIR",
            "All Data GMT hour ending",
            "Status: R = Ratified",
            ",,Example,,,,",
            'Date,time,"PM<sub>2.5</sub> particulate matter (Hourly measured)",status,unit',
        ]
        for hour in range(1, 25):
            status = "R" if hour <= 18 else "P"
            lines.append(f"01-01-2024,{hour:02d}:00,{hour},{status},ugm-3")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            path.write_text("\n".join(lines), encoding="utf-8")
            site = pd.Series(
                dict(site_id="TEST", site_name="Test", latitude=51.0, longitude=0.5, distance_km=50.0, bearing_deg=90.0)
            )
            result = parse_hourly_csv(path, site, 2024)
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.valid_hours.iloc[0]), 18)
        self.assertAlmostEqual(float(result.pm25.iloc[0]), 9.5)

    def test_feature_fill_never_reads_future_days(self) -> None:
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-01-01", "2021-01-03"]),
                "pm25": [10.0, 100.0],
                "distance_km": [50.0, 50.0],
                "bearing_deg": [270.0, 270.0],
            }
        )
        original, _, _ = build_daily_features(daily, np.ones(3), np.zeros(3), end=pd.Timestamp("2021-01-03"))
        changed = daily.copy()
        changed.loc[changed.date == pd.Timestamp("2021-01-03"), "pm25"] = 300.0
        updated, _, _ = build_daily_features(changed, np.ones(3), np.zeros(3), end=pd.Timestamp("2021-01-03"))
        np.testing.assert_allclose(original[:2], updated[:2])

    def test_lag_features_are_previous_day_values(self) -> None:
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                "pm25": [10.0, 20.0],
                "distance_km": [50.0, 50.0],
                "bearing_deg": [270.0, 270.0],
            }
        )
        values, names, _ = build_daily_features(daily, np.ones(2), np.zeros(2), end=pd.Timestamp("2021-01-02"))
        self.assertEqual(names[5], "aurn_regional_mean_lag1")
        self.assertEqual(float(values[0, 5]), 0.0)
        self.assertEqual(float(values[1, 5]), 10.0)


if __name__ == "__main__":
    unittest.main()
