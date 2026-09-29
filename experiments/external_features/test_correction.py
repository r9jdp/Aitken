from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_aurn_correction import gate, validate_prediction_frame  # noqa: E402


class CorrectionTests(unittest.TestCase):
    def test_gate_protects_normal_days(self) -> None:
        values = gate(np.array([2.0, 8.0, 11.5, 15.0, 30.0]), 8.0, 15.0)
        np.testing.assert_allclose(values, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_gate_rejects_invalid_bounds(self) -> None:
        with self.assertRaises(ValueError):
            gate(np.ones(3), 10.0, 10.0)

    def test_prediction_signature_rejects_missing_day(self) -> None:
        frame = pd.DataFrame(
            {
                "date_idx": [730],
                "date": [pd.Timestamp("2023-01-01")],
                "site_code": ["x"],
                "observed_pm25": [5.0],
                "predicted_pm25": [5.0],
            }
        )
        with self.assertRaises(ValueError):
            validate_prediction_frame(
                frame,
                label="selection",
                expected_rows=2,
                expected_days=2,
                minimum_date_idx=730,
                maximum_date_idx=731,
            )


if __name__ == "__main__":
    unittest.main()
