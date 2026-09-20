"""Synthetic checks for precise, scoped restoration without changing saved data."""
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from comparison.temporal_peak_sensitivity.render_report import (
    PrecisionReader, REPO, restored_prediction_reads, validate_precision, render,
)


def example():
    original = pd.DataFrame({
        "station_row_index": [0, 1], "date_idx": [730, 731],
        "date": ["2023-01-01", "2023-01-02"], "site_code": ["A", "A"],
        "row": [0, 0], "col": [0, 0],
        "observed_pm25": np.asarray([np.float32(12.3456789), np.float32(30.456789)], np.float32),
    })
    maps = np.asarray([[[11.1234567]], [[28.654321]]], dtype=np.float32)
    csv = original.copy()
    csv["predicted_pm25"] = maps[:, 0, 0]
    csv = pd.read_csv(StringIO(csv.to_csv(index=False)))
    return csv, original, maps


class RenderingTests(unittest.TestCase):
    def test_roundtrip_restores_exact_precision_without_mutating_frames(self):
        csv, original, maps = example()
        before = csv.copy()
        restored, record = validate_precision(csv, original, maps, 730)
        np.testing.assert_array_equal(restored.observed_pm25, original.observed_pm25)
        np.testing.assert_array_equal(restored.predicted_pm25, maps[:, 0, 0])
        pd.testing.assert_frame_equal(csv, before)
        self.assertGreater(record["maximum_observed_csv_roundtrip_error"], 0)

    def test_changed_observations_or_predictions_are_rejected(self):
        csv, original, maps = example()
        for key in ("observed_pm25", "predicted_pm25"):
            changed = csv.copy()
            changed.loc[0, key] += 0.001
            with self.assertRaises(ValueError):
                validate_precision(changed, original, maps, 730)

    def test_changed_keys_order_and_missing_rows_are_rejected(self):
        csv, original, maps = example()
        cases = [csv.iloc[::-1], csv.iloc[:1]]
        changed = csv.copy()
        changed.loc[0, "site_code"] = "B"
        cases.append(changed)
        for value in cases:
            with self.assertRaises(ValueError):
                validate_precision(value, original, maps, 730)

    def test_unrelated_csv_passes_through_and_patch_restores_on_failure(self):
        original_reader = pd.read_csv
        with tempfile.TemporaryDirectory(dir=REPO) as folder:
            root = Path(folder)
            run = root / "run"
            run.mkdir()
            other = root / "other.csv"
            pd.DataFrame({"value": [1.23456789]}).to_csv(other, index=False)
            reader = PrecisionReader(run, root, {})
            with self.assertRaises(RuntimeError):
                with restored_prediction_reads(reader):
                    pd.testing.assert_frame_equal(pd.read_csv(other), original_reader(other))
                    raise RuntimeError("simulate renderer failure")
            self.assertIs(pd.read_csv, original_reader)

    def test_unaudited_paths_inside_selected_run_fail(self):
        with tempfile.TemporaryDirectory(dir=REPO) as folder:
            run = Path(folder)
            reader = PrecisionReader(run, run, {})
            for path in (run / "other.csv", run / "screening" / "predictions" / "unknown.csv.gz"):
                with self.assertRaises(ValueError):
                    reader(path)

    def test_render_rejects_outside_run_scope_before_any_read_or_write(self):
        with self.assertRaises(ValueError):
            render(REPO.parent / "other_run", REPO, REPO / "not_audit.json")


if __name__ == "__main__":
    unittest.main()
