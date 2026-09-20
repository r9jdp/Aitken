"""Small fixtures test evaluation identity, guards and calendar-day uncertainty."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.temporal_peak_sensitivity.reporting import (
    REPO, _plots, block_interval, metrics, safe_candidate, write_report,
)


def frame(observed=(5, 8, 30, 7, 29, 10, 15, 6, 9, 33), predicted=None):
    observed = np.asarray(observed, float)
    if predicted is None:
        predicted = observed * 0.8
    n = len(observed)
    return pd.DataFrame({
        "station_row_index": np.arange(n), "date_idx": np.arange(n),
        "date": pd.date_range("2023-01-01", periods=n), "site_code": ["TEST"] * n,
        "row": np.zeros(n, int), "col": np.zeros(n, int),
        "observed_pm25": observed, "predicted_pm25": predicted,
    })


class ReportingTests(unittest.TestCase):
    def test_metrics_and_json(self):
        result = metrics(frame())
        self.assertEqual(result["n"], 10)
        self.assertEqual(result["bands"]["high_at_least_25"]["n"], 3)
        self.assertEqual(result["high_pollution"]["underprediction_count"], 3)
        self.assertEqual(result["high_pollution"]["true_exceedance_count"], 1)
        self.assertEqual(result["high_pollution"]["missed_exceedance_count"], 2)
        self.assertEqual(sum(x["n"] for x in result["calibration"]), 10)
        self.assertAlmostEqual(result["bias"], -3.04)
        json.dumps(result, allow_nan=False)

    def test_rapid_change_uses_same_station_previous_calendar_day(self):
        value = frame((5, 30, 10, 28))
        value.loc[3, "date"] = pd.Timestamp("2023-01-05")
        value.loc[3, "date_idx"] = 4
        rapid = metrics(value)["rapid_changes"]
        self.assertEqual(rapid["rise_at_least_10"]["n"], 1)
        self.assertEqual(rapid["fall_at_least_10"]["n"], 1)
        self.assertEqual(rapid["unavailable_previous_day_n"], 2)
        other = value.copy()
        other["site_code"] = "OTHER"
        other.station_row_index += 10
        self.assertEqual(metrics(pd.concat([value, other]))["rapid_changes"]["rise_at_least_10"]["n"], 2)

    def test_empty_bands_have_none_not_nan_and_gate_fails(self):
        value = metrics(frame((1, 2, 3)))
        self.assertIsNone(value["bands"]["high_at_least_25"]["mae"])
        self.assertFalse(safe_candidate(value, [value]))
        json.dumps(value, allow_nan=False)

    def test_gate_checks_all_controls_and_signed_bias_is_absolute(self):
        control = metrics(frame())
        candidate = metrics(frame(predicted=np.asarray((5, 8, 30, 7, 29, 10, 15, 6, 9, 33)) * 0.9))
        self.assertTrue(safe_candidate(candidate, [control]))
        self.assertFalse(safe_candidate(control, [candidate]))
        self.assertFalse(safe_candidate(candidate, [control, candidate]))
        candidate["bands"]["normal_below_15"]["bias"] = -100
        self.assertFalse(safe_candidate(candidate, [control]))

    def test_gate_rejects_overall_mae_worsening_even_if_rmse_better(self):
        control = metrics(frame())
        candidate = copy.deepcopy(control)
        candidate["rmse"] -= 0.1
        candidate["bands"]["high_at_least_25"]["mae"] -= 0.1
        candidate["mae"] += 0.0001
        self.assertFalse(safe_candidate(candidate, [control]))

    def test_pairing_rejects_changed_truth_or_keys(self):
        base = frame()
        for column, value in (("observed_pm25", 4), ("site_code", "DIFFERENT"), ("row", 1)):
            changed = base.copy()
            changed.loc[0, column] = value
            with self.assertRaises(ValueError):
                block_interval(changed, base, n=20)

    def test_bootstrap_is_paired_reproducible_and_handles_row_order(self):
        base = frame()
        perfect = base.copy()
        perfect.predicted_pm25 = perfect.observed_pm25
        result = block_interval(perfect, base, n=100)
        self.assertEqual(result, block_interval(perfect.iloc[::-1], base, n=100))
        self.assertEqual(result["replicates"], 100)
        self.assertLess(result["ci95"][1], 0)
        self.assertEqual(block_interval(base, base, n=100)["ci95"], [0, 0])

    def test_missing_calendar_days_are_retained(self):
        base = frame().drop(index=[3, 4, 5])
        result = block_interval(base, base, n=20)
        self.assertEqual(result["calendar_days"], 10)
        self.assertEqual(result["ci95"], [0, 0])
        self.assertIsNone(block_interval(frame((1, 2)), frame((1, 2)), n=20)["ci95"])

    def test_invalid_prediction_rows_are_rejected(self):
        for change in ("duplicate", "nonfinite", "negative", "missing"):
            value = frame()
            if change == "duplicate":
                value = pd.concat([value, value.iloc[:1]])
            elif change == "nonfinite":
                value.loc[0, "predicted_pm25"] = np.inf
            elif change == "negative":
                value.loc[0, "predicted_pm25"] = -1
            else:
                value = value.drop(columns="site_code")
            with self.assertRaises(ValueError):
                metrics(value)

    def test_report_partial_and_output_guard(self):
        # Temporary paths stay inside Aitken, matching production write guards.
        with tempfile.TemporaryDirectory(dir=REPO) as folder:
            root = Path(folder)
            run = root / "run"
            run.mkdir()
            (run / "summary.json").write_text(json.dumps({"status": "budget_exhausted", "phases": {}, "budget": {"seconds": 7200}}))
            path = write_report(run, root / "report")
            self.assertIn("budget_exhausted", path.read_text(encoding="utf-8"))
            saved = json.loads((path.parent / "metrics.json").read_text())
            self.assertEqual(saved["phases"]["final"], {})
            with self.assertRaises(ValueError):
                write_report(run, REPO.parent / "outside_report")

    def test_report_recomputes_and_rejects_metric_mismatch(self):
        with tempfile.TemporaryDirectory(dir=REPO) as folder:
            root = Path(folder)
            run = root / "run"
            predictions = run / "screening" / "predictions"
            predictions.mkdir(parents=True)
            value = frame()
            value.to_csv(predictions / "standalone_control.csv.gz", index=False)
            summary = {"status": "screening", "phases": {"screening": {"standalone_control": metrics(value)}},
                       "variants": {"standalone_control": {"family": "standalone", "is_control": True}}}
            (run / "summary.json").write_text(json.dumps(summary))
            with patch("experiments.temporal_peak_sensitivity.reporting._plots", return_value=[]):
                write_report(run, root / "report")
            summary["phases"]["screening"]["standalone_control"]["rmse"] += 1
            (run / "summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError), patch("experiments.temporal_peak_sensitivity.reporting._plots", return_value=[]):
                write_report(run, root / "report")

    def test_real_plot_render_includes_fixed_and_peak_dates(self):
        value = frame((5, 30, 10, 28))
        dates = pd.to_datetime(["2023-03-11", "2023-04-20", "2023-07-15", "2023-12-01"])
        value["date"] = dates
        value["date_idx"] = (dates - pd.Timestamp("2023-01-01")).days
        variant = {"standalone_control": {"family": "standalone", "is_control": True}}
        with tempfile.TemporaryDirectory(dir=REPO) as folder:
            names = _plots({"standalone_control": value}, variant, "screening", Path(folder))
            self.assertEqual(len(names), 2)
            for name in names:
                self.assertGreater((Path(folder) / name).stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
