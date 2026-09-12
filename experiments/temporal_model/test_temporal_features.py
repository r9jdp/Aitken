"""Focused checks for temporal leakage and normal-range protection."""
import unittest

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

from temporal_features import (
    TEMPORAL_NAMES,
    band_metrics,
    build_past_only_features,
    compose_prediction,
    fit_temporal_scaler,
    safe_candidate,
    scale_temporal,
)


class TemporalFeatureTests(unittest.TestCase):
    def test_features_use_no_future_rows(self):
        history = np.arange(10, dtype=np.float32)[:, None, None]
        original = build_past_only_features(history)
        changed = history.copy()
        changed[6:] = 10000
        modified = build_past_only_features(changed)
        np.testing.assert_array_equal(original[:6], modified[:6])

    def test_expected_rolling_values(self):
        history = np.asarray([1, 2, 4, 8, 16, 32, 64], np.float32)[:, None, None]
        result = build_past_only_features(history)
        self.assertEqual(result.shape, (7, len(TEMPORAL_NAMES), 1, 1))
        self.assertAlmostEqual(result[3, 0, 0, 0], (2 + 4 + 8) / 3)
        self.assertAlmostEqual(result[6, 1, 0, 0], sum([1, 2, 4, 8, 16, 32, 64]) / 7)
        self.assertEqual(result[6, 2, 0, 0], 64)
        self.assertAlmostEqual(result[3, 4, 0, 0], 8 - (1 + 2 + 4) / 3, places=5)

    def test_invalid_or_nonfinite_history_rejected(self):
        with self.assertRaises(ValueError):
            build_past_only_features(np.ones((2, 2)))
        broken = np.ones((2, 1, 1), np.float32)
        broken[1] = np.nan
        with self.assertRaises(ValueError):
            build_past_only_features(broken)

    def test_scaler_uses_training_days_only(self):
        values = np.ones((4, len(TEMPORAL_NAMES), 1, 2), np.float32)
        values[0, :, 0, 1] = 3
        centres, scales = fit_temporal_scaler(values, np.ones((1, 2), bool), 2)
        changed = values.copy()
        changed[2:] = 999
        other_centres, other_scales = fit_temporal_scaler(changed, np.ones((1, 2), bool), 2)
        np.testing.assert_array_equal(centres, other_centres)
        np.testing.assert_array_equal(scales, other_scales)
        scaled = scale_temporal(values, centres, scales)
        self.assertTrue(np.isfinite(scaled).all())


@unittest.skipIf(torch is None, "requires PyTorch from the CUDA training environment")
class PredictionTests(unittest.TestCase):
    def test_correction_is_added_to_the_hgb_baseline(self):
        baseline = torch.full((1, 1, 1, 1), 2.0)
        correction = torch.full_like(baseline, 4.0)
        result = compose_prediction(baseline, correction, 0.5)
        self.assertTrue(torch.allclose(result, torch.full_like(baseline, 4.0)))


class SelectionTests(unittest.TestCase):
    def test_band_metrics_keep_normal_and_high_separate(self):
        metrics = band_metrics(
            np.asarray([5, 10, 20, 30]), np.asarray([6, 8, 21, 20])
        )
        self.assertEqual(metrics["normal_below_15"]["n"], 2)
        self.assertEqual(metrics["elevated_15_to_25"]["n"], 1)
        self.assertEqual(metrics["high_at_least_25"]["mae"], 10)

    def test_selection_rejects_normal_range_damage(self):
        control = {
            "rmse": 4.0, "r2": 0.50,
            "bands": {
                "normal_below_15": {"mae": 1.5, "bias": -0.1},
                "high_at_least_25": {"mae": 14.0},
            },
        }
        safe = {
            "rmse": 3.8, "r2": 0.55,
            "bands": {
                "normal_below_15": {"mae": 1.5, "bias": 0.05},
                "high_at_least_25": {"mae": 12.0},
            },
        }
        damaging = {
            "rmse": 3.7, "r2": 0.57,
            "bands": {
                "normal_below_15": {"mae": 1.7, "bias": 0.3},
                "high_at_least_25": {"mae": 11.0},
            },
        }
        self.assertTrue(safe_candidate(safe, control))
        self.assertFalse(safe_candidate(damaging, control))


if __name__ == "__main__":
    unittest.main()
