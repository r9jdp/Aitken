"""Small CPU-only safety tests; no dataset files or checkpoints are written."""
from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from experiments.temporal_model.temporal_features import build_past_only_features
from experiments.temporal_peak_sensitivity.methods import (
    ALPHA, FEATURE_KINDS, apply_scaler, build_features, decode_tensor,
    fit_scaler, loss, matched_model, raw_training_scale,
)


class CausalFeatureTests(unittest.TestCase):
    def test_prefix_invariance_every_recipe(self):
        history = np.arange(36, dtype=np.float32).reshape(9, 2, 2)
        changed = history.copy()
        changed[5:] = -100.0
        for kind in FEATURE_KINDS:
            with self.subTest(kind=kind):
                values, names = build_features(history, kind)
                altered, _ = build_features(changed, kind)
                prefix, _ = build_features(history[:5], kind)
                np.testing.assert_array_equal(values[:5], altered[:5])
                np.testing.assert_array_equal(values[:5], prefix)
                self.assertEqual(values.shape, (9, len(names), 2, 2))

    def test_startup_and_no_second_lag(self):
        history = np.array([4., 8., 20.], dtype=np.float32)[:, None, None]
        ewma, _ = build_features(history, "ewma")
        np.testing.assert_allclose(ewma[:, 0, 0, 0], [4., 6., 13.])
        np.testing.assert_allclose(ewma[:, 2, 0, 0], [0., 2., 7.])
        alpha = 1.0 - 2.0 ** (-1.0 / 3.0)
        self.assertAlmostEqual(float(ewma[1, 1, 0, 0]), alpha * 8 + (1 - alpha) * 4, places=5)
        median, _ = build_features(history, "median")
        np.testing.assert_allclose(median[:, 0, 0, 0], [4., 6., 8.])
        np.testing.assert_allclose(median[:, 1, 0, 0], [0., 2., 12.])
        temporal, _ = build_features(history, "temporal")
        np.testing.assert_allclose(temporal[0, :, 0, 0], [4, 4, 4, 0, 0])

    def test_original_temporal_recipe_is_identical(self):
        history = np.random.default_rng(42).normal(size=(20, 3, 4)).astype(np.float32)
        values, _ = build_features(history, "temporal")
        np.testing.assert_array_equal(values, build_past_only_features(history))
        extended, _ = build_features(history, "temporal_ewma")
        np.testing.assert_array_equal(values, extended[:, :5])

    def test_invalid_input_rejected(self):
        for history in (np.zeros((2, 2)), np.empty((0, 2, 2)), np.full((2, 2, 2), np.nan)):
            with self.assertRaises(ValueError):
                build_features(history, "ewma")
        with self.assertRaises(ValueError):
            build_features(np.zeros((2, 2, 2)), "future_average")

    def test_scaler_only_training_inside_domain(self):
        features = np.array([[[[1., 999.]]], [[[3., 999.]]], [[[1000., 999.]]]])
        original = features.copy()
        inside = np.array([[True, False]])
        centre, scale = fit_scaler(features, inside, 2)
        np.testing.assert_allclose(centre, [2.])
        np.testing.assert_allclose(scale, [1.])
        scaled = apply_scaler(features, centre, scale, mask=inside)
        np.testing.assert_allclose(scaled[:2, 0, 0, 0], [-1., 1.])
        np.testing.assert_array_equal(scaled[:, :, :, 1], 0.)
        np.testing.assert_array_equal(features, original)
        changed = features.copy()
        changed[2:] = -9999.
        np.testing.assert_array_equal(fit_scaler(changed, inside, 2)[0], centre)
        empty = np.empty((3, 0, 1, 2), np.float32)
        c, s = fit_scaler(empty, inside, 2)
        self.assertEqual(apply_scaler(empty, c, s, inside).shape, empty.shape)
        constant = np.ones((3, 2, 1, 2), np.float32)
        np.testing.assert_array_equal(fit_scaler(constant, inside, 2)[1], [1., 1.])

    def test_raw_scale_unweighted_and_training_only(self):
        raw = np.array([[1., 3., np.nan, 100.], [1000., 2000., 3000., 4000.]])
        weight = np.array([[1., 100., 1., 0.], [1., 1., 1., 1.]])
        self.assertEqual(raw_training_scale(raw, weight, 1), 1.0)
        with self.assertRaises(ValueError):
            raw_training_scale(np.ones((2, 2)), np.ones((2, 2)), 2)
        with self.assertRaises(ValueError):
            raw_training_scale(raw, np.zeros_like(weight), 1)


class LossTests(unittest.TestCase):
    prep = {"target_log_mean": 0.0, "target_log_std": 1.0}

    def test_final_hybrid_alpha_and_raw_auxiliary(self):
        output = torch.tensor([[[[0.5]]]], requires_grad=True)
        baseline = torch.tensor([[[[math.log1p(10.)]]]])
        target = torch.tensor([[[[math.log1p(20.)]]]])
        weights = torch.ones_like(output)
        control, final = loss(output, target, weights, baseline, self.prep, 5., 0.)
        augmented, final_aux = loss(output, target, weights, baseline, self.prep, 5., .1)
        torch.testing.assert_close(final, baseline + ALPHA * output)
        torch.testing.assert_close(final, final_aux)
        expected = .1 * ((torch.expm1(final) - 20.) / 5.).square().mean()
        torch.testing.assert_close(augmented - control, expected)
        expected_control = torch.nn.functional.smooth_l1_loss(output, target - baseline)
        torch.testing.assert_close(control, expected_control)
        augmented.backward()
        self.assertTrue(torch.isfinite(output.grad).all())
        self.assertNotEqual(float(output.grad.item()), 0.)

    def test_standalone_and_missing_target_gradient(self):
        output = torch.tensor([1., 200., 3.], requires_grad=True)
        targets = torch.tensor([2., float("nan"), 4.])
        weights = torch.tensor([1., 1., 0.])
        actual, final = loss(output, targets, weights, None, self.prep, 3., .1)
        expected, _ = loss(output[:1], targets[:1], weights[:1], None, self.prep, 3., .1)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(final, output)
        actual.backward()
        torch.testing.assert_close(output.grad[1:], torch.zeros(2))

    def test_empty_observations_finite_zero_loss(self):
        output = torch.ones((2, 1), requires_grad=True)
        value, _ = loss(output, torch.full_like(output, float("nan")),
                        torch.zeros_like(output), None, self.prep, 1., .1)
        self.assertEqual(float(value.detach()), 0.)
        value.backward()
        torch.testing.assert_close(output.grad, torch.zeros_like(output))

    def test_float32_decode_and_bounds(self):
        z = torch.tensor([-100., 0., math.log1p(10), 100.], dtype=torch.float16)
        decoded = decode_tensor(z, 0., 1.)
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertEqual(float(decoded[0]), 0.)
        self.assertEqual(float(decoded[-1]), 500.)
        self.assertTrue(torch.isfinite(decoded).all())

    def test_existing_smooth_l1_source_weights_preserved(self):
        output = torch.tensor([.5, 2., -2.])
        target = torch.tensor([0., 0., 0.])
        weights = torch.tensor([1., .15, 2.5])
        actual, _ = loss(output, target, weights, None, self.prep, 1., 0.)
        expected = (torch.nn.functional.smooth_l1_loss(output, target, reduction="none")
                    * weights).sum() / weights.sum()
        torch.testing.assert_close(actual, expected)


class MatchedInitializationTests(unittest.TestCase):
    def test_shared_weights_extra_zeros_and_matching_rng(self):
        base_names = ("temperature", "history", "hgb_baseline_z")
        names = ("temperature", "history", "ewma", "innovation", "hgb_baseline_z")
        canonical = matched_model(42, base_names, base_names)
        expected_rng_draw = torch.rand(5)
        expanded = matched_model(42, base_names, names)
        torch.testing.assert_close(torch.rand(5), expected_rng_draw, rtol=0, atol=0)
        for key, value in canonical.state_dict().items():
            counterpart = expanded.state_dict()[key]
            if key in {"encoder1.conv1.weight", "encoder1.skip.weight"}:
                for index, name in enumerate(base_names):
                    torch.testing.assert_close(value[:, index], counterpart[:, names.index(name)], rtol=0, atol=0)
                torch.testing.assert_close(counterpart[:, 2:4], torch.zeros_like(counterpart[:, 2:4]), rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, counterpart, rtol=0, atol=0)

    def test_standalone_does_not_inject_hgb_channel(self):
        base_names = ("temperature", "pm25_lag1_idw_causal")
        names = base_names + ("history_ewma_half_life_1d",)
        model = matched_model(42, base_names, names)
        self.assertEqual(model.encoder1.conv1.in_channels, 3)
        self.assertFalse(any("hgb" in name.lower() for name in names))
        with self.assertRaises(ValueError):
            matched_model(42, base_names, ("temperature", "new_feature"))


if __name__ == "__main__":
    unittest.main()
