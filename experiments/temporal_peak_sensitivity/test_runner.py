"""Runner scope, cache, pairing and budget tests using only tiny ignored fixtures."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.temporal_peak_sensitivity import run_experiment as runner


class LocalFixtureTests(unittest.TestCase):
    def setUp(self):
        runner.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="unit_", dir=runner.ARTIFACT_ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_superseded_run_compute_is_carried_forward(self):
        budget = runner.Budget(self.folder / "budget.json", 7200, initial_used=4.5)
        self.assertEqual(budget.used, 4.5)
        runner.save_json(budget.path, {**budget.record(), "used_seconds": 7.5})
        resumed = runner.Budget(budget.path, 7200, initial_used=4.5)
        self.assertEqual(resumed.used, 7.5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA AMP regression test")
    def test_amp_overflow_skips_update_without_corrupting_parameters(self):
        model = torch.nn.Linear(1, 1).cuda()
        before = [p.detach().clone() for p in model.parameters()]
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        amp = torch.amp.GradScaler("cuda")
        with torch.amp.autocast("cuda"):
            value = model(torch.ones(2, 1, device="cuda")).float().sum() * 1e20
        amp.scale(value).backward()
        self.assertTrue(runner.optimizer_step(model, optimizer, amp))
        for old, new in zip(before, model.parameters()):
            self.assertTrue(torch.isfinite(new).all())
            torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_output_scope_rejects_roots_outside_and_traversal(self):
        self.assertEqual(runner.output_path(self.folder / "run" / "metrics.json"),
                         (self.folder / "run" / "metrics.json").resolve())
        for path in (runner.ARTIFACT_ROOT, runner.REPO, runner.REPO.parent / "code",
                     runner.ARTIFACT_ROOT / ".." / "outside.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runner.output_path(path)

    def test_frozen_record_cannot_be_changed(self):
        path = self.folder / "selection.json"
        record = {"selected": "standalone_ewma", "passed": True, "seed": 42}
        runner.freeze_json(path, record)
        first = path.read_bytes()
        runner.freeze_json(path, copy.deepcopy(record))
        self.assertEqual(first, path.read_bytes())
        with self.assertRaisesRegex(ValueError, "Immutable"):
            runner.freeze_json(path, {**record, "selected": "standalone_median"})
        self.assertEqual(first, path.read_bytes())

    def test_readonly_float32_memmap_inputs_unchanged_and_no_implicit_hgb(self):
        path = self.folder / "features.npy"
        values = np.arange(3 * 4 * 2 * 2, dtype=np.float32).reshape(3, 4, 2, 2)
        np.save(path, values)
        source = np.load(path, mmap_mode="r")
        before = runner.sha256(path)
        prep = {"channels": [2, 0], "feature_centres": [3., 5.], "feature_scales": [2., 4.]}
        extra = np.full((3, 1, 2, 2), 7., dtype=np.float32)
        original_extra = extra.copy()
        try:
            for days in ([0, 2], np.array([0, 2]), slice(0, 3)):
                with self.subTest(days=str(days)):
                    model_input = runner.model_inputs(source, days, prep, extra, None)
                    expected = values[days][:, [2, 0]].copy()
                    expected -= np.array([3., 5.], np.float32)[None, :, None, None]
                    expected /= np.array([2., 4.], np.float32)[None, :, None, None]
                    self.assertEqual(model_input.shape[1], 3)  # Two base + one extra, no HGB.
                    np.testing.assert_array_equal(model_input[:, :2], expected)
                    np.testing.assert_array_equal(model_input[:, 2], 7.)
                    model_input[:] = -12345.
                    np.testing.assert_array_equal(source, values)
                    np.testing.assert_array_equal(extra, original_extra)
            baseline = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
            original_baseline = baseline.copy()
            hybrid = runner.model_inputs(source, [0, 2], prep, extra, baseline)
            self.assertEqual(hybrid.shape[1], 4)
            np.testing.assert_array_equal(hybrid[:, -1], baseline[[0, 2]])
            hybrid[:] = 0.
            np.testing.assert_array_equal(baseline, original_baseline)
            self.assertEqual(runner.sha256(path), before)
        finally:
            source._mmap.close()

    def test_budget_persists_elapsed_compute_and_rejects_changed_limit(self):
        path = self.folder / "budget.json"
        with mock.patch.object(runner.torch.cuda, "synchronize"), \
                mock.patch.object(runner.time, "monotonic", side_effect=[100., 102., 104.]):
            budget = runner.Budget(path, 10.)
            budget.begin()
            budget.charge()
            budget.end()
        self.assertEqual(budget.used, 4.)
        self.assertIsNone(budget.started)
        self.assertEqual(runner.Budget(path, 10.).used, 4.)
        with self.assertRaisesRegex(ValueError, "change a resumed budget"):
            runner.Budget(path, 20.)
        resumed = runner.Budget(path, 10.)
        resumed.used = 10.
        with self.assertRaises(runner.BudgetExceeded):
            resumed.check()
        self.assertEqual(runner.Budget(self.folder / "capped.json", 99999.).limit, 7200.)
        for seconds in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                runner.Budget(self.folder / "invalid.json", seconds)

    def test_completed_cache_contract_checks_feature_scaler_and_inputs(self):
        prep = {"feature_names": ["a", "b"], "target_log_mean": 1., "target_log_std": .5}
        ctx = {
            "out": self.folder,
            "protocol": {"standalone_schedules": {"42": [3e-4]}, "hybrid_alpha": .425},
            "manifest_digest": "input-and-code-hash",
            "prepared": {("standalone", "selection"): {
                "prep": prep, "baseline": None, "raw_scale": 4., "baseline_hash": None}},
            "extras": {("selection", "none"): (
                np.empty((3, 0, 1, 1), np.float32), (),
                {"scaled_hash": "features-v1", "centres": [], "scales": []})},
        }

        def fake_train(context, out, name, seed, contract, contract_data, *unused):
            checkpoint = out / "final_ema_checkpoint.pt"
            runner.save_torch(checkpoint, {"state": {}, "contract": contract})
            runner.save_json(out / "completed.json", {
                "contract": contract, "checkpoint_sha256": runner.sha256(checkpoint)})

        values = np.zeros((365, 1, 1), np.float32)
        with mock.patch.object(runner, "train_member", side_effect=fake_train) as training, \
                mock.patch.object(runner, "infer", return_value=values) as inference:
            actual, _ = runner.run_member(ctx, "screening", "standalone_control", 42)
            np.testing.assert_array_equal(actual, values)
            runner.run_member(ctx, "confirmation", "standalone_control", 42)
            self.assertEqual(training.call_count, 1)
            self.assertEqual(inference.call_count, 1)
            # Same seed42 is correctly reused between screening and confirmation.
            for field, value in (("manifest_digest", "changed-source"),):
                bad = copy.deepcopy(ctx)
                bad[field] = value
                with self.assertRaisesRegex(ValueError, "Immutable experiment record"):
                    runner.run_member(bad, "screening", "standalone_control", 42)
            for change in ({"scaled_hash": "different-feature-recipe"},
                           {"centres": [0.25]}, {"scales": [2.]}):
                bad = copy.deepcopy(ctx)
                bad["extras"][("selection", "none")][2].update(change)
                with self.assertRaisesRegex(ValueError, "Immutable experiment record"):
                    runner.run_member(bad, "screening", "standalone_control", 42)
            bad = copy.deepcopy(ctx)
            bad["prepared"][("standalone", "selection")]["raw_scale"] = 8.
            with self.assertRaisesRegex(ValueError, "Immutable experiment record"):
                runner.run_member(bad, "screening", "standalone_control", 42)
            # Hash verification also catches prediction corruption with unchanged recipe.
            np.save(self.folder / "selection" / "standalone_control" / "seed_42" / "prediction_z.npy",
                    values + 1.)
            with self.assertRaisesRegex(ValueError, "Saved predictions conflict"):
                runner.run_member(ctx, "screening", "standalone_control", 42)


class SelectionAndInitializationTests(unittest.TestCase):
    @staticmethod
    def score(rmse=3., mae=2., high=7., normal=1.5, bias=-.2):
        return {"n": 100, "rmse": rmse, "mae": mae,
                "bands": {"normal_below_15": {"n": 80, "mae": normal, "bias": bias},
                          "high_at_least_25": {"n": 10, "mae": high}}}

    def test_hybrid_candidate_must_beat_both_controls(self):
        scores = {
            "hybrid_original_control": self.score(rmse=3.),
            "hybrid_temporal_control": self.score(rmse=2.8),
            "hybrid_ewma": self.score(rmse=2.9, high=6.),
        }
        self.assertEqual(runner.select_candidate(scores, "hybrid"), (None, []))
        scores["hybrid_ewma"] = self.score(rmse=2.7, high=6.)
        self.assertEqual(runner.select_candidate(scores, "hybrid")[0], "hybrid_ewma")

    def test_absolute_normal_bias_guard_and_complexity_tie_break(self):
        scores = {"standalone_control": self.score(),
                  "standalone_ewma": self.score(rmse=2.9, high=6., bias=-.21)}
        self.assertIsNone(runner.select_candidate(scores, "standalone")[0])
        scores["standalone_ewma"] = self.score(rmse=2.9, high=6., bias=-.19)
        scores["standalone_median"] = copy.deepcopy(scores["standalone_ewma"])
        self.assertEqual(runner.select_candidate(scores, "standalone")[0], "standalone_median")
        scores["standalone_final_loss"] = copy.deepcopy(scores["standalone_ewma"])
        self.assertEqual(runner.select_candidate(scores, "standalone")[0], "standalone_final_loss")

    def test_initialization_hash_identity_across_added_channels(self):
        base = ["temperature", "history", "fixed_hgb_z"]
        expanded = ["temperature", "history", "new_1", "new_2", "fixed_hgb_z"]
        control = runner.matched_model(42, base, base)
        candidate = runner.matched_model(42, base, expanded)
        control_hash = runner.init_hashes(control, base, base)
        candidate_hash = runner.init_hashes(candidate, base, expanded)
        self.assertEqual(control_hash["common_parameter_hash"], candidate_hash["common_parameter_hash"])
        self.assertTrue(candidate_hash["added_channels_zero"])
        altered = runner.matched_model(11, base, expanded)
        self.assertNotEqual(runner.init_hashes(altered, base, expanded)["common_parameter_hash"],
                            control_hash["common_parameter_hash"])

    def test_array_hash_accepts_zero_channel_recipe_and_distinguishes_shape_dtype(self):
        first = runner.array_hash(np.empty((3, 0, 2, 2), np.float32))
        self.assertEqual(first, runner.array_hash(np.empty((3, 0, 2, 2), np.float32)))
        self.assertNotEqual(first, runner.array_hash(np.empty((4, 0, 2, 2), np.float32)))
        self.assertNotEqual(first, runner.array_hash(np.empty((3, 0, 2, 2), np.float64)))


if __name__ == "__main__":
    unittest.main()
