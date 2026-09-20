"""Independent, CPU-only audit of the saved peak-sensitivity experiment.

No trainer, feature builder, or reporting implementation is imported. Original
station labels come directly from station_daily.npz; predictions are reconstructed
from member arrays and their recorded scalers. Nothing in the training run is
modified. The only write is the requested JSON audit inside this comparison folder.

Example (from Aitken):
  python comparison/temporal_peak_sensitivity/audit_saved_run.py \
    --run-dir artifacts/temporal_peak_sensitivity/run_20260920_v2 \
    --data-dir ../code/pm25_london_bundle/data/processed/london_1km_daily
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
KEYS = ("station_row_index", "date_idx", "date", "site_code", "row", "col")
SEEDS = (42, 11, 22)
BANDS = ("normal_below_15", "elevated_15_to_25", "high_at_least_25")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def numbers(y, p, total_sse=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    error = p - y
    squared = float(np.dot(error, error))
    n = len(y)
    variance_sum = float(np.square(y - y.mean()).sum()) if n else 0.
    return dict(n=n, mae=float(np.abs(error).mean()) if n else None,
                rmse=float(np.sqrt(squared / n)) if n else None,
                r2=1. - squared / variance_sum if variance_sum else None,
                bias=float(error.mean()) if n else None,
                observed_mean=float(y.mean()) if n else None,
                predicted_mean=float(p.mean()) if n else None,
                squared_error=squared,
                squared_error_percent=100. * squared / total_sse if total_sse else None)


def score(frame):
    """Recompute every reported group without calling the report implementation."""
    y, p = frame.observed_pm25.to_numpy(float), frame.predicted_pm25.to_numpy(float)
    require(np.isfinite(y).all() and np.isfinite(p).all(), "Nonfinite score input")
    total = float(np.square(p - y).sum())
    result = numbers(y, p, total)
    dates = pd.to_datetime(frame.date)
    result.update(n_dates=int(dates.nunique()), first_date=str(dates.min().date()),
                  last_date=str(dates.max().date()))
    result["bands"] = {name: numbers(y[use], p[use], total) for name, use in zip(
        BANDS, (y < 15, (y >= 15) & (y < 25), y >= 25))}
    result["high_pollution"] = dict(
        threshold=25., observed_at_least_25_count=int((y >= 25).sum()),
        underprediction_count=int(((y >= 25) & (p < y)).sum()),
        predicted_at_least_25_count=int((p >= 25).sum()),
        true_exceedance_count=int(((y >= 25) & (p >= 25)).sum()),
        false_exceedance_count=int(((y < 25) & (p >= 25)).sum()),
        missed_exceedance_count=int(((y >= 25) & (p < 25)).sum()))
    previous = {(str(site), pd.Timestamp(day)): float(value)
                for site, day, value in zip(frame.site_code, dates, y)}
    changes = np.array([value - previous.get((str(site), day - pd.Timedelta(days=1)), np.nan)
                        for site, day, value in zip(frame.site_code, dates, y)])
    available = np.isfinite(changes)
    result["rapid_changes"] = dict(
        available_previous_day_n=int(available.sum()),
        unavailable_previous_day_n=int((~available).sum()),
        rise_at_least_10=numbers(y[changes >= 10], p[changes >= 10], total),
        fall_at_least_10=numbers(y[changes <= -10], p[changes <= -10], total),
        other_changes=numbers(y[available & (np.abs(changes) < 10)],
                              p[available & (np.abs(changes) < 10)], total))
    result["calibration"] = []
    limits = (0., 5., 10., 15., 25., float("inf"))
    for low, high in zip(limits[:-1], limits[1:]):
        use = (p >= low) & (p < high)
        result["calibration"].append(dict(
            predicted_lower_inclusive=low,
            predicted_upper_exclusive=None if np.isinf(high) else high,
            **numbers(y[use], p[use], total)))
    return result


def equal_metrics(actual, saved, label):
    """Require every independently calculated metric to be present and equal."""
    if isinstance(actual, dict):
        require(isinstance(saved, dict), f"{label}: expected object")
        for key, value in actual.items():
            require(key in saved, f"{label}: missing metric {key}")
            equal_metrics(value, saved[key], f"{label}/{key}")
    elif isinstance(actual, list):
        require(isinstance(saved, list) and len(saved) == len(actual), f"{label}: list differs")
        for index, (left, right) in enumerate(zip(actual, saved)):
            equal_metrics(left, right, f"{label}/{index}")
    elif isinstance(actual, (float, int)):
        require(saved is not None and np.isclose(actual, saved, rtol=1e-8, atol=1e-9),
                f"{label}: recomputed {actual}, recorded {saved}")
    else:
        require(actual == saved, f"{label}: value differs")


def gates(candidate, control):
    normal, baseline_normal = candidate["bands"][BANDS[0]], control["bands"][BANDS[0]]
    high, baseline_high = candidate["bands"][BANDS[2]], control["bands"][BANDS[2]]
    return dict(
        same_counts=candidate["n"] == control["n"] and normal["n"] == baseline_normal["n"]
        and high["n"] == baseline_high["n"],
        rmse_improved=candidate["rmse"] < control["rmse"],
        high_mae_improved=bool(high["n"] and high["mae"] < baseline_high["mae"]),
        overall_mae_not_worse=candidate["mae"] <= control["mae"],
        normal_mae_not_worse=bool(normal["n"] and normal["mae"] <= baseline_normal["mae"]),
        absolute_normal_bias_guard=bool(normal["n"] and
            abs(normal["bias"]) <= max(.10, abs(baseline_normal["bias"]))))


def control_names(family):
    return ["standalone_control"] if family == "standalone" else [
        "hybrid_original_control", "hybrid_temporal_control"]


def passes(candidate, controls):
    return bool(controls) and all(all(gates(candidate, control).values()) for control in controls)


def original_rows(data, first, stop):
    with np.load(data / "station_daily.npz", allow_pickle=False) as station:
        selected = np.flatnonzero((station["source"] == 1)
                                 & (station["date_idx"] >= first)
                                 & (station["date_idx"] < stop))
        days = station["date_idx"][selected].astype(int)
        frame = pd.DataFrame(dict(
            station_row_index=selected, date_idx=days,
            date=(pd.Timestamp("2021-01-01") + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d"),
            site_code=station["site_codes"][station["site_idx"][selected]].astype(str),
            row=station["row"][selected], col=station["col"][selected],
            observed_pm25=station["value"][selected]))
    require(not frame.duplicated(["site_code", "date"]).any(), "Duplicate original station-day")
    require(not frame.station_row_index.duplicated().any(), "Duplicate original station row")
    require(np.isfinite(frame.observed_pm25).all(), "Nonfinite original observation")
    return frame


def decode(values, prep):
    # Preserve stored float32 arithmetic, rather than add precision absent at inference.
    logged = np.clip(values * prep["target_log_std"] + prep["target_log_mean"], 0., np.log1p(500.))
    return np.clip(np.expm1(logged), 0., 500.)


def take(maps, rows, first):
    return maps[rows.date_idx.to_numpy(int) - first,
                rows.row.to_numpy(int), rows.col.to_numpy(int)]


def bootstrap(candidate, control, n=5000, seed=20260920):
    """Independent seven-calendar-day paired RMSE interval for final comparisons."""
    dates = pd.to_datetime(candidate.date)
    days = (dates - dates.min()).dt.days.to_numpy(int)
    total_days = int(days.max()) + 1
    require(total_days >= 7, "Bootstrap needs seven calendar days")
    labels = candidate.observed_pm25.to_numpy(float)
    counts = np.bincount(days, minlength=total_days)
    candidate_sse = np.bincount(days, weights=np.square(candidate.predicted_pm25 - labels), minlength=total_days)
    control_sse = np.bincount(days, weights=np.square(control.predicted_pm25 - labels), minlength=total_days)
    rng, values = np.random.default_rng(seed), []
    for _ in range(n * 10):
        starts = rng.integers(0, total_days - 6, size=int(np.ceil(total_days / 7)))
        sampled = np.concatenate([np.arange(start, start + 7) for start in starts])[:total_days]
        size = counts[sampled].sum()
        if size:
            values.append(float(np.sqrt(candidate_sse[sampled].sum() / size)
                                - np.sqrt(control_sse[sampled].sum() / size)))
        if len(values) == n:
            break
    require(len(values) == n, "Insufficient valid bootstrap samples")
    return dict(replicates=n, block_days=7, seed=seed,
                rmse_delta=float(np.sqrt(candidate_sse.sum() / counts.sum())
                                 - np.sqrt(control_sse.sum() / counts.sum())),
                ci95=np.quantile(values, [.025, .975]).tolist())


def audit(run, data, with_bootstrap=False, bootstrap_all=False):
    run, data = Path(run).resolve(), Path(data).resolve()
    require(run.is_relative_to(REPO / "artifacts" / "temporal_peak_sensitivity"), "Run must be inside experiment artifacts")
    summary_path, manifest_path = run / "summary.json", run / "manifest.json"
    initial_summary_hash = digest(summary_path)
    summary, manifest = read_json(summary_path), read_json(manifest_path)
    require(summary["status"] in ("complete", "partial_budget_exhausted"),
            "Audit only a completed or budget-stopped run, never a running experiment")
    full = summary["status"] == "complete"
    metadata = read_json(data / "metadata.json")
    input_checks, code_checks = {}, {}
    for name, recorded in manifest["input_sha256"].items():
        actual = digest(name)
        require(actual == recorded, f"Input/reference changed: {name}")
        input_checks[name] = actual
    require(str(data / "station_daily.npz") in input_checks, "Data path does not match the frozen input")
    for name, recorded in manifest["code_sha256"].items():
        path = (REPO / name).resolve()
        require(path.is_relative_to(REPO), "Code manifest path escapes Aitken")
        require(digest(path) == recorded, f"Experiment code changed: {name}")
        code_checks[name] = recorded
    protocol, variants = manifest["protocol"], manifest["variants"]
    require(summary["variants"] == variants, "Summary and frozen recipes differ")
    require(protocol["hybrid_alpha"] == .425, "Hybrid alpha differs from the approved value")
    manifest_id = json_digest(manifest)
    raw = np.load(data / "target_pm25.npy", mmap_mode="r")
    weight = np.load(data / "target_weight.npy", mmap_mode="r")
    inside = np.load(data / "gla_mask.npy") > 0
    require(not np.any((weight > 0) & ~np.isfinite(raw)), "Missing target has positive weight")
    require(np.isfinite(weight).all(), "Nonfinite target weight")
    require(not np.any(weight[:, ~inside] > 0), "Observed targets outside London")
    expected_base = [metadata["feature_names"][i] for i in protocol["channels"]]
    reconstructed, members, frames = {}, {}, {}
    prediction_hashes, verified_members = {}, {}
    prep_sets, initialization_sets = defaultdict(set), defaultdict(set)
    for phase in ("screening", "confirmation", "final"):
        declared = summary["phases"][phase]
        folder = run / phase / "predictions"
        saved_paths = {path.name.removesuffix(".csv.gz"): path for path in folder.glob("*.csv.gz")}
        require(set(declared) == set(saved_paths), f"{phase}: declared and saved prediction names differ")
        first, stop = (1095, 1461) if phase == "final" else (730, 1095)
        original = original_rows(data, first, stop)
        require(len(original) == (9619 if phase == "final" else 11102), "Original benchmark count differs")
        reconstructed[phase], members[phase], frames[phase] = {}, {}, {}
        for name, saved in declared.items():
            require(name in variants, f"Unknown variant {name}")
            recipe = variants[name]
            family = recipe["family"]
            csv = pd.read_csv(saved_paths[name], dtype={"site_code": str})
            require(len(csv) == len(original), f"{phase}/{name}: benchmark rows removed or added")
            for column in KEYS:
                require(np.array_equal(csv[column].to_numpy(), original[column].to_numpy()),
                        f"{phase}/{name}: original key/order differs: {column}")
            require(np.allclose(csv.observed_pm25, original.observed_pm25, rtol=1e-7, atol=1e-7),
                    f"{phase}/{name}: observed concentrations changed")
            seeds = [42] if phase == "screening" else list(SEEDS)
            require(saved["seeds"] == seeds, f"{phase}/{name}: wrong seed set")
            require(set(saved["member_metrics"]) == {str(seed) for seed in seeds}, "Missing or extra member metrics")
            cache_phase = "final" if phase == "final" else "selection"
            z_members, member_scores = [], {}
            for seed in seeds:
                home = run / cache_phase / name / f"seed_{seed}"
                contract = read_json(home / "contract.json")
                completed = read_json(home / "completed.json")
                pred_record = read_json(home / "prediction.json")
                contract_id = json_digest(contract)
                checkpoint_sha = digest(home / "final_ema_checkpoint.pt")
                require(contract["manifest"] == manifest_id and contract["variant"] == name,
                        "Member manifest/variant mismatch")
                require(contract["recipe"] == recipe and contract["seed"] == seed, "Member recipe or seed differs")
                require(completed["contract"] == contract_id and completed["checkpoint_sha256"] == checkpoint_sha,
                        "Checkpoint identity/hash differs")
                require(contract["schedule"] == protocol[family + "_schedules"][str(seed)], "Frozen schedule changed")
                require(completed["epochs"] == len(contract["schedule"]), "Training duration incomplete")
                require(completed["added_channels_zero"], "Added channels were not zero initialized")
                initialization_sets[(cache_phase, family, seed)].add(completed["common_parameter_hash"])
                prep = contract["prep"]
                require(prep["train_stop_exclusive"] == first, "Wrong preprocessing training window")
                require(prep["feature_names"] == expected_base, "Base predictors changed")
                require(prep["channels"] == protocol["channels"], "Base channel positions changed")
                prep_sets[(cache_phase, family)].add(json_digest(prep))
                training_valid = np.isfinite(raw[:first]) & (weight[:first] > 0)
                expected_scale = float(np.asarray(raw[:first][training_valid], np.float64).std())
                require(np.isclose(contract["raw_scale"], expected_scale, rtol=1e-12), "Auxiliary target scale uses wrong targets")
                names = contract["feature_names"]
                require(names[:len(expected_base)] == expected_base, "Base input ordering changed")
                hgb_channels = [entry for entry in names if "hgb" in entry.lower()]
                if family == "standalone":
                    require(not hgb_channels and contract["baseline_hash"] is None, "Standalone received HGB input")
                else:
                    require(hgb_channels == ["fixed_hgb_z"] and names[-1] == "fixed_hgb_z", "Hybrid HGB input changed")
                    require(contract["baseline_hash"] == protocol["references"][cache_phase + "_baseline"]["sha256"],
                            "Hybrid does not use frozen archived baseline")
                    scale = protocol["hybrid_target_scalers"][cache_phase]
                    require(prep["target_log_mean"] == scale["mean"] and prep["target_log_std"] == scale["std"],
                            "Hybrid baseline and target scaling differ")
                require(contract["alpha"] == .425, "Residual multiplier changed")
                scaler = contract["feature_scaler"]
                require(scaler["train_stop"] == first, "Temporal scaler fitted on wrong window")
                require(contract["feature_array_hash"] == scaler["scaled_hash"], "Feature identity differs")
                require(pred_record["contract"] == dict(training_contract=contract_id,
                        checkpoint_sha256=checkpoint_sha, first=first, stop=stop), "Inference contract differs")
                prediction_file = home / "prediction_z.npy"
                require(digest(prediction_file) == pred_record["sha256"], "Member prediction hash changed")
                z = np.load(prediction_file, allow_pickle=False)
                require(z.shape == (stop - first, 48, 64) and np.isfinite(z).all(), "Member array shape/value invalid")
                member_frame = original.copy()
                member_frame["predicted_pm25"] = take(decode(z, prep), original, first)
                member_scores[str(seed)] = score(member_frame)
                equal_metrics(member_scores[str(seed)], saved["member_metrics"][str(seed)], f"{phase}/{name}/{seed}")
                z_members.append(z)
                verified_members[str(home.relative_to(run))] = dict(
                    contract_sha256=contract_id, checkpoint_sha256=checkpoint_sha,
                    prediction_sha256=pred_record["sha256"])
            ensemble_maps = decode(np.mean(z_members, axis=0), prep)
            map_file = folder / f"{name}_maps.npy"
            require(np.array_equal(np.load(map_file, allow_pickle=False), ensemble_maps), "Saved ensemble map differs from members")
            exact = original.copy()
            exact["predicted_pm25"] = take(ensemble_maps, original, first)
            require(np.allclose(csv.predicted_pm25, exact.predicted_pm25, rtol=1e-7, atol=1e-7), "CSV predictions differ from member ensemble")
            calculated = score(exact)
            equal_metrics(calculated, saved, f"{phase}/{name}")
            # Summary metrics are checked at exact original-label precision.
            # The runner's reported bootstrap is calculated after CSV reload;
            # reproduce that verified serialization precision, not a subtly
            # different interval from unrounded float32 observations.
            reconstructed[phase][name], members[phase][name], frames[phase][name] = calculated, member_scores, csv
            prediction_hashes[str(saved_paths[name].relative_to(run))] = digest(saved_paths[name])
    require(all(len(values) == 1 for values in initialization_sets.values()), "Common initialization differs across treatments")
    require(all(len(values) == 1 for values in prep_sets.values()), "Preprocessing differs within a matched family/phase")
    completed_paths = {str(path.parent.relative_to(run)) for path in run.glob("*/**/completed.json")}
    if full:
        require(completed_paths == set(verified_members), "Unexpected or unevaluated completed checkpoint")
    else:
        require(set(verified_members).issubset(completed_paths), "Evaluated member has no completion record")

    decisions, detailed_gates = {}, {}
    screening = reconstructed["screening"]
    frozen_screen = read_json(run / "screen_selection_frozen.json") if (run / "screen_selection_frozen.json").exists() else {}
    frozen_selection = read_json(run / "selection_frozen.json") if (run / "selection_frozen.json").exists() else {}
    if full:
        require(set(summary["selection"]) == {"standalone", "hybrid"}, "Missing completed family decision")
    for family in ("standalone", "hybrid"):
        refs = control_names(family)
        required_screen = refs + [f"{family}_{suffix}" for suffix in ("ewma", "median", "final_loss")]
        if not set(required_screen).issubset(screening):
            require(not full and family not in summary["selection"], "Selection made from incomplete screen")
            continue
        baseline_scores = [screening[name] for name in refs]
        combined_allowed = all(passes(screening[f"{family}_{suffix}"], baseline_scores)
                               for suffix in ("ewma", "final_loss"))
        combined_exists = f"{family}_combined" in screening
        require(not combined_exists or combined_allowed, "Combined treatment ran without both individual treatments passing")
        if full:
            require(combined_exists == combined_allowed, "Eligible combined experiment missing")
        candidate_names = [name for name in screening if variants[name]["family"] == family
                           and not variants[name]["is_control"]]
        detailed_gates[family] = {name: {ref: gates(screening[name], screening[ref]) for ref in refs}
                                  for name in candidate_names}
        eligible = [name for name in candidate_names if passes(screening[name], baseline_scores)]
        complexity = {"final_loss": 0, "median": 2, "ewma": 3, "combined": 4}
        selected = min(eligible, key=lambda name: (screening[name]["rmse"],
                       screening[name]["bands"][BANDS[2]]["mae"],
                       complexity[name.removeprefix(family + "_")], name)) if eligible else None
        decision = summary["selection"].get(family)
        if decision is None:
            require(not full, "Completed screening decision missing")
            continue
        require(decision["screen_selected"] == selected and decision["eligible"] == eligible,
                "Screening selection does not follow frozen metric gates/ranking")
        require(decision["combined_allowed"] == combined_allowed, "Combined eligibility record differs")
        if frozen_screen:
            require(frozen_screen[family]["screen_selected"] == selected
                    and frozen_screen[family]["eligible"] == eligible
                    and frozen_screen[family]["combined_allowed"] == combined_allowed
                    and frozen_screen[family]["confirmed"] is False, "Frozen screening decision changed")
        confirmation, improved, confirmed = reconstructed["confirmation"], [], False
        if selected and set(refs + [selected]).issubset(confirmation):
            for seed in SEEDS:
                if all(members["confirmation"][selected][str(seed)]["rmse"]
                       < members["confirmation"][ref][str(seed)]["rmse"] for ref in refs):
                    improved.append(seed)
            confirmed = passes(confirmation[selected], [confirmation[ref] for ref in refs]) and len(improved) >= 2
            require(decision["rmse_improved_seeds"] == improved, "Per-seed confirmation differs")
        elif selected:
            require(not full, "Completed run omitted selected candidate confirmation")
        require(decision["confirmed"] == confirmed, "Confirmation decision violates original guards")
        actual_final = [name for name in reconstructed["final"] if variants[name]["family"] == family]
        if not confirmed:
            require(not actual_final, "2024 evaluated for an unconfirmed family")
            require(not any(path.parts[-3] == name for path in run.glob("final/*/seed_*/completed.json")
                            for name, recipe in variants.items() if recipe["family"] == family),
                    "Final model trained for a failed family")
        elif full:
            require(set(actual_final) == set(refs + [selected]), "Final comparison omits or adds variants")
        decisions[family] = dict(screen_selected=selected, eligible=eligible,
                                combined_allowed=combined_allowed, confirmed=confirmed,
                                rmse_improved_seeds=improved)
    if full:
        require(frozen_selection == summary["selection"], "Frozen final selection differs from summary")
        require(bool(frozen_screen), "Frozen screening decision missing")
    if reconstructed["final"]:
        require(bool(frozen_selection), "2024 results exist without frozen validation decision")
        for path in (run / "final").glob("*/seed_*/final_ema_checkpoint.pt"):
            require((run / "selection_frozen.json").stat().st_mtime <= path.stat().st_mtime,
                    "Final checkpoint predates selection freeze")
    budget = read_json(run / "budget.json")
    require(budget == summary["budget"], "Final budget file and summary disagree")
    require(np.isfinite(budget["used_seconds"]) and budget["used_seconds"] >= 0, "Invalid compute accounting")
    require(0 < budget["limit_seconds"] <= 7200 and budget["limit_seconds"] == manifest["training_limit_seconds"],
            "Budget limit exceeds approved two hours or differs from manifest")
    prior = manifest.get("prior_budget")
    if prior:
        require(digest(prior["path"]) == prior["sha256"], "Prior stopped-run compute record changed")
        require(read_json(prior["path"])["used_seconds"] == prior["used_seconds"], "Carried compute differs")
        require(budget["used_seconds"] >= prior["used_seconds"], "Stopped-run compute not carried")
    if full:
        require(budget["used_seconds"] <= budget["limit_seconds"], "Completed run exceeded approved compute budget")
    intervals_by_phase = {}
    if with_bootstrap or bootstrap_all:
        selected_phases = ("screening", "confirmation", "final") if bootstrap_all else ("final",)
        for phase in selected_phases:
            intervals_by_phase[phase] = {}
            for name, frame in frames[phase].items():
                if variants[name]["is_control"]:
                    continue
                for ref in control_names(variants[name]["family"]):
                    require(ref in frames[phase], f"{phase}: paired control missing for bootstrap")
                    key = f"{name}_minus_{ref}"
                    value = bootstrap(frame, frames[phase][ref])
                    recorded = summary.get("intervals", {}).get(phase, {}).get(key)
                    if recorded is not None:
                        equal_metrics(value, recorded, f"interval/{phase}/{key}")
                    else:
                        require(not full, f"Completed run omitted recorded interval {phase}/{key}")
                    intervals_by_phase[phase][key] = dict(
                        **value, verified_against_runner=recorded is not None,
                        interpretation=(
                            "Conditional, descriptive selection-year comparison; not confirmatory evidence or an untouched test."
                            if phase != "final" else
                            "Conditional retrospective benchmark interval; not an untouched test and excludes selection/seed uncertainty."))
    require(digest(summary_path) == initial_summary_hash, "Run changed while being audited")
    return dict(
        audit_status="passed", run_status=summary["status"], source_run=str(run.relative_to(REPO)),
        auditor_sha256=digest(__file__), manifest_sha256=digest(manifest_path), summary_sha256=initial_summary_hash,
        original_data_labels_and_rows_verified=True, input_and_reference_hashes_verified=input_checks,
        code_hashes_verified=code_checks, saved_prediction_hashes=prediction_hashes,
        completed_members_verified=verified_members,
        common_initialization_record_matches=True, family_preprocessing_matches=True,
        recomputed_metrics=reconstructed, recomputed_decisions=decisions, screening_guard_details=detailed_gates,
        independently_recomputed_intervals_by_phase=intervals_by_phase,
        independently_recomputed_final_intervals=intervals_by_phase.get("final", {}),
        budget=budget, within_nominal_budget=budget["used_seconds"] <= budget["limit_seconds"],
        limitations=[
            "No retraining or checkpoint inference was performed; predictions were reconstructed from saved member arrays.",
            "Common initialization is verified against recorded parameter hashes, not independently regenerated training histories.",
            "Hashes verify current agreement with frozen records; timestamps do not independently authenticate protocol chronology.",
            "A budget stop may overrun by one batch. Without per-batch timing logs, that duration cannot be independently bounded.",
            "2024 remains a previously examined retrospective benchmark. Bootstrap intervals condition on fitted models.",
            "Screening/confirmation intervals describe reused 2023 selection data; they do not provide confirmatory evidence or correct for trying multiple candidates.",
            "Metrics use exact original NPZ labels; interval replication uses saved CSV precision after original-row/label checks (rtol=atol=1e-7).",
        ])


def self_test():
    """No file writes and no model/data imports: simple arithmetic and gate checks."""
    example = pd.DataFrame(dict(site_code=["A"] * 8,
        date=pd.date_range("2023-01-01", periods=8).strftime("%Y-%m-%d"),
        observed_pm25=[10., 12., 30., 9., 20., 25., 8., 10.],
        predicted_pm25=[9., 11., 20., 10., 18., 23., 8., 10.]))
    result = score(example)
    require(result["n"] == 8 and result["bands"][BANDS[2]]["n"] == 2, "Band self-test failed")
    require(result["rapid_changes"]["unavailable_previous_day_n"] == 1, "Missing previous-day self-test failed")
    perfect = example.copy()
    perfect["predicted_pm25"] = perfect.observed_pm25
    require(passes(score(perfect), [result]), "Improvement gate self-test failed")
    require(not passes(result, [result]), "Strict improvement self-test failed")
    interval = bootstrap(perfect, example, n=50)
    require(interval["rmse_delta"] < 0 and interval["ci95"][1] < 0, "Bootstrap self-test failed")
    print("Independent auditor self-tests passed; no files written.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--output", default=str(HERE / "audit.json"))
    parser.add_argument("--bootstrap-final", action="store_true", help="Independently repeat 5,000 paired seven-day resamples")
    parser.add_argument("--bootstrap-all", action="store_true",
                        help="Repeat and verify 5,000 paired seven-day resamples for every available phase; selection-year intervals are descriptive only")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.run_dir or not args.data_dir:
        parser.error("--run-dir and --data-dir are required unless --self-test is used")
    destination = Path(args.output).resolve()
    require(destination.is_relative_to(HERE) and destination.suffix.lower() == ".json",
            "Audit output must be a JSON file inside comparison/temporal_peak_sensitivity")
    try:
        result = audit(args.run_dir, args.data_dir, args.bootstrap_final, args.bootstrap_all)
    except Exception as exc:
        result = dict(audit_status="failed", error=f"{type(exc).__name__}: {exc}",
                      auditor_sha256=digest(__file__))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"audit_status": result["audit_status"], "output": str(destination),
                      "error": result.get("error")}, indent=2))
    return 0 if result["audit_status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
