"""Import the four-file AURN supplement and independently audit a local rerun.

The supplied temporal predictions are inputs, not regenerated neural outputs.
All writes stay in Aitken; original observations and collaborator records stay
unchanged. Archive documents are provenance, not executable instructions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import sklearn

from import_aurn_bundle import REPO, EXPECTED_FEATURE_SHA, inside, sha

PREFIX = "aurn_reproduction_supplement_20260930/"
SCREEN = "artifacts/temporal_tail_correction_20260912/screening/B_temporal_predictions.csv.gz"
FINAL = "artifacts/temporal_tail_correction_20260912/final/B_temporal_predictions.csv.gz"
SELECTION = "artifacts/external_features_aurn/selection_frozen.json"
SUMMARY = "artifacts/external_features_aurn/summary.json"
EXPECTED = {
    SCREEN: "e157b67262b1623810350b340e5f29fd291c864de5f63cd7ddc254aebe094f09",
    FINAL: "adeb406b666a58c9c3fb4bb7fd09a7d77022c4f7ecc8403c11a8c1d219715ee5",
    SELECTION: "45b87a0614c283a6ab80935ed4e9037cd87422f13dc48ae1f250d76a3afc1b43",
    SUMMARY: "002d5017409b75fa2fead98819ddba14174d71a8eda82df52829f6a4c457819a",
}


def supplement_payload(archive: zipfile.ZipFile) -> dict[str, bytes]:
    """Validate the whole archive before returning any payload for installation."""
    allowed = set(EXPECTED) | {"README.md", "CHECKSUMS.sha256"}
    content = {}
    total = 0
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if (path.is_absolute() or ".." in path.parts or ":" in member.filename
                or "\\" in member.filename or stat.S_ISLNK(member.external_attr >> 16)
                or not member.filename.startswith(PREFIX)):
            raise ValueError(f"Unsafe supplement member: {member.filename}")
        if member.is_dir():
            continue
        relative = member.filename[len(PREFIX):]
        if relative not in allowed or relative in content:
            raise ValueError(f"Unexpected or duplicate supplement member: {relative}")
        total += member.file_size
        if total > 5_000_000:
            raise ValueError("Unexpected supplement size")
        content[relative] = archive.read(member)
    if set(content) != allowed:
        raise ValueError("Supplement is incomplete")
    declared = {}
    for line in content["CHECKSUMS.sha256"].decode("utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if name in declared:
            raise ValueError("Duplicate checksum record")
        declared[name] = digest
    if declared != EXPECTED:
        raise ValueError("Supplement manifest differs from the recorded experiment")
    for name, digest in EXPECTED.items():
        if hashlib.sha256(content[name]).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
    return content


def install_supplement(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        content = supplement_payload(archive)
    planned = []
    for name, data in content.items():
        relative = name if name in EXPECTED else "artifacts/aurn_supplement_received_20260930/" + name
        target = inside(REPO / relative)
        if target.exists() and (not target.is_file() or target.read_bytes() != data):
            raise ValueError(f"Refusing to replace different existing data: {target}")
        planned.append((target, data))
    for target, data in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(data)
    return {"archive_sha256": sha(path), "verified_payload_sha256": EXPECTED,
            "payload_files": len(EXPECTED), "different_existing_files_overwritten": False}


def validate_rows(frame: pd.DataFrame, station, first: int, last: int) -> dict:
    """Check full evaluation membership, identity, dates and unchanged targets."""
    use = (station["source"] == 1) & (station["date_idx"] >= first) & (station["date_idx"] <= last)
    expected_index = np.flatnonzero(use)
    for column in ("station_row_index", "date_idx", "row", "col"):
        if not np.array_equal(frame[column], frame[column].astype(np.int64)):
            raise ValueError(f"Non-integral {column}")
    idx = frame.station_row_index.to_numpy(np.int64)
    if frame.duplicated(["site_code", "date_idx"]).any() or frame.station_row_index.duplicated().any():
        raise ValueError("Duplicate station/day or row ID")
    np.testing.assert_array_equal(np.sort(idx), expected_index)
    for column in ("date_idx", "row", "col"):
        np.testing.assert_array_equal(frame[column], station[column][idx])
    np.testing.assert_array_equal(frame.site_code, station["site_codes"][station["site_idx"][idx]])
    np.testing.assert_array_equal(pd.to_datetime(frame.date).to_numpy(),
                                  (pd.Timestamp("2021-01-01") + pd.to_timedelta(frame.date_idx, unit="D")).to_numpy())
    error = np.abs(frame.observed_pm25.to_numpy(float) - station["value"][idx].astype(float))
    np.testing.assert_allclose(frame.observed_pm25, station["value"][idx], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(frame.observed_pm25.to_numpy(np.float32), station["value"][idx])
    if not np.isfinite(frame[["observed_pm25", "predicted_pm25"]]).all().all():
        raise ValueError("Non-finite observations/predictions")
    return {"rows": len(frame), "days": int(frame.date_idx.nunique()),
            "sites": int(frame.site_code.nunique()), "first_date": str(frame.date.min().date()),
            "last_date": str(frame.date.max().date()), "complete_original_laqn_membership": True,
            "observations_exactly_match_original_float32": True,
            "duplicate_station_days": 0, "max_target_serialization_difference": float(error.max()),
            "target_comparison_absolute_tolerance": 1e-5}


def check_tree(actual, expected, path="root", tolerance=5e-5) -> float:
    """Compare all recorded recipe decisions, trials, fits and metrics recursively."""
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"Different record keys: {path}")
        return max((check_tree(actual[k], v, f"{path}.{k}", tolerance) for k, v in expected.items()), default=0.)
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Different array length: {path}")
        return max((check_tree(a, b, f"{path}[{i}]", tolerance) for i, (a, b) in enumerate(zip(actual, expected))), default=0.)
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not np.isfinite(actual) or abs(actual-expected) > tolerance:
            raise ValueError(f"Numeric mismatch at {path}: {actual} != {expected}")
        return float(abs(actual-expected))
    if actual != expected:
        raise ValueError(f"Mismatch at {path}: {actual} != {expected}")
    return 0.


def independent_metrics(observed, prediction) -> dict:
    y, p = np.asarray(observed, float), np.asarray(prediction, float)
    err = p-y
    return {"n": len(y), "mae": float(np.mean(abs(err))), "rmse": float(np.sqrt(np.mean(err**2))),
            "r2": float(1-np.sum(err**2)/np.sum((y-y.mean())**2)), "bias": float(err.mean())}


def independent_interval(frame: pd.DataFrame) -> list[float]:
    """Calendar moving-block bootstrap, preserving the published RNG protocol."""
    count = int(frame.date_idx.max()-frame.date_idx.min()+1)
    per_day = np.zeros((count, 3))
    offset = frame.date_idx.to_numpy(int)-int(frame.date_idx.min())
    np.add.at(per_day[:, 0], offset, 1)
    np.add.at(per_day[:, 1], offset, (frame.control_pm25-frame.observed_pm25)**2)
    np.add.at(per_day[:, 2], offset, (frame.predicted_pm25-frame.observed_pm25)**2)
    rng = np.random.default_rng(20260930)
    starts = rng.integers(0, count-6, size=(5000, int(np.ceil(count/7))))
    sampled = (starts[..., None]+np.arange(7)).reshape(5000, -1)[:, :count]
    totals = per_day[sampled].sum(axis=1)
    differences = np.sqrt(totals[:, 2]/totals[:, 0])-np.sqrt(totals[:, 1]/totals[:, 0])
    return np.quantile(differences, [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO / "artifacts/aurn_reproduction_20260930")
    args = parser.parse_args()
    output = inside(args.output_dir)
    supplied = [REPO / name for name in EXPECTED]
    if any(output == p.parent or output.is_relative_to(p.parent) for p in supplied):
        raise ValueError("Rerun outputs must be separate from supplied records")
    report = {"scope": "AURN correction reproduced from shared temporal predictions, not neural retraining",
              "import": install_supplement(args.archive)}
    features = REPO / "data/external_features/aurn/aurn_features.npz"
    if sha(features) != EXPECTED_FEATURE_SHA:
        raise ValueError("AURN features changed")
    data = args.bundle.resolve() / "data/processed/london_1km_daily"
    reference = [data / name for name in ("features_float16.npy", "target_pm25.npy", "target_weight.npy",
                 "laqn_cell_target.npy", "breathe_cell_target.npy", "station_daily.npz", "metadata.json")]
    before = {p.name: sha(p) for p in reference}
    with np.load(data / "station_daily.npz", allow_pickle=False) as station:
        screening = pd.read_csv(REPO / SCREEN, parse_dates=["date"])
        final = pd.read_csv(REPO / FINAL, parse_dates=["date"])
        report["screening_rows"] = validate_rows(screening, station, 730, 1094)
        report["benchmark_rows"] = validate_rows(final, station, 1095, 1443)
    output.mkdir(parents=True, exist_ok=True)
    runner = REPO / "experiments/external_features/run_aurn_correction.py"
    command = [sys.executable, "-B", str(runner), "--features", str(features),
               "--selection-predictions", str(REPO / SCREEN), "--test-predictions", str(REPO / FINAL),
               "--output-dir", str(output)]
    print("Observation identity checks passed; rerunning unchanged AURN selection/correction.", flush=True)
    with (output / "rerun.stdout.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    original_summary = json.loads((REPO / SUMMARY).read_text())
    local_summary = json.loads((output / "summary.json").read_text())
    original_selection = json.loads((REPO / SELECTION).read_text())
    local_selection = json.loads((output / "selection_frozen.json").read_text())
    selection_difference = check_tree(local_selection, original_selection)
    # A JSON hash can differ across platforms because float32 Ridge arithmetic
    # differs slightly. Compare every recipe decision and value, not just hashes.
    summary_difference = check_tree({k:v for k,v in local_summary.items() if k != "inputs"},
                                    {k:v for k,v in original_summary.items() if k != "inputs"})
    for key in ("features_sha256", "selection_predictions_sha256", "test_predictions_sha256"):
        if local_summary["inputs"][key] != original_summary["inputs"][key]:
            raise ValueError(f"Different source inputs: {key}")
    predictions = pd.read_csv(output / "aurn_same_day_2024_predictions.csv.gz", parse_dates=["date"])
    np.testing.assert_array_equal(predictions.station_row_index, final.station_row_index)
    np.testing.assert_allclose(predictions.observed_pm25, final.observed_pm25, rtol=0, atol=1e-12)
    np.testing.assert_allclose(predictions.control_pm25, final.predicted_pm25, rtol=0, atol=1e-12)
    control = independent_metrics(predictions.observed_pm25, predictions.control_pm25)
    corrected = independent_metrics(predictions.observed_pm25, predictions.predicted_pm25)
    for metrics, expected in ((control, original_summary["control"]),
                              (corrected, original_summary["sources"]["aurn_same_day"]["metrics"])):
        check_tree(metrics, {k:expected[k] for k in metrics}, tolerance=1e-5)
    interval = independent_interval(predictions)
    np.testing.assert_allclose(interval, original_summary["sources"]["aurn_same_day"]["paired_block_interval"]["ci95"],
                               rtol=0, atol=1e-5)
    march = predictions[predictions.date.eq("2024-03-11")]
    episode = {"date": "2024-03-11", "n": len(march), "observed_mean": float(march.observed_pm25.mean()),
               "control_mean": float(march.control_pm25.mean()), "corrected_mean": float(march.predicted_pm25.mean()),
               "control": independent_metrics(march.observed_pm25, march.control_pm25),
               "corrected": independent_metrics(march.observed_pm25, march.predicted_pm25)}
    high = predictions[predictions.observed_pm25.ge(25)]
    report.update({"performance_correction_reproduced": True, "temporal_neural_outputs_regenerated": False,
                   "control": control, "corrected": corrected,
                   "bands": local_summary["sources"]["aurn_same_day"]["metrics"]["bands"],
                   "independent_7day_5000_bootstrap_ci95": interval, "march_11_episode": episode,
                   "high_band_underpredictions": {"n": len(high),
                       "control": int((high.control_pm25 < high.observed_pm25).sum()),
                       "corrected": int((high.predicted_pm25 < high.observed_pm25).sum())},
                   "selection_all_decisions_match": True, "selection_max_numeric_difference": selection_difference,
                   "summary_max_numeric_difference": summary_difference, "comparison_atol": 5e-5,
                   "selection_json_byte_identical": sha(output / "selection_frozen.json") == EXPECTED[SELECTION],
                   "summary_json_byte_identical": sha(output / "summary.json") == EXPECTED[SUMMARY],
                   "runner_sha256": sha(runner), "audit_script_sha256": sha(Path(__file__)),
                   "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                                "scikit_learn": sklearn.__version__},
                   "official_model_changed": False, "wind_scaling_issue_fixed": False,
                   "original_data_sha256": before})
    after = {p.name: sha(p) for p in reference}
    if before != after or any(sha(REPO / name) != digest for name, digest in EXPECTED.items()):
        raise ValueError("Original or supplied reference changed during reproduction")
    report["original_data_and_supplied_records_unchanged"] = True
    (output / "reproduction_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"import", "original_data_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
