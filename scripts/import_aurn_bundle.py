"""Import a shared AURN ZIP unchanged, then audit it without fitting models.

All output stays in Aitken. The original London data and archive are read-only.
Mac metadata is skipped; different existing destination bytes are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments/external_features"))
from aurn import (  # noqa: E402
    build_daily_features, compile_daily, haversine_km,
    load_station_catalogue, select_regional_sites,
)

EXPECTED_FEATURE_SHA = "da2d3f5e4a9ededc28b6a738b1a9f17a6bc85e7720e0f4f7c36cea2063b16cda"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(path: Path) -> Path:
    result = path.resolve()
    if result == REPO or not result.is_relative_to(REPO):
        raise ValueError("All output must resolve inside Aitken")
    return result


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    kept = []
    seen = set()
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.parts and path.parts[0] == "__MACOSX":
            continue
        if (not path.parts or path.is_absolute() or path.parts[0] != "aurn"
                or ".." in path.parts or "\\" in member.filename
                or ":" in member.filename or stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError(f"Unsafe ZIP member: {member.filename}")
        if member.is_dir():
            continue
        if len(path.parts) < 2 or path.suffix.lower() not in {".csv", ".json", ".npz", ".md"}:
            raise ValueError(f"Unexpected payload: {member.filename}")
        relative = PurePosixPath(*path.parts[1:])
        key = str(relative).casefold()
        if key in seen:
            raise ValueError(f"Duplicate ZIP destination: {relative}")
        seen.add(key)
        kept.append(member)
    if sum(x.file_size for x in kept) > 250_000_000:
        raise ValueError("Unexpectedly large AURN archive")
    required = {"manifest.json", "aurn_sites.json", "aurn_features.npz", "selected_sites.csv",
                "coverage.csv", "aurn_daily.csv", "daily_availability.csv"}
    if not required.issubset(seen):
        raise ValueError("The archive is missing required AURN files")
    return kept


def install(archive_path: Path, destination: Path) -> dict:
    destination = inside(destination)
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive)
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        feature_hash = hashlib.sha256(archive.read("aurn/aurn_features.npz")).hexdigest()
        if feature_hash != EXPECTED_FEATURE_SHA:
            raise ValueError("Feature archive differs from the friend's published experiment hash")
        planned = []
        hashes = {}
        for member in members:
            relative = Path(*PurePosixPath(member.filename).parts[1:])
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("ZIP destination escapes import directory")
            content = archive.read(member)
            digest = hashlib.sha256(content).hexdigest()
            if target.exists() and (not target.is_file() or sha(target) != digest):
                raise ValueError(f"Refusing to replace different existing bytes: {target}")
            planned.append((target, content))
            hashes[relative.as_posix()] = digest
        # Verify the sender's raw-file manifest before making any dataset changes.
        manifest = json.loads(archive.read("aurn/manifest.json"))
        assert hashes["aurn_sites.json"] == manifest["catalogue"]["sha256"]
        successful = [r for r in manifest["downloads"] if r["status"] in {"cached", "downloaded"}]
        for record in successful:
            key = "raw/" + PurePosixPath(record["path"].replace("\\", "/")).name
            assert hashes[key] == record["sha256"], key
        for target, content in planned:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(content)
    return {"archive_sha256": sha(archive_path), "payload_files": len(hashes),
            "payload_bytes": sum(len(b) for _, b in planned), "file_sha256": hashes,
            "manifest_raw_hashes_verified": len(successful),
            "mac_metadata_extracted": False, "different_existing_files_overwritten": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=REPO / "data/external_features/aurn")
    parser.add_argument("--audit-dir", type=Path, default=REPO / "artifacts/aurn_import_20260930")
    args = parser.parse_args()
    output, destination = inside(args.audit_dir), inside(args.destination)
    bundle = args.bundle.resolve()
    data = bundle / "data/processed/london_1km_daily"
    originals = [data / name for name in ("features_float16.npy", "target_pm25.npy", "target_weight.npy",
                  "laqn_cell_target.npy", "breathe_cell_target.npy", "station_daily.npz", "metadata.json")]
    before = {p.name: sha(p) for p in originals}
    report = {"scope": "Dataset import and offline audit; no model training or promotion",
              "import": install(args.archive, destination)}
    manifest = json.loads((destination / "manifest.json").read_text())
    sites = pd.read_csv(destination / "selected_sites.csv")
    selected = select_regional_sites(load_station_catalogue(destination / "aurn_sites.json"))
    assert sites.site_id.tolist() == selected.site_id.tolist()
    daily = pd.read_csv(destination / "aurn_daily.csv", parse_dates=["date"])
    assert not daily.duplicated(["site_id", "date"]).any()
    assert daily.pm25.between(0, 500).all() and daily.valid_hours.between(18, 24).all()
    rebuilt = compile_daily(destination / "raw", sites, [2021, 2022, 2023, 2024])
    for column in ("date", "site_id", "valid_hours"):
        assert np.array_equal(daily[column].to_numpy(), rebuilt[column].to_numpy()), column
    np.testing.assert_allclose(daily.pm25, rebuilt.pm25, rtol=0, atol=1e-10)
    assert len(daily) == manifest["usable_station_days"] == 16413
    assert daily.site_id.nunique() == manifest["usable_site_count"] == 15
    assert len(sites) == manifest["selected_site_count"] == 18
    # Independently count distinct valid hourly timestamps; the shared parser
    # currently counts rows rather than explicitly removing duplicate hours.
    duplicated_hours = 0
    status_headers = True
    for path in sorted((destination / "raw").glob("*.csv")):
        hourly = pd.read_csv(path, skiprows=4, low_memory=False)
        pm_columns = [i for i, c in enumerate(hourly.columns) if "Hourly measured" in c and "PM<sub>2.5" in c]
        if not pm_columns:
            continue
        col = pm_columns[0]
        status_headers &= col + 1 < len(hourly.columns) and str(hourly.columns[col + 1]).lower().startswith("status")
        valid = pd.to_numeric(hourly.iloc[:, col], errors="coerce").between(0, 500)
        valid &= hourly.iloc[:, col + 1].astype(str).str.strip().str.upper().eq("R")
        valid &= pd.to_datetime(hourly.iloc[:, 0], format="%d-%m-%Y", errors="coerce").notna()
        duplicated_hours += int(hourly.loc[valid].duplicated(list(hourly.columns[:2])).sum())
    assert status_headers and duplicated_hours == 0
    metadata = json.loads((data / "metadata.json").read_text())
    features = np.load(data / "features_float16.npy", mmap_mode="r")
    mask = np.load(data / "gla_mask.npy", allow_pickle=False) > 0
    names = metadata["feature_names"]
    u = np.asarray(features[:, names.index("era5_u10"), mask], np.float32).mean(axis=1)
    v = np.asarray(features[:, names.index("era5_v10"), mask], np.float32).mean(axis=1)
    recomputed, feature_names, availability = build_daily_features(rebuilt, u, v)
    with np.load(destination / "aurn_features.npz", allow_pickle=False) as archive:
        saved = archive["values"].copy()
        assert archive["names"].astype(str).tolist() == feature_names
    assert saved.shape == (1461, 8) and np.isfinite(saved).all()
    np.testing.assert_allclose(saved, recomputed, rtol=0, atol=2e-5)
    np.testing.assert_array_equal(saved[1:, 5:], saved[:-1, :3])
    np.testing.assert_array_equal(saved[0, 5:], 0)
    assert availability.available.eq(1).all()
    report["data_checks"] = {
        "selected_stations": len(sites), "usable_stations": int(daily.site_id.nunique()),
        "valid_station_days": len(daily), "days": len(availability),
        "feature_shape": list(saved.shape), "features": feature_names,
        "raw_hourly_reaggregation_matches_daily": True, "duplicate_valid_hours": duplicated_hours,
        "ratification_status_column_present": bool(status_headers),
        "features_rebuilt_max_abs_difference": float(np.max(np.abs(saved - recomputed))),
        "lag1_exactly_previous_day": True,
        "minimum_sites_per_day": int(availability.valid_site_count.min()),
        "maximum_sites_per_day": int(availability.valid_site_count.max()),
    }
    # Diagnose physical-wind interpretation without modifying the imported data.
    scalers = metadata["feature_scalers_train_only"]
    u_scale, v_scale = scalers["era5_u10"], scalers["era5_v10"]
    physical_u = u * u_scale["std"] + u_scale["mean"]
    physical_v = v * v_scale["std"] + v_scale["mean"]
    corrected, _, _ = build_daily_features(rebuilt, physical_u, physical_v)
    a = np.degrees(np.arctan2(-u, -v))
    b = np.degrees(np.arctan2(-physical_u, -physical_v))
    direction_error = np.abs((a-b+180) % 360 - 180)
    report["wind_diagnostic"] = {
        "issue": "Shared downloader treats standardized ERA5 u/v channels as physical wind components",
        "u_scaler": u_scale, "v_scaler": v_scale,
        "median_direction_difference_degrees": float(np.median(direction_error)),
        "days_direction_difference_above_30_degrees": int((direction_error > 30).sum()),
        "upwind_mean_average_abs_change_ug_m3": float(np.mean(np.abs(saved[:, 2]-corrected[:, 2]))),
        "upwind_mean_max_abs_change_ug_m3": float(np.max(np.abs(saved[:, 2]-corrected[:, 2]))),
        "diagnostic_only": True, "imported_features_changed": False,
        "limitation": "Inverse scaling remains approximate because the original cube is float16 and clipped",
    }
    with np.load(data / "station_daily.npz", allow_pickle=False) as station:
        use = (station["source"] == 1) & (station["date_idx"] >= 1095) & (station["date_idx"] <= 1443)
        eval_sites = set(station["site_codes"][station["site_idx"][use]].tolist())
        assert int(use.sum()) == 9619
    laqn = pd.read_csv(bundle / "metadata/laqn_pm25_sites_gla_2021_2024.csv")
    laqn = laqn[laqn.site_code.isin(eval_sites)]
    assert set(laqn.site_code) == eval_sites
    nearest = []
    for aurn in sites.itertuples():
        distances = [(haversine_km(aurn.latitude, aurn.longitude, x.latitude, x.longitude), x.site_code, x.site_name)
                     for x in laqn.itertuples()]
        distance, code, name = min(distances)
        nearest.append({"aurn_site": aurn.site_id, "usable": aurn.site_id in set(daily.site_id),
                        "nearest_eval_site": code, "nearest_eval_name": name, "distance_km": distance})
    report["evaluation_station_check"] = {
        "evaluated_laqn_sites": len(eval_sites), "evaluated_station_days": 9619,
        "nearest_aurn_to_evaluation_site_km": min(x["distance_km"] for x in nearest),
        "aurn_sites_within_500m_of_evaluation_monitor": [x for x in nearest if x["distance_km"] < .5],
        "scope": "Coordinate proximity checks station co-location; not proof of all upstream independence",
    }
    report["missing_reproduction_files"] = [str(p.relative_to(REPO)).replace("\\", "/") for p in (
        REPO / "artifacts/temporal_tail_correction_20260912/screening/B_temporal_predictions.csv.gz",
        REPO / "artifacts/temporal_tail_correction_20260912/final/B_temporal_predictions.csv.gz",
        REPO / "artifacts/external_features_aurn/selection_frozen.json",
        REPO / "artifacts/external_features_aurn/summary.json",
    ) if not p.exists()]
    after = {p.name: sha(p) for p in originals}
    assert before == after
    report["original_data_unchanged"] = True
    report["original_data_sha256"] = before
    report["performance_independently_reproduced"] = False
    report["official_model_changed"] = False
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(nearest).to_csv(output / "nearest_evaluation_stations.csv", index=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"import", "original_data_sha256"}}, indent=2))
    print(f"Import and audit complete: {destination}")


if __name__ == "__main__":
    main()
