"""Publish only derived model grids and aggregate scores, never raw station rows.

Reads the existing London bundle; does not train, infer new dates or modify it.
The dashboard renders 2024 maps on demand from this frozen prediction archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECS = [
    ("hybrid", "Hybrid HGB + residual U-Net", "Hybrid · 3 U-Net seeds", "Residual U-Net (3-seed)", "#16796d", True),
    ("hgb", "HGB", "Unconstrained gradient boosting", "HGB (unconstrained)", "#5c6c86", True),
    ("hgb-monotonic", "HGB · limited monotonic", "Constrained gradient boosting", "HGB (limited monotonic)", "#8190a7", True),
    ("xgboost", "XGBoost", "Gradient-boosted tree baseline", "XGBoost", "#b68a58", False),
    ("ann", "ANN", "Five-seed neural ensemble", "ANN (exact sklearn, 5-seed)", "#927d9c", False),
    ("unet-ensemble", "Standalone U-Net · ensemble", "Three U-Nets · no HGB", "Standalone U-Net (3-seed)", "#3c93b8", True),
    ("unet", "Standalone U-Net", "Single network · pre-specified seed 42", "Standalone U-Net (seed 42)", "#6c9daf", True),
]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def inverse(z, mean, std):
    return np.clip(np.expm1(np.clip(z * std + mean, 0., np.log1p(500.))), 0., 500.)


def export(bundle: Path, destination: Path):
    standalone = bundle / "artifacts/london_standalone_unet_20260906"
    hybrid = bundle / "artifacts/london_3year_unet_refit_20260905_142452"
    trees = bundle / "artifacts/london_3year_tree_refit_20260905_141738"
    processed = bundle / "data/processed/london_1km_daily"
    source_files = []

    def source(path):
        source_files.append({"path": path.relative_to(bundle).as_posix(), "sha256": sha(path)})
        return path

    metadata = json.loads(source(processed / "metadata.json").read_text())
    metrics = json.loads(source(standalone / "metrics.json").read_text())["models"]
    mask = np.load(source(processed / "gla_mask.npy")) > 0
    assert mask.shape == (48, 64) and mask.sum() == 1719
    indices = np.flatnonzero(mask.ravel())
    geometry = {"rows": 48, "cols": 64, "indices": indices.tolist(), "crs": metadata["crs"],
                "transform": metadata["transform_gdal"], "cellSizeM": 1000}
    write_json(destination / "geometry.json", geometry)
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    station = pd.read_csv(source(standalone / "test_2024_laqn_all_models.csv.gz"))
    assert len(station) == 9619 and station.date.nunique() == 349
    ids = ["date_idx", "site_code", "row", "col"]
    assert not station.duplicated(ids).any()
    # Scores retain exact canonical observations; no station identities are exported.
    published_models = []
    for key, label, detail, column, color, has_maps in MODEL_SPECS:
        m = metrics[column]
        y = station.observed_pm25.to_numpy(dtype=float)
        p = station[column].to_numpy(dtype=float)
        assert np.isclose(np.sqrt(np.mean((p-y)**2)), m["rmse"], rtol=0, atol=1e-9)
        published_models.append({"id": key, "label": label, "detail": detail, "color": color,
                                "hasMaps": has_maps, "metrics": m})
    contract = json.loads(source(trees / "metrics.json").read_text())["contract"]
    prep = json.loads(source(standalone / "preprocessing_final_2021_2023.json").read_text())
    maps = {
        "hybrid": np.load(source(hybrid / "test_2024_unet_ensemble_maps_float16.npy")).astype(np.float32),
        "hgb": inverse(np.load(source(trees / "hgb_unconstrained_baseline_normalised.npy"))[1095:].astype(np.float32), contract["target_log_mean"], contract["target_log_std"]),
        "hgb-monotonic": inverse(np.load(source(trees / "hgb_monotonic_pm_lags_baseline_normalised.npy"))[1095:].astype(np.float32), contract["target_log_mean"], contract["target_log_std"]),
        "unet-ensemble": np.load(source(standalone / "test_2024_standalone_ensemble_pm25_float32.npy")).astype(np.float32),
        "unet": inverse(np.load(source(standalone / "seed_42/test_2024_direct_z_float32.npy")), prep["target_log_mean"], prep["target_log_std"]),
    }
    daily = {d.strftime("%Y-%m-%d"): {"n": 0, "scores": {}, "maps": {}} for d in dates}
    for date, rows in station.groupby("date", sort=True):
        daily[date]["n"] = len(rows)
        for key, _, _, column, _, _ in MODEL_SPECS:
            error = rows[column].to_numpy(dtype=float)-rows.observed_pm25.to_numpy(dtype=float)
            daily[date]["scores"][key] = {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}
    chunks = {}
    outputs = []
    for key, values in maps.items():
        assert values.shape == (366, 48, 64), (key, values.shape)
        inside = values.reshape(366, -1)[:, indices].astype("<f4")
        assert np.isfinite(inside).all(), f"Non-finite in-London prediction: {key}"
        assert inside.min() >= 0 and inside.max() <= 500
        chunks[key] = {}
        for month in range(1, 13):
            selected = dates.month == month
            path = destination / "grids" / f"{key}-2024-{month:02d}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            inside[selected].tofile(path)
            chunks[key][f"2024-{month:02d}"] = {"url": f"data/grids/{path.name}", "days": int(selected.sum()), "bytes": path.stat().st_size, "sha256": sha(path)}
            outputs.append({"path": path.relative_to(destination).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
            assert np.array_equal(np.fromfile(path, dtype="<f4").reshape(-1, len(indices)), inside[selected])
        for date, day_values in zip(dates, inside):
            daily[date.strftime("%Y-%m-%d")]["maps"][key] = {
                "min": float(day_values.min()), "max": float(day_values.max()),
                "mean": float(day_values.mean()), "median": float(np.median(day_values)),
                "p90": float(np.percentile(day_values, 90)),
            }
    catalog = {"version": 1, "title": "Aitken · London PM2.5", "training": "2021–2023",
               "start": "2024-01-01", "end": "2024-12-31", "defaultDate": "2024-07-15",
               "labelEnd": "2024-12-14", "days": 366, "labelledDays": 349, "stationDays": 9619,
               "cells": 1719, "models": published_models, "chunks": chunks,
               "mode": "Frozen retrospective prediction archive; not live forecasting",
               "encoding": "little-endian float32; day-major then geometry.indices order; units µg/m³",
               "mapMetricsNote": "Day scores use saved unrounded station predictions; display grids may be stored at float16 precision in the original hybrid archive.",
               "limitations": ["2024 was inspected in earlier experiments; this is a retrospective benchmark.",
                               "Monitoring-site evaluation does not establish accuracy in every grid cell.",
                               "The hybrid's small RMSE advantage over HGB is not decisive under the descriptive day bootstrap.",
                               "Quarterly satellite composites make this retrospective mapping, not a strict future forecast.",
                               "XGBoost and ANN are included in performance comparisons; full-grid archives are not available for these models."]}
    write_json(destination / "catalog.json", catalog)
    write_json(destination / "daily.json", daily)
    write_json(destination / "provenance.json", {"schemaVersion": 1, "modelsRetrained": False,
        "contents": "Derived prediction grids and aggregate scores only. No raw station observations, credentials or model weights.",
        "sourceBundle": "pm25_london_bundle", "sourceFiles": source_files, "gridFiles": outputs,
        "checks": {"stationRows": len(station), "labelledDates": station.date.nunique(), "gridCells": len(indices),
                   "mapModels": len(maps), "mapDates": len(dates), "binaryRoundTripExact": True}})
    print(f"Exported {len(maps)} models × {len(dates)} dates × {len(indices)} cells. {sum(x['bytes'] for x in outputs)/1e6:.2f} MB of exact float32 grid data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=ROOT.parent / "code/pm25_london_bundle")
    parser.add_argument("--output", type=Path, default=ROOT / "dashboard/public/data")
    args = parser.parse_args()
    export(args.bundle.resolve(), args.output.resolve())
