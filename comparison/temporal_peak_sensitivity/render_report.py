"""Render a frozen experiment using verified original numeric precision.

The trainer records exact float32 observations/maps but pandas exports float32
CSV columns with fewer decimal digits. Near-zero signed bias can consequently
fail the frozen report writer's tight relative check despite an unchanged run.
This wrapper validates every CSV value against the original station NPZ and
saved ensemble map at the already-audited float32 round-trip tolerance, then
restores those exact values IN MEMORY. It does not loosen metric checks, edit
the frozen code, rewrite predictions, or alter the source observations.

Run only after the experiment stops and audit_saved_run.py reports success:
  python comparison/temporal_peak_sensitivity/render_report.py --run-dir ... \
    --data-dir ... --audit comparison/temporal_peak_sensitivity/audit.json
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from comparison.temporal_peak_sensitivity.audit_saved_run import (  # noqa: E402
    KEYS, digest, original_rows, require, take,
)
from experiments.temporal_peak_sensitivity import reporting  # noqa: E402

PHASES = ("screening", "confirmation", "final")
RTOL = 1e-7
ATOL = 1e-7


def validate_precision(csv: pd.DataFrame, original: pd.DataFrame,
                       maps: np.ndarray, first: int) -> tuple[pd.DataFrame, dict]:
    """Reject any changed key/order or non-round-trip concentration difference."""
    required = {*KEYS, "observed_pm25", "predicted_pm25"}
    require(required.issubset(csv), "CSV is missing required prediction columns")
    require(len(csv) == len(original), "Prediction row count changed")
    require(not csv.station_row_index.duplicated().any(), "Duplicate CSV row identifier")
    for name in KEYS:
        require(np.array_equal(csv[name].to_numpy(), original[name].to_numpy()),
                f"Original station key/order differs: {name}")
    require(maps.ndim == 3 and np.isfinite(maps).all(), "Nonfinite or incorrectly shaped ensemble map")
    try:
        expected = take(maps, original, first)
    except (IndexError, ValueError) as exc:
        raise ValueError("Ensemble map does not cover prediction rows") from exc
    require(np.all(original.date_idx.to_numpy(int) >= first), "Map start is after evaluation rows")
    y, p = csv.observed_pm25.to_numpy(float), csv.predicted_pm25.to_numpy(float)
    true_y = original.observed_pm25.to_numpy(float)
    true_p = np.asarray(expected, float)
    require(np.isfinite(y).all() and np.isfinite(p).all(), "Nonfinite CSV concentration")
    require(np.allclose(y, true_y, rtol=RTOL, atol=ATOL), "CSV observations differ beyond float32 serialization tolerance")
    require(np.allclose(p, true_p, rtol=RTOL, atol=ATOL), "CSV predictions differ beyond float32 serialization tolerance")
    restored = csv.copy()
    restored["observed_pm25"] = original.observed_pm25.to_numpy(copy=True)
    restored["predicted_pm25"] = expected.copy()
    return restored, {
        "rows": len(csv), "keys_and_order_exact": True,
        "maximum_observed_csv_roundtrip_error": float(np.max(np.abs(y - true_y))) if len(csv) else 0.,
        "maximum_predicted_csv_roundtrip_error": float(np.max(np.abs(p - true_p))) if len(csv) else 0.,
        "relative_tolerance": RTOL, "absolute_tolerance": ATOL,
    }


class PrecisionReader:
    """Scoped replacement: only this run's audited prediction CSVs are restored."""

    def __init__(self, run: Path, data: Path, prediction_hashes: dict):
        self.run = Path(run).resolve()
        self.data = Path(data).resolve()
        self.original_read_csv = pd.read_csv
        self.expected = {str(name): sha for name, sha in prediction_hashes.items()}
        self.rows, self.cache, self.records = {}, {}, {}

    def __call__(self, filepath, *args, **kwargs):
        if not isinstance(filepath, (str, Path)):
            return self.original_read_csv(filepath, *args, **kwargs)
        path = Path(filepath).resolve()
        if not path.is_relative_to(self.run):
            return self.original_read_csv(filepath, *args, **kwargs)
        parts = path.relative_to(self.run).parts
        require(len(parts) == 3 and parts[0] in PHASES and parts[1] == "predictions" and path.name.endswith(".csv.gz"),
                "Only audited phase prediction CSVs can be read inside the selected run")
        key = str(path.relative_to(self.run))
        require(key in self.expected, "Prediction CSV was not included in the completed audit")
        require(digest(path) == self.expected[key], "Prediction CSV changed after audit")
        # The frozen writer uses whole-frame reads. Do not silently change the
        # semantics of a future caller requesting chunks, selected rows or columns.
        require(not args and not kwargs, "Precision restoration requires an unfiltered full CSV read")
        if key not in self.cache:
            phase = parts[0]
            first, stop = (1095, 1461) if phase == "final" else (730, 1095)
            if phase not in self.rows:
                self.rows[phase] = original_rows(self.data, first, stop)
            maps_path = path.with_name(path.name.removesuffix(".csv.gz") + "_maps.npy")
            require(maps_path.resolve().is_relative_to(self.run), "Map path escapes selected run")
            maps_hash = digest(maps_path)
            maps = np.load(maps_path, allow_pickle=False, mmap_mode="r")
            require(maps.shape == (stop-first, 48, 64), "Saved ensemble map shape differs from frozen grid")
            raw = self.original_read_csv(path, dtype={"site_code": str})
            frame, notes = validate_precision(raw, self.rows[phase], maps, first)
            self.cache[key] = frame
            self.records[key] = dict(notes, csv_sha256=self.expected[key],
                                     map_path=str(maps_path.relative_to(self.run)), map_sha256=maps_hash)
        return self.cache[key].copy()

    def verify_unchanged(self):
        for name, record in self.records.items():
            require(digest(self.run / name) == record["csv_sha256"], "Prediction CSV changed during report rendering")
            require(digest(self.run / record["map_path"]) == record["map_sha256"], "Ensemble map changed during report rendering")


@contextmanager
def restored_prediction_reads(reader: PrecisionReader):
    """Always restore pandas.read_csv, including when the report raises."""
    with patch.object(reporting.pd, "read_csv", side_effect=reader):
        yield


def render(run_dir: str | Path, data_dir: str | Path, audit_path: str | Path,
           destination: str | Path = HERE) -> Path:
    run, data, audit_file, dest = map(lambda value: Path(value).resolve(), (run_dir, data_dir, audit_path, destination))
    artifact_root = (REPO / "artifacts" / "temporal_peak_sensitivity").resolve()
    require(run.is_relative_to(artifact_root) and run != artifact_root, "Selected run must be inside the isolated Aitken experiment artifacts")
    require(dest.is_relative_to(HERE), "Curated report must remain inside comparison/temporal_peak_sensitivity")
    require(audit_file.is_relative_to(HERE), "Audit must be inside the comparison folder")
    audit = json.loads(audit_file.read_text(encoding="utf-8"))
    summary_path, manifest_path = run / "summary.json", run / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(summary["status"] in {"complete", "partial_budget_exhausted"}, "Do not render an active or failed run as a completed experiment")
    require(audit.get("audit_status") == "passed", "A successful independent audit is required before rendering")
    require(audit.get("run_status") == summary["status"], "Audit and run statuses differ")
    require(audit.get("source_run") == str(run.relative_to(REPO)), "Audit belongs to a different run")
    summary_hash, manifest_hash = digest(summary_path), digest(manifest_path)
    require(audit.get("summary_sha256") == summary_hash and audit.get("manifest_sha256") == manifest_hash,
            "Summary or manifest changed since independent audit")
    require(audit.get("original_data_labels_and_rows_verified") is True, "Original observations were not independently verified")
    source_path = data / "station_daily.npz"
    source_hash = digest(source_path)
    require(manifest["input_sha256"].get(str(source_path)) == source_hash, "Original station source is not the frozen audited input")
    prediction_hashes = audit.get("saved_prediction_hashes", {})
    reader = PrecisionReader(run, data, prediction_hashes)
    # Validate every read before the writer starts changing the curated report.
    for name in prediction_hashes:
        reader(run / name)
    with restored_prediction_reads(reader):
        report = reporting.write_report(run, dest)
    reader.verify_unchanged()
    require(digest(source_path) == source_hash, "Original station source changed during rendering")
    require(digest(summary_path) == summary_hash and digest(manifest_path) == manifest_hash,
            "Run records changed during rendering")
    notes = {
        "status": "passed", "reason": "Restore validated float32 CSV round-trip precision in memory; frozen metric checks unchanged",
        "source_run": str(run.relative_to(REPO)), "summary_sha256": summary_hash,
        "manifest_sha256": manifest_hash, "independent_audit_sha256": digest(audit_file),
        "station_source_sha256": source_hash, "renderer_sha256": digest(__file__),
        "reads": reader.records, "source_files_modified": False,
        "bootstrap_note": "Any final intervals rendered here use restored exact numeric values; runner summary intervals use CSV round-trip precision. Differences at rounding scale are not new model results.",
    }
    (dest / "render_audit.json").write_text(json.dumps(notes, indent=2, allow_nan=False), encoding="utf-8")
    with report.open("a", encoding="utf-8") as stream:
        stream.write("\n## Numeric-precision rendering note\n\n"
                     "The frozen trainer stores exact float32 observations and prediction maps. CSV export rounds some of those numbers, enough to trip a very tight check of near-zero signed bias. "
                     "After the independent audit passed, the reporting wrapper verified every row key, original observation and saved prediction, and restored their original numeric precision **in memory only**. "
                     "No target, prediction file, training result, safeguard or frozen experiment code was changed. `render_audit.json` records the CSV round-trip errors and file hashes. "
                     "Any final intervals in this report use restored precision; the run summary's intervals use validated CSV precision. This is a serialization correction, not a scientific change.\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--destination", default=str(HERE))
    args = parser.parse_args()
    print(render(args.run_dir, args.data_dir, args.audit, args.destination))


if __name__ == "__main__":
    main()
