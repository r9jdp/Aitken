"""Independent audit of the saved 2024 control and temporal predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = (
    "A_current_hybrid",
    "B_temporal",
)
SEEDS = (42, 11, 22)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(frame: pd.DataFrame) -> dict:
    y = frame.observed_pm25.to_numpy(float)
    prediction = frame.predicted_pm25.to_numpy(float)
    error = prediction - y
    sst = np.square(y - y.mean()).sum()

    def band(mask: np.ndarray) -> dict:
        selected = error[mask]
        count = int(mask.sum())
        return {
            "n": count,
            "mae": float(np.abs(selected).mean()),
            "rmse": float(np.sqrt(np.square(selected).mean())),
            "bias": float(selected.mean()),
        }

    return {
        "n": len(y),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / sst),
        "bias": float(error.mean()),
        "bands": {
            "normal_below_15": band(y < 15),
            "elevated_15_to_25": band((y >= 15) & (y < 25)),
            "high_at_least_25": band(y >= 25),
        },
    }


def compare(actual: dict, expected: dict) -> None:
    for key in ("n", "mae", "rmse", "r2", "bias"):
        if not np.isclose(actual[key], expected[key], rtol=0, atol=1e-6):
            raise AssertionError(f"Metric mismatch for {key}: {actual[key]} != {expected[key]}")
    for name in actual["bands"]:
        for key in ("n", "mae", "rmse", "bias"):
            if not np.isclose(
                actual["bands"][name][key], expected["bands"][name][key],
                rtol=0, atol=1e-6,
            ):
                raise AssertionError(f"Band metric mismatch for {name}/{key}")


def audit(run: Path, output: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    integrity = json.loads((run / "input_integrity.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("A complete 2024 run is required")
    if integrity["before"] != integrity["after"]:
        raise AssertionError("Input integrity record failed")
    for path, expected in integrity["before"].items():
        if sha256(Path(path)) != expected:
            raise AssertionError(f"Source input changed: {path}")

    identifiers = ["date_idx", "site_code", "row", "col", "observed_pm25"]
    reference = None
    records = []
    for name in VARIANTS:
        frame = pd.read_csv(run / "final" / f"{name}_predictions.csv.gz")
        if reference is None:
            reference = frame[identifiers]
        elif not frame[identifiers].equals(reference):
            raise AssertionError(f"Evaluation rows differ for {name}")
        calculated = metrics(frame)
        compare(calculated, summary["final"][name])
        checkpoints = []
        for seed in SEEDS:
            directory = run / "final" / name / f"seed_{seed}"
            completed = json.loads((directory / "completed.json").read_text())
            checkpoint = directory / "final_ema_checkpoint.pt"
            if sha256(checkpoint) != completed["checkpoint_sha256"]:
                raise AssertionError(f"Checkpoint hash mismatch: {name}/seed {seed}")
            checkpoints.append({"seed": seed, "sha256": completed["checkpoint_sha256"]})
        records.append({"variant": name, "metrics_recomputed": True, "checkpoints": checkpoints})

    assert reference is not None
    if len(reference) != 9619 or reference.date_idx.nunique() != 349:
        raise AssertionError("Unexpected 2024 evaluation population")
    result = {
        "passed": True,
        "station_rows": len(reference),
        "labelled_dates": int(reference.date_idx.nunique()),
        "same_rows_for_both_models": True,
        "source_inputs_unchanged": True,
        "selection_used_2024": False,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audited = audit(args.run_dir.resolve(), args.output.resolve())
    print(json.dumps(audited, indent=2))
