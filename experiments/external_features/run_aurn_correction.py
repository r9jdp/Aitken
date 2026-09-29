"""Fit a guarded AURN correction on 2023 and evaluate it once on 2024."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TEMPORAL = HERE.parent / "temporal_model"
sys.path.insert(0, str(TEMPORAL))
from temporal_features import band_metrics  # noqa: E402


SOURCE_COLUMNS = {
    "aurn_lag1": [
        "aurn_regional_mean_lag1",
        "aurn_regional_max_lag1",
        "aurn_upwind_mean_lag1",
    ],
    "aurn_same_day": [
        "aurn_regional_mean_same_day",
        "aurn_regional_max_same_day",
        "aurn_upwind_mean_same_day",
        "aurn_log_valid_site_count",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside_repo(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved == REPO or not resolved.is_relative_to(REPO):
        raise ValueError("Output must stay inside the repository")
    return resolved


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    y = np.asarray(observed, np.float64)
    p = np.asarray(predicted, np.float64)
    error = p - y
    return {
        "n": int(len(y)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / np.square(y - y.mean()).sum()),
        "bias": float(error.mean()),
        "bands": band_metrics(y, p),
    }


def block_interval(
    frame: pd.DataFrame, control: np.ndarray, candidate: np.ndarray, repeats: int = 5000
) -> dict:
    days = frame.date_idx.to_numpy(int)
    observed = frame.observed_pm25.to_numpy(float)
    first, count = int(days.min()), int(days.max() - days.min() + 1)
    table = np.zeros((count, 3), float)
    for column, values in enumerate((np.ones(len(days)), np.square(control - observed), np.square(candidate - observed))):
        np.add.at(table[:, column], days - first, values)
    rng = np.random.default_rng(20260930)
    differences = []
    for _ in range(repeats):
        starts = rng.integers(0, max(1, count - 6), size=int(np.ceil(count / 7)))
        sampled = (starts[:, None] + np.arange(7)).ravel()[:count]
        totals = table[sampled].sum(axis=0)
        differences.append(float(np.sqrt(totals[2] / totals[0]) - np.sqrt(totals[1] / totals[0])))
    return {
        "definition": "candidate minus control RMSE; negative favours candidate",
        "replicates": repeats,
        "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
    }


def load_feature_frame(path: Path) -> pd.DataFrame:
    archive = np.load(path, allow_pickle=False)
    names = archive["names"].astype(str).tolist()
    values = archive["values"]
    if values.shape != (1461, len(names)) or not np.isfinite(values).all():
        raise ValueError("Unexpected AURN feature archive")
    frame = pd.DataFrame(values, columns=names)
    frame["date_idx"] = np.arange(len(frame))
    frame["date"] = pd.date_range("2021-01-01", "2024-12-31")
    return frame


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    expected_rows: int,
    expected_days: int,
    minimum_date_idx: int,
    maximum_date_idx: int,
) -> None:
    required = {"date_idx", "date", "site_code", "observed_pm25", "predicted_pm25"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} predictions are missing columns: {sorted(missing)}")
    actual = (
        len(frame),
        frame.date_idx.nunique(),
        int(frame.date_idx.min()),
        int(frame.date_idx.max()),
    )
    expected = (expected_rows, expected_days, minimum_date_idx, maximum_date_idx)
    if actual != expected:
        raise ValueError(f"{label} prediction shape/date signature {actual} != {expected}")
    expected_date = pd.Timestamp("2021-01-01") + pd.to_timedelta(frame.date_idx, unit="D")
    if not np.array_equal(pd.to_datetime(frame.date).to_numpy(), expected_date.to_numpy()):
        raise ValueError(f"{label} date and date_idx columns disagree")
    numeric = frame[["observed_pm25", "predicted_pm25"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{label} predictions contain non-finite values")


def daily_training_table(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    daily = (
        predictions.groupby(["date_idx", "date"], as_index=False)
        .agg(
            observed_mean=("observed_pm25", "mean"),
            predicted_mean=("predicted_pm25", "mean"),
            station_count=("site_code", "size"),
        )
    )
    daily["mean_residual"] = daily.observed_mean - daily.predicted_mean
    return daily.merge(features, on=["date_idx", "date"], how="left", validate="one_to_one")


def gate(score: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if not upper > lower:
        raise ValueError("Gate upper bound must exceed its lower bound")
    return np.clip((np.asarray(score, float) - lower) / (upper - lower), 0.0, 1.0)


def station_correction(
    frame: pd.DataFrame,
    day_delta: dict[int, float],
    feature_days: pd.DataFrame,
    columns: list[str],
    lower: float,
    upper: float,
    shrink: float,
) -> np.ndarray:
    score = feature_days[columns[:3]].max(axis=1).to_numpy(float)
    gated = gate(score, lower, upper) * float(shrink)
    correction_by_day = {
        int(day): float(gated[index] * np.clip(day_delta[int(day)], -5.0, 15.0))
        for index, day in enumerate(feature_days.date_idx)
    }
    correction = frame.date_idx.map(correction_by_day).to_numpy(float)
    if not np.isfinite(correction).all():
        raise ValueError("Missing correction for one or more station rows")
    return np.clip(frame.predicted_pm25.to_numpy(float) + correction, 0.0, 500.0)


def select_recipe(
    predictions: pd.DataFrame, features: pd.DataFrame, source: str
) -> tuple[dict, pd.DataFrame]:
    columns = SOURCE_COLUMNS[source]
    daily = daily_training_table(predictions, features)
    groups = pd.to_datetime(daily.date).dt.month.to_numpy()
    splitter = GroupKFold(n_splits=12)
    control = calculate_metrics(predictions.observed_pm25, predictions.predicted_pm25)
    trials = []
    best = None
    for alpha in (0.1, 1.0, 10.0, 100.0):
        oof = np.full(len(daily), np.nan, float)
        for train, test in splitter.split(daily, groups=groups):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(daily.iloc[train][columns], daily.iloc[train].mean_residual)
            oof[test] = model.predict(daily.iloc[test][columns])
        day_delta = dict(zip(daily.date_idx.astype(int), oof))
        for lower in (8.0, 10.0, 12.0):
            for upper in (15.0, 20.0, 25.0):
                if upper <= lower:
                    continue
                for shrink in (0.25, 0.5, 0.75, 1.0):
                    candidate = station_correction(
                        predictions, day_delta, daily, columns, lower, upper, shrink
                    )
                    metrics = calculate_metrics(predictions.observed_pm25, candidate)
                    eligible = bool(
                        metrics["rmse"] < control["rmse"]
                        and metrics["r2"] > control["r2"]
                        and metrics["bands"]["normal_below_15"]["mae"]
                        <= control["bands"]["normal_below_15"]["mae"]
                        and metrics["bands"]["high_at_least_25"]["mae"]
                        < control["bands"]["high_at_least_25"]["mae"]
                    )
                    trial = {
                        "alpha": alpha,
                        "gate_lower": lower,
                        "gate_upper": upper,
                        "shrink": shrink,
                        "eligible": eligible,
                        "metrics": metrics,
                    }
                    trials.append(trial)
                    if eligible and (best is None or metrics["rmse"] < best["metrics"]["rmse"]):
                        best = trial
    selection = {
        "source": source,
        "columns": columns,
        "control_metrics": control,
        "selected": best,
        "rule": "lower 2023 blocked-OOF RMSE, higher R2, lower high-band MAE, and no worse normal-band MAE",
        "trial_count": len(trials),
        "trials": trials,
    }
    return selection, daily


def fit_frozen_delta(daily: pd.DataFrame, recipe: dict, columns: list[str]) -> tuple[dict[int, float], dict]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(recipe["alpha"])))
    model.fit(daily[columns], daily.mean_residual)
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    fitted = model.predict(daily[columns])
    return dict(zip(daily.date_idx.astype(int), fitted)), {
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "coefficients": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
    }


def predict_new_days(
    train_daily: pd.DataFrame, feature_days: pd.DataFrame, recipe: dict, columns: list[str]
) -> tuple[dict[int, float], dict]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(recipe["alpha"])))
    model.fit(train_daily[columns], train_daily.mean_residual)
    delta = model.predict(feature_days[columns])
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    record = {
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "coefficients": ridge.coef_.tolist(),
        "intercept": float(ridge.intercept_),
    }
    return dict(zip(feature_days.date_idx.astype(int), delta)), record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/external_features/aurn/aurn_features.npz"))
    parser.add_argument(
        "--selection-predictions",
        type=Path,
        default=Path("artifacts/temporal_tail_correction_20260912/screening/B_temporal_predictions.csv.gz"),
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=Path("artifacts/temporal_tail_correction_20260912/final/B_temporal_predictions.csv.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/external_features_aurn"))
    args = parser.parse_args()
    output = inside_repo(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features = load_feature_frame(args.features)

    # Selection reads only the 2023 out-of-year predictions. Freeze the complete
    # recipe on disk before loading any 2024 observations.
    selection_predictions = pd.read_csv(args.selection_predictions, parse_dates=["date"])
    validate_prediction_frame(
        selection_predictions,
        label="selection",
        expected_rows=11102,
        expected_days=365,
        minimum_date_idx=730,
        maximum_date_idx=1094,
    )
    frozen = {}
    train_tables = {}
    for source in SOURCE_COLUMNS:
        selection, daily = select_recipe(selection_predictions, features, source)
        frozen[source] = selection
        train_tables[source] = daily
    selection_path = output / "selection_frozen.json"
    selection_path.write_text(json.dumps(frozen, indent=2, allow_nan=False), encoding="utf-8")

    test_predictions = pd.read_csv(args.test_predictions, parse_dates=["date"])
    validate_prediction_frame(
        test_predictions,
        label="test",
        expected_rows=9619,
        expected_days=349,
        minimum_date_idx=1095,
        maximum_date_idx=1443,
    )
    control_values = test_predictions.predicted_pm25.to_numpy(float)
    results = {
        "scope": "2023-only recipe selection followed by one frozen 2024 evaluation",
        "inputs": {
            "features_sha256": sha256(args.features),
            "selection_predictions_sha256": sha256(args.selection_predictions),
            "test_predictions_sha256": sha256(args.test_predictions),
            "frozen_selection_sha256": sha256(selection_path),
        },
        "control": calculate_metrics(test_predictions.observed_pm25, control_values),
        "sources": {},
    }
    test_days = features[features.date_idx.isin(test_predictions.date_idx.unique())].copy()
    for source, columns in SOURCE_COLUMNS.items():
        selected = frozen[source]["selected"]
        if selected is None:
            candidate = control_values.copy()
            model_record = None
            status = "rejected_on_2023"
        else:
            delta, model_record = predict_new_days(train_tables[source], test_days, selected, columns)
            candidate = station_correction(
                test_predictions,
                delta,
                test_days,
                columns,
                float(selected["gate_lower"]),
                float(selected["gate_upper"]),
                float(selected["shrink"]),
            )
            status = "evaluated"
        saved = test_predictions.copy()
        saved["control_pm25"] = control_values
        saved["predicted_pm25"] = candidate
        saved.to_csv(output / f"{source}_2024_predictions.csv.gz", index=False)
        results["sources"][source] = {
            "status": status,
            "selection": selected,
            "fitted_model": model_record,
            "metrics": calculate_metrics(test_predictions.observed_pm25, candidate),
            "paired_block_interval": block_interval(test_predictions, control_values, candidate),
        }
    (output / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
