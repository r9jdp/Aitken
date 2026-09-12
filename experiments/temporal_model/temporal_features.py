"""Leakage-safe temporal features for the London hybrid PM2.5 model."""
from __future__ import annotations

import numpy as np
try:
    import torch
except ModuleNotFoundError:  # NumPy-only safety tests can run in the public checkout.
    torch = None


TEMPORAL_NAMES = (
    "history_mean_3d",
    "history_mean_7d",
    "history_max_7d",
    "history_std_7d",
    "history_trend_3d",
)


def build_past_only_features(history: np.ndarray) -> np.ndarray:
    """Build summaries using only history available on or before each day.

    ``history[d]`` must itself be a causal predictor (for example, a lag-1 PM2.5
    map). The function never reads rows after ``d``. This makes a lag-1 source
    remain safe: its newest information on target day d is still from day d-1.
    """
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("History must be a finite [day, row, column] array")
    result = np.empty((len(values), len(TEMPORAL_NAMES), *values.shape[1:]), np.float32)
    for day in range(len(values)):
        last3 = values[max(0, day - 2) : day + 1]
        last7 = values[max(0, day - 6) : day + 1]
        earlier = values[max(0, day - 3) : day]
        result[day, 0] = last3.mean(axis=0)
        result[day, 1] = last7.mean(axis=0)
        result[day, 2] = last7.max(axis=0)
        result[day, 3] = last7.std(axis=0)
        result[day, 4] = 0.0 if not len(earlier) else values[day] - earlier.mean(axis=0)
    return result


def fit_temporal_scaler(
    features: np.ndarray, mask: np.ndarray, train_stop: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit channel scaling on training days and in-domain cells only."""
    values = np.asarray(features)
    inside = np.asarray(mask, dtype=bool)
    if values.ndim != 4 or values.shape[1] != len(TEMPORAL_NAMES):
        raise ValueError("Unexpected temporal feature shape")
    if inside.shape != values.shape[-2:] or not 0 < train_stop <= len(values):
        raise ValueError("Invalid mask or training boundary")
    training = values[:train_stop, :, inside].astype(np.float64)
    centres = training.mean(axis=(0, 2)).astype(np.float32)
    scales = training.std(axis=(0, 2)).astype(np.float32)
    scales[scales < 1e-6] = 1.0
    return centres, scales


def scale_temporal(
    features: np.ndarray, centres: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    centres = np.asarray(centres, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    if values.ndim != 4 or centres.shape != (values.shape[1],) or scales.shape != centres.shape:
        raise ValueError("Temporal scaler does not match the features")
    if np.any(scales <= 0):
        raise ValueError("Temporal scales must be positive")
    return np.ascontiguousarray(
        (values - centres[None, :, None, None]) / scales[None, :, None, None]
    )


def compose_prediction(
    baseline_z: torch.Tensor,
    correction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if torch is None:
        raise RuntimeError("PyTorch is required to compose model predictions")
    return baseline_z + float(alpha) * correction


def band_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    """Report normal, elevated and high bands so overall R2 cannot hide damage."""
    y = np.asarray(observed, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.shape != p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Observed and predicted values must be matching finite arrays")

    def calculate(use: np.ndarray) -> dict[str, float | int | None]:
        error = p[use] - y[use]
        return {
            "n": int(use.sum()),
            "mae": float(np.abs(error).mean()) if use.any() else None,
            "rmse": float(np.sqrt(np.square(error).mean())) if use.any() else None,
            "bias": float(error.mean()) if use.any() else None,
        }

    return {
        "normal_below_15": calculate(y < 15.0),
        "elevated_15_to_25": calculate((y >= 15.0) & (y < 25.0)),
        "high_at_least_25": calculate(y >= 25.0),
    }


def safe_candidate(candidate: dict, control: dict, bias_ceiling: float = 0.10) -> bool:
    """Require better peaks/overall error without sacrificing normal predictions."""
    c_band = candidate["bands"]
    b_band = control["bands"]
    normal = c_band["normal_below_15"]
    baseline_normal = b_band["normal_below_15"]
    high = c_band["high_at_least_25"]
    baseline_high = b_band["high_at_least_25"]
    return bool(
        candidate["rmse"] < control["rmse"]
        and candidate["r2"] > control["r2"]
        and normal["mae"] <= baseline_normal["mae"]
        and normal["bias"] <= max(bias_ceiling, baseline_normal["bias"])
        and high["mae"] < baseline_high["mae"]
    )
