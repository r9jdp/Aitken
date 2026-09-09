"""Small unchanged helpers extracted from the existing London trainers.
See PROVENANCE.md. No imports or writes into the original London folder.
"""
from __future__ import annotations
import math
import random
import numpy as np
import pandas as pd
import torch
HIGH_PM25 = 25.0

def inverse_target(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    log_values = np.clip(values * std + mean, 0.0, np.log1p(500.0))
    return np.clip(np.expm1(log_values), 0.0, 500.0)

def gaussian_kernel(sigma_bins: float) -> np.ndarray:
    radius = max(1, int(math.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return kernel / kernel.sum()

def derive_lds_lookup(
    values: np.ndarray,
    source_weights: np.ndarray,
    cap: float,
    bins: int = 120,
    sigma_bins: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Create capped inverse-density weights from training targets only.

    Histogram counts are source weighted and Gaussian-smoothed.  The inverse
    square-root density is less aggressive than full inverse density.  A
    bisection scale gives mean weight one while respecting [0.5, cap].
    """

    if cap <= 1.0:
        return (
            np.asarray([0.0, 500.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
            {"enabled": 0, "cap": 1.0},
        )
    values = np.asarray(values, dtype=np.float64)
    source_weights = np.asarray(source_weights, dtype=np.float64)
    upper = max(60.0, float(math.ceil(np.percentile(values, 99.9) / 5.0) * 5.0))
    edges = np.linspace(0.0, upper, bins + 1, dtype=np.float64)
    clipped = np.clip(values, 0.0, np.nextafter(upper, 0.0))
    hist, _ = np.histogram(clipped, bins=edges, weights=source_weights)
    smooth = np.convolve(hist, gaussian_kernel(sigma_bins), mode="same")
    reference = float(np.average(smooth, weights=np.maximum(hist, 1e-12)))
    raw_lookup = np.sqrt(reference / np.maximum(smooth, 1e-12))

    value_bins = np.clip(np.searchsorted(edges, clipped, side="right") - 1, 0, bins - 1)
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        mean_weight = float(
            np.average(np.clip(mid * raw_lookup[value_bins], 0.5, cap), weights=source_weights)
        )
        if mean_weight < 1.0:
            lo = mid
        else:
            hi = mid
    lookup = np.clip(hi * raw_lookup, 0.5, cap).astype(np.float32)
    assigned = lookup[value_bins]
    stats: dict[str, float | int] = {
        "enabled": 1,
        "bins": bins,
        "sigma_bins": sigma_bins,
        "histogram_upper_pm25": upper,
        "cap": cap,
        "minimum": float(lookup.min()),
        "maximum": float(lookup.max()),
        "source_weighted_mean": float(np.average(assigned, weights=source_weights)),
        "mean_at_or_above_25": float(
            np.average(assigned[values >= HIGH_PM25], weights=source_weights[values >= HIGH_PM25])
        )
        if np.any(values >= HIGH_PM25)
        else float("nan"),
        "training_samples_at_or_above_25": int(np.sum(values >= HIGH_PM25)),
    }
    return edges.astype(np.float32), lookup, stats

def fit_preprocessing(data, features, metadata, channels, train_stop):
    """Fit ONLY days [0, train_stop); no validation/test moments or labels."""
    inside = np.load(data / "gla_mask.npy") > 0
    means = np.zeros(len(channels), np.float32)
    stds = np.ones(len(channels), np.float32)
    for position, channel in enumerate(channels):
        name = metadata["feature_names"][channel]
        if not metadata["feature_scalers_train_only"][name]["scaled"]:
            continue
        total = squared = 0.0
        count = 0
        for first in range(0, train_stop, 32):
            values = np.asarray(features[first:min(first + 32, train_stop), channel],
                                dtype=np.float32)[:, inside].astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite feature {name}")
            total += values.sum()
            squared += np.square(values).sum()
            count += values.size
        means[position] = total / count
        std = math.sqrt(max(squared / count - (total / count) ** 2, 0.0))
        stds[position] = std if std >= 1e-6 else 1.0
    raw = np.asarray(np.load(data / "target_pm25.npy", mmap_mode="r")[:train_stop])
    weights = np.load(data / "target_weight.npy", mmap_mode="r")[:train_stop]
    laqn = np.load(data / "laqn_cell_target.npy", mmap_mode="r")[:train_stop]
    observed = np.isfinite(raw) & (weights > 0)
    source = np.where(np.isfinite(laqn[observed]), 1.0, 0.15)
    values = raw[observed]
    transformed = np.log1p(values.astype(np.float64))
    mean = float(np.average(transformed, weights=source))
    std = float(np.sqrt(np.average((transformed - mean) ** 2, weights=source)))
    edges, lookup, lds = derive_lds_lookup(values, source, cap=2.5)
    return {
        "train_stop_exclusive": train_stop, "channels": channels,
        "feature_names": [metadata["feature_names"][i] for i in channels],
        "feature_centres": means.tolist(), "feature_scales": stds.tolist(),
        "target_log_mean": mean, "target_log_std": std,
        "lds_edges": edges.tolist(), "lds_lookup": lookup.tolist(), "lds_stats": lds,
        "observed_training_cell_days": int(observed.sum()),
        "scaling_note": "Affine refit of fixed float16 prepared cube, originally clipped at +/-8; not a raw-data reconstruction.",
    }

def station_rows(data, first, stop):
    with np.load(data / "station_daily.npz") as station:
        use = (station["source"] == 1) & (station["date_idx"] >= first) & (station["date_idx"] < stop)
        indices = np.flatnonzero(use)
        day = station["date_idx"][use].astype(int)
        return pd.DataFrame({
            "station_row_index": indices, "date_idx": day,
            "date": (pd.Timestamp("2021-01-01") + pd.to_timedelta(day, unit="D")).strftime("%Y-%m-%d"),
            "site_code": station["site_codes"][station["site_idx"][use]],
            "row": station["row"][use], "col": station["col"][use],
            "observed_pm25": station["value"][use],
        })

def gather(maps, frame, first):
    return maps[frame.date_idx.to_numpy() - first, frame.row.to_numpy(), frame.col.to_numpy()]
