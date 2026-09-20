"""Causal feature and loss primitives for the isolated peak-sensitivity study.

Provenance: the five temporal summaries reproduce temporal_model/temporal_features.py;
the backbone is imported unchanged from outlier_sensitivity/london_unet_model.py.
Input history is the prepared (already standardized/clipped) lag-1 feature. These
functions do not reconstruct concentrations or introduce another one-day shift.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from experiments.outlier_sensitivity.london_unet_model import LondonResidualUNet


ALPHA = 0.425
TEMPORAL_NAMES = (
    "history_mean_3d", "history_mean_7d", "history_max_7d",
    "history_std_7d", "history_trend_3d",
)
EWMA_NAMES = ("history_ewma_half_life_1d", "history_ewma_half_life_3d",
              "history_minus_fast_ewma")
MEDIAN_NAMES = ("history_median_3d", "history_minus_median_3d")
FEATURE_KINDS = ("none", "temporal", "ewma", "median", "temporal_ewma", "temporal_median")


def build_features(history: np.ndarray, kind: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return D,C,H,W summaries, reading history[:d+1] only for target day d."""
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 3 or not values.shape[0] or not np.isfinite(values).all():
        raise ValueError("History must be a nonempty, finite [day, row, column] array")
    if kind not in FEATURE_KINDS:
        raise ValueError(f"Unknown feature kind: {kind}")
    names = ()
    if kind.startswith("temporal"):
        names += TEMPORAL_NAMES
    if "ewma" in kind:
        names += EWMA_NAMES
    if "median" in kind:
        names += MEDIAN_NAMES
    output = np.empty((len(values), len(names), *values.shape[1:]), dtype=np.float32)
    fast = values[0].copy()
    slow = values[0].copy()
    alpha_fast = 1.0 - 2.0 ** -1.0
    alpha_slow = 1.0 - 2.0 ** (-1.0 / 3.0)
    for day in range(len(values)):
        index = 0
        last3 = values[max(0, day - 2):day + 1]
        if kind.startswith("temporal"):
            last7 = values[max(0, day - 6):day + 1]
            earlier = values[max(0, day - 3):day]
            output[day, :5] = np.stack((
                last3.mean(axis=0), last7.mean(axis=0), last7.max(axis=0),
                last7.std(axis=0), values[day] - earlier.mean(axis=0)
                if len(earlier) else np.zeros_like(values[day]),
            ))
            index = 5
        if "ewma" in kind:
            if day:
                fast = alpha_fast * values[day] + (1.0 - alpha_fast) * fast
                slow = alpha_slow * values[day] + (1.0 - alpha_slow) * slow
            output[day, index:index + 3] = np.stack((fast, slow, values[day] - fast))
        if "median" in kind:
            median = np.median(last3, axis=0)
            output[day, index:index + 2] = np.stack((median, values[day] - median))
    return np.ascontiguousarray(output), names


def fit_scaler(features: np.ndarray, mask: np.ndarray, train_stop: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-channel moments using only training days and in-domain cells."""
    values = np.asarray(features)
    inside = np.asarray(mask, dtype=bool)
    if values.ndim != 4 or inside.shape != values.shape[-2:]:
        raise ValueError("Feature or domain mask shape is invalid")
    if not isinstance(train_stop, (int, np.integer)) or not 0 < train_stop <= len(values):
        raise ValueError("Invalid training boundary")
    if not inside.any():
        raise ValueError("Training domain contains no cells")
    if not values.shape[1]:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    training = values[:train_stop, :, inside].astype(np.float64)
    if not np.isfinite(training).all():
        raise ValueError("Nonfinite training feature inside the study area")
    centres = training.mean(axis=(0, 2)).astype(np.float32)
    scales = training.std(axis=(0, 2)).astype(np.float32)
    scales[scales < 1e-6] = 1.0
    return centres, scales


def apply_scaler(features: np.ndarray, centres: np.ndarray, scales: np.ndarray,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """Apply frozen training moments; optional mask zeroes outside-domain channels."""
    values = np.asarray(features, dtype=np.float32)
    centres = np.asarray(centres, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    if (values.ndim != 4 or centres.shape != (values.shape[1],)
            or scales.shape != centres.shape):
        raise ValueError("Scaler does not match feature channels")
    if (not np.isfinite(centres).all() or not np.isfinite(scales).all()
            or np.any(scales <= 0)):
        raise ValueError("Scaler must have finite centres and positive finite scales")
    result = (values - centres[None, :, None, None]) / scales[None, :, None, None]
    if mask is not None:
        inside = np.asarray(mask, dtype=bool)
        if inside.shape != values.shape[-2:]:
            raise ValueError("Domain mask shape is invalid")
        result[:, :, ~inside] = 0.0
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite scaled feature")
    return np.ascontiguousarray(result)


def raw_training_scale(raw: np.ndarray, weights: np.ndarray, train_stop: int) -> float:
    """Unweighted population SD of finite positive-weight TRAINING targets only."""
    values, recorded_weight = np.asarray(raw), np.asarray(weights)
    if values.shape != recorded_weight.shape or values.ndim < 1:
        raise ValueError("Training target and weight shapes differ")
    if not isinstance(train_stop, (int, np.integer)) or not 0 < train_stop <= len(values):
        raise ValueError("Invalid training boundary")
    values, recorded_weight = values[:train_stop], recorded_weight[:train_stop]
    observed = np.isfinite(values) & np.isfinite(recorded_weight) & (recorded_weight > 0)
    if not observed.any():
        raise ValueError("No finite positive-weight training targets")
    result = float(values[observed].astype(np.float64).std())
    if not math.isfinite(result) or result == 0.0:
        raise ValueError("Degenerate training-target standard deviation")
    return max(result, 1e-6)


def decode_tensor(z: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Decode in float32 with the existing [0, 500] concentration bounds."""
    if not math.isfinite(float(mean)) or not math.isfinite(float(std)) or std <= 0:
        raise ValueError("Invalid target scaler")
    decoded = torch.expm1(torch.clamp(z.float() * float(std) + float(mean),
                                      min=0.0, max=math.log1p(500.0)))
    # Float32 expm1(log1p(500)) can round slightly above 500.
    return decoded.clamp(min=0.0, max=500.0)


def loss(model_output: torch.Tensor, target_z: torch.Tensor, weight: torch.Tensor,
         baseline_z: torch.Tensor | None, prep: dict, raw_scale: float,
         aux: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen SmoothL1 plus optional normalized raw MSE of the FINAL prediction.

    target_z is the full standardized log target for both model families (not a
    pre-subtracted residual). baseline_z=None selects standalone U-Net. Hybrid
    SmoothL1 still trains the full residual, exactly as the frozen recipe does;
    only the auxiliary term uses baseline + 0.425 * correction.
    """
    if model_output.shape != target_z.shape or model_output.shape != weight.shape:
        raise ValueError("Prediction, target and weight tensor shapes differ")
    if baseline_z is not None and baseline_z.shape != model_output.shape:
        raise ValueError("Baseline shape differs from the correction")
    if not math.isfinite(float(aux)) or aux < 0:
        raise ValueError("Auxiliary loss coefficient must be finite and nonnegative")
    if not math.isfinite(float(raw_scale)) or raw_scale <= 0:
        raise ValueError("Raw training scale must be finite and positive")
    prediction, target, weights = model_output.float(), target_z.float(), weight.float()
    baseline = None if baseline_z is None else baseline_z.float()
    final_z = prediction if baseline is None else baseline + ALPHA * prediction
    observed = torch.isfinite(target) & torch.isfinite(weights) & (weights > 0)
    # Index first: NaN missing targets must not poison either the loss or gradients.
    fitted_target = target[observed]
    if baseline is not None:
        if not torch.isfinite(baseline[observed]).all():
            raise ValueError("Nonfinite baseline at an observed training target")
        fitted_target = fitted_target - baseline[observed]
    selected_weights = weights[observed]
    denominator = selected_weights.sum().clamp_min(1.0)
    result = (F.smooth_l1_loss(prediction[observed], fitted_target,
                              beta=1.0, reduction="none") * selected_weights).sum() / denominator
    if aux:
        mean, std = prep["target_log_mean"], prep["target_log_std"]
        decoded_prediction = decode_tensor(final_z[observed], mean, std)
        decoded_target = decode_tensor(target[observed], mean, std)
        normalized_error = (decoded_prediction - decoded_target) / float(raw_scale)
        result = result + float(aux) * (normalized_error.square() * selected_weights).sum() / denominator
    return result, final_z


def matched_model(seed: int, base_names: Sequence[str], feature_names: Sequence[str]
                  ) -> LondonResidualUNet:
    """Create an expanded model sharing canonical weights and post-init RNG state.

    feature_names is the COMPLETE desired input-channel list, not just additions.
    All base_names must occur exactly once. Extra channels receive zero weights
    in both first-block input convolutions. Standalone callers supply only their
    environmental/history predictors: no HGB channel is created or inserted here.
    """
    base_names, feature_names = tuple(base_names), tuple(feature_names)
    if (not base_names or len(set(base_names)) != len(base_names)
            or len(set(feature_names)) != len(feature_names)
            or not set(base_names).issubset(feature_names)):
        raise ValueError("Channel names must be unique and include every canonical input")
    # With exactly 32 base inputs the frozen backbone has an identity skip, which
    # cannot be enlarged without changing the topology. Actual runs have 66/67.
    if len(base_names) == 32 and len(feature_names) != 32:
        raise ValueError("Cannot expand the identity input skip of a 32-channel canonical model")
    torch.manual_seed(int(seed))
    canonical = LondonResidualUNet(len(base_names))
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    if feature_names == base_names:
        return canonical
    expanded = LondonResidualUNet(len(feature_names))
    source, destination = canonical.state_dict(), expanded.state_dict()
    input_weights = {"encoder1.conv1.weight", "encoder1.skip.weight"}
    try:
        for name, value in destination.items():
            if name in input_weights:
                value.zero_()
                for canonical_index, channel_name in enumerate(base_names):
                    value[:, feature_names.index(channel_name)] = source[name][:, canonical_index]
            else:
                if name not in source or value.shape != source[name].shape:
                    raise ValueError(f"Backbone changed unexpectedly at {name}")
                value.copy_(source[name])
        expanded.load_state_dict(destination)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return expanded
