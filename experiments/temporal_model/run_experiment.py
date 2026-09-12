"""Compare the hybrid model with and without past-only temporal features.

Both versions are screened on 2023 after fitting 2021-2022, then freshly trained
on 2021-2023 and retrospectively evaluated on the same 2024 observations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTLIER = HERE.parent / "outlier_sensitivity"
import sys

sys.path.insert(0, str(OUTLIER))
from frozen_helpers import fit_preprocessing, gather, inverse_target, station_rows  # noqa: E402
from london_unet_model import LondonResidualUNet  # noqa: E402

from temporal_features import (  # noqa: E402
    TEMPORAL_NAMES,
    band_metrics,
    build_past_only_features,
    compose_prediction,
    fit_temporal_scaler,
    safe_candidate,
    scale_temporal,
)

VARIANTS = {
    "A_current_hybrid": {"temporal": False},
    "B_temporal": {"temporal": True},
}
SEEDS = (42, 11, 22)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def inside_repo(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved == REPO or not resolved.is_relative_to(REPO):
        raise ValueError("Experiment outputs must stay inside Aitken")
    if resolved.relative_to(REPO).parts[0] in {".git", ".codex", ".agents"}:
        raise ValueError("Protected output directory")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.shape, array.dtype)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def json_save(path: Path, value: object) -> None:
    path = inside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def torch_save(path: Path, value: object) -> None:
    path = inside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def gather_features(
    feature_map: np.ndarray,
    days: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    channels: list[int],
) -> np.ndarray:
    """Gather sparse station/cell predictors without loading the full cube."""
    channel_array = np.asarray(channels, dtype=int)
    output = np.empty((len(days), len(channel_array)), dtype=np.float32)
    order = np.argsort(days, kind="stable")
    sorted_days = days[order]
    starts = np.flatnonzero(np.r_[True, sorted_days[1:] != sorted_days[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        positions = order[start:end]
        plane = np.asarray(feature_map[int(days[positions[0]])], dtype=np.float32)
        output[positions] = plane[channel_array][:, rows[positions], cols[positions]].T
    return output


def build_or_load_baseline(
    data: Path,
    features: np.ndarray,
    prep: dict,
    mask: np.ndarray,
    train_stop: int,
    output: Path,
    validation: pd.DataFrame | None,
    frozen_log_weight: float | None,
) -> tuple[np.ndarray, float, dict]:
    """Create a matched log/raw HGB baseline when transferred artifacts are absent."""
    output = inside_repo(output)
    output.mkdir(parents=True, exist_ok=True)
    map_path = output / "baseline_normalised.npy"
    model_path = output / "hgb_bundle.joblib"
    manifest_path = output / "manifest.json"
    source_contract = {
        "train_stop": train_stop,
        "channels": prep["channels"],
        "target_log_mean": prep["target_log_mean"],
        "target_log_std": prep["target_log_std"],
        "requested_log_weight": frozen_log_weight,
        "recipe": {
            "log": {"learning_rate": 0.055, "max_iter": 350, "max_leaf_nodes": 31, "min_samples_leaf": 24, "l2_regularization": 2.0},
            "raw": {"learning_rate": 0.05, "max_iter": 450, "max_leaf_nodes": 63, "min_samples_leaf": 24, "l2_regularization": 3.0},
        },
    }
    contract = hashlib.sha256(json.dumps(source_contract, sort_keys=True).encode()).hexdigest()
    if manifest_path.exists() and map_path.exists() and model_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["contract"] == contract
            and sha256(map_path) == manifest["map_sha256"]
            and sha256(model_path) == manifest["model_sha256"]
        ):
            return np.load(map_path, mmap_mode="r"), float(manifest["log_weight"]), manifest
        raise ValueError(f"Existing baseline conflicts with the requested recipe: {output}")

    raw = np.load(data / "target_pm25.npy", mmap_mode="r")
    recorded_weight = np.load(data / "target_weight.npy", mmap_mode="r")
    laqn = np.load(data / "laqn_cell_target.npy", mmap_mode="r")
    observed = np.isfinite(raw[:train_stop]) & (recorded_weight[:train_stop] > 0)
    days, rows, cols = np.where(observed)
    x_train = gather_features(features, days, rows, cols, prep["channels"])
    y_raw = np.asarray(raw)[days, rows, cols].astype(np.float32)
    y_log = (
        (np.log1p(y_raw) - prep["target_log_mean"]) / prep["target_log_std"]
    ).astype(np.float32)
    weights = np.where(np.isfinite(laqn[days, rows, cols]), 1.0, 0.15).astype(np.float32)
    common = {"loss": "squared_error", "early_stopping": False, "random_state": 42}
    log_model = HistGradientBoostingRegressor(
        learning_rate=0.055, max_iter=350, max_leaf_nodes=31,
        min_samples_leaf=24, l2_regularization=2.0, **common,
    )
    raw_model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=450, max_leaf_nodes=63,
        min_samples_leaf=24, l2_regularization=3.0, **common,
    )
    log(f"Fitting matched HGB baseline on {len(y_raw):,} observed training cells")
    log_model.fit(x_train, y_log, sample_weight=weights)
    raw_model.fit(x_train, y_raw, sample_weight=weights)
    del x_train

    validation_metrics = None
    if frozen_log_weight is None:
        if validation is None:
            raise ValueError("Selection validation is required to choose the baseline blend")
        x_val = gather_features(
            features,
            validation.date_idx.to_numpy(int), validation.row.to_numpy(int),
            validation.col.to_numpy(int), prep["channels"],
        )
        y_val = validation.observed_pm25.to_numpy(float)
        log_raw = inverse_target(
            log_model.predict(x_val), prep["target_log_mean"], prep["target_log_std"]
        )
        raw_pred = np.clip(raw_model.predict(x_val), 0.0, 500.0)
        trials = []
        for weight in np.linspace(0.0, 1.0, 101):
            prediction = weight * log_raw + (1.0 - weight) * raw_pred
            trials.append((float(np.sqrt(np.mean(np.square(prediction - y_val)))), float(weight)))
        _, log_weight = min(trials)
        validation_metrics = calculate_metrics(
            y_val, log_weight * log_raw + (1.0 - log_weight) * raw_pred
        )
    else:
        log_weight = float(frozen_log_weight)

    inside = np.flatnonzero(mask.ravel())
    maps = np.zeros((len(features), *features.shape[-2:]), dtype=np.float16)
    log("Generating matched HGB baseline maps")
    for first in range(0, len(features), 32):
        stop = min(first + 32, len(features))
        block = np.asarray(features[first:stop][:, prep["channels"]], dtype=np.float32)
        values = (
            block.reshape(stop - first, len(prep["channels"]), -1)[:, :, inside]
            .transpose(0, 2, 1).reshape(-1, len(prep["channels"]))
        )
        log_raw = inverse_target(
            log_model.predict(values), prep["target_log_mean"], prep["target_log_std"]
        )
        raw_pred = np.clip(raw_model.predict(values), 0.0, 500.0)
        blended = log_weight * log_raw + (1.0 - log_weight) * raw_pred
        normalised = (
            (np.log1p(blended) - prep["target_log_mean"]) / prep["target_log_std"]
        ).reshape(stop - first, len(inside))
        maps[first:stop].reshape(stop - first, -1)[:, inside] = normalised.astype(np.float16)
    np.save(map_path, maps)
    joblib.dump(
        {"log_model": log_model, "raw_model": raw_model, "log_weight": log_weight, "contract": source_contract},
        model_path, compress=3,
    )
    manifest = {
        "contract": contract, "log_weight": log_weight,
        "map_sha256": sha256(map_path), "model_sha256": sha256(model_path),
        "validation_metrics": validation_metrics,
    }
    json_save(manifest_path, manifest)
    return np.load(map_path, mmap_mode="r"), log_weight, manifest


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path: Path, hours: float) -> None:
        self.path = path
        self.limit = hours * 3600.0
        self.used = json.loads(path.read_text())["used_seconds"] if path.exists() else 0.0

    def check(self) -> None:
        if self.used >= self.limit:
            raise BudgetExceeded(f"GPU budget reached ({self.used / 3600:.2f} hours)")

    def add(self, seconds: float) -> None:
        self.used += seconds
        json_save(self.path, {"used_seconds": self.used, "limit_seconds": self.limit})


def prepare(
    data: Path,
    features: np.ndarray,
    metadata: dict,
    protocol: dict,
    phase: str,
) -> dict:
    stop = 730 if phase == "selection" else 1095
    prep = fit_preprocessing(data, features, metadata, protocol["channels"], stop)
    target = protocol["hybrid_target_scalers"][phase]
    prep["target_log_mean"], prep["target_log_std"] = target["mean"], target["std"]
    if phase == "selection":
        prep["feature_centres"], prep["feature_scales"] = [0.0] * 66, [1.0] * 66
    return prep


def resolve_history_channel(metadata: dict, protocol: dict, requested: str) -> int:
    names = metadata["feature_names"]
    if requested not in names:
        suggestions = [n for n in names if "pm" in n.lower() and any(k in n.lower() for k in ("lag", "prior", "history"))]
        raise ValueError(
            f"Unknown --history-feature {requested!r}. Causal PM history candidates: {suggestions}"
        )
    index = names.index(requested)
    if index not in protocol["channels"]:
        raise ValueError("The history feature must be one of the frozen 66 predictors")
    lowered = requested.lower()
    if "pm" not in lowered or not any(k in lowered for k in ("lag", "prior", "history")):
        raise ValueError("History feature name must clearly identify a prior/lagged PM predictor")
    return index


def model_inputs(
    features: np.ndarray,
    days: np.ndarray,
    prep: dict,
    baseline: np.ndarray,
    temporal: np.ndarray | None,
) -> np.ndarray:
    x = np.asarray(features[days], dtype=np.float32)[:, prep["channels"]]
    x -= np.asarray(prep["feature_centres"], np.float32)[None, :, None, None]
    x /= np.asarray(prep["feature_scales"], np.float32)[None, :, None, None]
    parts = [x]
    if temporal is not None:
        parts.append(np.asarray(temporal[days], dtype=np.float32))
    parts.append(np.asarray(baseline[days, None], dtype=np.float32))
    result = np.ascontiguousarray(np.concatenate(parts, axis=1))
    if not np.isfinite(result).all():
        raise ValueError("Non-finite model input")
    return result


def targets_for_day(
    raw: np.ndarray,
    recorded_weight: np.ndarray,
    laqn: np.ndarray,
    baseline_z: np.ndarray,
    prep: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.isfinite(raw) & (recorded_weight > 0)
    original = np.where(observed, raw, 0.0).astype(np.float32)
    bins = np.clip(
        np.searchsorted(np.asarray(prep["lds_edges"]), original, side="right") - 1,
        0,
        len(prep["lds_lookup"]) - 1,
    )
    source = np.where(observed & np.isfinite(laqn), 1.0, np.where(observed, 0.15, 0.0))
    weights = (source * np.asarray(prep["lds_lookup"], np.float32)[bins]).astype(np.float32)
    target_z = (np.log1p(original) - prep["target_log_mean"]) / prep["target_log_std"]
    target_z = np.where(observed, target_z, 0.0).astype(np.float32)
    residual = np.where(observed, target_z - baseline_z, 0.0).astype(np.float32)
    return residual, target_z, weights


class TrainingDays(Dataset):
    def __init__(
        self,
        data: Path,
        features: np.ndarray,
        prep: dict,
        baseline: np.ndarray,
        temporal: np.ndarray | None,
    ) -> None:
        self.features, self.prep = features, prep
        self.baseline, self.temporal = baseline, temporal
        self.raw = np.load(data / "target_pm25.npy", mmap_mode="r")
        self.weight = np.load(data / "target_weight.npy", mmap_mode="r")
        self.laqn = np.load(data / "laqn_cell_target.npy", mmap_mode="r")

    def __len__(self) -> int:
        return self.prep["train_stop_exclusive"]

    def __getitem__(self, day: int):
        x = model_inputs(self.features, np.asarray([day]), self.prep, self.baseline, self.temporal)[0]
        residual, target_z, weight = targets_for_day(
            self.raw[day], self.weight[day], self.laqn[day],
            np.asarray(self.baseline[day], np.float32), self.prep,
        )
        return tuple(torch.from_numpy(v) for v in (x, residual[None], target_z[None], weight[None]))


def make_model(input_channels: int) -> torch.nn.Module:
    return LondonResidualUNet(input_channels, 32, 0.12)


def rng_state(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": generator.get_state(),
    }


def restore_rng(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def train(
    data: Path,
    features: np.ndarray,
    prep: dict,
    baseline: np.ndarray,
    temporal: np.ndarray | None,
    variant_name: str,
    seed: int,
    schedule: list[float],
    output: Path,
    alpha: float,
    budget: Budget,
    device: torch.device,
) -> Path:
    variant = VARIANTS[variant_name]
    output = inside_repo(output)
    output.mkdir(parents=True, exist_ok=True)
    input_channels = 67 + (len(TEMPORAL_NAMES) if variant["temporal"] else 0)
    contract_data = {
        "variant": variant_name, "seed": seed, "schedule": schedule,
        "prep": prep, "input_channels": input_channels, "alpha": alpha,
    }
    contract = hashlib.sha256(json.dumps(contract_data, sort_keys=True).encode()).hexdigest()
    final = output / "final_ema_checkpoint.pt"
    completed = output / "completed.json"
    if completed.exists():
        record = json.loads(completed.read_text())
        if record["contract"] != contract or sha256(final) != record["checkpoint_sha256"]:
            raise ValueError("Completed run conflicts with the requested configuration")
        return final

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(seed)
    model = make_model(input_channels).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-4, weight_decay=1e-4,
    )
    amp = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    loader = DataLoader(
        TrainingDays(data, features, prep, baseline, temporal if variant["temporal"] else None),
        batch_size=8, shuffle=True, generator=generator, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    first, history = 1, []
    resume = output / "resume_state.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["contract"] != contract:
            raise ValueError("Resume configuration mismatch")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        amp.load_state_dict(state["amp"])
        restore_rng(state["rng"], generator)
        first, history = state["epoch"] + 1, state["history"]

    for epoch in range(first, len(schedule) + 1):
        budget.check()
        synchronize(device)
        started = time.monotonic()
        for group in optimizer.param_groups:
            group["lr"] = schedule[epoch - 1]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for index, (x, residual, target_z, weight) in enumerate(loader):
            x, residual, target_z, weight = [
                v.to(device, non_blocking=device.type == "cuda")
                for v in (x, residual, target_z, weight)
            ]
            baseline_z = x[:, -1:]
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                correction = model(x)
                prediction_z = compose_prediction(baseline_z, correction, alpha)
                loss = (
                    F.smooth_l1_loss(
                        correction, residual, beta=1.0, reduction="none"
                    )
                    * weight
                ).sum() / weight.sum().clamp_min(1.0)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            amp.scale(loss / 2.0).backward()
            epoch_loss += float(loss.detach())
            if (index + 1) % 2 == 0 or index + 1 == len(loader):
                amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                amp.step(optimizer)
                amp.update()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for averaged, current in zip(ema.parameters(), model.parameters()):
                        averaged.lerp_(current, 0.02)
                    for averaged, current in zip(ema.buffers(), model.buffers()):
                        averaged.copy_(current)
        synchronize(device)
        elapsed = time.monotonic() - started
        history.append({"epoch": epoch, "loss": epoch_loss / len(loader), "seconds": elapsed})
        budget.add(elapsed)
        torch_save(resume, {
            "contract": contract, "model": model.state_dict(), "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(), "amp": amp.state_dict(),
            "rng": rng_state(generator), "epoch": epoch, "history": history,
        })
        json_save(output / "history.json", history)
        log(f"{variant_name}/seed {seed}: epoch {epoch}/{len(schedule)}, loss={epoch_loss / len(loader):.5f}")

    torch_save(final, {"state": ema.state_dict(), "contract": contract, "variant": variant_name})
    json_save(completed, {
        "contract": contract, "checkpoint_sha256": sha256(final),
        "epochs": len(schedule), "seed": seed,
    })
    return final


@torch.inference_mode()
def infer(
    checkpoint: Path,
    features: np.ndarray,
    prep: dict,
    baseline: np.ndarray,
    temporal: np.ndarray | None,
    variant_name: str,
    first: int,
    stop: int,
    alpha: float,
    budget: Budget,
    device: torch.device,
) -> np.ndarray:
    budget.check()
    started = time.monotonic()
    variant = VARIANTS[variant_name]
    channels = 67 + (len(TEMPORAL_NAMES) if variant["temporal"] else 0)
    model = make_model(channels).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state"])
    predictions = np.empty((stop - first, *features.shape[-2:]), np.float32)
    for begin in range(first, stop, 16):
        days = np.arange(begin, min(begin + 16, stop))
        x = torch.from_numpy(model_inputs(
            features, days, prep, baseline, temporal if variant["temporal"] else None
        )).to(device)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            final_z = compose_prediction(x[:, -1:], model(x), alpha)
        position = begin - first
        predictions[position : position + len(days)] = final_z[:, 0].float().cpu().numpy()
    synchronize(device)
    budget.add(time.monotonic() - started)
    return predictions


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    y, p = np.asarray(observed, np.float64), np.asarray(predicted, np.float64)
    error = p - y
    sst = np.square(y - y.mean()).sum()
    return {
        "n": len(y), "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": float(1.0 - np.square(error).sum() / sst),
        "bias": float(error.mean()), "bands": band_metrics(y, p),
    }


def save_evaluation(directory: Path, frame: pd.DataFrame, predicted: np.ndarray, name: str) -> dict:
    metrics = calculate_metrics(frame.observed_pm25.to_numpy(), predicted)
    rows = frame.copy()
    rows["predicted_pm25"] = predicted
    rows.to_csv(inside_repo(directory / f"{name}_predictions.csv.gz"), index=False)
    json_save(directory / f"{name}_metrics.json", metrics)
    return metrics


def block_interval(frame: pd.DataFrame, control: np.ndarray, candidate: np.ndarray, repeats: int = 5000) -> dict:
    day = frame.date_idx.to_numpy(int)
    observed = frame.observed_pm25.to_numpy(float)
    count = int(day.max() - day.min() + 1)
    table = np.zeros((count, 3), float)
    for column, values in enumerate((np.ones(len(day)), np.square(control - observed), np.square(candidate - observed))):
        np.add.at(table[:, column], day - day.min(), values)
    rng = np.random.default_rng(20260912)
    differences = []
    for _ in range(repeats):
        starts = rng.integers(0, max(1, count - 6), size=int(np.ceil(count / 7)))
        totals = table[(starts[:, None] + np.arange(7)).ravel()[:count]].sum(axis=0)
        if totals[0]:
            differences.append(float(np.sqrt(totals[2] / totals[0]) - np.sqrt(totals[1] / totals[0])))
    return {
        "definition": "candidate minus control RMSE; negative favours candidate",
        "replicates": len(differences), "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
    }


def execute(args: argparse.Namespace) -> dict:
    data, reference, output = Path(args.data_dir).resolve(), Path(args.reference_bundle).resolve(), inside_repo(args.output_dir)
    if output.is_relative_to(data) or output.is_relative_to(reference):
        raise ValueError("Output overlaps a read-only input")
    device = resolve_device()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = OUTLIER / "frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    metadata = json.loads((data / "metadata.json").read_text())
    history_channel = resolve_history_channel(metadata, protocol, args.history_feature)
    required = [
        protocol_path, data / "metadata.json", data / "features_float16.npy",
        data / "target_pm25.npy", data / "target_weight.npy",
        data / "laqn_cell_target.npy", data / "station_daily.npz", data / "gla_mask.npy",
    ]
    before = {str(path): sha256(path) for path in required}
    old_manifest = output / "input_manifest.json"
    if old_manifest.exists() and json.loads(old_manifest.read_text())["sha256"] != before:
        raise ValueError("Inputs changed since this run began")
    json_save(old_manifest, {"sha256": before, "history_feature": args.history_feature})

    features = np.load(data / "features_float16.npy", mmap_mode="r")
    if features.shape != (1461, 71, 48, 64):
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    mask = np.load(data / "gla_mask.npy") > 0
    temporal_raw = build_past_only_features(np.asarray(features[:, history_channel], np.float32))
    budget = Budget(output / "budget.json", args.gpu_hours)
    summary = {
        "status": "running", "history_feature": args.history_feature,
        "device": str(device),
        "temporal_features": list(TEMPORAL_NAMES), "screening": {}, "selection": {}, "final": {},
        "normal_range_protection": "candidate must not worsen MAE below 15 ug/m3 or introduce positive bias above the guard",
        "test_labels_modified": False, "same_day_pm25_added": False,
    }
    try:
        phase = "selection"
        prep = prepare(data, features, metadata, protocol, phase)
        centres, scales = fit_temporal_scaler(temporal_raw, mask, 730)
        temporal = scale_temporal(temporal_raw, centres, scales)
        validation = station_rows(data, 730, 1095)
        baseline, selected_log_weight, baseline_manifest = build_or_load_baseline(
            data, features, prep, mask, 730, output / "baselines" / phase,
            validation, None,
        )
        summary["baselines"] = {phase: baseline_manifest}
        predictions: dict[str, np.ndarray] = {}
        for name, variant in VARIANTS.items():
            checkpoint = train(
                data, features, prep, baseline, temporal, name, 42,
                protocol["hybrid_schedules"]["42"], output / "screening" / name,
                protocol["hybrid_alpha"], budget, device,
            )
            z = infer(
                checkpoint, features, prep, baseline, temporal, name, 730, 1095,
                protocol["hybrid_alpha"], budget, device,
            )
            maps = inverse_target(z, prep["target_log_mean"], prep["target_log_std"])
            predicted = gather(maps, validation, 730)
            predictions[name] = predicted
            metrics = save_evaluation(output / "screening", validation, predicted, name)
            summary["screening"][name] = metrics
            json_save(output / "summary.json", summary)
            log(f"{name}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

        control = summary["screening"]["A_current_hybrid"]
        eligible = [
            "B_temporal"
        ] if safe_candidate(summary["screening"]["B_temporal"], control) else []
        chosen = min(eligible, key=lambda name: summary["screening"][name]["rmse"]) if eligible else None
        summary["selection"] = {
            "eligible": eligible, "selected": chosen,
            "rule": "lower overall RMSE, higher R2, lower >=25 MAE, no worse <15 MAE, and guarded <15 bias",
        }
        json_save(output / "selection_frozen.json", summary["selection"])
        json_save(output / "summary.json", summary)

        summary["final_scope"] = (
            "retrospective fixed-recipe temporal-versus-control comparison; "
            "2024 does not select settings"
        )
        if args.evaluate_2024:
            prep = prepare(data, features, metadata, protocol, "final")
            centres, scales = fit_temporal_scaler(temporal_raw, mask, 1095)
            temporal = scale_temporal(temporal_raw, centres, scales)
            test = station_rows(data, 1095, 1461)
            baseline, _, baseline_manifest = build_or_load_baseline(
                data, features, prep, mask, 1095, output / "baselines" / "final",
                None, selected_log_weight,
            )
            summary["baselines"]["final"] = baseline_manifest
            final_predictions = {}
            for name in VARIANTS:
                members = []
                for seed in SEEDS:
                    checkpoint = train(
                        data, features, prep, baseline, temporal, name, seed,
                        protocol["hybrid_schedules"][str(seed)], output / "final" / name / f"seed_{seed}",
                        protocol["hybrid_alpha"], budget, device,
                    )
                    z = infer(
                        checkpoint, features, prep, baseline, temporal, name, 1095, 1461,
                        protocol["hybrid_alpha"], budget, device,
                    )
                    members.append(z)
                ensemble_z = np.mean(members, axis=0)
                maps = inverse_target(ensemble_z, prep["target_log_mean"], prep["target_log_std"])
                final_predictions[name] = gather(maps, test, 1095)
                summary["final"][name] = save_evaluation(output / "final", test, final_predictions[name], name)
            summary["final"]["paired_block_interval"] = {
                "B_temporal": block_interval(
                    test,
                    final_predictions["A_current_hybrid"],
                    final_predictions["B_temporal"],
                )
            }
        summary["status"] = "complete"
    except BudgetExceeded as error:
        summary["status"], summary["reason"] = "partial_budget_exhausted", str(error)
    except Exception as error:
        summary["status"], summary["reason"] = "failed", str(error)
        raise
    finally:
        after = {str(path): sha256(path) for path in required}
        summary["input_integrity_passed"] = before == after
        summary["gpu_budget_seconds_used"] = budget.used
        json_save(output / "input_integrity.json", {"before": before, "after": after, "unchanged": before == after})
        json_save(output / "summary.json", summary)
        if before != after:
            raise AssertionError("Read-only inputs changed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--history-feature", required=True, help="Exact metadata name of a causal lag/prior PM2.5 predictor")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-hours", type=float, default=4.0)
    parser.add_argument(
        "--evaluate-2024", action="store_true",
        help="Retrospectively refit and score the control and temporal model on 2024",
    )
    args = parser.parse_args()
    if not 0 < args.gpu_hours <= 8:
        parser.error("GPU budget must be positive and at most eight hours")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    execute(args)


if __name__ == "__main__":
    main()
