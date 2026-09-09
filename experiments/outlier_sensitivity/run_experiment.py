"""Pre-registered, training-target-only winsorisation experiment. No HGB fitting.

Run from Aitken; all inputs outside it are read-only, all outputs stay inside it.
The test-year evaluation is gated on a written 2023 selection decision.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from frozen_helpers import fit_preprocessing, gather, inverse_target, station_rows
from london_unet_model import LondonResidualUNet, parameter_count

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TREATMENTS = {"control": None, "cap_995": 0.995, "cap_990": 0.99}
SEEDS = [42, 11, 22]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def inside_repo(path):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(REPO) or resolved == REPO:
        raise ValueError(f"Output must be a descendant of Aitken: {path}")
    # Never allow experiment outputs to modify repository controls.
    if resolved.relative_to(REPO).parts[0] in {".git", ".codex", ".agents"}:
        raise ValueError("Protected output directory")
    return resolved


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def json_save(path, value):
    path = inside_repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def torch_save(path, value):
    path = inside_repo(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def fit_cap(targets, recorded_weights, train_stop, quantile):
    """No validation/test values influence the cap; zero-weight/NaNs excluded."""
    raw = np.asarray(targets[:train_stop])
    observed = np.isfinite(raw) & (recorded_weights[:train_stop] > 0)
    values = raw[observed]
    if not values.size:
        raise ValueError("No observed training targets")
    if quantile is not None and quantile not in (0.99, 0.995):
        raise ValueError("Only the pre-registered caps may be tested")
    cap = None if quantile is None else float(np.quantile(values, quantile))
    affected = np.zeros(raw.shape, bool) if cap is None else observed & (raw > cap)
    return {"quantile": quantile, "cap_pm25": cap,
            "training_cell_days": int(values.size), "affected_cell_days": int(affected.sum()),
            "affected_percent": float(100 * affected.sum() / values.size),
            "uncapped_training_target_hash": array_hash(raw),
            "training_window_stop_exclusive": train_stop}


def capped_copy(raw, observed, cap):
    capped = np.array(raw, copy=True, dtype=np.float32)
    if cap is not None:
        use = observed & (capped > cap)
        capped[use] = cap
    return capped


def training_target(raw, recorded_weight, laqn, prep, cap, baseline=None):
    """Weights are ALWAYS derived from the original, uncapped target."""
    observed = np.isfinite(raw) & (recorded_weight > 0)
    if np.any(raw[observed] < 0):
        raise ValueError("Negative observed target")
    original = np.where(observed, raw, 0).astype(np.float32)
    lookup = np.asarray(prep["lds_lookup"], np.float32)
    edges = np.asarray(prep["lds_edges"], np.float32)
    bins = np.clip(np.searchsorted(edges, original, side="right") - 1, 0, len(lookup) - 1)
    source = np.where(observed & np.isfinite(laqn), 1., np.where(observed, .15, 0.))
    weights = (source * lookup[bins]).astype(np.float32)
    clean = capped_copy(original, observed, cap)
    target = (np.log1p(clean) - prep["target_log_mean"]) / prep["target_log_std"]
    if baseline is not None:
        target = target - baseline
    target = np.where(observed, target, 0.).astype(np.float32)
    return target, weights


def inputs(features, days, prep, baseline):
    x = np.asarray(features[days], dtype=np.float32)[:, prep["channels"]]
    x -= np.asarray(prep["feature_centres"], np.float32)[None, :, None, None]
    x /= np.asarray(prep["feature_scales"], np.float32)[None, :, None, None]
    if baseline is not None:
        x = np.concatenate([x, np.asarray(baseline[days, None], np.float32)], axis=1)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite input")
    return np.ascontiguousarray(x)


class TrainingDays(Dataset):
    def __init__(self, data, features, prep, cap, baseline):
        self.features, self.prep, self.cap, self.baseline = features, prep, cap, baseline
        self.raw = np.load(data / "target_pm25.npy", mmap_mode="r")
        self.weight = np.load(data / "target_weight.npy", mmap_mode="r")
        self.laqn = np.load(data / "laqn_cell_target.npy", mmap_mode="r")

    def __len__(self):
        return self.prep["train_stop_exclusive"]

    def __getitem__(self, day):
        x = inputs(self.features, [day], self.prep, self.baseline)[0]
        base = None if self.baseline is None else np.asarray(self.baseline[day], np.float32)
        y, w = training_target(self.raw[day], self.weight[day], self.laqn[day], self.prep, self.cap, base)
        return torch.from_numpy(x), torch.from_numpy(y[None]), torch.from_numpy(w[None])


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Cumulative wall time spent in training/inference, including resumed work."""
    def __init__(self, path, limit):
        self.path, self.limit = path, limit
        self.used = json.loads(path.read_text())["used_seconds"] if path.exists() else 0.

    def check(self):
        if self.used >= self.limit:
            raise BudgetExceeded(f"Two-GPU-hour budget reached ({self.used:.1f}s)")

    def add(self, seconds):
        self.used += seconds
        json_save(self.path, {"used_seconds": self.used, "limit_seconds": self.limit})


def train(data, features, prep, cap, baseline, family, seed, schedule, out, budget):
    out = inside_repo(out)
    out.mkdir(parents=True, exist_ok=True)
    config = {"family": family, "seed": seed, "schedule": schedule, "cap": cap, "prep": prep,
              "runner_sha256": sha256(Path(__file__)), "backbone_sha256": sha256(HERE / "london_unet_model.py")}
    contract = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    final = out / "final_ema_checkpoint.pt"
    if (out / "completed.json").exists():
        done = json.loads((out / "completed.json").read_text())
        if done["contract"] != contract or sha256(final) != done["checkpoint_sha256"]:
            raise ValueError("Completed run conflicts with configuration")
        return final
    budget.check()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Same architecture/optimiser as before; deterministic kernels improve pairing.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(seed)
    model = LondonResidualUNet(66 if family == "standalone" else 67, 32, .12).cuda()
    initial_hash = array_hash(torch.cat([p.detach().flatten().cpu() for p in model.parameters()]).numpy())
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    amp = torch.amp.GradScaler("cuda")
    loader = DataLoader(TrainingDays(data, features, prep, cap, baseline), batch_size=8,
                        shuffle=True, generator=generator, num_workers=0, pin_memory=True)
    first, history = 1, []
    resume = out / "resume_state.pt"
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
    log(f"{family}/{out.name}: seed {seed}, cap {cap}, {len(schedule)} fixed epochs, {parameter_count(model):,} parameters")
    for epoch in range(first, len(schedule) + 1):
        budget.check()
        torch.cuda.synchronize()
        started = time.monotonic()
        for group in optimizer.param_groups:
            group["lr"] = schedule[epoch - 1]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for index, (x, y, weight) in enumerate(loader):
            x, y, weight = [t.cuda(non_blocking=True) for t in (x, y, weight)]
            # Preserve each trainer's handling of the final partial accumulation.
            divisor = min(2, len(loader) - (index // 2) * 2) if family == "standalone" else 2
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = (F.smooth_l1_loss(pred, y, beta=1., reduction="none") * weight).sum() / weight.sum().clamp_min(1.)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            amp.scale(loss / divisor).backward()
            total_loss += float(loss.detach())
            if (index + 1) % 2 == 0 or index + 1 == len(loader):
                amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                amp.step(optimizer)
                amp.update()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for a, b in zip(ema.parameters(), model.parameters()):
                        a.lerp_(b, .02)
                    for a, b in zip(ema.buffers(), model.buffers()):
                        a.copy_(b)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        history.append({"epoch": epoch, "lr": schedule[epoch - 1], "loss": total_loss / len(loader), "seconds": elapsed})
        torch_save(resume, {"contract": contract, "model": model.state_dict(), "ema": ema.state_dict(),
                           "optimizer": optimizer.state_dict(), "amp": amp.state_dict(), "rng": rng_state(generator),
                           "epoch": epoch, "history": history})
        budget.add(elapsed)
        json_save(out / "history.json", history)
        log(f"  {epoch}/{len(schedule)} loss={total_loss/len(loader):.5f}, {elapsed:.1f}s")
    torch_save(final, {"state": ema.state_dict(), "config": config, "contract": contract})
    json_save(out / "completed.json", {"contract": contract, "initial_parameter_hash": initial_hash,
              "checkpoint_sha256": sha256(final), "epochs": len(schedule), "seed": seed,
              "training_seconds": sum(x["seconds"] for x in history)})
    return final


@torch.inference_mode()
def infer(checkpoint, features, prep, baseline, family, first, stop, budget):
    budget.check()
    started = time.monotonic()
    model = LondonResidualUNet(66 if family == "standalone" else 67, 32, .12).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state"])
    z = np.empty((stop - first, *features.shape[-2:]), np.float32)
    for begin in range(first, stop, 16):
        days = np.arange(begin, min(begin + 16, stop))
        x = torch.from_numpy(inputs(features, days, prep, baseline)).cuda()
        with torch.amp.autocast("cuda"):
            p = model(x)[:, 0]
        z[begin-first:begin-first+len(days)] = p.float().cpu().numpy()
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite prediction")
    torch.cuda.synchronize()
    budget.add(time.monotonic() - started)
    return z


def metrics(y, pred):
    y, pred = np.asarray(y, np.float64), np.asarray(pred, np.float64)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Empty/nonfinite evaluation; rows must not be silently dropped")
    e = pred - y
    high = y >= 25
    def subset(mask):
        return {"n": int(mask.sum()), "mae": float(np.abs(e[mask]).mean()) if mask.any() else None,
                "rmse": float(np.sqrt(np.square(e[mask]).mean())) if mask.any() else None,
                "bias": float(e[mask].mean()) if mask.any() else None}
    sst = np.square(y - y.mean()).sum()
    return {"n": len(y), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(np.square(e).mean())),
            "r2": float(1 - np.square(e).sum() / sst) if sst else None, "bias": float(e.mean()),
            "high": subset(high), "below25": subset(~high),
            "high_recall": float((pred[high] >= 25).mean()) if high.any() else None,
            "high_precision": float(high[pred >= 25].mean()) if (pred >= 25).any() else None}


def eligible(candidate, control):
    return (candidate["rmse"] < control["rmse"] and candidate["mae"] <= control["mae"]
            and candidate["high"]["mae"] is not None and control["high"]["mae"] is not None
            and candidate["high"]["mae"] <= control["high"]["mae"])


def block_ci(frame, control, candidate, repeats=5000):
    """Paired moving blocks of seven consecutive calendar days, including gaps."""
    days = frame.date_idx.to_numpy(int)
    y = frame.observed_pm25.to_numpy(float)
    n = int(days.max() - days.min() + 1)
    table = np.zeros((n, 3), float)
    for j, values in enumerate([np.ones(len(days)), (control-y)**2, (candidate-y)**2]):
        np.add.at(table[:, j], days-days.min(), values)
    rng = np.random.default_rng(20260909)
    differences = []
    for _ in range(repeats):
        starts = rng.integers(0, max(1, n-6), size=int(np.ceil(n/7)))
        indices = (starts[:, None] + np.arange(7)).ravel()[:n]
        totals = table[np.minimum(indices, n-1)].sum(axis=0)
        if totals[0] > 0:
            differences.append(float(np.sqrt(totals[2]/totals[0]) - np.sqrt(totals[1]/totals[0])))
    return {"block_calendar_days": 7, "replicates": len(differences),
            "definition": "candidate minus matched control RMSE; negative favours cap",
            "difference": metrics(y,candidate)["rmse"] - metrics(y,control)["rmse"],
            "ci95": np.quantile(differences, [.025, .975]).tolist()}


def save_evaluation(out, frame, pred, name):
    result = frame.copy()
    result["predicted_pm25"] = pred
    result.to_csv(inside_repo(out / f"{name}_predictions.csv.gz"), index=False)
    scored = metrics(result.observed_pm25, pred)
    monthly = {str(k): metrics(v.observed_pm25, v.predicted_pm25)
               for k, v in result.groupby(result.date.str[:7])}
    daily = []
    for day, rows in result.groupby("date"):
        m = metrics(rows.observed_pm25, rows.predicted_pm25)
        daily.append({"date": day, "observed_mean": float(rows.observed_pm25.mean()),
                      "predicted_mean": float(rows.predicted_pm25.mean()), **{k:m[k] for k in ["n","mae","rmse","bias"]}})
    pd.DataFrame(daily).to_csv(inside_repo(out / f"{name}_daily.csv"), index=False)
    json_save(out / f"{name}_metrics.json", {"overall": scored, "months": monthly})
    return scored


def prepare(data, features, metadata, config, family, phase):
    stop = 730 if phase == "selection" else 1095
    prep = fit_preprocessing(data, features, metadata, config["channels"], stop)
    if family == "hybrid":
        scaler = config["hybrid_target_scalers"][phase]
        prep["target_log_mean"], prep["target_log_std"] = scaler["mean"], scaler["std"]
        if phase == "selection":
            # The original two-year hybrid uses the prepared cube directly.
            prep["feature_centres"], prep["feature_scales"] = [0.]*66, [1.]*66
    return prep


def concentration(z, prep, baseline, first, alpha):
    combined = z if baseline is None else np.asarray(baseline[first:first+len(z)], np.float32) + alpha*z
    return inverse_target(combined, prep["target_log_mean"], prep["target_log_std"])


def execute(args):
    out = inside_repo(args.output_dir)
    data, reference = Path(args.data_dir).resolve(), Path(args.reference_bundle).resolve()
    if out.is_relative_to(data) or out.is_relative_to(reference):
        raise ValueError("Output overlaps a read-only input directory")
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch required; use the existing Python 3.10 installation")
    torch.set_num_threads(4)
    config = json.loads((HERE / "frozen_protocol.json").read_text())
    metadata = json.loads((data / "metadata.json").read_text())
    required = [data / name for name in ["metadata.json", "features_float16.npy", "target_pm25.npy",
                "target_weight.npy", "laqn_cell_target.npy", "station_daily.npz", "gla_mask.npy"]]
    required += [reference / v["path"] for v in config["references"].values()]
    before = {str(p): sha256(p) for p in required}
    for entry in config["references"].values():
        if before[str(reference / entry["path"])] != entry["sha256"]:
            raise ValueError(f"Frozen reference changed: {entry['path']}")
    old_manifest = out / "input_manifest.json"
    if old_manifest.exists() and json.loads(old_manifest.read_text())["sha256"] != before:
        raise ValueError("Inputs changed since initial run")
    json_save(old_manifest, {"sha256": before, "device": torch.cuda.get_device_name(0), "torch": torch.__version__})
    features = np.load(data / "features_float16.npy", mmap_mode="r")
    raw = np.load(data / "target_pm25.npy", mmap_mode="r")
    weights = np.load(data / "target_weight.npy", mmap_mode="r")
    if features.shape != (1461, 71, 48, 64) or raw.shape != (1461,48,64):
        raise ValueError("Unexpected data dimensions")
    budget = Budget(out / "budget.json", args.gpu_hours*3600)
    summary = {"status": "running", "screening": {}, "selection": {}, "final": {},
               "scope": "training-target-only sensitivity test; not a correction of verified measurement errors",
               "same_day_pm25_inputs_added": False, "hgb_retrained": False, "test_labels_modified": False,
               "existing_dashboard_or_results_replaced": False,
               "recipe": "fixed duration/LR; deterministic matched controls; no best-epoch or alpha retuning"}
    validation = station_rows(data, 730, 1095)
    evaluation_hash = array_hash(validation.observed_pm25.to_numpy())
    try:
        for family in ("standalone", "hybrid"):
            family_dir = inside_repo(out / "screening" / family)
            family_dir.mkdir(parents=True, exist_ok=True)
            prep = prepare(data, features, metadata, config, family, "selection")
            json_save(family_dir / "preprocessing.json", prep)
            baseline = None if family == "standalone" else np.load(reference / config["references"]["selection_baseline"]["path"], mmap_mode="r")
            scores = {}
            for treatment, q in TREATMENTS.items():
                cap = fit_cap(raw, weights, 730, q)
                json_save(family_dir / f"{treatment}_cap.json", cap)
                schedule = config[family+"_schedules"]["42"]
                ckpt = train(data, features, prep, cap["cap_pm25"], baseline, family, 42, schedule,
                             family_dir / treatment, budget)
                z = infer(ckpt, features, prep, baseline, family, 730, 1095, budget)
                maps = concentration(z, prep, baseline, 730, config["hybrid_alpha"])
                pred = gather(maps, validation, 730)
                np.save(inside_repo(family_dir / f"{treatment}_maps.npy"), maps)
                scores[treatment] = {"metrics": save_evaluation(family_dir, validation, pred, treatment), "cap": cap}
                summary["screening"][family] = scores
                json_save(out / "summary.json", summary)
                log(f"{family}/{treatment}: RMSE={scores[treatment]['metrics']['rmse']:.5f}, MAE={scores[treatment]['metrics']['mae']:.5f}, high MAE={scores[treatment]['metrics']['high']['mae']:.5f}")
            eligible_names = [n for n in TREATMENTS if n != "control" and eligible(scores[n]["metrics"], scores["control"]["metrics"])]
            chosen = min(eligible_names, key=lambda n:scores[n]["metrics"]["rmse"]) if eligible_names else None
            initial_hashes = {json.loads((family_dir / n / "completed.json").read_text())["initial_parameter_hash"] for n in TREATMENTS}
            if len(initial_hashes) != 1:
                raise AssertionError("Paired treatments did not start with identical parameters")
            summary["selection"][family] = {"selected": chosen, "eligible": eligible_names,
                  "rule": "2023 RMSE strictly lower; overall and >=25 MAE no worse than matched control"}
            log(f"Selection for {family}: {chosen or 'NO CAP QUALIFIES'}")
        if array_hash(validation.observed_pm25.to_numpy()) != evaluation_hash:
            raise AssertionError("Validation observations changed")
        # This decision is saved before selecting or scoring any 2024 station rows.
        json_save(out / "selection_frozen.json", summary["selection"])
        json_save(out / "summary.json", summary)
        for family, decision in summary["selection"].items():
            chosen = decision["selected"]
            if chosen is None:
                continue
            family_dir = inside_repo(out / "final" / family)
            family_dir.mkdir(parents=True, exist_ok=True)
            prep = prepare(data, features, metadata, config, family, "final")
            json_save(family_dir / "preprocessing.json", prep)
            baseline = None if family == "standalone" else np.load(reference / config["references"]["final_baseline"]["path"], mmap_mode="r")
            maps_by_treatment = {}
            for treatment in ["control", chosen]:
                cap = fit_cap(raw, weights, 1095, TREATMENTS[treatment])
                json_save(family_dir / f"{treatment}_cap.json", cap)
                members = []
                for seed in SEEDS:
                    ckpt = train(data, features, prep, cap["cap_pm25"], baseline, family, seed,
                              config[family+"_schedules"][str(seed)], family_dir / treatment / f"seed_{seed}", budget)
                    members.append(infer(ckpt, features, prep, baseline, family, 1095, 1461, budget))
                maps = concentration(np.mean(members, axis=0), prep, baseline, 1095, config["hybrid_alpha"])
                maps_by_treatment[treatment] = maps
                np.save(inside_repo(family_dir / f"{treatment}_maps.npy"), maps)
            json_save(family_dir / "prediction_freeze.json", {n:array_hash(m) for n,m in maps_by_treatment.items()})
            test = station_rows(data, 1095, 1461)
            pred = {n:gather(m, test, 1095) for n,m in maps_by_treatment.items()}
            summary["final"][family] = {n:save_evaluation(family_dir, test, p, n) for n,p in pred.items()}
            summary["final"][family]["paired_block_interval"] = block_ci(test, pred["control"], pred[chosen])
            json_save(out / "summary.json", summary)
        summary["status"] = "complete"
    except BudgetExceeded as exc:
        summary["status"], summary["reason"] = "partial_budget_exhausted", str(exc)
        log(str(exc))
    except Exception as exc:
        summary["status"], summary["reason"] = "failed", str(exc)
        raise
    finally:
        after = {str(p):sha256(p) for p in required}
        summary["input_integrity_passed"] = before == after
        summary["validation_observations_unchanged"] = array_hash(validation.observed_pm25.to_numpy()) == evaluation_hash
        summary["gpu_budget_seconds_used"] = budget.used
        json_save(out / "input_integrity.json", {"unchanged": before == after, "before": before, "after": after})
        json_save(out / "summary.json", summary)
        if before != after:
            raise AssertionError("Read-only input integrity failed")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--reference-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-hours", type=float, default=2.)
    args = parser.parse_args()
    if not 0 < args.gpu_hours <= 2:
        parser.error("Budget must be positive and at most two GPU-hours")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    execute(args)


if __name__ == "__main__":
    main()
