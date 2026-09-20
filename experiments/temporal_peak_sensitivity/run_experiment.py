"""Bounded causal-history and final-output-loss ablation; no HGB fitting.

All source data/reference models are read-only. Only this isolated experiment's
artifacts and curated report are written. 2024 is gated and retrospective.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FROZEN = HERE.parent / "outlier_sensitivity"
sys.path.insert(0, str(FROZEN))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from experiments.outlier_sensitivity.run_experiment import (
    prepare, training_target, rng_state, restore_rng,
)
from frozen_helpers import gather, inverse_target, station_rows
from london_unet_model import LondonResidualUNet
from methods import (
    build_features, fit_scaler, apply_scaler, raw_training_scale,
    matched_model, loss as training_loss,
)
from reporting import metrics, safe_candidate, block_interval, write_report

SEEDS = (42, 11, 22)
HISTORY_FEATURE = "pm25_lag1_idw_causal"
ARTIFACT_ROOT = REPO / "artifacts" / "temporal_peak_sensitivity"
REPORT_ROOT = REPO / "comparison" / "temporal_peak_sensitivity"
VARIANTS = {
    "standalone_control": dict(family="standalone", features="none", aux=0., is_control=True),
    "hybrid_original_control": dict(family="hybrid", features="none", aux=0., is_control=True),
    "hybrid_temporal_control": dict(family="hybrid", features="temporal", aux=0., is_control=True),
}
for _family in ("standalone", "hybrid"):
    _prefix = "temporal_" if _family == "hybrid" else ""
    for _name, _features, _aux in (
        ("ewma", _prefix + "ewma", 0.),
        ("median", _prefix + "median", 0.),
        ("final_loss", "temporal" if _prefix else "none", .1),
        ("combined", _prefix + "ewma", .1),
    ):
        VARIANTS[f"{_family}_{_name}"] = dict(
            family=_family, features=_features, aux=_aux, is_control=False)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def output_path(path):
    value = Path(path).resolve()
    root = ARTIFACT_ROOT.resolve()
    if not value.is_relative_to(root) or value == root:
        raise ValueError("Run outputs must be below Aitken/artifacts/temporal_peak_sensitivity")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((value.shape, value.dtype)).encode())
    if value.size:
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def save_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def save_torch(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def freeze_json(path, value):
    """A pre-existing decision/manifest is immutable, including on resume."""
    path = output_path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Immutable experiment record mismatch: {path.name}")
    else:
        save_json(path, value)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Cumulative training/inference wall time; setup/reporting is excluded.

    Charge at minibatch boundaries, including interrupted partial epochs. A
    resumed partial epoch restarts from its last complete checkpoint, but all
    previously consumed compute remains charged. One batch is the stop latency.
    """
    def __init__(self, path, seconds, initial_used=0.):
        self.path = output_path(path)
        self.limit = min(float(seconds), 7200.)
        if not 0 < self.limit <= 7200:
            raise ValueError("Budget must be positive and at most two GPU-hours")
        self.used = float(initial_used)
        if not np.isfinite(self.used) or self.used < 0:
            raise ValueError("Invalid carried-forward compute time")
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if old["limit_seconds"] != self.limit:
                raise ValueError("Cannot silently change a resumed budget")
            self.used = old["used_seconds"]
        self.started = None

    def record(self):
        return dict(used_seconds=self.used, limit_seconds=self.limit,
                    accounting="training/inference elapsed wall seconds, one batch stop latency")

    def check(self):
        if self.used >= self.limit:
            raise BudgetExceeded("Two-GPU-hour training/inference budget exhausted")

    def begin(self):
        self.check()
        torch.cuda.synchronize()
        self.started = time.monotonic()

    def charge(self):
        if self.started is not None:
            torch.cuda.synchronize()
            now = time.monotonic()
            self.used += now - self.started
            self.started = now
            save_json(self.path, self.record())

    def end(self):
        self.charge()
        self.started = None


def model_inputs(features, days, prep, extra, baseline):
    base = np.asarray(features[days], np.float32)[:, prep["channels"]]
    base -= np.asarray(prep["feature_centres"], np.float32)[None, :, None, None]
    base /= np.asarray(prep["feature_scales"], np.float32)[None, :, None, None]
    parts = [base]
    if extra.shape[1]:
        parts.append(extra[days])
    if baseline is not None:
        parts.append(np.asarray(baseline[days, None], np.float32))
    result = np.ascontiguousarray(np.concatenate(parts, axis=1))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite model inputs")
    return result


class TrainingDays(Dataset):
    def __init__(self, data, features, prep, extra, baseline):
        self.features, self.prep, self.extra, self.baseline = features, prep, extra, baseline
        self.raw = np.load(data / "target_pm25.npy", mmap_mode="r")
        self.weight = np.load(data / "target_weight.npy", mmap_mode="r")
        self.laqn = np.load(data / "laqn_cell_target.npy", mmap_mode="r")

    def __len__(self):
        return self.prep["train_stop_exclusive"]

    def __getitem__(self, day):
        x = model_inputs(self.features, [day], self.prep, self.extra, self.baseline)[0]
        # No capping, unchanged weights; absolute standardized target for both families.
        z, weight = training_target(self.raw[day], self.weight[day], self.laqn[day],
                                    self.prep, cap=None, baseline=None)
        return tuple(torch.from_numpy(v) for v in (x, z[None], weight[None]))


def init_hashes(model, base_names, full_names):
    common, added = {}, {}
    positions = [full_names.index(n) for n in base_names]
    new_positions = [i for i, n in enumerate(full_names) if n not in base_names]
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().numpy()
        if name in ("encoder1.conv1.weight", "encoder1.skip.weight"):
            common[name] = array_hash(value[:, positions])
            if new_positions:
                added[name] = bool(np.all(value[:, new_positions] == 0))
        else:
            common[name] = array_hash(value)
    return dict(common_parameter_hash=digest_json(common), added_channels_zero=all(added.values()))


def optimizer_step(model, optimizer, amp):
    """Preserve GradScaler's normal overflow/skip handling during warmup.

    unscale_ records nonfinite gradients BEFORE clipping. The scaler then skips
    that optimizer update and reduces its scale, preserving finite parameters.
    Raising on the gradient norm would prevent this inherited AMP behavior.
    """
    amp.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=False)
    if not amp.is_enabled() and not torch.isfinite(norm):
        raise ValueError("Nonfinite gradient without an active loss scaler")
    scale = amp.get_scale()
    amp.step(optimizer)
    amp.update()
    optimizer.zero_grad(set_to_none=True)
    return amp.get_scale() < scale


def run_member(ctx, phase, name, seed):
    variant = VARIANTS[name]
    family = variant["family"]
    config = ctx["protocol"]
    phase_key = "final" if phase == "final" else "selection"
    prepared = ctx["prepared"][(family, phase_key)]
    prep, baseline = prepared["prep"], prepared["baseline"]
    extra, extra_names, scaler = ctx["extras"][(phase_key, variant["features"])]
    base_names = list(prep["feature_names"])
    if family == "hybrid":
        base_names.append("fixed_hgb_z")
    full_names = list(prep["feature_names"]) + list(extra_names)
    if family == "hybrid":
        full_names.append("fixed_hgb_z")
    schedule = config[family + "_schedules"][str(seed)]
    contract_data = dict(
        version=1, manifest=ctx["manifest_digest"], variant=name, recipe=variant,
        seed=seed, schedule=schedule, prep=prep, raw_scale=prepared["raw_scale"],
        feature_names=full_names, feature_scaler=scaler, feature_array_hash=scaler["scaled_hash"],
        baseline_hash=prepared["baseline_hash"], alpha=config["hybrid_alpha"],
        initialization="canonical common weights, zero added input weights, canonical RNG state",
        decoding="float32 log clip [0,log1p(500)] then expm1; ensemble in z space",
    )
    contract = digest_json(contract_data)
    out = output_path(ctx["out"] / phase_key / name / f"seed_{seed}")
    out.mkdir(parents=True, exist_ok=True)
    freeze_json(out / "contract.json", contract_data)
    checkpoint = out / "final_ema_checkpoint.pt"
    completed = out / "completed.json"
    if completed.exists():
        done = json.loads(completed.read_text())
        if done["contract"] != contract or sha256(checkpoint) != done["checkpoint_sha256"]:
            raise ValueError("Completed checkpoint contract/hash mismatch")
    else:
        train_member(ctx, out, name, seed, contract, contract_data, prep, extra,
                     baseline, base_names, full_names, schedule, prepared["raw_scale"])
    prediction = out / "prediction_z.npy"
    pred_record = out / "prediction.json"
    first, stop = (1095, 1461) if phase == "final" else (730, 1095)
    inference_contract = dict(training_contract=contract, checkpoint_sha256=sha256(checkpoint),
                              first=first, stop=stop)
    if pred_record.exists():
        record = json.loads(pred_record.read_text())
        if record["contract"] != inference_contract or sha256(prediction) != record["sha256"]:
            raise ValueError("Saved predictions conflict with model or recipe")
        z = np.load(prediction)
    else:
        z = infer(ctx, checkpoint, prep, extra, baseline, full_names, first, stop)
        np.save(output_path(prediction), z)
        save_json(pred_record, dict(contract=inference_contract, sha256=sha256(prediction)))
    return z, prep


def train_member(ctx, out, name, seed, contract, contract_data, prep, extra,
                 baseline, base_names, full_names, schedule, raw_scale):
    budget = ctx["budget"]
    budget.check()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    model = matched_model(seed, base_names, full_names).cuda()
    initial = init_hashes(model, base_names, full_names)
    if not initial["added_channels_zero"]:
        raise AssertionError("New input channels did not start at zero")
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    amp = torch.amp.GradScaler("cuda")
    loader = DataLoader(TrainingDays(ctx["data"], ctx["features"], prep, extra, baseline),
                        batch_size=8, shuffle=True, generator=generator, num_workers=0, pin_memory=True)
    resume = out / "resume_state.pt"
    first, history = 1, []
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["contract"] != contract:
            raise ValueError("Resume checkpoint contract mismatch")
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        amp.load_state_dict(state["amp"])
        restore_rng(state["rng"], generator)
        first, history = state["epoch"] + 1, state["history"]

    def snapshot(epoch):
        save_torch(resume, dict(contract=contract, model=model.state_dict(), ema=ema.state_dict(),
                               optimizer=optimizer.state_dict(), amp=amp.state_dict(),
                               rng=rng_state(generator), epoch=epoch, history=history))

    if not resume.exists():
        snapshot(0)
    log(f"TRAIN {name} seed={seed} epochs={len(schedule)} starting={first}")
    for epoch in range(first, len(schedule) + 1):
        budget.begin()
        started = time.monotonic()
        total = 0.
        skipped_updates = 0
        try:
            for group in optimizer.param_groups:
                group["lr"] = schedule[epoch-1]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for index, (x, target_z, weight) in enumerate(loader):
                budget.check()
                x, target_z, weight = [v.cuda(non_blocking=True) for v in (x, target_z, weight)]
                base_z = None if baseline is None else x[:, -1:]
                with torch.amp.autocast("cuda"):
                    prediction = model(x)
                    loss, _ = training_loss(prediction, target_z, weight, base_z, prep,
                                            raw_scale, VARIANTS[name]["aux"])
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite loss: {name}")
                divisor = (min(2, len(loader) - (index//2)*2)
                           if VARIANTS[name]["family"] == "standalone" else 2)
                amp.scale(loss/divisor).backward()
                total += float(loss.detach())
                if (index+1) % 2 == 0 or index+1 == len(loader):
                    skipped_updates += int(optimizer_step(model, optimizer, amp))
                    with torch.no_grad():
                        for averaged, current in zip(ema.parameters(), model.parameters()):
                            averaged.lerp_(current, .02)
                        for averaged, current in zip(ema.buffers(), model.buffers()):
                            averaged.copy_(current)
                budget.charge()
        finally:
            budget.end()
        history.append(dict(epoch=epoch, lr=schedule[epoch-1], loss=total/len(loader),
                            seconds=time.monotonic()-started, amp_skipped_updates=skipped_updates))
        snapshot(epoch)
        save_json(out / "history.json", history)
        log(f"  {name}/{seed} {epoch}/{len(schedule)} loss={total/len(loader):.5f}")
    checkpoint = out / "final_ema_checkpoint.pt"
    save_torch(checkpoint, dict(state=ema.state_dict(), config=contract_data, contract=contract))
    save_json(out / "completed.json", dict(contract=contract, checkpoint_sha256=sha256(checkpoint),
              epochs=len(schedule), seed=seed, **initial))


@torch.inference_mode()
def infer(ctx, checkpoint, prep, extra, baseline, names, first, stop):
    model = LondonResidualUNet(len(names), 32, .12).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state"])
    result = np.empty((stop-first, *ctx["features"].shape[-2:]), np.float32)
    budget = ctx["budget"]
    budget.begin()
    try:
        for begin in range(first, stop, 16):
            budget.check()
            days = np.arange(begin, min(begin+16, stop))
            x = torch.from_numpy(model_inputs(ctx["features"], days, prep, extra, baseline)).cuda()
            with torch.amp.autocast("cuda"):
                prediction = model(x)
            final_z = prediction.float() if baseline is None else x[:, -1:] + .425*prediction.float()
            result[begin-first:begin-first+len(days)] = final_z[:, 0].cpu().numpy()
            budget.charge()
    finally:
        budget.end()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite predictions")
    return result


def evaluate(ctx, phase, name, seeds):
    members, prep = [], None
    for seed in seeds:
        z, prep = run_member(ctx, phase, name, seed)
        members.append(z)
    maps = inverse_target(np.mean(members, axis=0), prep["target_log_mean"], prep["target_log_std"])
    first, stop = (1095, 1461) if phase == "final" else (730, 1095)
    rows = station_rows(ctx["data"], first, stop)
    expected = 9619 if phase == "final" else 11102
    if len(rows) != expected:
        raise ValueError(f"Unexpected evaluation count: {len(rows)} != {expected}")
    rows["predicted_pm25"] = gather(maps, rows, first)
    folder = output_path(ctx["out"] / phase / "predictions")
    folder.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path(folder / f"{name}.csv.gz"), index=False)
    np.save(output_path(folder / f"{name}_maps.npy"), maps)
    result = metrics(rows)
    result["member_metrics"] = {}
    for seed, member in zip(seeds, members):
        single = rows.copy()
        single["predicted_pm25"] = gather(inverse_target(member, prep["target_log_mean"], prep["target_log_std"]), rows, first)
        result["member_metrics"][str(seed)] = metrics(single)
    result["seeds"] = list(seeds)
    ctx["summary"]["phases"][phase][name] = result
    save_json(ctx["out"] / "summary.json", ctx["summary"])
    log(f"SCORE {phase}/{name}: RMSE={result['rmse']:.6f} MAE={result['mae']:.6f} highMAE={result['bands']['high_at_least_25']['mae']:.6f}")
    return result


def controls(family):
    return ["standalone_control"] if family == "standalone" else ["hybrid_original_control", "hybrid_temporal_control"]


def select_candidate(scores, family):
    refs = [scores[name] for name in controls(family)]
    eligible = [name for name in scores if VARIANTS[name]["family"] == family
                and not VARIANTS[name]["is_control"] and safe_candidate(scores[name], refs)]
    # Fixed, explicit complexity tie break: fewer extra channels, then no aux.
    complexity = {"final_loss": 0, "median": 2, "ewma": 3, "combined": 4}
    chosen = min(eligible, key=lambda n:(scores[n]["rmse"], scores[n]["bands"]["high_at_least_25"]["mae"],
                    complexity[n.removeprefix(family+"_")], n)) if eligible else None
    return chosen, eligible


def verify_initializations(out):
    grouped = {}
    count = 0
    for path in out.glob("*/**/completed.json"):
        data = json.loads(path.read_text())
        contract = json.loads((path.parent / "contract.json").read_text())
        if sha256(path.parent / "final_ema_checkpoint.pt") != data["checkpoint_sha256"]:
            raise AssertionError("Checkpoint changed after completion")
        key = (path.relative_to(out).parts[0], contract["recipe"]["family"], contract["seed"])
        grouped.setdefault(key, set()).add(data["common_parameter_hash"])
        if not data["added_channels_zero"]:
            raise AssertionError("Added weights not zero initialized")
        count += 1
    if any(len(hashes) != 1 for hashes in grouped.values()):
        raise AssertionError("Matched models have different common initial parameters")
    return dict(checkpoints_verified=count, common_initialization_matches=True, added_weights_zero=True)


def setup(args):
    out = output_path(args.output_dir)
    data, reference = Path(args.data_dir).resolve(), Path(args.reference_bundle).resolve()
    for source in (data, reference):
        if out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("Output overlaps read-only input")
    if not torch.cuda.is_available():
        raise RuntimeError("Existing CUDA PyTorch installation required")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    protocol = json.loads((FROZEN / "frozen_protocol.json").read_text())
    if protocol["hybrid_alpha"] != .425:
        raise ValueError("Unexpected frozen hybrid alpha")
    metadata = json.loads((data / "metadata.json").read_text())
    required = [data / n for n in ("metadata.json", "features_float16.npy", "target_pm25.npy",
                "target_weight.npy", "laqn_cell_target.npy", "station_daily.npz", "gla_mask.npy")]
    required += [reference / entry["path"] for entry in protocol["references"].values()]
    sources = list(HERE.glob("*.py")) + [FROZEN / n for n in (
        "run_experiment.py", "frozen_helpers.py", "london_unet_model.py", "frozen_protocol.json")]
    before = {str(p):sha256(p) for p in required}
    for entry in protocol["references"].values():
        if before[str(reference / entry["path"])] != entry["sha256"]:
            raise ValueError(f"Frozen reference mismatch: {entry['path']}")
    manifest = dict(input_sha256=before, code_sha256={str(p.relative_to(REPO)):sha256(p) for p in sources},
        history_feature=HISTORY_FEATURE, variants=VARIANTS, protocol=protocol,
        torch=str(torch.__version__), numpy=np.__version__, device=torch.cuda.get_device_name(0),
        raw_scale_definition="unweighted population standard deviation of finite positive-weight training targets",
        feature_semantics="prepared standardized/clipped lag-1 input; no extra shift; per-pixel causal filters",
        training_limit_seconds=min(args.gpu_hours*3600, 7200.),
        gate="RMSE and high MAE strictly lower; MAE and normal MAE nonincreased; abs normal bias <= max(.1,abs control)")
    carried = 0.
    prior_budget = getattr(args, "prior_budget", None)
    if prior_budget:
        prior = output_path(prior_budget)
        if prior.is_relative_to(out):
            raise ValueError("Prior budget must belong to a different, stopped run")
        carried = float(json.loads(prior.read_text())["used_seconds"])
        manifest["prior_budget"] = dict(path=str(prior), sha256=sha256(prior), used_seconds=carried)
    freeze_json(out / "manifest.json", manifest)
    features = np.load(data / "features_float16.npy", mmap_mode="r")
    raw = np.load(data / "target_pm25.npy", mmap_mode="r")
    weights = np.load(data / "target_weight.npy", mmap_mode="r")
    mask = np.load(data / "gla_mask.npy") > 0
    if features.shape != (1461,71,48,64) or raw.shape != (1461,48,64):
        raise ValueError("Unexpected input shape")
    index = metadata["feature_names"].index(HISTORY_FEATURE)
    if index not in protocol["channels"]:
        raise ValueError("History input not in frozen predictors")
    history = np.asarray(features[:, index], np.float32)
    summary = dict(status="running", phases={p:{} for p in ("screening", "confirmation", "final")},
                   selection={}, variants=VARIANTS, integrity={}, budget={}, intervals={},
                   hgb_retrained=False, targets_modified=False, benchmark="2024 retrospective",
                   temporal_control_note="Friend's five-channel recipe rebased on archived fixed HGB and paired initialization; not a numerical reproduction of the friend's refitted-HGB run")
    ctx = dict(out=out, data=data, reference=reference, features=features, protocol=protocol,
               prepared={}, extras={}, manifest_digest=digest_json(manifest), before=before,
               budget=Budget(out / "budget.json", args.gpu_hours*3600, carried), summary=summary)
    # Preparation for final training is intentionally deferred until validation gates pass.
    ctx.update(metadata=metadata, raw=raw, weights=weights, mask=mask, history=history)
    prepare_phase(ctx, "selection")
    save_json(out / "summary.json", summary)
    return ctx


def prepare_phase(ctx, phase):
    train_stop = 730 if phase == "selection" else 1095
    scale = raw_training_scale(ctx["raw"], ctx["weights"], train_stop)
    for family in ("standalone", "hybrid"):
        prep = prepare(ctx["data"], ctx["features"], ctx["metadata"], ctx["protocol"], family, phase)
        baseline, baseline_hash = None, None
        if family == "hybrid":
            entry = ctx["protocol"]["references"][phase + "_baseline"]
            baseline = np.load(ctx["reference"] / entry["path"], mmap_mode="r")
            baseline_hash = entry["sha256"]
        ctx["prepared"][(family, phase)] = dict(prep=prep, baseline=baseline, raw_scale=scale, baseline_hash=baseline_hash)
        save_json(ctx["out"] / phase / family / "preprocessing.json", dict(prep=prep, raw_scale=scale, baseline_hash=baseline_hash))
    for kind in sorted({v["features"] for v in VARIANTS.values()}):
        values, names = build_features(ctx["history"], kind)
        centres, scales = fit_scaler(values, ctx["mask"], train_stop)
        # Prepared history outside London is zero padding, not an observation.
        # Preserve padding after centering instead of creating artificial signals.
        extra = apply_scaler(values, centres, scales, mask=ctx["mask"])
        scaler = dict(names=list(names), centres=centres.tolist(), scales=scales.tolist(),
                      train_stop=train_stop, history_hash=array_hash(ctx["history"]),
                      unscaled_hash=array_hash(values), scaled_hash=array_hash(extra))
        ctx["extras"][(phase, kind)] = (extra, names, scaler)
        save_json(ctx["out"] / phase / "features" / f"{kind}.json", scaler)


def execute(args):
    ctx = setup(args)
    summary, out = ctx["summary"], ctx["out"]
    try:
        if args.check_only:
            summary["status"] = "preflight_complete"
            return
        for family in ("standalone", "hybrid"):
            for name in controls(family):
                evaluate(ctx, "screening", name, [42])
            for treatment in ("ewma", "median", "final_loss"):
                evaluate(ctx, "screening", f"{family}_{treatment}", [42])
            scores = summary["phases"]["screening"]
            refs = [scores[n] for n in controls(family)]
            combined_allowed = all(safe_candidate(scores[f"{family}_{n}"], refs) for n in ("ewma", "final_loss"))
            if combined_allowed:
                evaluate(ctx, "screening", f"{family}_combined", [42])
            selected, eligible = select_candidate(scores, family)
            summary["selection"][family] = dict(screen_selected=selected, eligible=eligible,
                combined_allowed=combined_allowed, confirmed=False,
                reason="No candidate passed all 2023 screen guards" if selected is None else "Awaiting three-seed validation")
            log(f"SCREEN SELECTION {family}: {selected or 'NONE'}")
        freeze_json(out / "screen_selection_frozen.json", summary["selection"])
        for family, decision in summary["selection"].items():
            chosen = decision["screen_selected"]
            if chosen is None:
                continue
            names = controls(family) + [chosen]
            for name in names:
                evaluate(ctx, "confirmation", name, SEEDS)
            scores = summary["phases"]["confirmation"]
            passing_seeds = [seed for seed in SEEDS if all(
                scores[chosen]["member_metrics"][str(seed)]["rmse"] < scores[c]["member_metrics"][str(seed)]["rmse"]
                for c in controls(family))]
            passed = safe_candidate(scores[chosen], [scores[c] for c in controls(family)]) and len(passing_seeds) >= 2
            decision.update(confirmed=passed, rmse_improved_seeds=passing_seeds,
                            reason="Passed three-seed 2023 confirmation" if passed else "Failed three-seed 2023 confirmation; do not evaluate on 2024")
            log(f"CONFIRMATION {family}: {passed}, improved seeds={passing_seeds}")
        # No new test observations have been scored; decision cannot be changed on resume.
        freeze_json(out / "selection_frozen.json", summary["selection"])
        if any(d["confirmed"] for d in summary["selection"].values()):
            prepare_phase(ctx, "final")
            for family, decision in summary["selection"].items():
                if not decision["confirmed"]:
                    continue
                names = controls(family) + [decision["screen_selected"]]
                for name in names:
                    evaluate(ctx, "final", name, SEEDS)
        for phase in ("screening", "confirmation", "final"):
            summary["intervals"][phase] = {}
            for name in summary["phases"][phase]:
                if VARIANTS[name]["is_control"]:
                    continue
                for control in controls(VARIANTS[name]["family"]):
                    folder = out / phase / "predictions"
                    summary["intervals"][phase][f"{name}_minus_{control}"] = block_interval(
                        pd.read_csv(folder / f"{name}.csv.gz"), pd.read_csv(folder / f"{control}.csv.gz"))
        summary["status"] = "complete"
    except BudgetExceeded as exc:
        summary.update(status="partial_budget_exhausted", reason=str(exc))
        log(str(exc))
    except BaseException as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", reason=str(exc))
        raise
    finally:
        summary["budget"] = ctx["budget"].record()
        after = {p:sha256(p) for p in ctx["before"]}
        summary["integrity"] = dict(source_hashes_unchanged=ctx["before"] == after,
                                     **verify_initializations(out))
        save_json(out / "summary.json", summary)
        if not summary["integrity"]["source_hashes_unchanged"]:
            raise AssertionError("Source data/reference hashes changed")
        if not args.check_only:
            write_report(out, REPORT_ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--reference-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-hours", type=float, default=2.)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prior-budget", help="Carry compute charged to a stopped, superseded run; no checkpoint reuse")
    execute(parser.parse_args())
