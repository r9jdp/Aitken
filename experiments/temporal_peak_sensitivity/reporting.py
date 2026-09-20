"""Original-observation evaluation and reporting for temporal peak experiments.

This module never changes labels, selects a threshold using the benchmark, or
loads checkpoints. Counts describe station-days, not independent observations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
KEYS = ("station_row_index", "date_idx", "date", "site_code", "row", "col")
REQUIRED = (*KEYS, "observed_pm25", "predicted_pm25")
BANDS = ("normal_below_15", "elevated_15_to_25", "high_at_least_25")


def _checked(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Nonempty prediction rows required; missing columns: {sorted(missing)}")
    result = frame.loc[:, REQUIRED].copy().reset_index(drop=True)
    if result.isna().any().any():
        raise ValueError("Missing values in prediction rows")
    for name in ("observed_pm25", "predicted_pm25"):
        result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
        if not np.isfinite(result[name]).all() or (result[name] < 0).any():
            raise ValueError(f"{name} must contain finite nonnegative concentrations")
    for name in ("station_row_index", "date_idx", "row", "col"):
        values = pd.to_numeric(result[name], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(f"Invalid integer key {name}")
        result[name] = values.astype(np.int64)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].dt.tz is not None or not result["date"].equals(result["date"].dt.normalize()):
        raise ValueError("Dates must be timezone-free calendar days")
    result["site_code"] = result["site_code"].astype(str)
    if result["site_code"].str.strip().eq("").any():
        raise ValueError("Empty station code")
    if result["station_row_index"].duplicated().any() or result.duplicated(["site_code", "date"]).any():
        raise ValueError("Duplicate station row or station-day")
    # A date index must preserve calendar-day distances, including missing dates.
    offsets = result["date"] - pd.to_timedelta(result["date_idx"], unit="D")
    if offsets.nunique() != 1:
        raise ValueError("date_idx and calendar dates disagree")
    return result


def _numbers(y: np.ndarray, p: np.ndarray, total_sse: float | None = None) -> dict:
    n = len(y)
    error = p - y
    sse = float(np.square(error).sum())
    sst = float(np.square(y - y.mean()).sum()) if n else 0.0
    return {
        "n": int(n),
        "mae": float(np.abs(error).mean()) if n else None,
        "rmse": float(np.sqrt(sse / n)) if n else None,
        "r2": float(1.0 - sse / sst) if sst > 0 else None,
        "bias": float(error.mean()) if n else None,
        "observed_mean": float(y.mean()) if n else None,
        "predicted_mean": float(p.mean()) if n else None,
        "squared_error": sse,
        "squared_error_percent": float(100.0 * sse / total_sse) if total_sse else None,
    }


def metrics(frame: pd.DataFrame) -> dict:
    """Evaluate untouched, paired station-day concentrations in micrograms/m3."""
    checked = _checked(frame)
    y = checked.observed_pm25.to_numpy(float)
    p = checked.predicted_pm25.to_numpy(float)
    sse = float(np.square(p - y).sum())
    result = _numbers(y, p, sse)
    result.update({
        "n_dates": int(checked.date.nunique()),
        "first_date": checked.date.min().strftime("%Y-%m-%d"),
        "last_date": checked.date.max().strftime("%Y-%m-%d"),
        "bands": {
            name: _numbers(y[use], p[use], sse)
            for name, use in zip(BANDS, (y < 15, (y >= 15) & (y < 25), y >= 25))
        },
        "high_pollution": {
            "threshold": 25.0,
            "observed_at_least_25_count": int((y >= 25).sum()),
            "underprediction_count": int(((y >= 25) & (p < y)).sum()),
            "predicted_at_least_25_count": int((p >= 25).sum()),
            "true_exceedance_count": int(((y >= 25) & (p >= 25)).sum()),
            "false_exceedance_count": int(((y < 25) & (p >= 25)).sum()),
            "missed_exceedance_count": int(((y >= 25) & (p < 25)).sum()),
        },
    })
    # Join by exact calendar date rather than shift across a missing station-day.
    previous = checked[["site_code", "date", "observed_pm25"]].copy()
    previous["date"] += pd.Timedelta(days=1)
    previous = previous.rename(columns={"observed_pm25": "previous_observed"})
    joined = checked.merge(previous, on=["site_code", "date"], how="left", validate="one_to_one", sort=False)
    change = joined.observed_pm25.to_numpy(float) - joined.previous_observed.to_numpy(float)
    available = np.isfinite(change)
    result["rapid_changes"] = {
        "definition": "Same-station observed change from the previous calendar day, for diagnosis only; never an input",
        "available_previous_day_n": int(available.sum()),
        "unavailable_previous_day_n": int((~available).sum()),
        "rise_at_least_10": _numbers(y[change >= 10], p[change >= 10], sse),
        "fall_at_least_10": _numbers(y[change <= -10], p[change <= -10], sse),
        "other_changes": _numbers(y[available & (np.abs(change) < 10)], p[available & (np.abs(change) < 10)], sse),
    }
    boundaries = (0.0, 5.0, 10.0, 15.0, 25.0, float("inf"))
    result["calibration"] = []
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        use = (p >= low) & (p < high)
        result["calibration"].append({
            "predicted_lower_inclusive": low,
            "predicted_upper_exclusive": high if np.isfinite(high) else None,
            **_numbers(y[use], p[use], sse),
        })
    # Explicitly prohibit invalid JSON, including undefined R2 in empty bands.
    json.dumps(result, allow_nan=False)
    return result


def safe_candidate(candidate: dict, controls: list[dict]) -> bool:
    """All predeclared safeguards must pass against every matched control."""
    if not controls:
        return False
    try:
        for control in controls:
            normal = candidate["bands"]["normal_below_15"]
            base_normal = control["bands"]["normal_below_15"]
            high = candidate["bands"]["high_at_least_25"]
            base_high = control["bands"]["high_at_least_25"]
            if candidate["n"] != control["n"] or normal["n"] != base_normal["n"] or high["n"] != base_high["n"]:
                return False
            required = (candidate["rmse"], control["rmse"], candidate["mae"], control["mae"],
                        normal["mae"], base_normal["mae"], normal["bias"], base_normal["bias"],
                        high["mae"], base_high["mae"])
            if not normal["n"] or not high["n"] or any(value is None or not np.isfinite(value) for value in required):
                return False
            if not (
                candidate["rmse"] < control["rmse"]
                and high["mae"] < base_high["mae"]
                and candidate["mae"] <= control["mae"]
                and normal["mae"] <= base_normal["mae"]
                and abs(normal["bias"]) <= max(0.10, abs(base_normal["bias"]))
            ):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _paired(candidate: pd.DataFrame, control: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = _checked(candidate).sort_values("station_row_index").reset_index(drop=True)
    b = _checked(control).sort_values("station_row_index").reset_index(drop=True)
    identity = [*KEYS, "observed_pm25"]
    if not a[identity].equals(b[identity]):
        raise ValueError("Paired evaluation requires identical station keys and unchanged observations")
    return a, b


def block_interval(candidateframe: pd.DataFrame, controlframe: pd.DataFrame,
                   n: int = 5000, seed: int = 20260920) -> dict:
    """Moving seven-calendar-day paired bootstrap, weighted by station-day count.

    Dates without observations remain empty days rather than being compressed.
    This estimates date-sampling uncertainty conditional on fitted models, not
    model-selection or training-seed uncertainty.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("Positive integer bootstrap replicate count required")
    candidate, control = _paired(candidateframe, controlframe)
    day = (candidate.date - candidate.date.min()).dt.days.to_numpy(int)
    count = int(day.max()) + 1
    y = candidate.observed_pm25.to_numpy(float)
    a = candidate.predicted_pm25.to_numpy(float)
    b = control.predicted_pm25.to_numpy(float)
    delta = float(np.sqrt(np.mean((a - y) ** 2)) - np.sqrt(np.mean((b - y) ** 2)))
    result = {
        "definition": "candidate minus control RMSE; negative favours candidate",
        "rmse_delta": delta, "unit": "micrograms/m3", "block_days": 7,
        "calendar_days": count, "replicates": 0, "requested_replicates": n,
        "seed": seed, "ci95": None,
        "limitation": "Conditional on fitted models; excludes selection and training-seed uncertainty",
    }
    if count < 7:
        result["unavailable_reason"] = "Fewer than seven calendar days"
        return result
    table = np.zeros((count, 3), dtype=float)
    for column, values in enumerate((np.ones(len(day)), (b - y) ** 2, (a - y) ** 2)):
        np.add.at(table[:, column], day, values)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n * 10):
        starts = rng.integers(0, count - 6, size=int(np.ceil(count / 7)))
        totals = table[(starts[:, None] + np.arange(7)).ravel()[:count]].sum(axis=0)
        if totals[0] > 0:
            differences.append(float(np.sqrt(totals[2] / totals[0]) - np.sqrt(totals[1] / totals[0])))
        if len(differences) == n:
            break
    result["replicates"] = len(differences)
    if differences:
        result["ci95"] = np.quantile(differences, [0.025, 0.975]).tolist()
    else:
        result["unavailable_reason"] = "No resample contained observations"
    return result


def _inside(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if resolved == REPO or not resolved.is_relative_to(REPO):
        raise ValueError("Experiment reporting paths must resolve inside Aitken")
    if resolved.relative_to(REPO).parts[0] in {".git", ".codex", ".agents"}:
        raise ValueError("Protected reporting path")
    return resolved


def _dump(path: Path, value: Any) -> None:
    _inside(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _control(name: str, meta: dict) -> bool:
    return bool(meta.get("is_control", "control" in name.lower()))


def _plots(frames: dict[str, pd.DataFrame], variants: dict, phase: str, destination: Path) -> list[str]:
    if not frames:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    links = []
    families = sorted({variants.get(name, {}).get("family", "model") for name in frames})
    for family in families:
        names = [name for name in frames if variants.get(name, {}).get("family", "model") == family]
        controls = [name for name in names if _control(name, variants.get(name, {}))]
        candidates = [name for name in names if name not in controls]
        # Best RMSE is shown descriptively; this does not apply or replace gates.
        best = min(candidates, key=lambda name: metrics(frames[name])["rmse"]) if candidates else None
        names = controls + ([best] if best else [])
        if not names:
            names = [min(frames, key=lambda name: metrics(frames[name])["rmse"])]
        first = _checked(frames[names[0]])
        for name in names[1:]:
            _paired(first, frames[name])
        daily = first.groupby("date").observed_pm25.mean()
        year = int(first.date.min().year)
        calendar = pd.date_range(f"{year}-01-01", f"{year}-12-31")
        fig, axis = plt.subplots(figsize=(13, 4.6))
        axis.plot(calendar, daily.reindex(calendar), color="#172332", linewidth=1.3, label="Observed at available stations")
        for name in names:
            values = _checked(frames[name]).groupby("date").predicted_pm25.mean()
            axis.plot(calendar, values.reindex(calendar), linewidth=1.0, alpha=0.85, label=name)
        axis.set(xlabel="Calendar date (gaps mean no evaluated observations)", ylabel="Daily mean PM2.5 (micrograms/m³)",
                 title=f"{phase}: {family} — full calendar year at matched stations")
        axis.xaxis.set_major_locator(mdates.MonthLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        axis.set_xlim(calendar[0], calendar[-1])
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in f"{phase}_{family}")
        path = _inside(destination / f"{stem}_daily.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        links.append(path.name)

        dates = [pd.Timestamp(f"{year}-03-11"), pd.Timestamp(f"{year}-07-15")]
        if year == 2023:
            dates.append(daily.idxmax())
        dates = list(dict.fromkeys(day for day in dates if day in daily.index))
        if dates:
            fig, axes = plt.subplots(len(dates), 1, figsize=(12, 3.6 * len(dates)), squeeze=False)
            for ax, day in zip(axes[:, 0], dates):
                rows = first[first.date.eq(day)].sort_values("site_code")
                ax.plot(np.arange(len(rows)), rows.observed_pm25, "o-", color="#172332", label="Observed", markersize=4)
                for name in names:
                    current = _checked(frames[name]).set_index("station_row_index").loc[rows.station_row_index]
                    ax.plot(np.arange(len(rows)), current.predicted_pm25, ".-", label=name, alpha=0.8)
                ax.set_xticks(np.arange(len(rows)), rows.site_code, rotation=70, fontsize=7)
                role = "largest observed 2023 station-average day; diagnostic selection" if year == 2023 and day == daily.idxmax() else "fixed diagnostic date"
                ax.set(title=f"{day:%Y-%m-%d} — {role}", ylabel="PM2.5 (micrograms/m³)")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            path = _inside(destination / f"{stem}_examples.png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            links.append(path.name)
    return links


def write_report(outputrunpath: str | Path, destinationreportfolder: str | Path) -> Path:
    """Recompute saved predictions and produce an auditable, partial-safe report."""
    output, destination = _inside(outputrunpath), _inside(destinationreportfolder)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    variants = summary.get("variants", {})
    recomputed, manifest, intervals, links = {}, {}, {}, []
    report = [
        "# Temporal smoothing and peak sensitivity — experiment report", "",
        f"Run status: **{summary.get('status', 'unknown')}**.", "",
        "## Verdict and safeguards", "",
        "This experiment preserves the original PM2.5 observations. It tests historical input summaries and a small final-output loss term, not removal of pollution episodes.", "",
        "Selection is based on 2023 safeguards, followed by matched-seed confirmation. Any 2024 figures are retrospective: this benchmark has previously been examined. Missing phases are not inferred or presented as completed.", "",
    ]
    selection = summary.get("selection", {})
    for family, decision in selection.items():
        if isinstance(decision, dict):
            report.append(f"- **{family}:** screened choice `{decision.get('screen_selected')}`; confirmation `{decision.get('confirmed')}`. {decision.get('reason', '')}")
        else:
            report.append(f"- **{family}:** {decision}")
    if not selection:
        report.append("No completed selection decision is recorded yet.")
    report.extend(["", "The official model, published performance table, dashboard and presentation are not changed by this report.", "",
        "## Design and comparability", "",
        "- Training: 2021–2022; validation: 2023. Only confirmed candidates are eligible for a 2021–2023 fit and unchanged-observation 2024 evaluation.",
        "- HGB remains fixed to archived maps. The temporal hybrid is a matched rebase onto those maps, not a reproduction of the friend's differently refitted HGB experiment.",
        "- Three-day median and one-/three-day-half-life exponential summaries retain the raw lagged input. These use yesterday and earlier, never today's PM2.5.",
        "- The history source is already standardized/clipped; it is not a reconstructed raw concentration series. New scalers use training days only.",
        "- Added channels start at zero weight; common parameters share initialization. Controls and candidates use the same frozen schedules and seed set.",
        "- The extra loss, when enabled, is 0.1 times source/density-weighted squared final-output concentration error, normalized by training-target standard deviation. Hybrid final output includes the frozen 0.425 correction multiplier.",
        "- A threshold of 25 means micrograms per cubic metre, not AQI, a measurement-error rule or a claim of safe exposure.", "",
    ])
    for phase in ("screening", "confirmation", "final"):
        declared = summary.get("phases", {}).get(phase, {})
        prediction_dir = output / phase / "predictions"
        frames = {}
        if prediction_dir.is_dir():
            for path in sorted(prediction_dir.glob("*.csv.gz")):
                name = path.name.removesuffix(".csv.gz")
                frames[name] = _checked(pd.read_csv(path))
                manifest[str(path.relative_to(REPO))] = _digest(path)
        recomputed[phase] = {}
        for name, frame in frames.items():
            value = metrics(frame)
            recomputed[phase][name] = value
            saved = declared.get(name)
            if isinstance(saved, dict):
                for key in ("n", "mae", "rmse", "r2", "bias"):
                    if key in saved and saved[key] is not None and not np.isclose(saved[key], value[key], rtol=1e-8, atol=1e-10):
                        raise ValueError(f"Saved summary disagrees with prediction rows: {phase}/{name}/{key}")
        missing = sorted(set(declared) - set(frames))
        report.extend([f"## {phase.capitalize()}", ""])
        if not frames:
            report.extend(["No saved prediction rows for this phase. No performance claim is made.", ""])
            continue
        if missing:
            report.extend([f"Prediction files unavailable for: {', '.join(missing)}. Their summary numbers are not independently rechecked here.", ""])
        report.extend(["All errors and bias are in micrograms/m³. Negative bias means underprediction.", "",
                       "| Variant | N | MAE | RMSE | R² | Bias | MAE <15 | Bias <15 | MAE ≥25 |",
                       "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for name, value in recomputed[phase].items():
            normal, high = value["bands"][BANDS[0]], value["bands"][BANDS[2]]
            report.append(f"| {name} | {value['n']} | {_f(value['mae'])} | {_f(value['rmse'])} | {_f(value['r2'])} | {_f(value['bias'])} | {_f(normal['mae'])} | {_f(normal['bias'])} | {_f(high['mae'])} |")
        report.extend(["", "### Matched-control comparisons", "",
                       "Negative error changes favour the candidate. Passing the screening guard alone is not final confirmation.", "",
                       "| Candidate | Control | RMSE change | Overall MAE change | High MAE change | All safeguards against this control |",
                       "|---|---|---:|---:|---:|---|"])
        for name, value in recomputed[phase].items():
            meta = variants.get(name, {})
            if _control(name, meta):
                continue
            controls = [other for other in frames if _control(other, variants.get(other, {})) and variants.get(other, {}).get("family") == meta.get("family")]
            for other in controls:
                baseline = recomputed[phase][other]
                high, base_high = value["bands"][BANDS[2]]["mae"], baseline["bands"][BANDS[2]]["mae"]
                high_delta = high - base_high if high is not None and base_high is not None else None
                report.append(f"| {name} | {other} | {_f(value['rmse'] - baseline['rmse'])} | {_f(value['mae'] - baseline['mae'])} | {_f(high_delta)} | {'Pass' if safe_candidate(value, [baseline]) else 'Fail'} |")
        report.extend(["", "### Errors by observed concentration", "",
                       "| Variant | Observed band | N | MAE | RMSE | Bias | Squared-error share % |",
                       "|---|---|---:|---:|---:|---:|---:|"])
        for name, value in recomputed[phase].items():
            for band_name, band in value["bands"].items():
                report.append(f"| {name} | {band_name} | {band['n']} | {_f(band['mae'])} | {_f(band['rmse'])} | {_f(band['bias'])} | {_f(band['squared_error_percent'])} |")
        report.extend(["", "### Peaks and rapid changes", "",
                       "Rapid-change groups use same-station observations on the preceding calendar day, solely for diagnosis. They overlap the ≥25 group; squared-error shares must not be added.", "",
                       "| Variant | High N | High underpredicted | High SSE % | Rise ≥10 N / MAE | Fall ≥10 N / MAE | Previous day unavailable |",
                       "|---|---:|---:|---:|---:|---:|---:|"])
        for name, value in recomputed[phase].items():
            high, counts, changes = value["bands"][BANDS[2]], value["high_pollution"], value["rapid_changes"]
            rise, fall = changes["rise_at_least_10"], changes["fall_at_least_10"]
            report.append(f"| {name} | {high['n']} | {counts['underprediction_count']} | {_f(high['squared_error_percent'])} | {rise['n']} / {_f(rise['mae'])} | {fall['n']} / {_f(fall['mae'])} | {changes['unavailable_previous_day_n']} |")
        report.extend(["", "### Calibration by predicted concentration", "",
                       "These groups are based on the model's prediction, not the observed target. Sparse groups should not be overinterpreted.", "",
                       "| Variant | Predicted band | N | Mean predicted | Mean observed | MAE | Bias |",
                       "|---|---|---:|---:|---:|---:|---:|"])
        for name, value in recomputed[phase].items():
            for band in value["calibration"]:
                upper = band["predicted_upper_exclusive"]
                label = f"[{band['predicted_lower_inclusive']:g}, {upper:g})" if upper is not None else "[25, infinity)"
                report.append(f"| {name} | {label} | {band['n']} | {_f(band['predicted_mean'])} | {_f(band['observed_mean'])} | {_f(band['mae'])} | {_f(band['bias'])} |")
        report.append("")
        for name, frame in frames.items():
            meta = variants.get(name, {})
            if _control(name, meta):
                continue
            controls = [other for other in frames if _control(other, variants.get(other, {})) and variants.get(other, {}).get("family") == meta.get("family")]
            for other in controls:
                _paired(frame, frames[other])
                if phase == "final":
                    key = f"{name}_minus_{other}"
                    intervals[key] = block_interval(frame, frames[other])
                    interval = intervals[key]
                    report.append(f"- {key}: RMSE change {_f(interval['rmse_delta'])}; paired seven-day 95% interval {interval['ci95']}.")
        charts = _plots(frames, variants, phase, destination)
        for chart in charts:
            report.extend(["", f"![{chart.replace('_', ' ').removesuffix('.png')}]({chart})", ""])
        links.extend(charts)
    report.extend(["## Limitations and audit", "",
        "- Bootstrap intervals use 5,000 paired moving seven-calendar-day resamples, retaining station-day weighting and missing calendar days. They condition on the fitted models and do not include selection or seed uncertainty.",
        "- Full-year plots average the available matched stations each day, not the entire geographic population. Coverage changes can affect these averages.",
        "- March 11 and July 15 are fixed diagnostic dates. The largest observed 2023 daily station mean is shown descriptively, not used to tune parameters.",
        "- Curves show matched controls and the lowest-RMSE candidate for readability. A plotted candidate is not necessarily eligible or promoted.",
        "- Original labels are evaluated without smoothing, capping or winsorisation. Remaining peak errors cannot by themselves establish sensor noise or an atmospheric cause.", "",
        "### Budget", "", "```json", json.dumps(summary.get("budget", {}), indent=2, allow_nan=False), "```", "",
        "### Integrity", "", "```json", json.dumps(summary.get("integrity", {}), indent=2, allow_nan=False), "```", "",
        "Prediction hashes and recomputed metrics are in `metrics.json`. The report does not substitute for the runner's input-hash and checkpoint-contract checks.", "",
        "## Research basis", "",
        "1. [NIST — exponential smoothing](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc431.htm): recent values receive greater weight; retaining innovations avoids treating smooth levels as the whole signal.",
        "2. [MathWorks — Hampel filtering](https://www.mathworks.com/help/signal/ref/hampel.html): robust filtering can also flag real extrema; a flag is not proof of invalid measurement.",
        "3. [PyTorch — SmoothL1 loss](https://docs.pytorch.org/docs/main/generated/torch.nn.modules.loss.SmoothL1Loss.html) and [forecast scoring/loss alignment](https://arxiv.org/abs/0912.0902): objective choice affects the point prediction being learned.",
        "4. [PM2.5 correction study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7788047/): temporal correction has evidence in a different setting; not an expected effect size for this model.",
        "5. [Time-series decomposition leakage study](https://www.nature.com/articles/s41598-024-80018-9): whole-series preprocessing can leak future information; all tested filters are causal.",
        "6. [Seo and Gupta, 2026](https://www.nature.com/articles/s41612-026-01437-1): temporal and extreme-pollution modeling is relevant, but different data and evaluation prevent a direct score ranking.", "",
    ])
    payload = {
        "status": summary.get("status"), "summary_sha256": _digest(summary_path),
        "source_run": str(output.relative_to(REPO)), "prediction_sha256": manifest,
        "phases": recomputed, "selection": selection, "variants": variants,
        "paired_intervals": intervals, "budget": summary.get("budget", {}),
        "integrity": summary.get("integrity", {}), "charts": links,
    }
    _dump(destination / "metrics.json", payload)
    report_path = _inside(destination / "REPORT.md")
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path
