# Verdict: keep the existing models

Completed 20 September 2026. **The proposed tweaks did not earn promotion under
the agreed acceptance rules.** Some reduced peak error, but none improved all
required measures. No real pollution values were flattened or removed.

For a presentation, the conclusion is:

> We tested whether extra summaries of past pollution, or a training loss that
> pays more attention to large concentration errors, could improve peak
> prediction. Some peak errors decreased, but the improvements either harmed
> other predictions or were very small and uncertain. None passed our predefined
> validation rules, so we retained the existing models and preserved the real
> observations. High pollution was not treated as proof of bad data.

## What was actually evaluated

We completed **nine matched fits**, all with seed 42, trained on 2021–2022 and
evaluated on **11,102 original LAQN station-days in 2023**. These comprise 9,954
observations below 15, 884 from 15 to below 25, and 264 at least 25 µg/m³.

HGB was **not retrained**. Both hybrid controls use the verified archived HGB
maps. The temporal control adds the friend's five historical summary features,
rebased onto these fixed maps and matched initialization. It is not a numerical
reproduction of the friend's earlier refitted-HGB experiment.

These are **2023 screening results**, not replacements for the published 2024
performance table. They cannot be directly compared with the earlier models
trained on three years and evaluated on a different year. The 2023 validation
year has also been examined previously.

## Results at a glance

MAE, RMSE and peak MAE are in µg/m³; lower is better. Peak MAE uses original
observations ≥25. R² is unitless and higher is better.

| Model / treatment | Overall MAE | Overall RMSE | R² | Peak MAE | Decision |
|---|---:|---:|---:|---:|---|
| Standalone U-Net control | 2.1774 | 3.2410 | 0.6735 | 7.7847 | Keep as matched reference |
| Standalone + exponential history | 2.2932 | 3.3654 | 0.6479 | 7.2634 | Reject |
| Standalone + median history | 2.3096 | 3.4115 | 0.6382 | 7.2513 | Reject |
| Standalone + final-output squared loss | 2.3268 | 3.4675 | 0.6262 | 7.9257 | Reject |
| Original hybrid control | 2.1042 | 3.1449 | 0.6925 | 6.9576 | Keep as matched reference |
| Temporal hybrid control | 2.1041 | 3.1434 | 0.6928 | 6.9332 | Keep as second reference |
| Temporal hybrid + exponential history | 2.1019 | 3.1406 | 0.6934 | 6.8669 | Reject: bias safeguard |
| Temporal hybrid + median history | 2.1034 | 3.1422 | 0.6931 | 6.9082 | Reject: ordinary-day safeguards |
| Temporal hybrid + final-output squared loss | 2.1023 | 3.1462 | 0.6923 | 6.8816 | Reject: RMSE safeguard |

### Standalone U-Net: a clear trade-off, not an overall improvement

Exponential history reduced peak MAE by **0.5213**, but increased overall RMSE by
**0.1244 (3.84%)**. Its error during rapid rises improved from **9.4329 to
7.5335**, while error during rapid falls worsened from **3.8344 to 4.4277**.
Ordinary-concentration MAE and bias also worsened.

Median history similarly reduced peak MAE by **0.5334**, but increased overall
RMSE by **0.1704 (5.26%)**. The final-output loss worsened both overall RMSE and
peak MAE. Under these frozen schedules and settings, the unchanged standalone
control was the better overall model.

### Hybrid: tiny numerical gains, but no qualifying candidate

Exponential history had the lowest numerical RMSE: **3.1406**, compared with
**3.1449** for the original hybrid and **3.1434** for the temporal control. That
is approximately **0.14%** and **0.09%** lower, respectively. Peak MAE fell by
**0.0907** against the original control.

However, signed bias below 15 rose from **+0.4009** (original control) and
**+0.4026** (temporal control) to **+0.4062**. The agreed rule did not permit that
increase. The difference is small; we do **not** claim it is a large or proven
practical harm. We simply did not relax the rule after observing the scores.
The uncertainty interval also includes no improvement.

The median treatment failed the ordinary-concentration MAE and bias rules
against both controls. The loss treatment improved several MAE measures but
increased overall RMSE against both controls. No candidate passed every rule.

## Exactly which safeguards failed

All candidate observations and counts matched their controls. The table lists
only failed numerical safeguards; hybrid failures apply against **both** controls.

| Candidate | Failed safeguards |
|---|---|
| Standalone exponential history | Overall RMSE, overall MAE, MAE below 15, absolute bias below 15 |
| Standalone median history | Overall RMSE, overall MAE, MAE below 15, absolute bias below 15 |
| Standalone final-output loss | Overall RMSE, peak MAE, overall MAE, MAE below 15, absolute bias below 15 |
| Hybrid exponential history | Absolute bias below 15 |
| Hybrid median history | MAE below 15, absolute bias below 15 |
| Hybrid final-output loss | Overall RMSE |

The full Boolean checks are in `audit.json` under `screening_guard_details`.
The bias bound was `max(0.10, absolute control bias)` for each control; there was
no added tolerance chosen after the results.

## Uncertainty: are the small gains convincing?

These **2023, single-seed, descriptive** intervals use 5,000 paired moving
seven-calendar-day bootstrap resamples. All stations within sampled day blocks
remain together, with station-day weighting. A negative RMSE change favours the
candidate. Intervals condition on the fitted models and do not account for
selection, multiple candidates or training-seed uncertainty.

| Candidate minus control | RMSE change | 95% interval |
|---|---:|---|
| Standalone exponential − standalone control | +0.1244 | [−0.1242, +0.3829] |
| Standalone median − standalone control | +0.1704 | [−0.0091, +0.3581] |
| Standalone loss − standalone control | +0.2265 | [+0.0811, +0.3739] |
| Hybrid exponential − original control | −0.0043 | [−0.0115, +0.0026] |
| Hybrid exponential − temporal control | −0.0028 | [−0.0080, +0.0024] |
| Hybrid median − original control | −0.0027 | [−0.0078, +0.0024] |
| Hybrid median − temporal control | −0.0012 | [−0.0064, +0.0060] |
| Hybrid loss − original control | +0.0013 | [−0.0041, +0.0062] |
| Hybrid loss − temporal control | +0.0028 | [−0.0032, +0.0098] |

In particular, **every hybrid interval includes zero**. These tests do not
establish a reliable improvement, and are not a substitute for three-seed
confirmation or an independent evaluation.

## Peak prediction is still a weakness

The exponential-history hybrid still underpredicted **233 of 264** high
observations. On rapid-rise station-days, MAE changed only from **9.6816 to
9.6132**; rapid-fall MAE worsened from **4.9039 to 4.9400**. The diagnostic groups
overlap, so their error shares must not be added.

There were **172 rapid rises**, **320 rapid falls**, and **220 observations with
no same-station observation on the previous day inside the evaluation slice**.
Those missing pairs were disclosed rather than treated as zero change. These
groups are evaluation diagnostics, never inputs to the model.

The [full report](REPORT.md) includes all observed bands, predicted-concentration
calibration, full-year curves and fixed ordinary-day examples. It also shows
**22 January 2023**, the largest observed daily station-average episode in this
validation year, explicitly selected for diagnosis rather than tuning. The
hybrid curves remain almost identical and miss substantial peaks at some sites.
This experiment does not identify whether the cause is missing atmospheric
information, measurement problems or another modeling limitation.

## Why we stopped here

- Neither family had an eligible screening candidate.
- The combined exponential-plus-loss treatment was therefore **not run**.
- Three-seed confirmation was **not run**.
- No new 2021–2023 refit or 2024 prediction evaluation was performed.
- No thresholds, schedules or selection rules were revised to chase 2024 scores.

This is the plan's completed **negative screening branch**, not an unfinished
attempt or a claim that temporal processing can never help. The official model,
dashboard, presentation and published 2024 table remain unchanged. The code and
negative findings are retained to prevent repeating the same experiment and to
make the decision auditable.

## Verification and execution record

- Frozen training code: commit `2415c6a`, following initial code/tests commit
  `ee3957c`; individual source and reference hashes are preserved in the manifest
  and independent audit.
- Completed run: `artifacts/temporal_peak_sensitivity/run_20260920_v2`.
- Charged training/inference time: **1,109.936 seconds (18.50 minutes)** of the
  7,200-second allowance. This includes **4.594 seconds** from the superseded
  initial attempt; setup, CPU tests and reporting are additional.
- The initial attempt exposed overly strict handling of expected AMP gradient
  overflow. The original scaler's skip/backoff behavior was restored and tested
  before the v2 manifest was frozen; incompatible weights were not reused.
- A transient Windows file lock interrupted the last hybrid screening fit after epoch
  14. The identical run resumed at epoch 15 with model, optimizer, scaler and
  random state restored. Eight completed fits were reused, not retrained;
  consumed compute remained charged.
- Report generation encountered a float32 CSV round-trip consistency issue.
  The audited [precision wrapper](render_report.py) recovered exact original
  label/map precision in memory only. It changed no scientific result, input,
  saved prediction or frozen training code. See `render_audit.json`.
- **36 experiment tests and 6 report-wrapper tests passed**, as did the
  independent auditor's self-tests. Tests cover future-prefix invariance,
  training-only scales, missing-target masks, standalone/no-HGB inputs, final
  hybrid composition, matched initialization, checkpoint rejection and output
  boundaries. All four generated plots were visually checked.
- The independent saved-run audit verified all nine completed members, original
  observation rows, metrics, selection decisions, source/reference hashes and
  all nine bootstrap comparisons. It reconstructs predictions from saved member
  arrays; it does not claim to rerun checkpoint inference independently.

For reproduction and file descriptions, see [README.md](README.md). Keep
monitoring the running process output rather than holding progress JSON files
open during atomic replacement on Windows.
