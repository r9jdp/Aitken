# AURN data import and performance review

Date: 30 September 2026. Reviewed collaborator commit: `9842330`.

## Verdict

The shared AURN dataset has been imported and independently checked against its
raw hourly files. After receiving the reproduction supplement, the AURN
correction was rerun locally: **RMSE 3.7679 -> 3.2309**, with
**R2 0.5201 -> 0.6471**. The improvement is confirmed to the reported precision.
All selection decisions, overall/band scores and the bootstrap interval match
within small numerical tolerances. Both prediction tables match the complete
original LAQN evaluation membership and original float32 observations.

**Verification boundary:** this reproduces selection and fitting of the small
AURN correction from supplied temporal-model predictions. It does not retrain
or independently regenerate the underlying HGB/U-Net predictions. The original
import audit is retained as a historical pre-supplement snapshot; the current
verification record is [reproduction_audit.json](reproduction_audit.json).

This is **temporal HGB + U-Net plus a same-day AURN correction**, not a newly
trained standalone U-Net. It is better on this retrospective station-day
benchmark, but not evidence of a better next-day forecast. Official model,
dashboard and paper remain unchanged. A wind-scaling defect still needs a
separately labelled controlled rerun before promotion.

## What changed

`git pull --ff-only origin main` brought Aitken from `c7963e2` to `9842330`.
The collaborator added source acquisition/preparation, a small correction
experiment, tests, source documentation and an aggregate performance report.

The correction learns the average daily residual of the temporal hybrid on
2023 observations, using Ridge regression. It adds a single day-specific
adjustment to each location's temporal-hybrid prediction. Its inputs are
same-day regional mean, maximum, an upwind-weighted mean and reporting-site
count. A smooth pollution gate and shrinkage limit the adjustment on ordinary
days. The unscaled correction is clipped to [-5, 15] micrograms/m3 and final
predictions to [0, 500]. No HGB or U-Net weights were retrained in this step.

This primarily changes each day's background pollution level. Apart from
effects of clipping, adding the same amount to every cell does not change
within-day spatial differences or hotspot ordering. A better station-day RMSE
does not demonstrate that U-Net learned more detailed neighbourhood hotspots.

## Locally reproduced comparison

Both rows use the same 9,619 LAQN station-days at 38 sites over 349 dates,
1 January to 14 December 2024. All error measures are in micrograms/m3.
The local rerun agrees with the collaborator's published values at the table's
four-decimal precision; percentage changes below also agree at shown precision.

| Metric | Temporal hybrid | + same-day AURN correction | Change |
|---|---:|---:|---:|
| RMSE | 3.7679 | 3.2309 | 14.25% lower |
| MAE | 2.3361 | 2.1343 | 8.64% lower |
| R2 | 0.5201 | 0.6471 | +0.1271 |
| Signed bias | -0.7685 | -0.4832 | Less underprediction |
| MAE, observed below 15 | 1.7174 | 1.6967 | 1.20% lower |
| MAE, observed 15 to below 25 | 5.9437 | 4.7401 | 20.25% lower |
| MAE, observed at least 25 | 14.0490 | 10.2219 | 27.24% lower |

The original official hybrid remains a separate reference (RMSE 3.7765,
MAE 2.3405, R2 0.5179); it is not the matched control for this experiment.

An independent 5,000-resample, seven-calendar-day block-bootstrap computation
reproduced **[-0.8294, -0.2373]** for corrected-minus-control RMSE. The interval
excludes zero for this sample and protocol, but remains descriptive because
2024 was previously examined. The lag-1 alternative again failed the 2023
selection gate and returned the unchanged control; it is not another improved
forecast model.

Sources: [local reproduction audit](reproduction_audit.json),
[recorded metrics](../../results/external_features_aurn_20260930/metrics.csv)
and [collaborator report](../../results/external_features_aurn_20260930/REPORT.md).

## Local dataset update and checks completed

Imported to `Aitken/data/external_features/aurn/`:

- 63 payload files, 77,785,649 bytes; no Mac resource forks extracted.
- 55 raw station-year CSV hashes and the catalogue hash match the sender's manifest.
- Feature archive SHA-256 matches the exact checksum in the experiment report:
  `da2d3f5e4a9ededc28b6a738b1a9f17a6bc85e7720e0f4f7c36cea2063b16cda`.
- 18 selected background/rural sites, 15 with usable PM2.5 measurements.
- 16,413 unique valid station-days, 2021-2024. Every study date has 6-15 sites.
- Ratification-status columns were present, valid hourly timestamps had no
  duplicates, and accepted station-days had 18-24 hours with PM2.5 in [0, 500].
- Reaggregating the hourly measurements reproduced the supplied daily values.
- Rebuilding all eight features with the collaborator's exact procedure gave
  **zero difference** from the supplied `(1461, 8)` array.
- Lag-one columns exactly match the preceding day's same-day columns.
- None of the 18 selected sites is within 500 m of the 38 evaluated LAQN
  monitors. The closest pair is 9.30 km apart. This rules out obvious physical
  co-location in this set, not every possible upstream data dependency.
- Original feature cube, targets, weights, station rows and metadata retain
  their before/after SHA-256 hashes. The original London bundle was not changed.

The archive is installed as an **external feature dataset**, not merged into
London ground-truth labels. The original 444,202 station-days and model input
contracts remain unchanged. Raw observations and all large arrays stay ignored
by Git. See [audit.json](audit.json) and [local data registry](../../data/README.md).

## Important issue: wind components are in the wrong scale

The collaborator's `download_aurn.py` reads `era5_u10` and `era5_v10` directly
from `features_float16.npy` and uses them to infer wind direction. That cube
contains **standardized** values, not wind components in metres/second. The
stored scalers are:

| Component | Mean | Standard deviation |
|---|---:|---:|
| u10 | 1.027996 | 2.641867 |
| v10 | 0.663049 | 2.772196 |

Direction must use physical components. Undoing those stored scalers in a
diagnostic copy changes the inferred direction by a median **10.52 degrees**;
**202 dates** change by more than 30 degrees. The resulting upwind PM2.5 feature
changes by **0.396 micrograms/m3 on average**, with maximum absolute change
**16.343 micrograms/m3**. These are feature differences, not model-score changes.
The reconstruction is approximate because the cube is float16 and previously
clipped; original raw ERA5 winds are preferable for a corrected experiment.

The imported array and collaborator code were deliberately left unchanged to
preserve reproducibility. This defect does not by itself invalidate the
reported numerical gain: regional mean/max may supply useful information even
with an imperfect directional feature. But the current upwind field should
not be claimed as physically correct. Fixing it and rerunning the frozen
evaluation must be a separately labelled result, not silently substituted.

## Evaluation interpretation

1. **Same-day information changes the task.** The improved model sees other
   stations' pollution on the target day; the previous hybrid relied on earlier
   PM2.5 measurements. This is a meaningful information advantage for regional
   episode reconstruction, not proof of a superior U-Net architecture.
2. **Ratified daily data are retrospective.** The experiment requires at least
   18 hours per day and ratified observations. It has not tested data availability
   at a forecast issue time. The lag-1 field is prior-day by construction, but
   operational availability of ratified readings would still need checking.
3. **The 2023 folds are leave-one-month-out, not forward-only.** Later 2023
   months can train a correction evaluated on an earlier held-out month. The
   code saves recipe selection before loading 2024 labels, but this validation
   is not a chronological rolling-origin evaluation. It tries 144 recipes per
   source; the selected cross-validation score is not an untouched estimate.
4. **Large peaks remain difficult.** A reported high-band MAE of 10.22 is a
   substantial improvement over 14.05, but still a large miss. It is evidence
   consistent with useful missing regional information, not proof that all
   previous errors were caused by one atmospheric process or by noisy labels.
5. **This archive supplies only AURN.** CAMS, MAIAC AOD and HYSPLIT are listed as
   future/blocked sources; DfT daily traffic was rejected. They did not produce
   the reported gain.

For example, the imported AURN regional mean on 11 March 2024 is **29.97**, and
the regional maximum **39.70 micrograms/m3**, from 13 sites. The new inputs
therefore contain a strong signal on a previously difficult episode day.
The local correction raises the mean prediction across 33 evaluated London
stations from **11.17 to 15.83**, versus an observed mean of **34.76**.
Day-specific MAE falls from **23.60 to 18.94** and RMSE from **23.77 to 19.15**.
This is useful but far from solving this major episode. Across the benchmark,
**213 of 216** observations at or above 25 remain underpredicted, compared with
all 216 previously. Do not present the lower aggregate error as peak prediction
being solved.

Official source context: [DEFRA AURN](https://uk-air.defra.gov.uk/networks/network-info?view=aurn).

## Supplement verification and reproducibility

The supplied `AURN_REPRODUCTION_SUPPLEMENT_20260930_READY.zip` contains all four
previously missing files. Each matches both the supplement checksum manifest
and the previously recorded experiment hashes:

```text
artifacts/temporal_tail_correction_20260912/screening/B_temporal_predictions.csv.gz
artifacts/temporal_tail_correction_20260912/final/B_temporal_predictions.csv.gz
artifacts/external_features_aurn/selection_frozen.json
artifacts/external_features_aurn/summary.json
```

The supplied artifacts were installed at those ignored paths without changing
their bytes. The rerun uses the unchanged shared AURN features and unmodified
collaborator correction code, with new outputs under the separate ignored
directory `artifacts/aurn_reproduction_20260930/`.

Checks completed:

- The 2023 selection table has all 11,102 original LAQN rows, 365 dates and
  43 stations; the 2024 table has all 9,619 original rows, 349 dates and 38 stations.
- Station row IDs, dates, site codes and grid locations agree with the original
  archive. No duplicate station-days or changed/omitted evaluation observations.
  Decimal CSV serialization differs by at most 0.00000153 micrograms/m3 when
  compared as float64; converting back to original float32 matches exactly.
- Rerunning every trial reproduces all 2023 eligibility decisions and the same
  selected recipe: Ridge alpha **1**, gate **8 to 15**, shrinkage **0.75**.
  Lag-1 is again rejected. No recipe is chosen using new 2024 results.
- The rerun writes its selection before loading 2024 labels, as verified in
  the reviewed code. This checks the procedure, not the collaborator's historical
  execution order or the upstream neural model's entire training provenance.
- Local RMSE is **3.2308853247**, versus the supplied **3.2308846809**; absolute
  difference is **0.0000006439**. Local R2 is **0.6471160626**. These are numerically
  matching results, not byte-identical JSON. The maximum numeric difference
  across all selection trial records is 0.00001937; exact categorical decisions
  match. Such small differences are consistent with floating-point/library
  variation between the sender's environment and this Windows installation.
- Independently calculated MAE, RMSE, R2, bias, bootstrap interval and episode
  metrics confirm the rerun output. The original data and all four supplied
  artifacts retain their checksums after the run.

Repeat from Aitken with the existing global Python:

```powershell
python scripts/reproduce_aurn_supplement.py --archive C:/Users/Rajdeep/Downloads/AURN_REPRODUCTION_SUPPLEMENT_20260930_READY.zip --bundle ../code/pm25_london_bundle
python -m unittest discover -s scripts -p "test_*aurn*.py" -v
python -m unittest discover -s experiments/external_features -p "test_*.py" -v
```

Fourteen import/reproduction tests and seven collaborator tests passed
(**21 total**). Small provenance reports and code/tests are tracked; raw
observations, prediction tables and large arrays remain Git-ignored. No HGB or
U-Net retraining, official model promotion, dashboard replacement, presentation
edit or research paper update was performed. Only the lightweight Ridge
selection/refit was rerun for verification.

## Remaining handoff boundary

For the next change, correct the physical wind inputs and run a separately
labelled comparison, preserving this reproduced result as its reference.
Do not retune against 2024. A complete trained-model/dashboard handoff would
also need temporal checkpoints or full-grid temporal predictions: the supplied
station tables are sufficient for this score audit, not city-wide inference.
