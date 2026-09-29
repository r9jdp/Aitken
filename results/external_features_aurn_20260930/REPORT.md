# Regional AURN PM2.5 experiment

## Result

Adding an external, same-day regional pollution signal to the temporal hybrid's
saved output improved the retrospective 2024 benchmark. This is a lightweight
day-level residual correction, not a full HGB or U-Net retraining.

| 2024 result | Temporal hybrid | + regional AURN correction | Change |
|---|---:|---:|---:|
| R2 | 0.5201 | **0.6471** | +0.1271 |
| RMSE (micrograms/m3) | 3.7679 | **3.2309** | -14.3% |
| MAE (micrograms/m3) | 2.3361 | **2.1343** | -8.6% |
| MAE below 15 | 1.7174 | **1.6967** | -1.2% |
| MAE from 15 to below 25 | 5.9437 | **4.7401** | -20.3% |
| MAE at or above 25 | 14.0490 | **10.2219** | -27.2% |

The paired seven-day block-bootstrap interval for candidate-minus-control RMSE
was **[-0.8294, -0.2373] micrograms/m3**. Negative values favour the AURN
correction. This is a descriptive interval on the existing retrospective
benchmark, not a claim about a new untouched test set.

## What was added

The downloader selected rural/background AURN PM2.5 stations 30-110 km from
central London. Hourly readings had to be ratified, between 0 and 500
micrograms/m3, with at least 18 valid hours per station-day. Fifteen stations
provided 16,413 valid station-days. The resulting daily fields were:

- regional mean and maximum PM2.5;
- a wind-aligned upwind mean using the existing ERA5 winds;
- the number of reporting sites and an availability indicator;
- explicit lag-1 versions for a forecast-safe comparison.

The source files and downloader manifest remain in ignored local `data/`
storage because the repository does not redistribute the measurements.

## Guard against harming ordinary days

Only 2023 saved out-of-year temporal-model errors were used to choose the Ridge
strength, pollution gate and shrinkage. Selection used twelve leave-one-month-
out folds. A recipe was eligible only if it improved overall RMSE and R2,
reduced high-band MAE, and did not worsen normal-band MAE on those held-out 2023
rows. The complete recipe was saved before the script loaded 2024 observations.

The selected correction turns on smoothly when regional AURN pollution is high
and is off on low-regional-pollution days. This is why improving the episode
range did not cause general overprediction in the normal range.

The lag-1-only AURN version failed the 2023 eligibility gate. It was rejected
before 2024 evaluation and therefore returned the unchanged temporal control.

## Interpretation and limits

This result supports the earlier diagnosis: the model was missing information
about region-wide pollution episodes, not simply lacking more temporal
smoothing. The same-day AURN observations provide that information.

This corrected model is suitable for **retrospective daily mapping/nowcasting**.
It is not a next-day forecast, because the strongest input is same-day regional
PM2.5. Also, the project had already examined 2024 in earlier experiments. In
this experiment, 2024 was held out from AURN recipe training and selection, but
it is not globally untouched. Promotion should wait for a later untouched year
and station-held-out testing.

## Other source decisions

- **DfT traffic counts:** rejected as a daily feature. The 2021-2024 London
  file has only 92-105 survey dates per year (25-29% calendar coverage), no
  weekends and long seasonal gaps.
- **CAMS European reanalysis:** next high-priority experiment; currently needs
  a Copernicus ADS account/API key.
- **MODIS MAIAC AOD:** next high-priority experiment; currently needs a Google
  Earth Engine project or NASA Earthdata access.
- **HYSPLIT trajectories:** retained as a later experiment after CAMS and MAIAC
  because it requires a separate trajectory engine and meteorology archive.

## Reproduction

```bash
python -m unittest discover -s experiments/external_features -p "test_*.py" -v

python experiments/external_features/download_aurn.py \
  --output-dir data/external_features/aurn \
  --data-dir pm25_london_bundle/data/processed/london_1km_daily

python experiments/external_features/run_aurn_correction.py \
  --output-dir artifacts/external_features_aurn

python experiments/external_features/audit_dft_traffic.py
```

Saved local artifact hashes:

- AURN feature archive: `da2d3f5e4a9ededc28b6a738b1a9f17a6bc85e7720e0f4f7c36cea2063b16cda`
- frozen 2023 selection: `45b87a0614c283a6ab80935ed4e9037cd87422f13dc48ae1f250d76a3afc1b43`
- 2024 result summary: `002d5017409b75fa2fead98819ddba14174d71a8eda82df52829f6a4c457819a`
