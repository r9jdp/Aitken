# External-information experiment

This experiment tests whether genuinely new pollution information reduces the
hybrid model's compression of high PM2.5 values. The frozen study split remains:

- fit/select using 2021-2023;
- evaluate every surviving fixed recipe on the same 2024 LAQN observations;
- report overall R2, RMSE, MAE and separate errors below 15, from 15-25, and at
  or above 25 micrograms per cubic metre.

The sources are added one group at a time. A source must pass coverage and
timing checks before it is allowed into a model.

| Group | Source | State |
|---|---|---|
| Regional PM2.5 | DEFRA AURN sites 30-110 km from central London | Automated |
| Atmospheric PM2.5 | CAMS European air-quality reanalysis | Needs ECMWF/ADS credentials |
| Aerosol optical depth | MODIS MAIAC MCD19A2.061 | Needs Earth Engine or Earthdata credentials |
| Air-mass path | NOAA HYSPLIT with GDAS1 | Planned after the higher-priority fields |
| Traffic | DfT raw counts / TfL SCOOT | DfT rejected as a daily feature after coverage audit |

## AURN preparation

The downloader selects only rural or background PM2.5 sites outside London,
downloads official hourly files, accepts ratified measurements in 0-500
micrograms per cubic metre, and requires at least 18 hours for a station-day.

```bash
python -m unittest discover -s experiments/external_features -p "test_*.py" -v

python experiments/external_features/download_aurn.py \
  --output-dir data/external_features/aurn \
  --data-dir pm25_london_bundle/data/processed/london_1km_daily
```

The prepared file contains both same-day regional signals for retrospective
mapping and explicit lag-1 alternatives for a stricter forecasting experiment.
Raw downloads and prepared arrays stay under the ignored `data/` directory.

## Fast evidence gate

Before paying the cost of another full U-Net refit, the experiment trains a
small guarded day-level correction on the saved out-of-year 2023 temporal-model
errors. Recipe selection uses blocked 2023 folds only. The selected recipe is
written to disk before the script loads and evaluates the frozen 2024 output.

```bash
python experiments/external_features/run_aurn_correction.py \
  --output-dir artifacts/external_features_aurn
```

Both a same-day nowcast source and a lag-1 forecast-safe source are evaluated.
The correction is smoothly gated off on low-regional-pollution days, directly
addressing the risk of improving episode days by overpredicting normal days.

## Traffic coverage gate

The public DfT London file is useful for road exposure, but it is made from
occasional manual survey days rather than continuous daily measurements. Audit
it with:

```bash
python experiments/external_features/audit_dft_traffic.py
```

The 2021-2024 file covers only 92-105 distinct dates per year, has no weekends,
and omits winter and late-autumn months. It is therefore rejected as a daily
time-series input: filling those long gaps would create synthetic traffic data.
The machine-readable source/access status is recorded in `sources.json`.
