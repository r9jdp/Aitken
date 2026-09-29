# Local data registry

Large observations and prepared arrays in this directory are Git-ignored.
This registry is tracked; raw data are not published by this update.

## AURN extension imported 30 September 2026

- Local directory: `data/external_features/aurn/` relative to Aitken.
- Input archive: the user-supplied `Downloads/aurn.zip`.
- ZIP SHA-256: `c1743c290e88a12a5108d1dd094e34f969c5574a84d6cdbd61eb6df683cdc13a`.
- 63 payload files; 77,785,649 uncompressed bytes. Mac resource forks excluded.
- 18 selected stations; 15 usable stations; 16,413 valid station-days.
- All 1,461 dates in 2021-2024 have observations from 6-15 regional sites.
- `aurn_features.npz` contains eight daily inputs, shape `(1461, 8)`.
- Features and source files preserve the sender's exact bytes. Local audit
  reproduced daily observations and prepared features from the shared raw files.

These are **additional regional inputs**, not extra London target labels. The
original processed London cube at
`../code/pm25_london_bundle/data/processed/london_1km_daily` remains read-only
and unchanged. Do not append eight channels to its model input without a
separately defined training contract: the shared experiment applies an external
day-level correction after the temporal hybrid's prediction.

Read the [local import and performance audit](../comparison/aurn_import_20260930/REPORT.md)
before using the new result. The upwind feature has a documented wind-scaling
issue. The additional reproduction supplement has now been imported; rerunning
the correction confirms RMSE 3.2309 and R2 0.6471 from the supplied temporal
predictions. This is not an independent regeneration of the neural predictions.
The original model, dashboard and paper have not been promoted.

## Repeat import and verification

From Aitken, using the existing global Python installation:

```powershell
python scripts/import_aurn_bundle.py --archive C:/Users/Rajdeep/Downloads/aurn.zip --bundle ../code/pm25_london_bundle
python -m unittest discover -s scripts -p test_import_aurn_bundle.py -v
python -m unittest discover -s experiments/external_features -p "test_*.py" -v
```

The four supplement files are installed under their original `artifacts/`
paths, unchanged and Git-ignored. To repeat the score audit without overwriting
the sender's outputs:

```powershell
python scripts/reproduce_aurn_supplement.py --archive C:/Users/Rajdeep/Downloads/AURN_REPRODUCTION_SUPPLEMENT_20260930_READY.zip --bundle ../code/pm25_london_bundle
python -m unittest discover -s scripts -p "test_*aurn*.py" -v
```

Local rerun results are under `artifacts/aurn_reproduction_20260930/`.

The importer is pinned to the published September 30 feature checksum. Repeating
it skips identical files and refuses to overwrite different existing bytes.
Original files and supplied ZIP are never changed. Audit outputs go to
`artifacts/aurn_import_20260930/`, also Git-ignored.

The friend's `manifest.json` preserves its original Mac paths as provenance;
these are not used as local extraction destinations. Source attribution is
[DEFRA UK-AIR AURN](https://uk-air.defra.gov.uk/networks/network-info?view=aurn).
