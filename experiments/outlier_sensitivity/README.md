# Training-target capping sensitivity test

Small, pre-registered experiment for the London standalone and hybrid U-Nets.
This is **not** a declaration that high pollution readings are measurement errors.
PM2.5 >=25 µg/m³ is a diagnostic group, not AQI and not an automatic rejection rule.

## Frozen comparison

Fit 2021–2022, score all original 2023 LAQN observations. Compare `control`,
`cap_995` and `cap_990`, in that order, with seed 42. The percentile is fitted on
finite, positively weighted **training grid-cell targets**, not test stations.
Only training supervision is capped. Observations, input-history fields, density
weights (computed from uncapped targets), source weights and target scalers remain
unchanged. No model architecture, HGB fit, new features, or losses are introduced.

The standalone schedule is the previously frozen 25-epoch sequence. The hybrid
uses its previously frozen seed-specific durations (42:16, 11:6, 22:27 epochs)
and learning rates. Alpha is fixed to the existing ensemble value 0.425, even in
the seed-42 screen; no best-epoch/alpha retuning takes place. EMA=.98, width=32,
dropout=.12, batch=8, accumulation=2, AdamW decay=1e-4, gradient clip=5 and log-space
SmoothL1 beta=1 are retained. Deterministic kernels are used for matched controls;
therefore new controls, not historic scores, are the experimental reference.

Hybrid selection uses the original un-refitted feature cube; standalone selection
and both final trainers use their existing affine feature-scaler refits. Capping
does not change these preprocessing settings. Existing clipped float16 features
and in-sample HGB training maps are retained deliberately for this limited test.

**Selection gate:** lower 2023 RMSE AND no worsening in overall MAE AND no worsening
in MAE for observed PM2.5 >=25. Choose the lowest-RMSE eligible cap per family.
If none qualifies, stop for that family and do not score new 2024 predictions.
If one qualifies, refit both it and a matched uncapped control from fresh seeds
42/11/22 on 2021–2023, average outputs in normalised-log space, freeze predictions,
then score unchanged 2024 station observations. Report a paired, seven-calendar-day
moving-block bootstrap (5,000 replicates) for the ensemble RMSE difference.
2024 is a retrospective benchmark, not a newly untouched test.

## Running locally

Use an existing CUDA-enabled Python with torch, numpy and pandas; no new virtual
environment is required. The study used Python 3.10 and PyTorch 2.10.0+cu126.
From the Aitken root, replace the input paths below with the existing London bundle:

```powershell
python -m unittest discover -s experiments/outlier_sensitivity -p 'test_*.py' -v
python -u experiments/outlier_sensitivity/run_experiment.py --data-dir ../code/pm25_london_bundle/data/processed/london_1km_daily --reference-bundle ../code/pm25_london_bundle --output-dir artifacts/outlier_sensitivity_20260909 --gpu-hours 2
```

`python` must resolve to the CUDA-enabled installation (on the study machine this
is the installed Python310 executable, not the default Python312 executable).
No source imports execute from the old London directory. Explicit input paths are
read-only; SHA-256 hashes are checked before/after. The runner rejects output paths
outside Aitken and overlapping input directories. Checkpoints, data and local logs
are already excluded by the repository's `.gitignore`.

Run one GPU process. Reusing the same output directory resumes full optimizer,
EMA, shuffle and RNG state, checks contracts, and skips verified completed fits.
The cumulative budget is at most 7,200 seconds of training/inference wall time
(checked at epoch boundaries, so a final epoch can overrun slightly). An exhausted
budget produces a partial report; do not reset its ledger to bypass the limit.
The old dashboard, model table, presentation and original London bundle are not
updated. Publish only code/tests, provenance, aggregate reports and selected figures.

## Outputs

`summary.json` contains screening metrics and the pass/fail selection decision.
`selection_frozen.json` is written before any new 2024 scoring. Each trial retains
its cap/affected count, uncapped preprocessing, checkpoint, daily metrics and
station predictions. `input_integrity.json` proves read-only source files match.
Results must include unsuccessful treatments; no threshold search using 2024.

Reference: [SciPy winsorisation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.winsorize.html).
