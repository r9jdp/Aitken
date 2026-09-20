# Peak-preserving temporal sensitivity experiment

This is a bounded experiment, not a replacement for the published model. It
tests the user-approved plan while preserving original measurements and the
standalone/residual U-Net backbones. All writes are restricted to Aitken.

## Frozen comparison

- Train 2021–2022; screen on original 2023 LAQN station-days with seed 42.
- Controls: standalone U-Net, original hybrid, and five-channel temporal hybrid.
- Candidates for standalone and temporal hybrid: EWMA features (half-lives 1/3
  days plus fast innovation), trailing-three-day median plus innovation, and
  normalized raw squared loss on the final output (coefficient 0.1).
- The combined EWMA+loss treatment runs only if both components individually
  pass. No window search, target capping, oversampling, new data or HGB fitting.
- Gate: overall RMSE and MAE >=25 strictly improve; overall MAE and MAE <15 do
  not increase; absolute bias <15 is at most max(0.10, absolute control bias).
  Hybrid candidates must pass against BOTH hybrid controls. Thresholds are
  concentration diagnostic bands, not AQI or evidence of measurement error.
- Confirm the best eligible treatment per family with seeds 42, 11 and 22 on
  2023: ensemble passes, and RMSE beats every family control in at least two
  seeds. Do not choose a different candidate after confirmation fails.
- Freeze the decision before refitting eligible treatments/controls on 2021–2023
  and scoring 2024. That benchmark is retrospective, not an untouched test.
- Existing seed-specific epochs/LR, optimizer, masks, weights, log target
  scaling, backbone and hybrid alpha 0.425 are fixed. Added input weights start
  at zero and common tensors/RNG initialization are shared within each family.
- Raw auxiliary scale is the unweighted population SD of finite positive-weight
  training targets; both loss terms preserve existing source/LDS weights. The
  auxiliary term decodes the actual final prediction in float32, including the
  hybrid's 0.425 multiplier and original [0,500] prediction bounds. Training
  targets are not capped or smoothed.

## Provenance and limitations

The unchanged backbone, preprocessing, target-mask/source/density-weight logic,
and schedules are reused from `experiments/outlier_sensitivity`. The five
temporal channels reproduce `experiments/temporal_model/temporal_features.py`.
`methods.py` implements the additional causal features, matched initialization,
and final-output loss. `run_experiment.py` orchestrates gates, bounded/resumable
training, immutable manifests/decisions, checkpoint validation, and reporting.

The original London bundle is read-only through explicit input paths. The
phase-specific archived HGB artifacts listed in the frozen protocol are loaded
and hash-verified, not refitted. The friend's temporal experiment refitted HGB
and its raw checkpoints are not available in this checkout. Thus our temporal
control is a matched rebase, not a numerical reproduction of its reported score.

History comes from the existing standardized/clipped float16 lag-1 IDW feature,
not a reconstruction of raw concentrations. It already ends at yesterday; no
second shift is applied. Temporal filters act independently at each grid cell.
Outside-London cells are padding: extra channels stay zero there after scaling;
only in-London cells contribute to scaler moments. This padding safeguard also
applies to the rebased temporal control. Existing availability indicators/fallbacks
are retained. No same-day PM input is added. Existing environmental inputs keep
the project's retrospective rolling-nowcast definition, not a strict live forecast.

The earlier auxiliary-loss sweep used raw Huber loss on an unshrunk residual.
The new term is squared error on the final composed prediction. Any improvement
is an empirical result, not proof that a particular atmospheric or noise process
caused the original errors. The 2023 validation year has also been used before.

## Run with the existing global Python/CUDA installation

From the Aitken repository root in PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -m unittest discover -s experiments/temporal_peak_sensitivity -p 'test_*.py' -v
python -u experiments/temporal_peak_sensitivity/run_experiment.py `
  --data-dir ../code/pm25_london_bundle/data/processed/london_1km_daily `
  --reference-bundle ../code/pm25_london_bundle `
  --output-dir artifacts/temporal_peak_sensitivity/run_20260920 `
  --gpu-hours 2
```

Use the existing Python 3.10 interpreter if `python` points to an installation
without CUDA PyTorch. No new environment or package installation is required.
`--check-only` hashes inputs and validates feature/preprocessing contracts without
training. Repeating the exact command resumes valid checkpoints; a changed
recipe/code/data hash rejects reuse. Use a new run directory for changed code.

The budget charges training/inference elapsed wall time at minibatch boundaries,
including consumed partial epochs. Setup, hashing, report generation and CPU
tests are separate. The stop latency is at most the currently running batch;
partial epochs restart from their last complete checkpoint with prior compute
still charged. No automatic extension beyond two GPU-hours is allowed.

## Outputs

Large checkpoint/map/prediction files stay in the ignored experiment artifact
directory. The curated research/results report, metrics, audit and plots are
written only to `comparison/temporal_peak_sensitivity`. The report includes
negative/partial results, error bands, rapid rises/falls, predicted-value
calibration, daily plots and paired seven-day block-bootstrap intervals (5000
resamples). Intervals concern these fitted models and correlated day blocks,
not all training randomness or an independent test.

The official model, dashboard, published performance table and presentation are
not modified or automatically promoted by this experiment.
