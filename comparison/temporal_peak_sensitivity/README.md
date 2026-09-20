# Peak-preserving temporal sensitivity test

This folder records a controlled follow-up, **not a new official model release**.
The experiment adds summaries of historical inputs or a small final-output loss
term. It does not smooth, cap, remove or replace the real PM2.5 observations.

- [Plain-language verdict and compact comparison](VERDICT.md)
- [Full results, error bands, calibration and plots](REPORT.md)
- [Recomputed aggregate metrics](metrics.json)
- [Independent saved-run audit and uncertainty intervals](audit.json)
- [Numeric-precision rendering audit](render_audit.json)
- [Frozen experiment design and training instructions](../../experiments/temporal_peak_sensitivity/README.md)

## How to reproduce the checks

Run these commands from the Aitken repository using the existing global Python
installation. The original London bundle is an explicit **read-only** input.
Do not run the audit while training is active.

```powershell
python comparison/temporal_peak_sensitivity/audit_saved_run.py `
  --run-dir artifacts/temporal_peak_sensitivity/run_20260920_v2 `
  --data-dir ../code/pm25_london_bundle/data/processed/london_1km_daily `
  --bootstrap-all

python comparison/temporal_peak_sensitivity/render_report.py `
  --run-dir artifacts/temporal_peak_sensitivity/run_20260920_v2 `
  --data-dir ../code/pm25_london_bundle/data/processed/london_1km_daily `
  --audit comparison/temporal_peak_sensitivity/audit.json

python -m unittest discover -s experiments/temporal_peak_sensitivity -p 'test_*.py' -v
python -m unittest comparison.temporal_peak_sensitivity.test_render_report -v
python comparison/temporal_peak_sensitivity/audit_saved_run.py --self-test
```

Large checkpoints, full-grid maps and station-level predictions remain in the
Git-ignored experiment directory. This folder publishes aggregates and figures,
not raw monitoring observations or credentials.

## Why there is a separate report renderer

The frozen trainer's CSV export introduces tiny float32 round-trip differences.
These can trigger its very tight consistency check for a signed bias near zero.
The renderer first requires a passed independent audit. It then validates every
saved CSV row against the original station file and saved prediction map,
restoring their exact numeric precision **only in memory** before calling the
unchanged report generator. It neither relaxes the experiment's selection rules
nor edits any observation, prediction file, model or frozen training code.

The audit independently reconstructs the saved metrics and selection decisions.
Its seven-calendar-day block intervals use 5,000 paired resamples and the
runner's verified CSV precision. Selection-year intervals are descriptive and
conditional on those fitted models; they do not include selection or training
seed uncertainty and are not evidence from an untouched test.

The existing published model comparison, dashboard and presentation are not
updated by this experiment. The hand-written verdict supplements, rather than
replaces, the reproducible full report.
