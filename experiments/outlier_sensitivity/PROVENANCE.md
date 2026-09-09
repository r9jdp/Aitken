# Provenance and scope

The experiment is adapted from the existing, locally completed London project.
It does not import its training scripts at runtime or modify their files.

- `london_unet_model.py`: complete unchanged backbone from the old bundle.
- `frozen_helpers.py`: unchanged `inverse_target` from `train_london_unet.py`;
  `gaussian_kernel` and `derive_lds_lookup` from `tune_london_unet_2023.py`;
  `fit_preprocessing`, `station_rows` and `gather` from
  `train_london_standalone_unet.py`. Only the module/import wrapper is new.
- `run_experiment.py`: scoped runner adapted from standalone and
  `train_london_unet_3year_final.py` training logic, with capping, deterministic
  pairing, input/output safety, validation gating and cumulative budget tracking.
- `frozen_protocol.json`: unchanged schedules recovered from the standalone
  protocol and hybrid three-year `dry_run_protocol.json`; reference artifact
  paths and SHA-256 values, source module hashes and hybrid target scalers.

The read-only source bundle is supplied using `--reference-bundle`; all new files
belong to Aitken. No atmospheric explanation for a pollution episode is assumed.
The caps are transformations of experimental training supervision, **not corrected
ground truth**, and are never applied to evaluation observations or shown as real
measurements. Histories still use the original earlier-day observations.

Original scientific limitations remain: sparse station supervision, mixed-source
labels, prior examination of 2024, retrospective environmental inputs, clipped
float16 covariates and in-sample HGB maps during residual training. This experiment
does not change those limitations or claim production promotion.
